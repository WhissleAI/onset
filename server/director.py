"""Search director: a tiny streaming Mamba-3 that reads the voice-search event stream (every Flux update + the tagger
poll + Arc audio state) and decides, per update, whether to answer now (end of turn), fire a speculative search,
start a draft, and which words to keep in the final query.

Trained in PromptingNemo-private recipes/headstart_director (train.py; real Flux traces of ~3,000 voiced queries).
This module is the RUNTIME and the single source of truth for the step features (the recipe's build_dataset.py imports
step_record / step_feats from here, so training and serving compute the same numbers). The forward pass is numpy
(int8 weights, one f32 scale per row, dequantized at load; no torch): one Mamba-3 recurrent step per update.

DIRECTOR=on|off (config.DIRECTOR); FLUX_SOFT=director uses p_eot; a failure falls back to FLUX_SOFT=fused.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "director"
K, W = 8, 24
EVENTS = ["Update", "StartOfTurn", "EagerEndOfTurn", "TurnResumed", "EndOfTurn"]
FINE = ["discourse.intent.qa_factoid", "discourse.intent.weather_query", "discourse.intent.news_query",
        "discourse.intent.qa_stock", "discourse.intent.qa_definition", "discourse.intent.recommendation_locations",
        "discourse.intent.cooking_recipe", "intent.request_information", "intent.command", "intent.report_inform"]
ACTS = ["discourse.act.question", "discourse.act.statement", "discourse.act.command_request",
        "discourse.assistant.device_directed"]
FEATS = (["eot"] + [f"ev_{e}" for e in EVENTS] + ["pause", "nw", "conf_last", "conf_min", "conf_mean3", "fconf_last",
         "fconf_min", "changed", "n_changed", "first_changed_rel", "age_last", "revs_any", "cc_last", "cc_score",
         "cc_dist", "gate", "act_q", "slot_end", "has_content", "last_content", "n_content", "n_slots", "query_changed",
         "intent_top_p"] + [f"in_{t.split('.')[-1]}" for t in FINE] + [f"act_{t.split('.')[-1]}" for t in ACTS] +
         ["sp_mention", "sp_number", "sp_pii", "sp_score", "sp_ver", "sp_alt", "arc_cc_p", "arc_cc_missing",
          "arc_cc_recent", "arc_for_me"])
F = len(FEATS)
_P = re.compile(r"[^\w'\- ]+")


def norm(s: str) -> str:
    return _P.sub("", s.lower()).strip()


def hash_tok(w: str, buckets: int = 1 << 20) -> int:
    """Stable word hash (0 = pad); the model folds it into its own table size."""
    w = norm(w)
    return 1 + int(hashlib.blake2b(w.encode(), digest_size=4).hexdigest(), 16) % (buckets - 1) if w else 0


# ---------------------------------------------------------------------------------------------- features
def step_feats(r: dict, prev: dict | None, fc: list[float]) -> np.ndarray:
    """r: one update as a feature record (eval/conf_features.py fields); prev: the previous update of the same turn."""
    from .frame import STOP
    n = len(r["words"])
    toks = [norm(w) for w in r["words"]]
    pw = [norm(w) for w in (prev["words"] if prev else [])]
    k = 0
    while k < min(len(pw), n) and pw[k] == toks[k]:
        k += 1
    changed = k < len(pw) or n != len(pw)
    revised = k < len(pw)
    sp = None
    for s in r["spans"]:
        if s["end"] == n and s.get("family") in ("entity", "number", "pii") and s["type"] != "entity.numeral":
            if sp is None or (s.get("score") or 0) > (sp.get("score") or 0):
                sp = s
    intents = {x["tag"]: x["p"] for x in r["intent"]}
    acts = {x["tag"]: x["p"] for x in r["act"]}
    cc_last = r["cc_at"] is not None and r["cc_at"] == n - 1
    w_end = r["ends"][-1]
    conf = [1.0 if c is None else float(c) for c in r["conf"]]
    d = {"eot": r["eot"], "pause": min(max(r["pause"], 0), 1.5), "nw": min(n, 30) / 30,
         "conf_last": conf[-1], "conf_min": min(conf), "conf_mean3": float(np.mean(conf[-3:])),
         "fconf_last": fc[-1], "fconf_min": min(fc), "changed": float(changed),
         "n_changed": min(n - k, 6) / 6 if changed else 0.0, "first_changed_rel": (k / n if revised else 1.0),
         "age_last": min(r["age"][-1], 6) / 6, "revs_any": float(any(x > 0 for x in r["revs"])),
         "cc_last": float(cc_last), "cc_score": (r.get("cc_score") or 0.0) if cc_last else 0.0,
         "cc_dist": min(n - 1 - r["cc_at"], 8) / 8 if r["cc_at"] is not None else 1.0,
         "gate": float(r["gate"]), "act_q": float(r["act_question"]),
         "slot_end": float(any(s.get("family") in ("entity", "number") and s["type"] != "entity.numeral" and
                               s["end"] == n for s in r["spans"])),
         "has_content": float(r["has_content"]), "last_content": float(bool(toks) and toks[-1] not in STOP),
         "n_content": min(len(r["content_idx"]), 8) / 8, "n_slots": min(len(r["slots"]), 4) / 4,
         "query_changed": float(prev is None or prev["query"] != r["query"]),
         "intent_top_p": max(intents.values()) if intents else 0.0,
         "sp_mention": float(sp is not None and sp["type"].startswith("entity.mention")),
         "sp_number": float(sp is not None and sp.get("family") == "number"),
         "sp_pii": float(sp is not None and sp.get("family") == "pii"),
         "sp_score": (sp.get("score") or 0) if sp else 0.0, "sp_ver": (sp.get("ver") or 0) if sp else 0.0,
         "sp_alt": ((sp.get("alt") or {}).get("p") or 0) if sp else 0.0,
         "arc_cc_p": r.get("arc_cc_p") or 0.0, "arc_cc_missing": float(r.get("arc_cc_p") is None),
         "arc_cc_recent": float(r.get("arc_cc_t") is not None and r["arc_cc_t"] >= w_end - 0.6),
         "arc_for_me": r.get("arc_for_me") or 0.0}
    for e in EVENTS:
        d[f"ev_{e}"] = float(r["event"] == e)
    for t in FINE:
        d[f"in_{t.split('.')[-1]}"] = intents.get(t, 0.0)
    for t in ACTS:
        d[f"act_{t.split('.')[-1]}"] = acts.get(t, 0.0)
    return np.array([d[k] for k in FEATS], np.float32)


def step_tokens(r: dict, fc: list[float]):
    """(toks [K], wtok [W], wfeat [W, 4]): hashes of the newest K words, of the last W words + per-word Flux conf,
    fused conf, in a typed span, stop word."""
    from .frame import STOP
    ws = r["words"]
    toks = np.zeros(K, np.int64)
    tk = [hash_tok(w) for w in ws[-K:]]
    if tk:
        toks[K - len(tk):] = tk
    wtok = np.zeros(W, np.int64)
    wfeat = np.zeros((W, 4), np.float32)
    n = len(ws)
    for j, i in enumerate(range(max(0, n - W), n)):
        wtok[j] = hash_tok(ws[i])
        insp = any(s["start"] <= i < s["end"] and s.get("family") in ("entity", "number") for s in r["spans"])
        c = r["conf"][i]
        wfeat[j] = (1.0 if c is None else float(c), fc[i], float(insp), float(norm(ws[i]) in STOP))
    return toks, wtok, wfeat


def session_record(sess, f, ws: list[dict], ev: str) -> dict:
    """The live Session's state at a Flux update as a feature record (the fields eval/conf_features.py saves)."""
    from .session import gate_open
    a = sess.arc or {}
    cc_t = None
    for e in (a.get("cc_events") or []):
        if e.get("t") is not None:
            cc_t = e["t"] if cc_t is None else max(cc_t, e["t"])
    ends = [w.get("end") or 0.0 for w in ws]
    return {"words": list(sess.words), "conf": list(getattr(sess, "confs", []) or [None] * len(sess.words)),
            "ends": ends or [0.0], "revs": list(sess.track.revs), "age": list(sess.track.age), "spans": sess.spans,
            "intent": (sess.current or {}).get("intent") or [], "act": (sess.current or {}).get("act") or [],
            "cc_at": sess.cc_at, "cc_score": sess.cc_score,
            "gate": bool(gate_open(f, sess.cc_at, sess.spans) and f.has_content()),
            "act_question": f.act_question, "has_content": f.has_content(), "content_idx": list(f.content_idx),
            "slots": [list(s) for s in f.slots], "query": f.query(), "event": ev, "eot": float(sess.eot_p or 0.0),
            "pause": max(0.0, sess._audio_end - (ends[-1] if ends else 0.0)), "turn": 0,
            "arc_cc_p": a.get("cc_p"), "arc_cc_t": cc_t, "arc_for_me": a.get("for_me"),
            # v2 streams (None = not running: the model reads them as missing)
            "starts": [(t[0] or 0.0) for t in (getattr(sess, "times", None) or [])], "t": sess._audio_end,
            "x2": getattr(sess, "arc_x2", None), "aw": getattr(sess, "arc_aw", None),
            "af": getattr(sess, "arc_af", None),
            # ArcState across presses (server/arcstate.py): recorded now, read by the "state" group once retrained
            "arcstate": (sess.arcstate.features() if getattr(sess, "arcstate", None) is not None else None)}


# ---------------------------------------------------------------------------------------------- v2 input streams
# Director v2 adds four input streams to the per-update step (each its own group, so the recipe can ablate it):
#   pros  pause / prosody timing from Flux's word times (no audio needed)
#   tag   the rest of the Arc Tagger read-out: PII / safety, typed number spans, entity kinds, the router's plan
#         (lane, vertical, route conf: server/router.py on the live frame) and whether the plan just changed
#   arc   Arc Mini audio heads the assistant domain doesn't ship (overlap, speaker change, emotion, arousal, statement /
#         backchannel acts, music / laughter / degraded audio) + the audio level (server/arcmeta.ArcExtra)
#   ear   the Arc ear's OWN words next to Flux's: agreement, who is ahead in time, Arc's own endpoint
# Records without a stream (traces featurized before v2, or a live session without ARC_META) carry its `*_ok` = 0.
LANES = ["card", "facts", "web", "wait"]
VERTS = ["units", "fx", "time", "sun", "weather", "wikidata", "define", "whois", "leads"]
PII_K = ["phone_number", "email", "street_address", "card_account_number", "date_of_birth"]
NUM_K = ["date", "time_of_day", "duration", "money", "percent", "count", "measure", "ordinal", "ref_number"]
MEN_K = ["person", "organization", "place", "work_product"]
X2F = ["speech", "sil", "ovl", "degr", "music", "laugh", "noneng", "chg", "intr", "yld"]
X2U = ["st", "q", "op", "cr", "bc", "ans", "dd", "e_neu", "e_hap", "e_ang", "e_sad", "e_sur", "a_lo", "a_hi"]
GROUPS = {
    "pros": ["last_dur", "prev_dur", "lengthen", "gap_last", "gap_max", "rate", "since_start", "pause_rel"],
    "tag": ([f"pii_{k}" for k in PII_K] + ["pii_other", "pii_any", "profanity"] + [f"num_{k}" for k in NUM_K] +
            ["n_num"] + [f"men_{k}" for k in MEN_K] + [f"lane_{k}" for k in LANES] + ["local"] +
            [f"vert_{k}" for k in VERTS] + ["route_conf", "plan_changed", "skip_web"]),
    "arc": (["x2_ok"] + [f"x_{k}" for k in X2F] + ["x_nspk", "x_spk_other", "x_lvl_rel", "x_lvl3_rel", "x_utt_ok"] +
            [f"u_{k}" for k in X2U]),
    "ear": ["aw_ok", "aw_ratio", "aw_jacc", "aw_last_in", "aw_ahead", "aw_more", "aw_final_recent", "aw_open",
            "aw_n_final"],
}
GROUP_ORDER = ["pros", "tag", "arc", "ear"]
# "state": ArcState carried across presses (server/arcstate.py ArcState.features: enrolled voice / another voice,
# this question vs the voice's usual pace / hesitation / level). Not in GROUP_ORDER: the shipped director weights were
# trained without it; a retrain adds it with step_feats2(..., groups=GROUP_ORDER + ["state"]).
ARCSTATE_F = ["as_ok", "as_other", "as_sim", "as_nvoices", "as_pace_rel", "as_hes_rel", "as_level_rel", "as_turns"]
GROUPS["state"] = ARCSTATE_F
WF2 = 6          # per-word features of the pii head: in a PII span, hidden by frame._pii_hide, in an entity mention,
                 # has a digit, digit count, email-ish token


def frame_of(r: dict):
    """The tagger frame a record describes (frame.build on its words / spans / intents: == the live Session frame)."""
    from . import frame as fr
    ev = {"discourse.assistant.command_complete"} if (r.get("cc_at") is not None and
                                                       r["cc_at"] == len(r["words"]) - 1) else set()
    return fr.build(r["words"], r["spans"], {"intent": r.get("intent") or [], "act": r.get("act") or []}, ev, None,
                    local_ok=bool(r.get("local_ok")))


def plan_of(f):
    from . import router
    try:
        p = router.plan(f)
    except Exception:  # noqa: BLE001
        return None
    return p


def step_feats2(r: dict, prev: dict | None, f=None, plan=None, groups=GROUP_ORDER) -> np.ndarray:
    """The v2 streams for one update (see GROUPS). f / plan: the live frame and router plan when the Session has them
    (else rebuilt from the record). Returns the concatenation of `groups` in GROUP_ORDER."""
    d = {}
    n = len(r["words"])
    ends = r["ends"]
    starts = r.get("starts")
    if "pros" in groups:
        if starts and len(starts) == n:
            durs = [max(0.0, e - s) for s, e in zip(starts, ends)]
            gaps = [max(0.0, starts[i] - ends[i - 1]) for i in range(1, n)]
            pd_ = float(np.mean(durs[-6:-1])) if n > 1 else 0.25
            med_gap = float(np.median(gaps)) if gaps else 0.05
            d.update(last_dur=min(durs[-1], 1.0), prev_dur=min(pd_, 1.0), lengthen=min(durs[-1] / (pd_ + 0.05), 4) / 4,
                     gap_last=min(gaps[-1], 1.0) if gaps else 0.0, gap_max=min(max(gaps), 1.5) / 1.5 if gaps else 0.0,
                     rate=min(n / (ends[-1] - starts[0] + 0.1), 6) / 6,
                     since_start=min(r["t"] - starts[0], 15) / 15 if "t" in r else 0.0,
                     pause_rel=min(max(r["pause"], 0) / (med_gap + 0.1), 10) / 10)
    if "tag" in groups:
        if f is None:
            f = frame_of(r)
        if plan is None:
            plan = plan_of(f)
        sp = r["spans"]
        for k in PII_K:
            d[f"pii_{k}"] = float(any(s["type"] == f"safety.pii.{k}" for s in sp))
        d["pii_other"] = float(any(s.get("family") == "pii" and s["type"].split(".")[-1] not in PII_K for s in sp))
        d["pii_any"] = float(bool(f.redact_tok))
        from .frame import PROFANITY
        d["profanity"] = float(any(norm(w) in PROFANITY for w in r["words"]))
        for k in NUM_K:
            d[f"num_{k}"] = float(any(s["type"] == f"entity.number.{k}" for s in sp))
        d["n_num"] = min(sum(s.get("family") == "number" and s["type"] != "entity.numeral" for s in sp), 4) / 4
        for k in MEN_K:
            d[f"men_{k}"] = float(any(s["type"] == f"entity.mention.{k}" for s in sp))
        lane = plan.lane if plan is not None else ""
        for k in LANES:
            d[f"lane_{k}"] = float(lane == k)
        d["local"] = float(getattr(f, "route", "web") == "local")
        for k in VERTS:
            d[f"vert_{k}"] = float(plan is not None and plan.vertical == k)
        d["route_conf"] = float(plan.conf) if plan is not None else 0.0
        sig = (lane, plan.vertical if plan is not None else "")
        d["plan_changed"] = float(prev is None or prev.get("_plan_sig") != sig)
        d["skip_web"] = float(plan is not None and plan.skip_web)
        r["_plan_sig"] = sig
    if "arc" in groups:
        x = r.get("x2")
        if x:
            d["x2_ok"] = 1.0
            for k in X2F:
                d[f"x_{k}"] = float(x.get(k) or 0.0)
            d["x_nspk"] = min(x.get("nspk") or 0, 4) / 4
            d["x_spk_other"] = float((x.get("spk") or 0) > 0)
            lv, ref, l3 = x.get("lvl"), x.get("lvl_ref"), x.get("lvl3")
            d["x_lvl_rel"] = float(np.clip(((lv if lv is not None else -100) - (ref or -100)) / 30, -1, 1))
            d["x_lvl3_rel"] = float(np.clip(((l3 if l3 is not None else -100) - (ref or -100)) / 30, -1, 1))
            u = x.get("utt")
            d["x_utt_ok"] = float(bool(u))
            for k in X2U:
                d[f"u_{k}"] = float((u or {}).get(k) or 0.0)
    if "ear" in groups:
        aw, af = r.get("aw"), r.get("af")
        if aw is not None:
            d["aw_ok"] = 1.0
            t0 = (starts[0] if starts else (ends[0] - 0.3)) - 0.3
            fin = [u for u in (af or []) if u[2] is not None and u[2] > t0]
            arc_w = [norm(w) for u in fin for w in (u[0] or "").split()] + [norm(w[0]) for w in aw]
            arc_w = [w for w in arc_w if w]
            fw = [norm(w) for w in r["words"]]
            A, B = set(arc_w), set(w for w in fw if w)
            d["aw_ratio"] = min(len(arc_w) / max(1, n), 2) / 2
            d["aw_jacc"] = len(A & B) / max(1, len(A | B))
            d["aw_last_in"] = float(bool(fw) and fw[-1] in A)
            a_end = max([w[2] for w in aw if w[2] is not None] + [u[2] for u in fin], default=None)
            d["aw_ahead"] = float(np.clip((a_end - ends[-1]) if a_end is not None else -1.0, -1, 1))
            d["aw_more"] = float(any(w[1] is not None and w[1] > ends[-1] + 0.05 for w in aw))
            d["aw_final_recent"] = float(any(u[2] is not None and abs(u[2] - ends[-1]) <= 0.6 for u in fin))
            d["aw_open"] = float(bool(aw))
            d["aw_n_final"] = min(len(fin), 3) / 3
    if "state" in groups:
        d.update({k: float(v) for k, v in (r.get("arcstate") or {}).items() if k in ARCSTATE_F})
    names = [k for g in GROUP_ORDER + ["state"] if g in groups for k in GROUPS[g]]
    return np.array([d.get(k, 0.0) for k in names], np.float32)


def word_feats2(r: dict, f=None) -> np.ndarray:
    """[W, WF2] per word of the last W (same alignment as step_tokens' wtok): the pii head's word inputs."""
    if f is None:
        f = frame_of(r)
    ws = r["words"]
    n = len(ws)
    out = np.zeros((W, WF2), np.float32)
    for j, i in enumerate(range(max(0, n - W), n)):
        w = str(ws[i])
        in_pii = any(s.get("family") == "pii" and s["start"] <= i < s["end"] for s in r["spans"])
        in_men = any(s["type"].startswith("entity.mention") and s["start"] <= i < s["end"] for s in r["spans"])
        nd = sum(c.isdigit() for c in w)
        out[j] = (float(in_pii), float(i in f.redact_tok), float(in_men), float(nd > 0), min(nd, 10) / 10,
                  float("@" in w or "_" in w or norm(w) in ("dot", "gmail", "com", "yahoo", "hotmail", "outlook")))
    return out


# ---------------------------------------------------------------------------------------------- numpy forward
def _gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class Model:
    """Numpy streaming forward of recipes/headstart_director train.Director (verified against torch by export.py)."""

    def __init__(self, path: Path = MODEL_DIR):
        meta = json.loads((path / "director.json").read_text())
        z = np.load(path / "director.npz")
        p = {}
        for k in z.files:
            if k.endswith("::q"):
                name = k[:-3]
                p[name] = z[k].astype(np.float32) * z[name + "::s"][:, None]
            elif not k.endswith("::s"):
                p[k] = z[k].astype(np.float32)
        self.p, self.meta = p, meta
        c = meta["size"]
        self.d, self.L, self.N, self.P = c["d"], c["layers"], c["d_state"], c["head_dim"]
        self.H = self.d // self.P
        self.buckets = c["buckets"]
        self.thr = meta.get("thr", {})
        self.eps = float(np.finfo(np.float32).eps)
        self.v2 = bool(meta.get("v2"))
        self.groups = list(meta.get("groups") or [])          # v2 input streams this model reads (GROUP_ORDER order)

    def lin(self, name, x):
        y = x @ self.p[name + ".weight"].T
        b = self.p.get(name + ".bias")
        return y + b if b is not None else y

    def rms(self, name, x):
        return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + self.eps) * self.p[name + ".weight"]

    def ln(self, name, x, eps=1e-5):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps) * self.p[name + ".weight"] + self.p[name + ".bias"]

    def emb(self, t):
        t = np.where(t > 0, np.remainder(t, self.buckets), 0)
        return self.p["emb.weight"][t]

    def block(self, i, u, h, bx):
        pre = f"blocks.{i}."
        H, N, P = self.H, self.N, self.P
        nu = self.rms(pre + "norm", u)
        dt = _softplus(self.lin(pre + "dt_proj", nu))                       # H
        A = -np.exp(self.p[pre + "A_log"])
        alpha = np.exp(dt * A)
        lam = _sigmoid(self.lin(pre + "lam_proj", nu))
        gamma, beta = lam * dt, (1 - lam) * dt * alpha
        Bm = self.rms(pre + "B_norm", self.lin(pre + "B_proj", nu).reshape(H, 1, N)).transpose(0, 2, 1) + self.p[pre + "B_bias"]
        Cm = self.rms(pre + "C_norm", self.lin(pre + "C_proj", nu).reshape(H, 1, N)).transpose(0, 2, 1) + self.p[pre + "C_bias"]
        x = self.lin(pre + "x_proj", nu).reshape(H, P, 1)
        theta = self.lin(pre + "theta_proj", nu).reshape(H, N // 2) * dt[:, None]
        h = np.zeros((H, N, P), np.float32) if h is None else h
        s = alpha[:, None, None] * h
        if bx is not None:
            s = s + beta[:, None, None] * bx
        c, sn = np.cos(theta)[:, :, None], np.sin(theta)[:, :, None]     # H, N/2, 1 (same angle for every P)
        a_, b_ = s[:, 0::2, :], s[:, 1::2, :]
        r = np.empty_like(s)
        r[:, 0::2, :] = c * a_ - sn * b_
        r[:, 1::2, :] = sn * a_ + c * b_
        bxn = Bm @ x.transpose(0, 2, 1)                                     # H,N,1 @ H,1,P -> H,N,P
        h = r + gamma[:, None, None] * bxn
        y = (Cm.transpose(0, 2, 1) @ h).reshape(-1)                         # H,1,N @ H,N,P
        out = self.lin(pre + "out_proj", y * _silu(self.lin(pre + "in_proj", nu)))
        return out, h, bxn

    def step(self, state, feats, toks, wtok, wfeat, wfeat2=None):
        p = self.p
        e = self.emb(toks)
        m = (toks > 0).astype(np.float32)[:, None]
        bag = (e * m).sum(0) / max(1.0, float(m.sum())) * float(p.get("use_tok", np.ones(1))[0])
        x = self.lin("inp", np.concatenate([(feats - p["mu"]) / p["sd"], bag]))
        state = state or [(None, None)] * self.L
        new = []
        for i in range(self.L):
            h, bx = state[i]
            y, h, bx = self.block(i, x, h, bx)
            x = x + y
            mm = f"mlps.{i}."
            x = x + self.lin(mm + "3", _gelu(self.lin(mm + "1", self.ln(mm + "0", x))))
            new.append((h, bx))
        hN = self.ln("norm", x)
        o = self.lin("head", hN)
        we = self.emb(wtok)
        keep = (we @ self.lin("wq", hN)) + self.lin("wf", wfeat)[:, 0] + self.lin("wb", hN)[0]
        keep = np.where(wtok > 0, keep, -1e4)
        r = {"eot": float(_sigmoid(o[0])), "fire": float(_sigmoid(o[1])), "draft": float(_sigmoid(o[2])),
             "keep": _sigmoid(keep)}
        if self.v2:
            r["addr"] = float(_sigmoid(o[3]))
            z = o[4:9] - o[4:9].max()
            e = np.exp(z)
            r["route"] = e / e.sum()                        # card, facts, web, local, none
            wf = np.concatenate([wfeat, wfeat2 if wfeat2 is not None else np.zeros((len(wfeat), WF2), np.float32)], 1)
            pii = (we @ self.lin("wq2", hN)) + self.lin("wf2", wf)[:, 0] + self.lin("wb2", hN)[0]
            r["pii"] = _sigmoid(np.where(wtok > 0, pii, -1e4))
        return r, new


_model = None
_failed = None


def model() -> Model | None:
    global _model, _failed
    if _model is None and _failed is None:
        try:
            _model = Model()
        except Exception as e:  # noqa: BLE001 (no model shipped / bad file: the session falls back to fused)
            _failed = f"{type(e).__name__}: {e}"
    return _model


ROUTES = ["card", "facts", "web", "local", "none"]


def _same_tail(out: dict, words: list) -> bool:
    ow = [norm(w) for w in out.get("words") or []]
    return bool(ow) and out.get("n") == len(words) and ow == [norm(w) for w in words][-len(ow):]


def apply_pii(f, out: dict | None, thr: float | None = None) -> bool:
    """DIRECTOR_PII=1: the pii head's hidden words replace frame._pii_hide's (profanity stays hidden). Only when the
    director has read exactly these words; else the rule stands. Slots were already built from the rule's spans."""
    if not out or "pii" not in out or not _same_tail(out, f.words):
        return False
    m = model()
    thr = thr if thr is not None else float(((m.thr if m else {}) or {}).get("director_pii", 0.5))
    n = out["n"]
    k = min(n, W)
    hide = {n - k + j for j in range(k) if out["pii"][j] >= thr}
    # entity names stay searchable ("colorado", "starbucks", "oakland" next to a phone number): a word inside a
    # tagged entity mention is never hidden unless it carries digits (offline: over-hide of entity words 10 % -> 0 %)
    men = {i for sp in (getattr(f, "spans", None) or []) if str(sp.get("type", "")).startswith("entity.mention")
           for i in range(sp["start"], sp["end"])}
    hide = {i for i in hide if not (i in men and not any(c.isdigit() for c in str(f.words[i])))}
    from .frame import PROFANITY
    hide |= {i for i, w in enumerate(f.words) if norm(w) in PROFANITY}
    f.redact_tok = frozenset(hide)
    return True


class Stream:
    """One utterance's director state (Session keeps one per question)."""

    def __init__(self):
        self.state = None
        self.prev = None
        self.out: dict | None = None
        self.turn_words = None

    def step(self, r: dict, fc: list[float], f=None, plan=None) -> dict | None:
        """r: session_record (+ for v2: "starts", and "x2" / "aw" / "af" when those streams run); f / plan: the live
        frame and router plan (v2 tag stream; rebuilt from r when not given)."""
        m = model()
        if m is None or not r["words"]:
            return None
        x = step_feats(r, self.prev, fc)
        wfeat2 = None
        if m.v2:
            if f is None:
                f = frame_of(r)
            if m.groups:
                x = np.concatenate([x, step_feats2(r, self.prev, f, plan, groups=m.groups)])
            wfeat2 = word_feats2(r, f)
        toks, wtok, wfeat = step_tokens(r, fc)
        self.out, self.state = m.step(self.state, x, toks, wtok, wfeat, wfeat2)
        self.out["n"] = len(r["words"])
        self.out["words"] = list(r["words"][-W:])
        self.prev = r
        return self.out

    # ---------------------------------------------------------------- v2 read-outs (every hook defaults OFF)
    def addressed(self) -> float | None:
        """p(the utterance is a request to the assistant) at the newest update (open mic gate), None without v2."""
        return None if not self.out or "addr" not in self.out else self.out["addr"]

    def route(self) -> tuple[str, float] | None:
        """(lane, p) the router will take for this question's final words, called from the partial words."""
        if not self.out or "route" not in self.out:
            return None
        k = int(np.argmax(self.out["route"]))
        return ROUTES[k], float(self.out["route"][k])

    def pii_hide(self, thr: float) -> set:
        """Indices (into the current words) the pii head keeps out of the search."""
        if not self.out or "pii" not in self.out:
            return set()
        n = self.out["n"]
        p = self.out["pii"]
        off = n - min(n, W)
        return {off + j for j in range(min(n, W)) if p[j] >= thr}
