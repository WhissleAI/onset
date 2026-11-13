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
            "arc_cc_p": a.get("cc_p"), "arc_cc_t": cc_t, "arc_for_me": a.get("for_me")}


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

    def step(self, state, feats, toks, wtok, wfeat):
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
        return {"eot": float(_sigmoid(o[0])), "fire": float(_sigmoid(o[1])), "draft": float(_sigmoid(o[2])),
                "keep": _sigmoid(keep)}, new


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


class Stream:
    """One utterance's director state (Session keeps one per question)."""

    def __init__(self):
        self.state = None
        self.prev = None
        self.out: dict | None = None
        self.turn_words = None

    def step(self, r: dict, fc: list[float]) -> dict | None:
        m = model()
        if m is None or not r["words"]:
            return None
        x = step_feats(r, self.prev, fc)
        toks, wtok, wfeat = step_tokens(r, fc)
        self.out, self.state = m.step(self.state, x, toks, wtok, wfeat)
        self.out["n"] = len(r["words"])
        self.prev = r
        return self.out
