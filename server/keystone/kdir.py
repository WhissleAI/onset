"""Keystone director v1 (KDIR=1): a learned policy over the answer graph that decides, per question, among

    local      serve the best local candidate (typed key / hash / question NN / dense passages)
    local_rc   serve it AND re-check it in the background on the free lanes (Wikidata facts, the local leads index,
               the Wikipedia summary); a disagreement corrects the answer and lowers the cell's trust
    free       try the free lanes first (cards / Wikidata / leads / summary), pay only if they come back weak
    pay        today's path (the router's plan: a free lane when it planned one, else the paid web lane)
    wait       (per-word only, eval) defer one more word

It replaces the VOI gate's single "p(local is no worse than paying) >= theta" with calibrated outcome models and an
expected-cost rule (eval/keystone/kdir_train.py fits them on the counterfactually labelled workload):

    p_ok     p(serving candidate c locally is right | lookup features, graph state, Arc / director signals)
    p_catch  p(the free lanes would answer this question right | question features)   (re-check / free-first)

    J(local)    = R (1 - p_ok)
    J(local_rc) = eps + (1 - p_ok) (C p_catch + R (1 - p_catch))
    J(free)     = R q_wrong + (1 - q_strong)          q from the free-lane model; not for live / fresh questions
    J(pay)      = paid(plan)                           1 for a web plan, 0 when the router's plan is a free lane

R (a wrong answer, in paid calls), C (a corrected-after answer) and eps are chosen on DEV by regret, then frozen.
Features are computed here (one source of truth for the replay and the app). The models are tiny gradient-boosted
trees evaluated in numpy (kdir_model.json, tens of KB). Any failure falls back to the VOI gate.

Per-cell trust (the feedback loop): re-check confirmations raise it, re-check catches and spoken corrections lower it;
it is a model input. Learned per-relation TTLs (ttl_model.json, eval/keystone/fit_ttl.py) replace the fixed
90 d / 30 d / 10 min when present.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import voi

MODEL_PATH = Path(__file__).with_name("kdir_model.json")
TTL_PATH = Path(__file__).with_name("ttl_model.json")
ACTIONS = ["pay", "local", "local_rc", "free"]
INTENTS = ["qa_factoid", "qa_definition", "weather_query", "news_query", "qa_stock", "qa_currency",
           "recommendation_locations", "datetime_query"]
ENT = {"person": ("person",), "place": ("place", "city", "country", "location", "gpe"),
       "org": ("organization", "org", "company", "team")}
VCL = ["stable", "slow", "live", "general"]
LANES = ["card", "facts", "web", "wait"]
DIR = ["d_eot", "d_fire", "d_addr", "d_r_card", "d_r_facts", "d_r_web", "d_r_none", "conf_min", "fconf_min"]

Q_NAMES = (["q_live", "q_fresh", "q_key_known", "q_key_hop", "q_has_edge_any", "q_subj_known", "q_n_cells",
            "q_nwords", "q_followup", "q_intent_p", "q_act_q", "q_act_cmd", "q_n_slots", "q_n_nums", "q_plan_conf",
            "q_leads_top", "q_leads_gap", "q_n_cands"] + [f"q_in_{x}" for x in INTENTS] +
           [f"q_ent_{x}" for x in ENT] + [f"q_lane_{x}" for x in LANES] + ["q_has_dir"] + [f"q_{x}" for x in DIR])
C_NAMES = (voi.NAMES + ["c_trust", "c_hits", "c_age_learned", "c_agree", "c_src_fact", "c_src_card"] +
           [f"c_vc_{x}" for x in VCL])
NAMES = Q_NAMES + C_NAMES


# ------------------------------------------------------------------------------------------- learned TTLs
_ttl: dict | None = None


def ttl_table() -> dict:
    global _ttl
    if _ttl is None:
        try:
            _ttl = json.loads(TTL_PATH.read_text()).get("ttl_s", {})
        except (OSError, ValueError):
            _ttl = {}
    return _ttl


def learned_ttl(relation: str | None, vclass: str, default: float) -> float:
    t = ttl_table()
    if relation and relation in t:
        return float(t[relation])
    return float(t.get("class:" + vclass, default))


# ------------------------------------------------------------------------------------------- features
def question_vec(q, frame, plan: dict, ks, feats: dict, cands: list, leads: tuple, dsig: dict | None) -> list[float]:
    """Question-level features. frame: server.frame.Frame (or anything with its fields); plan: {"lane", "conf"};
    leads: (top cosine, gap) of the local leads index for the question vector; dsig: director outputs or None."""
    key = q.key
    has_edge_any = 0.0
    if key is not None and key.complete and not key.subject.startswith("@"):
        node = ks.g.resolve(key.subject)
        has_edge_any = float(bool(node and ks.g.edges.get((node, key.relation))))
    intent = (getattr(frame, "intent", "") or "").split(".")[-1]
    acts = set(getattr(frame, "acts", ()) or ())
    et = [str(t).lower() for t in (getattr(frame, "ent_types", ()) or ())]
    x = [float(q.live), float(bool(getattr(frame, "when", ""))), feats.get("key_known", 0.0),
         float(bool(key is not None and key.subject.startswith("@"))), has_edge_any, feats.get("subj_known", 0.0),
         feats.get("n_cells", 0.0) / 10.0, min(len(q.words), 20) / 20.0,
         float(bool(getattr(frame, "carried", "") or getattr(frame, "inline", ""))),
         float(getattr(frame, "intent_p", 0.0) or 0.0), float("question" in acts), float("command_request" in acts),
         min(len(q.slots), 4) / 4.0, min(len(getattr(frame, "num_types", ()) or ()), 3) / 3.0,
         float(plan.get("conf", 0.0) or 0.0), float(leads[0]), float(leads[1]), min(len(cands), 5) / 5.0]
    x += [float(intent == i) for i in INTENTS]
    x += [float(any(t in v for t in et)) for v in ENT.values()]
    x += [float(plan.get("lane") == ln) for ln in LANES]
    x += [1.0 if dsig else 0.0] + [float((dsig or {}).get(k, 0.0)) for k in DIR]
    return x


def cand_vec(c, feats: dict, ks, cands: list, now: float) -> list[float]:
    if c.path == "shared_q":         # another user's verified question -> answer cell: a question-NN candidate
        x = voi.vector({**feats, "qnn_cos": c.score}, "qnn")
        x[voi.NAMES.index("eq_p")] = max(0.0, c.eq_p)
        x[voi.NAMES.index("has_eq")] = 1.0 if c.eq_p >= 0 else 0.0
    else:
        x = voi.vector(feats, c)
    top = c.cells[0][0] if c.cells else None
    trust = ks_trust(ks, top.id) if top is not None else 0.0
    hits = math.log1p(top.hits) if top is not None else 0.0
    if top is not None:
        rel = (top.arc or {}).get("relation") if top.key else None
        ttl = learned_ttl(rel, top.vclass, top.ttl or 86400.0)
        age = min(1.0, max(0.0, now - top.t) / max(ttl, 1.0))
    else:
        age = 1.0
    agree = sum(1 for o in cands if o is not c and o.cells and top is not None and o.cells[0][0].id == top.id)
    src = (top.source or "").lower() if top is not None else ""
    vc = top.vclass if top is not None else "general"
    x += [trust, hits, age, min(agree, 3) / 3.0, float(src in ("wikidata", "wiktionary")),
          float(vc == "live" and src not in ("wikipedia", "local wikipedia"))]
    x += [float(vc == v) for v in VCL]
    return x


def ks_trust(ks, cell_id: int) -> float:
    return float(getattr(ks, "trust", {}).get(cell_id, 0.0))


def feedback(ks, cell_id: int, signal: str):
    """Per-cell trust (log-odds-like): re-check confirmed +1, verified by a cited source +0.5, re-check caught an
    error -2, spoken correction -2. Bounded to [-4, 4]."""
    d = {"confirm": 1.0, "verified": 0.5, "caught": -2.0, "correction": -2.0}.get(signal, 0.0)
    if not hasattr(ks, "trust"):
        ks.trust = {}
    ks.trust[cell_id] = max(-4.0, min(4.0, ks.trust.get(cell_id, 0.0) + d))


# ------------------------------------------------------------------------------------------- tiny tree ensembles
class Trees:
    """A gradient-boosted tree ensemble (binary logistic) as flat arrays: per node feature, threshold, left, right,
    leaf value (feature -1 = leaf). Exported from xgboost by eval/keystone/kdir_train.py. Optional Platt calibration
    (a, b): p = sigmoid(a * margin + b)."""

    def __init__(self, d: dict):
        self.base = float(d.get("base", 0.0))
        self.trees = [(np.asarray(t["f"], np.int32), np.asarray(t["t"], np.float64), np.asarray(t["l"], np.int32),
                       np.asarray(t["r"], np.int32), np.asarray(t["v"], np.float64), np.asarray(t["m"], np.int32))
                      for t in d["trees"]]
        self.cal = d.get("cal") or [1.0, 0.0]
        self.cols = d.get("cols")            # feature indices into NAMES this model reads (None = all)

    def margin(self, x: np.ndarray) -> float:
        s = self.base
        for f, t, l, r, v, m in self.trees:
            i = 0
            while f[i] >= 0:
                xv = x[f[i]]
                if xv != xv:                 # NaN: the default (missing) branch
                    i = m[i]
                else:
                    i = l[i] if xv < t[i] else r[i]
            s += v[i]
        return s

    def p(self, x) -> float:
        x = np.asarray(x, np.float64)
        if self.cols is not None:
            x = x[self.cols]
        z = self.cal[0] * self.margin(x) + self.cal[1]
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


class Policy:
    def __init__(self, d: dict):
        self.ok = Trees(d["ok"])
        self.catch = Trees(d["catch"]) if d.get("catch") else None
        self.free_strong = Trees(d["free_strong"]) if d.get("free_strong") else None
        self.R = float(d.get("R", 10.0))
        self.C = float(d.get("C", 1.0))
        self.eps = float(d.get("eps", 0.02))
        self.use = set(d.get("actions", ACTIONS))
        self.allow_free = True
        self.meta = d

    def decide(self, qx: list[float], cands: list, cxs: list[list[float]], plan_paid: int, free_ok_q: bool):
        """-> (action, candidate, info). qx: question features; cxs: per-candidate features (same order as cands);
        plan_paid: 1 if today's plan pays (web), 0 if it is a free lane; free_ok_q: the free-first action is allowed
        (not live / fresh, a web plan)."""
        info = {}
        best, bp = None, -1.0
        for c, cx in zip(cands, cxs):
            if c.path == "shared":
                best, bp = c, 1.0
                break
            p = self.ok.p(qx + cx)
            if p > bp:
                best, bp = c, p
        # the free lanes can only check / answer a non-live question the router did not already send there
        pc = self.catch.p(qx) if (self.catch is not None and free_ok_q) else 0.0
        info.update(p_ok=round(bp, 4), p_catch=round(pc, 4))
        J = {"pay": float(plan_paid)}
        if best is not None:
            J["local"] = self.R * (1.0 - bp)
            if "local_rc" in self.use:
                J["local_rc"] = self.eps + (1.0 - bp) * (self.C * pc + self.R * (1.0 - pc))
        if free_ok_q and plan_paid and "free" in self.use and self.free_strong is not None and self.allow_free:
            qs = self.free_strong.p(qx)
            info["p_free_strong"] = round(qs, 4)
            # served free when strong; wrong when strong but not right: q_wrong ~ max(0, q_strong - p_catch)
            J["free"] = self.R * max(0.0, qs - pc) + (1.0 - qs) * 1.0 + self.eps
        a = min(J, key=lambda k: (J[k], ACTIONS.index(k)))
        info["J"] = {k: round(v, 4) for k, v in J.items()}
        if best is not None:
            best.p = bp
        return a, (best if a in ("local", "local_rc") else None), info


_policy: Policy | None = None
_failed: str | None = None


def load(path: Path = MODEL_PATH) -> Policy | None:
    global _policy, _failed
    if _policy is None and _failed is None:
        try:
            _policy = Policy(json.loads(path.read_text()))
        except Exception as e:  # noqa: BLE001
            _failed = f"{type(e).__name__}: {e}"
    return _policy


def reset():
    global _policy, _failed, _ttl
    _policy, _failed, _ttl = None, None, None


def leads_signal(qvec) -> tuple[float, float]:
    """Top cosine and top-2 gap of the local Wikipedia leads index for this question vector (offline, numpy)."""
    if qvec is None:
        return (0.0, 0.0)
    try:
        from .. import knowledge
        ix = knowledge.leads_index()
        r = ix.search(np.asarray(qvec, np.float32)[: ix.vecs.shape[1]], 2) if len(ix) else []
    except Exception:  # noqa: BLE001
        return (0.0, 0.0)
    if not r:
        return (0.0, 0.0)
    return (r[0][1], r[0][1] - (r[1][1] if len(r) > 1 else 0.0))


def director_signal(frame) -> dict | None:
    """The search director's latest outputs on the frame (DIRECTOR=on), in the replay's feature names."""
    d = getattr(frame, "director", None)
    if not d:
        return None
    rt = d.get("route")
    out = {"d_eot": d.get("eot", 0.0), "d_fire": d.get("fire", 0.0), "d_addr": d.get("addr", 0.0)}
    if rt is not None and len(rt) >= 5:
        out.update(d_r_card=float(rt[0]), d_r_facts=float(rt[1]), d_r_web=float(rt[2]), d_r_none=float(rt[4]))
    wc = getattr(frame, "wconf", ()) or ()
    if wc:
        out["fconf_min"] = float(min(wc))
    return out
