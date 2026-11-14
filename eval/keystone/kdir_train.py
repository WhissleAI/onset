"""Train the Keystone director (server/keystone/kdir.py) on the counterfactually labelled workload and pick its operating
point on DEV by regret.

  1. log    replay the TRAIN stream (kdir_train.jsonl: held-out users and families from dev / test) under a behaviour
            policy, recording for every question the features of every local candidate and the outcome of EVERY
            action (kdir_replay.simulate(log=True)). DAgger: round 0 = the VOI gate with exploration, round 1 = the
            round-0 policy; rows are aggregated.
  2. fit    tiny gradient-boosted trees (xgboost, depth 4) for p_ok (per candidate: serving it is no worse than
            paying), p_catch (the free lanes answer this question right: a re-check would catch a wrong local
            answer / free-first would serve it), p_free_strong (the free lanes come back strong); Platt calibration
            on dev; exported as flat arrays for the numpy evaluator.
  3. pick   R (the price of a wrong answer, in paid calls) on DEV: the cheapest R whose dev regret is no worse than
            today's search cache on dev (the VOI gate's rule). Frozen for test.
  Also: an imitation policy (multiclass trees on the oracle's action) and feature / action ablations, scored the
  same way.

    .venv/bin/python eval/keystone/kdir_train.py [--rounds 2]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval/keystone"))

import numpy as np  # noqa: E402

import kdir_replay as KR  # noqa: E402
from server.keystone import kdir, voi  # noqa: E402

OUT = ROOT / "runs/index"
NQ = len(kdir.Q_NAMES)
IDX = {n: i for i, n in enumerate(kdir.NAMES)}
VOI21 = [NQ + i for i in range(len(voi.NAMES))]
GRAPH_Q = ["q_live", "q_fresh", "q_key_known", "q_key_hop", "q_has_edge_any", "q_subj_known", "q_n_cells", "q_n_cands"]
GRAPH_C = ["c_trust", "c_hits", "c_age_learned", "c_agree", "c_src_fact", "c_src_card"] + [f"c_vc_{v}" for v in kdir.VCL]
ARC_Q = [n for n in kdir.Q_NAMES if n not in GRAPH_Q]
SETS = {
    "voi21": VOI21,
    "voi21+graph": VOI21 + [IDX[n] for n in GRAPH_Q + GRAPH_C],
    "all": list(range(len(kdir.NAMES))),
    "all-no-director": [i for i, n in enumerate(kdir.NAMES) if not (n.startswith("q_d_") or n in
                                                                    ("q_has_dir", "q_conf_min", "q_fconf_min"))],
}
Q_SETS = {"voi21": [IDX[n] for n in ("q_live", "q_key_known", "q_subj_known", "q_n_cells")],
          "voi21+graph": [IDX[n] for n in GRAPH_Q],
          "all": list(range(NQ)),
          "all-no-director": [i for i in range(NQ) if not (kdir.NAMES[i].startswith("q_d_") or kdir.NAMES[i] in
                                                            ("q_has_dir", "q_conf_min", "q_fconf_min"))]}


# ------------------------------------------------------------------------------------------------ logging
def log_stream(path, policy, tag, **kw):
    cp = OUT / f"kdir_log_{tag}.pkl"
    if cp.exists():
        return pickle.loads(cp.read_bytes())
    data = KR.KData(path)
    vm = voi.load()
    t = time.time()
    rows, logs, st = KR.simulate(data, policy, log=True, voi_model=vm, theta=vm.theta, **kw)
    out = {"rows": rows, "logs": logs, "stats": st}
    cp.write_bytes(pickle.dumps(out))
    print(f"logged {tag}: {len(logs)} decisions in {time.time() - t:.0f}s", flush=True)
    return out


def datasets(logs):
    ok_X, ok_y, ok_g, q_X, catch_y, strong_y, q_g = [], [], [], [], [], [], []
    for L in logs:
        for cx, o in zip(L["cx"], L["cand"]):
            if o["y"] is None:
                continue
            ok_X.append(L["qx"] + cx)
            ok_y.append(float(o["y"]))
            ok_g.append(L["pos"])
        if L["has_free"] and not L["lof"]:
            q_X.append(L["qx"])
            catch_y.append(float(L["catch"]))
            strong_y.append(float(L["free_strong"]))
            q_g.append(L["pos"])
    return (np.array(ok_X, np.float32), np.array(ok_y), np.array(q_X, np.float32), np.array(catch_y),
            np.array(strong_y))


# ------------------------------------------------------------------------------------------------ trees
def fit_trees(X, y, Xd, yd, cols, n=150, depth=4):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
                          min_child_weight=4, reg_lambda=2.0, tree_method="hist", n_jobs=2, eval_metric="logloss",
                          early_stopping_rounds=20)
    m.fit(X[:, cols], y, eval_set=[(Xd[:, cols], yd)], verbose=False)
    d = export(m, cols)
    # Platt calibration on dev margins
    md = m.predict(Xd[:, cols], output_margin=True)
    a, b = platt(md, yd)
    d["cal"] = [a, b]
    return d


def export(m, cols) -> dict:
    """xgboost booster -> flat arrays (kdir.Trees). Thresholds use xgboost's `x < split` rule."""
    bst = m.get_booster()
    best = getattr(m, "best_iteration", None)
    dumps = bst.get_dump(dump_format="json")
    if best is not None:
        dumps = dumps[: best + 1]
    trees = []
    for js in dumps:
        t = json.loads(js)
        nodes = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for c in n.get("children", []):
                walk(c)
        walk(t)
        ids = sorted(nodes)
        pos = {k: i for i, k in enumerate(ids)}
        f, th, l, r, v, mi = [], [], [], [], [], []
        for k in ids:
            n = nodes[k]
            if "leaf" in n:
                f.append(-1), th.append(0.0), l.append(-1), r.append(-1), v.append(float(n["leaf"])), mi.append(-1)
            else:
                f.append(int(n["split"].lstrip("f"))), th.append(float(n["split_condition"]))
                l.append(pos[n["yes"]]), r.append(pos[n["no"]]), v.append(0.0), mi.append(pos[n["missing"]])
        trees.append({"f": f, "t": th, "l": l, "r": r, "v": v, "m": mi})
    cfg = json.loads(bst.save_config())
    bs = float(cfg["learner"]["learner_model_param"]["base_score"].strip("[]"))
    base = float(np.log(bs / (1 - bs))) if 0 < bs < 1 else 0.0
    return {"base": base, "trees": trees, "cols": [int(c) for c in cols]}


def platt(m, y, iters=200):
    a, b = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * m + b)))
        ga, gb = np.sum((p - y) * m), np.sum(p - y)
        w = p * (1 - p)
        H = np.array([[np.sum(w * m * m) + 1e-6, np.sum(w * m)], [np.sum(w * m), np.sum(w) + 1e-6]])
        step = np.linalg.solve(H, [ga, gb])
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-7:
            break
    return float(a), float(b)


def ece(p, y, bins=10):
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & ((p < (b + 1) / bins) if b < bins - 1 else (p <= 1))
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(e)


# ------------------------------------------------------------------------------------------------ offline eval
def offline(logs, pol: kdir.Policy, allow=("pay", "local", "local_rc", "free")):
    """Score a policy on logged decisions (the store state of the logging policy; a fast approximation used to pick R;
    the chosen point is then re-run with a real replay)."""
    n = len(logs)
    paid = reg = cor = 0
    for L in logs:
        cands = [SimpleCand(p) for p in L["paths"]]
        free_ok_q = L["has_free"] and not L["lof"]
        a, c, _ = pol.decide(L["qx"], cands, L["cx"], L["plan_paid"], free_ok_q)
        if a not in allow:
            a = "local" if a == "local_rc" and "local" in allow else "pay"
        if a in ("local", "local_rc") and c is not None:
            o = L["cand"][cands.index(c)]
            if a == "local_rc":
                reg += o["rc_regret"]
                cor += o["rc_corrected"]
            else:
                reg += o["regret"]
        elif a == "free" and "free" in L["oc"]:
            paid += L["oc"]["free"]["paid"]
            reg += L["oc"]["free"]["regret"]
        else:
            paid += L["plan_paid"]
    return {"paid_per_q": paid / n, "regret_rate": reg / n, "corrected_rate": cor / n}


class SimpleCand:
    def __init__(self, path):
        self.path = path
        self.p = 0.0


def policy_dict(ok, catch, strong, R, C=1.0, eps=0.02, actions=kdir.ACTIONS):
    return {"ok": ok, "catch": catch, "free_strong": strong, "R": R, "C": C, "eps": eps, "actions": list(actions)}


def pick_R(dev_logs, base: dict, target_regret: float, n_dev_total: int, grid=None):
    """The cheapest R (fewest paid calls) whose dev regret <= target. Rates are over every dev question (card
    questions included: n_dev_total), like the replay's."""
    grid = grid or [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 120]
    pts = []
    for R in grid:
        r = offline(dev_logs, kdir.Policy({**base, "R": R}), base.get("actions", kdir.ACTIONS))
        k = len(dev_logs) / n_dev_total
        pts.append({"R": R, "paid_per_q": round((r["paid_per_q"] * len(dev_logs) + base.get("_card_paid", 0)) /
                                                n_dev_total, 4),
                    "regret_rate": round(r["regret_rate"] * k, 4), "corrected_rate": round(r["corrected_rate"] * k, 4)})
    ok = [p for p in pts if p["regret_rate"] <= target + 1e-9]
    best = min(ok, key=lambda p: (p["paid_per_q"], p["R"])) if ok else max(pts, key=lambda p: p["R"])
    return best["R"], pts


# ------------------------------------------------------------------------------------------------ main
def train(train_logs, dev_logs, fset="all", actions=kdir.ACTIONS):
    X, y, Q, cy, sy = datasets(train_logs)
    Xd, yd, Qd, cyd, syd = datasets(dev_logs)
    ok = fit_trees(X, y, Xd, yd, SETS[fset])
    cat = fit_trees(Q, cy, Qd, cyd, Q_SETS[fset]) if len(set(cy)) > 1 and len(set(cyd)) > 1 else None
    st = fit_trees(Q, sy, Qd, syd, Q_SETS[fset]) if len(set(sy)) > 1 and len(set(syd)) > 1 else None
    ms, msd = sy > 0, syd > 0
    fo = fit_trees(Q[ms], cy[ms], Qd[msd], cyd[msd], Q_SETS[fset], n=80, depth=3) \
        if ms.sum() > 20 and len(set(cy[ms])) > 1 and len(set(cyd[msd])) > 1 else None
    rep = {"rows_ok": int(len(y)), "rows_q": int(len(cy)), "pos_ok": round(float(y.mean()), 3),
           "pos_catch": round(float(cy.mean()), 3) if len(cy) else None}
    pt = kdir.Trees(ok)
    p = np.array([pt.p(x) for x in Xd])
    rep["dev_ok_ece"], rep["dev_ok_brier"] = round(ece(p, yd), 4), round(float(np.mean((p - yd) ** 2)), 4)
    if cat is not None:
        pc = np.array([kdir.Trees(cat).p(x) for x in Qd])
        rep["dev_catch_ece"] = round(ece(pc, cyd), 4)
    d = policy_dict(ok, cat, st, 10.0, actions=actions)
    d["free_ok"] = fo
    rep["free_strong_rows"] = int(ms.sum())
    return d, rep


def main(a):
    vm = voi.load()
    t0 = time.time()
    tr0 = log_stream(OUT / "kdir_train.jsonl", "behav", "train_r0")
    dv = log_stream(OUT / "kdir_dev.jsonl", "voi", "dev_voi")
    pol, rep = train(tr0["logs"], dv["logs"])
    print("round 0", rep, flush=True)
    logs = list(tr0["logs"])
    for r in range(1, a.rounds):
        kdir.MODEL_PATH.with_name("kdir_model_r.json").write_text(json.dumps(pol))
        tr = log_stream(OUT / "kdir_train.jsonl", kdir.Policy(pol), f"train_r{r}")
        logs += tr["logs"]
        pol, rep = train(logs, dv["logs"])
        print(f"round {r}", rep, flush=True)
    pickle.dump({"train_logs": logs, "dev_logs": dv["logs"]}, open(OUT / "kdir_train_logs.pkl", "wb"))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    main(ap.parse_args())
