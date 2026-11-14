"""KDIR ablation, step by step, on dev and test (old 800-question streams s2 / s1 and the new held-out kdir streams).

  step 0  baselines: no reuse, today's search cache, the shipped VOI gate (theta 0.95), oracles (local only / all actions)
  step 1  bigger workload: the VOI gate (same 21 features, same logistic model) refit on the 7k-question train stream,
          theta re-chosen on dev by regret
  step 2  richer actions: pay / local / local + re-check / free-first, expected-cost rule over trees on the SAME 21
          lookup features (2a: pay/local only, 2b: + re-check, 2c: + free-first)
  step 3  learned policy: + graph state (3a), + Arc / director signals (3b = all); 3c the same without the director
          signals; 3d imitation of the oracle's action (multiclass trees) instead of the expected-cost rule
  step 4  feedback loop: per-cell trust from re-checks + simulated spoken corrections, and the p_ok trees refit online
          every 500 questions with the feedback labels
  step 5  learned volatility: relation TTLs from ttl_model.json (store expiry + the age feature)
  step 6  network effect: per-user / + shared edges / + shared question -> answer cells / pooled (VOI vs KDIR)

Operating points (theta / R) are chosen on DEV by regret (no worse than today's cache on dev) and frozen for test.

    .venv/bin/python eval/keystone/kdir_eval.py --part train|ablation|network|curves|plots
"""
from __future__ import annotations

import argparse
import copy
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
import kdir_train as T  # noqa: E402
from server.keystone import kdir, voi  # noqa: E402

OUT = ROOT / "runs/index"
RES = OUT / "kdir_results.json"
STREAMS = {"old_dev": OUT / "stream_s2.jsonl", "old_test": OUT / "stream.jsonl", "dev": OUT / "kdir_dev.jsonl",
           "test": OUT / "kdir_test.jsonl"}


def res_load():
    return json.loads(RES.read_text()) if RES.exists() else {}


def res_save(r):
    RES.write_text(json.dumps(r, indent=1))


def logs_for(tag, path, policy="voi"):
    return T.log_stream(path, policy, tag)


# ------------------------------------------------------------------------------------------------ offline scoring
def card_paid(rows):
    return sum(r["paid"] for r in rows if r["tier"] == "card")


def offline_fn(L_all, n_total, fn):
    paid = reg = cor = 0
    for L in L_all:
        a, j = fn(L)
        if a in ("local", "local_rc") and j is not None:
            o = L["cand"][j]
            reg += o["rc_regret"] if a == "local_rc" else o["regret"]
            cor += o["rc_corrected"] if a == "local_rc" else 0
        elif a == "free" and "free" in L["oc"]:
            paid += L["oc"]["free"]["paid"]
            reg += L["oc"]["free"]["regret"]
        else:
            paid += L["plan_paid"]
    return paid / n_total, reg / n_total, cor / n_total


def voi_fn(model, theta):
    def fn(L):
        best, bp = None, -1.0
        for j, (p, cx) in enumerate(zip(L["paths"], L["cx"])):
            if p == "shared":
                return "local", j
            pr = model.p(cx[:len(voi.NAMES)])
            if pr > bp:
                best, bp = j, pr
        return ("local", best) if best is not None and bp >= theta else ("pay", None)
    return fn


def kdir_fn(pol, allow):
    def fn(L):
        cands = [T.SimpleCand(p) for p in L["paths"]]
        a, c, _ = pol.decide(L["qx"], cands, L["cx"], L["plan_paid"], L["has_free"] and not L["lof"])
        if a not in allow:
            a = "local" if a == "local_rc" and "local" in allow else "pay"
        return a, (cands.index(c) if c is not None else None)
    return fn


def choose(points, target):
    ok = [p for p in points if p["regret_rate"] <= target + 1e-9]
    return min(ok, key=lambda p: (p["paid_per_q"], -p["knob"])) if ok else max(points, key=lambda p: p["knob"])


# ------------------------------------------------------------------------------------------------ train everything
def part_train(a):
    r = res_load()
    tr = logs_for("train_r0", STREAMS["dev"].with_name("kdir_train.jsonl"), "behav")
    dv = logs_for("dev_voi", STREAMS["dev"], "voi")
    odv = logs_for("old_dev_voi", STREAMS["old_dev"], "voi")
    n_dev = len(dv["rows"])
    cache_dev = KR.baseline(KR.KData(STREAMS["dev"]), "a", 58.7)
    cache_odev = KR.baseline(KR.KData(STREAMS["old_dev"]), "a", 58.7)
    r["cache_dev"], r["cache_old_dev"] = cache_dev, cache_odev
    target = {"dev": cache_dev["regret_rate"], "old_dev": max(cache_odev["regret_rate"], 0.0)}
    devsets = {"dev": (dv["logs"], len(dv["rows"]), card_paid(dv["rows"])),
               "old_dev": (odv["logs"], len(odv["rows"]), card_paid(odv["rows"]))}
    print("targets (cache regret on dev)", target, flush=True)

    def pick(name, fn_of_knob, knobs):
        """knob chosen on dev (both dev sets must hold regret <= today's cache)."""
        pts = {}
        for k in knobs:
            row = {"knob": k}
            ok_all = True
            for dn, (logs, n, cp) in devsets.items():
                p, rg, co = offline_fn(logs, n, fn_of_knob(k))
                row[dn] = {"paid_per_q": round(p + cp / n, 4), "regret_rate": round(rg, 4), "corrected_rate": round(co, 4)}
                ok_all &= rg <= target[dn] + 1e-9
            row["ok"] = ok_all
            pts[k] = row
        ok = [v for v in pts.values() if v["ok"]]
        best = min(ok, key=lambda v: (v["dev"]["paid_per_q"] + v["old_dev"]["paid_per_q"], -v["knob"])) if ok else \
            max(pts.values(), key=lambda v: v["knob"])
        print(f"{name}: knob {best['knob']} dev {best['dev']} old_dev {best['old_dev']}", flush=True)
        return best["knob"], list(pts.values())

    models = {}
    # step 1: VOI refit on the big train stream
    X, y = [], []
    for L in tr["logs"]:
        for cx, o in zip(L["cx"], L["cand"]):
            if o["y"] is not None:
                X.append(cx[:len(voi.NAMES)])
                y.append(float(o["y"]))
    w = voi.fit(np.array(X), np.array(y), l2=1.0)
    vb = voi.VOI(list(map(float, w)))
    th, pts = pick("step1 voi_big", lambda t: voi_fn(vb, t), [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97,
                                                               0.98, 0.99])
    models["voi_big"] = {"w": list(map(float, w)), "theta": th, "curve": pts}
    # steps 2-3: trees over feature sets, actions
    grid = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 80, 120, 200]
    specs = [("2a_voi21_local", "voi21", ("pay", "local")), ("2b_voi21_rc", "voi21", ("pay", "local", "local_rc")),
             ("2c_voi21_all", "voi21", tuple(kdir.ACTIONS)), ("3a_graph", "voi21+graph", tuple(kdir.ACTIONS)),
             ("3b_all", "all", tuple(kdir.ACTIONS)), ("3c_nodir", "all-no-director", tuple(kdir.ACTIONS)),
             ("3b_all_local", "all", ("pay", "local"))]
    trained = {}
    for name, fs, acts in specs:
        if fs not in trained:
            pol, rep = T.train(tr["logs"], dv["logs"] + odv["logs"], fs)
            trained[fs] = (pol, rep)
            print("  trained", fs, rep, flush=True)
        pol, rep = trained[fs]
        base = {**pol, "actions": list(acts)}
        R, pts = pick(name, lambda R, b=base, ac=acts: kdir_fn(kdir.Policy({**b, "R": R}), ac), grid)
        models[name] = {**base, "R": R, "curve": pts, "train_report": rep}
    # 3d: imitation of the oracle's action
    models["3d_imitation"] = imitation(tr["logs"], devsets, target)
    pickle.dump(models, open(OUT / "kdir_models.pkl", "wb"))
    r["models"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("ok", "catch", "free_strong", "m")}
                   for k, v in models.items()}
    res_save(r)
    ship = models["3b_all"]
    kdir.MODEL_PATH.write_text(json.dumps({k: v for k, v in ship.items() if k in ("ok", "catch", "free_strong", "R",
                                                                                  "C", "eps", "actions")} |
                                          {"features": kdir.NAMES, "R_rule": "dev regret <= today's cache (s2 + kdir_dev)",
                                           "trained": "eval/keystone/kdir_eval.py --part train"}))
    print("model ->", kdir.MODEL_PATH, kdir.MODEL_PATH.stat().st_size, "bytes")


def oracle_action(L):
    best = None
    for a in KR.LAT_ORDER:
        if a in ("local", "local_rc"):
            for j, o in enumerate(L["cand"]):
                if o["ok"] is None:
                    continue
                J = (KR.COST["eps"] if a == "local_rc" else 0) + KR.COST["R"] * (o["rc_regret"] if a == "local_rc"
                                                                                 else o["regret"]) + \
                    (KR.COST["C"] * o["rc_corrected"] if a == "local_rc" else 0)
                if best is None or J < best[0] - 1e-9:
                    best = (J, a, j)
        elif a == "free":
            if "free" in L["oc"]:
                J = L["oc"]["free"]["paid"] + KR.COST["R"] * L["oc"]["free"]["regret"] + KR.COST["eps"]
                if best is None or J < best[0] - 1e-9:
                    best = (J, a, None)
        else:
            J = L["plan_paid"]
            if best is None or J < best[0] - 1e-9:
                best = (J, a, None)
    return best[1], best[2]


def imitation(train_logs, devsets, target):
    """Multiclass trees on [question features + the top candidate's features] -> the oracle's action; the policy
    pays unless the chosen action's probability beats `pay`'s by a margin (the knob, chosen on dev)."""
    import xgboost as xgb
    A = kdir.ACTIONS
    nq = len(kdir.Q_NAMES)
    nc = len(kdir.NAMES) - nq

    def rows(logs):
        X, y = [], []
        for L in logs:
            a, j = oracle_action(L)
            jj = j if j is not None else (int(np.argmax([cx[voi.NAMES.index("qnn_cos")] + cx[voi.NAMES.index("eq_p")]
                                                         for cx in L["cx"]])) if L["cx"] else None)
            X.append(L["qx"] + (L["cx"][jj] if jj is not None else [0.0] * nc) + [float(jj is not None)])
            y.append(A.index(a))
        return np.array(X, np.float32), np.array(y)
    X, y = rows(train_logs)
    m = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.85, objective="multi:softprob",
                          num_class=len(A), n_jobs=2, tree_method="hist")
    m.fit(X, y)

    def fn_of(margin):
        def fn(L):
            jj = (int(np.argmax([cx[voi.NAMES.index("qnn_cos")] + cx[voi.NAMES.index("eq_p")] for cx in L["cx"]]))
                  if L["cx"] else None)
            x = np.array([L["qx"] + (L["cx"][jj] if jj is not None else [0.0] * nc) + [float(jj is not None)]],
                         np.float32)
            p = m.predict_proba(x)[0]
            k = int(np.argmax(p))
            if A[k] == "pay" or p[k] < p[0] + margin:
                return "pay", None
            if A[k] in ("local", "local_rc") and jj is None:
                return "pay", None
            if A[k] == "free" and not (L["has_free"] and not L["lof"]):
                return "pay", None
            return A[k], jj
        return fn
    pts = []
    for mg in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        row = {"knob": mg, "ok": True}
        for dn, (logs, n, cp) in devsets.items():
            p, rg, co = offline_fn(logs, n, fn_of(mg))
            row[dn] = {"paid_per_q": round(p + cp / n, 4), "regret_rate": round(rg, 4), "corrected_rate": round(co, 4)}
            row["ok"] &= rg <= target[dn] + 1e-9
        pts.append(row)
    ok = [v for v in pts if v["ok"]]
    best = min(ok, key=lambda v: v["dev"]["paid_per_q"] + v["old_dev"]["paid_per_q"]) if ok else pts[-1]
    print(f"3d imitation: margin {best['knob']} dev {best['dev']} old_dev {best['old_dev']}", flush=True)
    m.save_model(str(OUT / "kdir_imitation.json"))
    return {"margin": best["knob"], "curve": pts, "m": m, "fn_of": None}


# ------------------------------------------------------------------------------------------------ replays
class ImitationPolicy(kdir.Policy):
    """kdir.Policy's interface over the imitation trees (multiclass on the oracle's action)."""

    def __init__(self, m, margin):
        self.m, self.margin = m, margin
        self.use = set(kdir.ACTIONS)

    def decide(self, qx, cands, cxs, plan_paid, free_ok_q):
        nc = len(kdir.NAMES) - len(kdir.Q_NAMES)
        jj = (int(np.argmax([cx[voi.NAMES.index("qnn_cos")] + cx[voi.NAMES.index("eq_p")] for cx in cxs]))
              if cxs else None)
        x = np.array([qx + (cxs[jj] if jj is not None else [0.0] * nc) + [float(jj is not None)]], np.float32)
        p = self.m.predict_proba(x)[0]
        k = int(np.argmax(p))
        a = kdir.ACTIONS[k]
        if a == "pay" or p[k] < p[0] + self.margin or (a in ("local", "local_rc") and jj is None) or \
                (a == "free" and not (free_ok_q and plan_paid)):
            return "pay", None, {}
        return a, (cands[jj] if a in ("local", "local_rc") else None), {}


def run_policy(stream_key, name, models, store="pooled", feedback=True, online=False, ttl=True, seed=0):
    data = KR.KData(STREAMS[stream_key])
    vm = voi.load()
    kdir._ttl = None if ttl else {}
    from server.keystone import relkey
    orig = relkey.RelKey.ttl_s
    if ttl and kdir.ttl_table():
        relkey.RelKey.ttl_s = property(lambda self: kdir.learned_ttl(self.relation, self.vclass,
                                                                     relkey.TTL_S.get(self.vclass, relkey.TTL_S["slow"])))
    try:
        if name in ("fresh", "voi", "oracle", "oracle_local"):
            rows, _, st = KR.simulate(data, name, store=store, voi_model=vm, theta=vm.theta, feedback=feedback, seed=seed)
        elif name == "3d_imitation":
            md = models[name]
            rows, _, st = KR.simulate(data, ImitationPolicy(md["m"], md["margin"]), store=store, voi_model=vm,
                                      theta=vm.theta, feedback=feedback, seed=seed)
        elif name == "voi_big":
            m = voi.VOI(models["voi_big"]["w"], models["voi_big"]["theta"])
            rows, _, st = KR.simulate(data, "voi", store=store, voi_model=m, theta=m.theta, seed=seed)
        else:
            md = models[name]
            pol = kdir.Policy({k: v for k, v in md.items() if k in ("ok", "catch", "free_strong", "R", "C", "eps",
                                                                     "actions")})
            onl = None
            if online:
                onl = online_refitter(md)
            rows, _, st = KR.simulate(data, pol, store=store, voi_model=vm, theta=vm.theta, feedback=feedback,
                                      online=onl, allow=tuple(md.get("actions", kdir.ACTIONS)), seed=seed)
    finally:
        relkey.RelKey.ttl_s = orig
        kdir._ttl = None
    s = KR.summarize(rows, 500)
    return s, st


def online_refitter(md):
    """Every `every` questions: refit the p_ok trees on the train rows + the feedback rows seen so far (corrections,
    re-check catches = 0; re-check confirmations = 1), Platt kept."""
    tr = pickle.loads((OUT / "kdir_log_train_r0.pkl").read_bytes())["logs"]
    X, y, _, _, _ = T.datasets(tr)
    cols = md["ok"]["cols"]
    state = {"n": 0}

    def refit(pol, fb_rows):
        new = fb_rows[state["n"]:]
        if len(new) < 5:
            return
        state["n"] = len(fb_rows)
        Xf = np.array([q + c for q, c, _ in fb_rows], np.float32)
        yf = np.array([v for _, _, v in fb_rows])
        Xa = np.concatenate([X, np.repeat(Xf, 5, 0)])            # feedback rows weighted x5 (they are on-policy)
        ya = np.concatenate([y, np.repeat(yf, 5)])
        d = T.fit_trees(Xa, ya, Xa[-len(Xf) * 5:], ya[-len(Xf) * 5:], cols)
        d["cal"] = md["ok"].get("cal", [1.0, 0.0])
        pol.ok = kdir.Trees(d)
    return {"every": 500, "refit": refit}


def part_ablation(a):
    r = res_load()
    models = pickle.load(open(OUT / "kdir_models.pkl", "rb"))
    out = r.get("ablation", {})
    jobs = a.jobs.split(",") if a.jobs else None
    plan = [("fresh", {}), ("voi", {}), ("oracle_local", {}), ("oracle", {}), ("voi_big", {}),
            ("2a_voi21_local", {"ttl": False, "feedback": False}), ("2b_voi21_rc", {"ttl": False, "feedback": False}),
            ("2c_voi21_all", {"ttl": False, "feedback": False}), ("3a_graph", {"ttl": False, "feedback": False}),
            ("3b_all", {"ttl": False, "feedback": False}), ("3c_nodir", {"ttl": False, "feedback": False}),
            ("3d_imitation", {"ttl": False, "feedback": False}),
            ("4_feedback", {"ttl": False, "feedback": True}), ("4_feedback_online", {"ttl": False, "feedback": True,
                                                                                    "online": True}),
            ("5_ttl", {"ttl": True, "feedback": True, "online": True})]
    for sk in a.streams.split(","):
        for name, kw in plan:
            key = f"{sk}:{name}"
            if jobs and name not in jobs or key in out:
                continue
            mname = "3b_all" if name.startswith(("4_", "5_")) else name
            t = time.time()
            s, st = run_policy(sk, mname, models, **kw)
            out[key] = {"summary": s, "stats": {k: v for k, v in st.items() if k != "path"}}
            al = s["all"]
            print(f"{key:28s} paid/q {al['paid_per_q']:.3f} regret {al['regret_rate']:.4f} corr {al['corrected_rate']:.4f} "
                  f"acc(exp) {al['acc_expect']} acc {al['acc_all']} p50 {al['p50_ms']} dec {al['dec_ms_p50']} "
                  f"{al['actions']} [{time.time() - t:.0f}s]", flush=True)
            r = res_load()
            r.setdefault("ablation", {})[key] = out[key]
            res_save(r)


def part_network(a):
    r = res_load()
    models = pickle.load(open(OUT / "kdir_models.pkl", "rb"))
    for sk in a.streams.split(","):
        for store in ("user", "user_shared", "user_shared_cells"):
            for name in ("voi", "3b_all"):
                key = f"{sk}:{store}:{name}"
                if key in r.get("network", {}):
                    continue
                t = time.time()
                s, st = run_policy(sk, name, models, store=store, ttl=False, feedback=name != "voi")
                al = s["all"]
                r = res_load()
                r.setdefault("network", {})[key] = {"summary": {"all": al}, "stats": json.loads(json.dumps(st,
                                                                                                          default=str))}
                res_save(r)
                print(f"{key:36s} paid/q {al['paid_per_q']:.3f} regret {al['regret_rate']:.4f} acc {al['acc_all']} "
                      f"local {al['local_share']} {st.get('shared_cells') or ''} [{time.time() - t:.0f}s]", flush=True)


def part_curves(a):
    """Operating curves on the TEST streams (offline, on decisions logged under the VOI gate: the store state is the
    VOI gate's, so these are approximations; the policies' own replays are the measured points)."""
    r = res_load()
    models = pickle.load(open(OUT / "kdir_models.pkl", "rb"))
    vm = voi.load()
    for sk in ("old_test", "test"):
        lg = logs_for(f"{sk}_voi", STREAMS[sk], "voi")
        n, cp = len(lg["rows"]), card_paid(lg["rows"])

        def pts(fn_of, knobs):
            out = []
            for k in knobs:
                p, rg, co = offline_fn(lg["logs"], n, fn_of(k))
                out.append({"knob": k, "paid_per_q": round(p + cp / n, 4), "regret_rate": round(rg, 4),
                            "corrected_rate": round(co, 4)})
            return out
        r.setdefault("test_curves", {})[f"{sk}:voi"] = pts(lambda t: voi_fn(vm, t), [0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                                                                                      0.93, 0.95, 0.97, 0.99])
        grid = [1, 2, 3, 5, 8, 12, 20, 30, 60, 120, 300]
        for name in ("2c_voi21_all", "3b_all"):
            md = models[name]
            r["test_curves"][f"{sk}:{name}"] = pts(lambda R, md=md: kdir_fn(kdir.Policy({**md, "R": R}),
                                                                            tuple(md["actions"])), grid)
        im = models["3d_imitation"]
        ip = lambda mg: (lambda L: _imit_fn(im["m"], mg)(L))  # noqa: E731
        r["test_curves"][f"{sk}:3d_imitation"] = pts(ip, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        print(sk, {k: v[len(v) // 2] for k, v in r["test_curves"].items() if k.startswith(sk)}, flush=True)
    res_save(r)


def _imit_fn(m, margin):
    pol = ImitationPolicy(m, margin)

    def fn(L):
        cands = [T.SimpleCand(p) for p in L["paths"]]
        a, c, _ = pol.decide(L["qx"], cands, L["cx"], L["plan_paid"], L["has_free"] and not L["lof"])
        return a, (cands.index(c) if c is not None else None)
    return fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="train")
    ap.add_argument("--streams", default="old_test,test,old_dev,dev")
    ap.add_argument("--jobs", default="")
    a = ap.parse_args()
    {"train": part_train, "ablation": part_ablation, "network": part_network, "curves": part_curves}[a.part](a)
