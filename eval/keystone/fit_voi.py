"""Fit the value-of-information gate on one workload stream (dev: runs/index/stream_s2.jsonl) and evaluate it on
another (test: runs/index/stream.jsonl): calibration (reliability bins, Brier, ECE) and the precision / coverage
operating curve. Theta is chosen on DEV as the lowest threshold whose served-local precision is at least the fresh
path's own accuracy on dev (accuracy held), then frozen for test.

Training rows: every local candidate (key / hash / qnn / dense) of every question, with the lookup features and
whether serving that candidate would have been correct (replay.correct). Logged while replaying policy e (hand gate).

    .venv/bin/python eval/keystone/fit_voi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval/keystone"))

import numpy as np  # noqa: E402

import replay as R  # noqa: E402
from server.keystone import voi  # noqa: E402

OUT = ROOT / "runs/index"


def logged(stream, emb):
    data = R.Data(stream)
    rows, feats, _ = R.run(data, "e", emb, log_feats=True)
    fresh, _, _ = R.run(data, "fresh", emb)
    acc = [r["correct"] for r in fresh if r["correct"] is not None]
    fr = [x for x in feats if x["y"] is not None]
    return fr, sum(acc) / len(acc)


def per_question(feat_rows, w):
    best = {}
    for x in feat_rows:
        p = 1 / (1 + np.exp(-np.dot(w, x["x"])))
        if x["pos"] not in best or p > best[x["pos"]][0]:
            best[x["pos"]] = (p, x["y"], x["path"])
    return best


def curve(best, n_q):
    out = []
    any_ok = sum(1 for p, y, _ in best.values())
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        sel = [y for p, y, _ in best.values() if p >= th]
        out.append({"theta": th, "served_share": round(len(sel) / n_q, 3), "precision": round(sum(sel) / len(sel), 3)
                    if sel else None, "n": len(sel)})
    return out


def calib(feat_rows, w, bins=10):
    p = np.array([1 / (1 + np.exp(-np.dot(w, x["x"]))) for x in feat_rows])
    y = np.array([float(x["y"]) for x in feat_rows])
    out, ece = [], 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1.0)
        if m.sum():
            out.append({"bin": b, "p_mean": round(float(p[m].mean()), 3), "y_rate": round(float(y[m].mean()), 3),
                        "n": int(m.sum())})
            ece += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return out, float(np.mean((p - y) ** 2)), float(ece)


def main():
    emb = 20.0
    dev, dev_fresh = logged(OUT / "stream_s2.jsonl", emb)
    test, test_fresh = logged(OUT / "stream.jsonl", emb)
    Xd = np.array([x["x"] for x in dev])
    yd = np.array([float(x["y"]) for x in dev])
    w = voi.fit(Xd, yd, l2=1.0)
    bd = per_question(dev, w)
    bt = per_question(test, w)
    n_dev = len({x["pos"] for x in dev})
    n_test = len({x["pos"] for x in test})
    cd, ct = curve(bd, n_dev), curve(bt, n_test)
    target = max(dev_fresh, 0.92)      # no-regret precision: at most ~8 % of local answers worse than paying
    theta = next((c["theta"] for c in cd if c["precision"] is not None and c["precision"] >= target), 0.95)
    cal_d, brier_d, ece_d = calib(dev, w)
    cal_t, brier_t, ece_t = calib(test, w)
    rep = {"names": voi.NAMES, "w": list(map(float, w)), "theta": theta, "target_precision": round(target, 3),
           "fresh_acc": {"dev": round(dev_fresh, 3), "test": round(test_fresh, 3)},
           "dev": {"rows": len(dev), "questions": n_dev, "curve": cd, "calibration": cal_d, "brier": round(brier_d, 4),
                   "ece": round(ece_d, 4)},
           "test": {"rows": len(test), "questions": n_test, "curve": ct, "calibration": cal_t,
                    "brier": round(brier_t, 4), "ece": round(ece_t, 4)}}
    voi.MODEL_PATH.write_text(json.dumps({"w": rep["w"], "names": voi.NAMES, "theta": theta}))
    (OUT / "voi_fit.json").write_text(json.dumps(rep, indent=1))
    print(f"theta {theta} (dev fresh acc {dev_fresh:.3f}); dev brier {brier_d:.3f} ece {ece_d:.3f}; "
          f"test brier {brier_t:.3f} ece {ece_t:.3f}")
    print("test curve:", ct)
    print("weights:", {n: round(x, 2) for n, x in zip(voi.NAMES, w)})


if __name__ == "__main__":
    main()
