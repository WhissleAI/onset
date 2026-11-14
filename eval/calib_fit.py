"""Fit the answer calibrator (server/calibrate.py) on eval/calib_data.py rows; report reliability before vs after.

  pre   features known before the answer model runs (router use)
  post  + verify + the answer itself (the badge)
Logistic regression (standardized, L2) -> isotonic regression fitted on 5-fold out-of-fold logits of the TRAIN split.
The TEST split (30% of question families, by family id: paraphrases of one question never straddle) is untouched by
the fit. Reports ECE (10 equal-width bins), Brier, AUC and the reliability bins for:
  raw rerank   the top EmbeddingGemma cosine taken as a probability (what a "relevance score" badge would show)
  raw route    the router plan's own conf
  pre / post   the calibrated stages
Writes server/calib_model.json and runs/calib/fit.json.

  .venv/bin/python eval/calib_fit.py [--rows runs/calib/rows.jsonl ...] [--no-write]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("WEB_BACKEND", "wikipedia")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from server import calibrate, lanes  # noqa: E402

PRE = ["rr_top", "rr_margin", "rr_mean3", "n_docs", "agree_subj", "facts", "card", "ks_hit", "ks_shared", "ks_age_h",
       "route_conf", "lane_web", "lane_facts", "lane_card", "key_conf", "corrected", "followup", "heard_fix",
       "fillers", "n_words", "intent_p", "i_factoid", "i_live", "src_top", "src_margin", "src_age_d"]
POST = PRE + ["v_agree", "v_supported", "v_partial", "v_unverified", "v_disagree", "v_found_frac", "non_answer",
              "ans_words", "n_cites", "extractive"]


def doc_of(j: dict):
    return lanes.Doc(id=j["id"], title=j.get("title", ""), text=j.get("text", ""), url=j.get("url", ""),
                     source=j.get("source", ""), score=float(j.get("score") or 0.0), meta=dict(j.get("meta") or {}))


def row_features(r: dict, frames: dict) -> dict:
    """Features of a calib_data row (text path) or a live metric row that already carries conf_feats."""
    if "conf_feats" in r:
        return dict(r["conf_feats"])
    from replay_frames import frame_for
    f = frames.get(r["text"])
    if f is None:
        f = frames[r["text"]] = frame_for(r["text"])
    ranked = [(doc_of(d), d.get("rr")) for d in r.get("ranked") or []]
    return calibrate.features(f, ranked, r.get("route") or {}, v=r.get("verify") or {}, answer=r.get("answer") or "",
                              fallback=r.get("fallback"))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list]:
    edges = np.linspace(0, 1, bins + 1)
    out, e = [], 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.sum():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
            out.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": int(m.sum()),
                        "mean_conf": round(float(p[m].mean()), 3), "accuracy": round(float(y[m].mean()), 3)})
    return round(float(e), 4), out


def report(name: str, p, y) -> dict:
    p = np.clip(np.asarray(p, float), 0, 1)
    e, rel = ece(p, y)
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else None
    return {"name": name, "n": int(len(y)), "ece": e, "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": None if auc is None else round(float(auc), 3), "bins": rel}


def matrix(F: list[dict], names: list[str], fill: dict | None = None):
    X = np.array([[np.nan if f.get(k) is None else float(f[k]) for k in names] for f in F], float)
    if fill is None:
        fill = {k: (float(np.nanmean(X[:, j])) if np.isfinite(X[:, j]).any() else 0.0) for j, k in enumerate(names)}
    for j, k in enumerate(names):
        X[np.isnan(X[:, j]), j] = fill[k]
    return X, fill


def fit_stage(Ftr, ytr, gtr, names, C: float):
    X, fill = matrix(Ftr, names)
    mu, sd = X.mean(0), X.std(0)
    keep = [j for j in range(len(names)) if sd[j] > 1e-9]           # constant on train (e.g. no Tavily score): dropped
    names = [names[j] for j in keep]
    X, mu, sd = X[:, keep], mu[keep], sd[keep]
    Z = (X - mu) / sd
    oof = np.zeros(len(ytr))
    for tr, va in GroupKFold(5).split(Z, ytr, gtr):
        lr = LogisticRegression(C=C, max_iter=2000).fit(Z[tr], ytr[tr])
        oof[va] = lr.predict_proba(Z[va])[:, 1]
    lr = LogisticRegression(C=C, max_iter=2000).fit(Z, ytr)
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(oof, ytr)
    xs, ys = list(iso.X_thresholds_), list(iso.y_thresholds_)
    spec = {"features": ["bias"] + names, "w": [float(lr.intercept_[0])] + [float(w) for w in lr.coef_[0]],
            "mu": [0.0] + [float(v) for v in mu], "sd": [1.0] + [float(v) for v in sd],
            "fill": [0.0] + [fill[k] for k in names],
            "iso": {"x": [round(float(v), 5) for v in xs], "y": [round(float(v), 5) for v in ys]}}
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=[str(ROOT / "runs/calib/rows.jsonl")])
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "runs/calib/fit.json"))
    a = ap.parse_args()
    rows = []
    for p in a.rows:
        rows += [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("correct") is not None and not r.get("error")]
    frames: dict = {}
    F = [row_features(r, frames) for r in rows]
    y = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    groups = np.array([r.get("group") or r["text"] for r in rows])
    te = np.array([r.get("split") == "test" for r in rows])
    tr = ~te
    out = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "acc_train": round(float(y[tr].mean()), 3),
           "acc_test": round(float(y[te].mean()), 3), "sets": {}, "test": [], "model": {}}
    for r in rows:
        out["sets"].setdefault(r.get("set", "?"), [0, 0])
        out["sets"][r.get("set", "?")][0] += 1
        out["sets"][r.get("set", "?")][1] += int(bool(r["correct"]))
    Ftr = [F[i] for i in np.where(tr)[0]]
    Fte = [F[i] for i in np.where(te)[0]]
    model = {"version": 1, "bands": {"low": calibrate.CALIB_LOW, "high": calibrate.CALIB_HIGH},
             "trained_on": {"rows": int(tr.sum()), "backend": "wikipedia", "label": "keyword scorer vs gold"}}
    for stage, names in (("pre", PRE), ("post", POST)):
        model[stage] = fit_stage(Ftr, y[tr], groups[tr], names, a.C)
    calibrate.reload(model)
    yte = y[te]
    rr = [f.get("rr_top") or 0.0 for f in Fte]
    rc = [f.get("route_conf") if f.get("route_conf") is not None else 0.5 for f in Fte]
    pre = [calibrate.raw(f, "pre") for f in Fte]
    post = [calibrate.raw(f, "post") for f in Fte]
    for name, p in (("raw rerank cosine", rr), ("raw router conf", rc), ("calibrated pre-answer", pre),
                    ("calibrated post-answer", post)):
        out["test"].append(report(name, p, yte))
    # bands on test: how often each band is right
    for stage, ps in (("pre", pre), ("post", post)):
        b = {}
        for p, yy in zip(ps, yte):
            k = calibrate.band(p)
            b.setdefault(k, [0, 0])
            b[k][0] += 1
            b[k][1] += int(yy)
        out[f"bands_{stage}"] = {k: {"n": v[0], "accuracy": round(v[1] / v[0], 3)} for k, v in b.items()}
    out["model"] = {s: dict(zip(model[s]["features"], [round(w, 3) for w in model[s]["w"]])) for s in ("pre", "post")}
    print(json.dumps({k: v for k, v in out.items() if k != "model"}, indent=1))
    print(json.dumps(out["model"], indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    if not a.no_write:
        (ROOT / "server/calib_model.json").write_text(json.dumps(model, indent=1))


if __name__ == "__main__":
    main()
