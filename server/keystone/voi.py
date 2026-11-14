"""Value-of-information gate: "can the local store answer this correctly?" as a calibrated probability.

One logistic model scores each local candidate (typed-key hit, hash hit, question NN, dense passages) from the lookup's
features: which path, the top passage cosine and its margin, BM25, the subject's coverage in the top passages, the
hash distance and the matched question's cosine, the question-NN cosine, edge age as a share of its TTL, activation of
the question's entities, whether the subject node is known, whether the question is live, store size. The best
candidate is served when p >= theta; otherwise Onset pays for the web. Live / volatile questions can only be served
from cells still inside their (minutes) TTL; the gate learns how often that is right.

Fit offline on one workload stream (eval/keystone/fit_voi.py), evaluated on another; coefficients in voi_model.json
(a few hundred bytes).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).with_name("voi_model.json")
PATHS = ["key", "hash", "qnn", "dense"]
NAMES = (["path_" + p for p in PATHS] + ["dense_top", "dense_gap", "bm25", "ent_cov", "hash_close", "hash_cos",
                                         "qnn_cos", "age_frac", "act", "subj_known", "key_known", "live", "n_cells",
                                         "live_x_key", "eq_p", "has_eq", "bias"])


def vector(feats: dict, cand) -> list[float]:
    p = cand.path if hasattr(cand, "path") else cand
    x = [1.0 if p == q else 0.0 for q in PATHS]
    x += [feats["dense_top"], feats["dense_gap"], min(feats["bm25_top"], 30.0) / 10.0, feats["ent_cov"],
          max(0.0, 1.0 - feats["hash_ham"] / 16.0), feats["hash_cos"], feats["qnn_cos"], feats["age_frac"],
          math.log1p(feats["act"]), feats["subj_known"], feats["key_known"], feats["live"], feats["n_cells"] / 10.0,
          feats["live"] * (1.0 if p == "key" else 0.0), max(0.0, getattr(cand, "eq_p", -1.0)),
          1.0 if getattr(cand, "eq_p", -1.0) >= 0 else 0.0, 1.0]
    return x


class VOI:
    def __init__(self, w: list[float], theta: float = 0.5, meta: dict | None = None):
        self.w = np.asarray(w, np.float64)
        self.theta = theta
        self.meta = meta or {}

    def p(self, x) -> float:
        z = float(np.dot(self.w, np.asarray(x, np.float64)))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def load(path: Path = MODEL_PATH) -> VOI | None:
    try:
        d = json.loads(path.read_text())
        return VOI(d["w"], d.get("theta", 0.5), d)
    except (OSError, ValueError, KeyError):
        return None


def choose(model: VOI, cands: list, feats: dict, theta: float | None = None):
    """The candidate with the highest p(correct), if it clears theta; else None (pay for the web)."""
    best, bp = None, -1.0
    for c in cands:
        p = model.p(vector(feats, c))
        if p > bp:
            best, bp = c, p
    th = model.theta if theta is None else theta
    if best is not None:
        best.p = bp
    return best if best is not None and bp >= th else None


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300) -> np.ndarray:
    """L2-regularised logistic regression (Newton / IRLS), bias unregularised (last column)."""
    w = np.zeros(X.shape[1])
    R = np.eye(X.shape[1]) * l2
    R[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + R @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w
