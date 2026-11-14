"""Learned answer-equivalence verifier: is a stored question's answer the answer to this question?

Candidates come from the question-vector NN (or the hash's bucket); the verifier reads the embedding cosine plus the
Arc agreement between the two questions: same answer type (who / when / where / how tall), same typed relation, same
typed key, subject overlap, same entity / number types, both live or not. One logistic model (weights in
verify_eq.json, fit by eval/keystone/train_ahash.py on question families; held-out families for evaluation).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("verify_eq.json")
NAMES = ["cos", "cos2", "ans_same", "rel_same", "rel_conflict", "key_same", "key_conflict", "subj_jacc", "ent_same",
         "num_same", "live_same", "bias"]


def _subj(k) -> set:
    return set(k.subject.replace("@", " ").replace("(", " ").replace(")", " ").split()) if k is not None else set()


def pair_features(cos: float, a: dict, b: dict, ka, kb) -> list[float]:
    ra, rb = a.get("relation") or "", b.get("relation") or ""
    sa, sb = _subj(ka), _subj(kb)
    kca = ka is not None and ka.complete
    kcb = kb is not None and kb.complete
    return [cos, cos * cos, float(a.get("answer_type") == b.get("answer_type")), float(bool(ra) and ra == rb),
            float(bool(ra) and bool(rb) and ra != rb), float(kca and kcb and ka.text() == kb.text()),
            float(kca and kcb and ka.text() != kb.text()),
            len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0,
            float(sorted(a.get("ent_types") or []) == sorted(b.get("ent_types") or [])),
            float(sorted(a.get("num_types") or []) == sorted(b.get("num_types") or [])),
            float(bool(a.get("live")) == bool(b.get("live"))), 1.0]


class Verifier:
    def __init__(self, w):
        self.w = list(w)

    def p(self, x) -> float:
        z = sum(wi * xi for wi, xi in zip(self.w, x))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def load(path: Path = MODEL_PATH) -> Verifier | None:
    try:
        return Verifier(json.loads(path.read_text())["w"])
    except (OSError, ValueError, KeyError):
        return None
