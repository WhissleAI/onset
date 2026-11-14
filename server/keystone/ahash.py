"""Learned answer-equivalence hashing: a 64-bit code per question such that questions with the SAME answer collide
(or land within a small Hamming radius) and questions with different answers don't.

Input: the question's EmbeddingGemma vector (Matryoshka-truncated to Q_DIM) concatenated with its Arc read-out
features (intent, answer type, typed relation, entity types, number types, dialogue act, freshness). Model: one
linear projection to 64 dims, sign -> bits. Trained offline (eval/keystone/train_ahash.py, torch) with a contrastive
objective on question families (same family = same answer: paraphrases, ASR renderings) with hard negatives (same
entity + different relation, same relation + different entity, the family's near-miss). Runtime is numpy: one
(Q_DIM + F) x 64 matmul, ~10 us.

Why the Arc features help: the embedding of "when was nvidia founded" and "who founded nvidia" is nearly the same
(same words); their answer types (date vs person) and relations (inception vs founded_by) differ, and the projection
learns to put those bits far apart. The embedding still carries the subject, which the typed key can miss when Arc's
span is off.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .relkey import RELATIONS, answer_type

MODEL_PATH = Path(__file__).with_name("ahash_model.npz")
INTENTS = ["discourse.intent.qa_factoid", "discourse.intent.qa_definition", "discourse.intent.qa_maths",
           "discourse.intent.qa_currency", "discourse.intent.qa_stock", "discourse.intent.weather_query",
           "discourse.intent.news_query", "discourse.intent.datetime_query", "discourse.intent.cooking_recipe",
           "discourse.intent.cooking_query", "discourse.intent.recommendation_locations",
           "discourse.intent.recommendation_movies", "discourse.intent.recommendation_events",
           "discourse.intent.transport_query", "discourse.intent.music_query", "discourse.intent.general_quirky",
           "discourse.intent.play_music", "discourse.intent.takeaway_query", "intent.request_information",
           "intent.request_action"]
ANS = ["person", "date", "place", "measure_h", "measure", "count", "age", "thing"]
RELS = sorted(RELATIONS)
ENT = ["person", "team", "organization", "place", "work_product", "event_name", "legal_ref", "game_entity", "term"]
NUM = ["date", "time", "time_of_day", "duration", "money", "percent", "cardinal", "ordinal", "quantity", "number",
       "year"]
ACTS = ["question", "command_request", "statement", "opinion_assessment", "suggestion_proposal"]
F_DIM = len(INTENTS) + 1 + len(ANS) + len(RELS) + 1 + len(ENT) + len(NUM) + len(ACTS) + 2


def arc_features(arc: dict) -> np.ndarray:
    """arc: {"intent", "answer_type", "relation", "ent_types", "num_types", "acts", "live"} -> F_DIM floats."""
    x = np.zeros(F_DIM, np.float32)
    o = 0
    it = arc.get("intent") or ""
    x[o + (INTENTS.index(it) if it in INTENTS else len(INTENTS))] = 1.0
    o += len(INTENTS) + 1
    at = arc.get("answer_type") or "thing"
    x[o + (ANS.index(at) if at in ANS else ANS.index("thing"))] = 1.0
    o += len(ANS)
    r = arc.get("relation") or ""
    x[o + (RELS.index(r) if r in RELS else len(RELS))] = 1.0
    o += len(RELS) + 1
    for t in arc.get("ent_types") or ():
        if t in ENT:
            x[o + ENT.index(t)] = 1.0
    o += len(ENT)
    for t in arc.get("num_types") or ():
        if t in NUM:
            x[o + NUM.index(t)] = 1.0
    o += len(NUM)
    for t in arc.get("acts") or ():
        if t in ACTS:
            x[o + ACTS.index(t)] = 1.0
    o += len(ACTS)
    x[o] = 1.0 if arc.get("live") else 0.0
    x[o + 1] = 1.0
    return x


def arc_of(words, intent: str = "", slots=(), acts=(), live: bool = False, relkey=None) -> dict:
    """The Arc signature of a question (what a cell stores and the hash reads)."""
    low = " ".join(w.lower() for w in words)
    return {"intent": intent, "answer_type": answer_type(low), "relation": relkey.relation if relkey else "",
            "ent_types": sorted({t for t, _ in slots if t in ENT}), "num_types": sorted({t for t, _ in slots
                                                                                        if t in NUM}),
            "acts": list(acts), "live": bool(live)}


class AHash:
    def __init__(self, W: np.ndarray, q_dim: int, bits: int, w_arc: float = 1.0, mean: np.ndarray | None = None):
        self.W, self.q_dim, self.bits, self.w_arc = W.astype(np.float32), q_dim, bits, w_arc
        self.mean = mean.astype(np.float32) if mean is not None else None

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "AHash | None":
        try:
            z = np.load(path)
        except (OSError, ValueError):
            return None
        return cls(z["W"], int(z["q_dim"]), int(z["bits"]), float(z["w_arc"]), z["mean"] if "mean" in z else None)

    def inputs(self, qvec: np.ndarray, arc: dict) -> np.ndarray:
        q = np.asarray(qvec, np.float32)[: self.q_dim]
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        x = np.concatenate([q, self.w_arc * arc_features(arc)])
        return x - self.mean if self.mean is not None else x

    def real(self, qvec: np.ndarray, arc: dict) -> np.ndarray:
        return self.inputs(qvec, arc) @ self.W

    def code(self, qvec: np.ndarray, arc: dict) -> np.ndarray:
        """Packed bits (uint8[bits/8])."""
        return np.packbits(self.real(qvec, arc) > 0)


_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming(codes: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Hamming distance of each row of `codes` (uint8[n, b]) to `c` (uint8[b])."""
    return _POP[np.bitwise_xor(codes, c)].sum(axis=1)
