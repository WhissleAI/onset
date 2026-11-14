"""Memory cells: what Keystone keeps of every passage and answer Onset retrieves.

A cell is small on purpose. It holds meaning and structure, not documents:

  vec      EmbeddingGemma 2 vector (the one the reranker already computed: never embedded twice), kept twice:
           a first-pass copy Matryoshka-truncated to FP_DIM = 256 dims (float32: numpy has BLAS kernels for float32
           only; int8 / float16 matmuls measured 5-30x SLOWER, runs/index/ann_bench.json; 1 bit per dim is an
           option but loses recall) and the full 768-dim float16 vector for re-scoring the shortlist (64 rows)
  qvec     (value cells) the vector of the QUESTION that produced it: answer-equivalent questions land near it
  code     the learned answer-equivalence hash of that question (keystone/ahash.py), 64 bits
  arc      the Arc read-out signature of the question: intent, answer type, entity types, number types, act, freshness
  key      its typed relation key (keystone/relkey.py), when the question had one: "founded_by(nvidia)"
  ents     entity keys (QIDs when known, else canonical names)
  payload  the answer span or the 1-2 key sentences for the question (<= PAYLOAD_CHARS)
  full     the source passage (optional; the first thing evicted under memory pressure)
  t, ttl   fetch time and time-to-live from the relation's volatility (stable: months, slow: weeks, live: minutes)
  hits     times it was served (value = hits x freshness decides eviction)

Search is two-stage: a dense first pass over the truncated / quantized codes of every live cell (numpy matmul: at
Onset's sizes brute force beats an ANN graph, runs/index/ann_bench.json), then exact re-scoring of the top M with the
full vectors, plus a small lexical (BM25) term over payload + title for names the embedding blurs.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FULL_DIM = 768
PAYLOAD_CHARS = 320
FULL_CHARS = 1200
_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("""a an the is are was were be been of in on at to for from by with and or it its this that as which who
what when where how why do does did has have had not no but if then than so such into about over after before also
can could would should will""".split())


def toks(s: str) -> list[str]:
    return [w for w in _TOK.findall((s or "").lower()) if w not in _STOP and len(w) > 1]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text or "") if s.strip()]


def key_sentences(text: str, q_terms: list[str], n: int = 2, limit: int = PAYLOAD_CHARS) -> str:
    """The n sentences of `text` that share the most terms with the question (ties: earlier first), in text order."""
    ss = sentences(text)
    if not ss:
        return (text or "")[:limit]
    qs = set(q_terms)
    sc = [(len(qs & set(toks(s))), -i) for i, s in enumerate(ss)]
    pick = sorted(sorted(range(len(ss)), key=lambda i: sc[i], reverse=True)[:n])
    out = " ".join(ss[i] for i in pick)
    return out[:limit]


@dataclass
class Cell:
    id: int
    kind: str                     # passage | value
    title: str
    payload: str
    source: str = ""
    url: str = ""
    full: str = ""
    key: str = ""                 # RelKey.text() for value cells / typed passages
    question: str = ""
    arc: dict = field(default_factory=dict)
    ents: list = field(default_factory=list)
    t: float = 0.0
    ttl: float = 0.0
    vclass: str = "general"
    hits: int = 0
    last_hit: float = 0.0
    doc_id: str = ""
    links: list = field(default_factory=list)     # value cell -> its source passage cells

    def fresh(self, now: float) -> bool:
        return self.ttl <= 0 or now - self.t <= self.ttl

    def value(self, now: float) -> float:
        """Eviction value: (1 + hits) x freshness. Freshness decays over the cell's TTL (stable cells barely)."""
        age = max(0.0, now - self.t)
        fr = math.exp(-age / self.ttl) if self.ttl > 0 else 1.0
        return (1.0 + self.hits) * fr

    def nbytes(self) -> int:
        return len(self.title) + len(self.payload) + len(self.full) + len(self.url) + len(self.key) + \
            len(self.question) + 64


def quantize_i8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-9) / 127.0
    return np.round(x / s).astype(np.int8), s.astype(np.float32).reshape(-1)


def first_pass(x: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation + renormalisation."""
    x = np.asarray(x, np.float32)[..., :dim]
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


class CellStore:
    """Growable arrays + a free list. Thread-safe for one writer and readers (a lock around mutations)."""

    def __init__(self, cap: int = 1024, fp_dim: int = 256, fp_quant: str = "fp32", max_cells: int = 20000,
                 max_bytes: int = 64 << 20, shortlist: int = 64):
        self.fp_dim, self.fp_quant, self.max_cells, self.max_bytes, self.M = fp_dim, fp_quant, max_cells, max_bytes, \
            shortlist
        self.cells: dict[int, Cell] = {}
        self.full = np.zeros((cap, FULL_DIM), np.float16)
        self.qv = np.zeros((cap, FULL_DIM), np.float32)          # the question vector (value cells; float32: BLAS)
        self.has_q = np.zeros(cap, bool)
        self.fpf = np.zeros((cap, fp_dim), np.float32)
        self.bits = np.zeros((cap, fp_dim // 8), np.uint8)
        self.alive = np.zeros(cap, bool)
        self.free: list[int] = list(range(cap - 1, -1, -1))
        self.by_doc: dict[str, int] = {}
        self.by_key: dict[str, list[int]] = {}
        self.post: dict[str, set] = {}                            # term -> cell ids (BM25)
        self.tf: dict[int, Counter] = {}
        self.len_sum = 0
        self.text_bytes = 0
        self.lock = threading.RLock()
        self.evicted = 0
        self.full_dropped = 0

    def __len__(self):
        return len(self.cells)

    # ------------------------------------------------------------------ write
    def _grow(self):
        n = len(self.alive)
        m = n * 2
        for name in ("full", "qv", "fpf", "bits"):
            a = getattr(self, name)
            b = np.zeros((m,) + a.shape[1:], a.dtype)
            b[:n] = a
            setattr(self, name, b)
        for name in ("alive", "has_q"):
            a = getattr(self, name)
            b = np.zeros(m, a.dtype)
            b[:n] = a
            setattr(self, name, b)
        self.free.extend(range(m - 1, n - 1, -1))

    def add(self, cell: Cell, vec: np.ndarray, qvec: np.ndarray | None = None) -> int:
        """Insert (or refresh, by doc_id / value key) a cell. Returns its id."""
        with self.lock:
            old = self.by_doc.get(cell.doc_id) if cell.doc_id else None
            if old is not None and old in self.cells:
                c = self.cells[old]
                c.t, c.ttl = max(c.t, cell.t), max(c.ttl, cell.ttl)
                if cell.key and not c.key:
                    c.key = cell.key
                    self.by_key.setdefault(cell.key, []).append(old)
                c.ents = list(dict.fromkeys(c.ents + cell.ents))
                if qvec is not None and not self.has_q[old]:
                    self.qv[old] = qvec[:FULL_DIM]
                    self.has_q[old] = True
                return old
            if not self.free:
                self._grow()
            i = self.free.pop()
            cell.id = i
            v = np.asarray(vec, np.float32)[:FULL_DIM]
            self.full[i] = v
            fp = first_pass(v, self.fp_dim)
            self.fpf[i] = fp
            self.bits[i] = np.packbits(fp > 0)
            if qvec is not None:
                self.qv[i] = np.asarray(qvec, np.float32)[:FULL_DIM]
                self.has_q[i] = True
            else:
                self.has_q[i] = False
            self.alive[i] = True
            self.cells[i] = cell
            if cell.doc_id:
                self.by_doc[cell.doc_id] = i
            if cell.key:
                self.by_key.setdefault(cell.key, []).append(i)
            tf = Counter(toks(cell.title + " " + cell.payload))
            self.tf[i] = tf
            self.len_sum += sum(tf.values())
            for w in tf:
                self.post.setdefault(w, set()).add(i)
            self.text_bytes += cell.nbytes()
            self._evict()
            return i

    def remove(self, i: int):
        with self.lock:
            c = self.cells.pop(i, None)
            if c is None:
                return
            self.alive[i] = False
            self.has_q[i] = False
            if c.doc_id and self.by_doc.get(c.doc_id) == i:
                del self.by_doc[c.doc_id]
            if c.key and i in self.by_key.get(c.key, []):
                self.by_key[c.key].remove(i)
                if not self.by_key[c.key]:
                    del self.by_key[c.key]
            tf = self.tf.pop(i, Counter())
            self.len_sum -= sum(tf.values())
            for w in tf:
                s = self.post.get(w)
                if s is not None:
                    s.discard(i)
                    if not s:
                        del self.post[w]
            self.text_bytes -= c.nbytes()
            self.free.append(i)

    def _evict(self):
        now = time.time()
        if self.text_bytes > self.max_bytes:         # full source text goes first
            for c in sorted(self.cells.values(), key=lambda c: c.value(now)):
                if self.text_bytes <= self.max_bytes * 0.9:
                    break
                if c.full:
                    self.text_bytes -= len(c.full)
                    c.full = ""
                    self.full_dropped += 1
        over = len(self.cells) - self.max_cells
        if over > 0 or self.text_bytes > self.max_bytes:
            order = sorted(self.cells.values(), key=lambda c: c.value(now))
            k = max(over, 0) + max(1, len(self.cells) // 50)           # evict in small batches
            for c in order[:k]:
                self.remove(c.id)
                self.evicted += 1

    def hit(self, i: int):
        c = self.cells.get(i)
        if c is not None:
            c.hits += 1
            c.last_hit = time.time()

    # ------------------------------------------------------------------ read
    def bm25(self, q_terms: list[str], ids: np.ndarray, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        n = max(1, len(self.cells))
        avg = self.len_sum / n if n else 1.0
        out = np.zeros(len(ids), np.float32)
        if not q_terms:
            return out
        for j, i in enumerate(ids):
            tf = self.tf.get(int(i))
            if not tf:
                continue
            L = sum(tf.values())
            s = 0.0
            for w in q_terms:
                f = tf.get(w, 0)
                if f:
                    df = len(self.post.get(w, ()))
                    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                    s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * L / avg))
            out[j] = s
        return out

    def search(self, qvec: np.ndarray, k: int = 5, q_terms: list[str] | None = None, now: float | None = None,
               w_bm25: float = 0.02, allow=None, use: str | None = None) -> list[tuple[Cell, float, dict]]:
        """[(cell, score, parts)] best first, live (unexpired) cells only. use: first-pass override for benchmarks
        ("fp32" truncated | "bits" | "fp16" exact)."""
        now = time.time() if now is None else now
        with self.lock:
            idx = np.flatnonzero(self.alive)
            if not len(idx):
                return []
            use = use or self.fp_quant
            qf = np.asarray(qvec, np.float32)[:FULL_DIM]
            if use == "fp16":
                s1 = self.full[idx].astype(np.float32) @ qf
            elif use == "bits":
                qb = np.packbits(first_pass(qf, self.fp_dim) > 0)
                ham = np.unpackbits(self.bits[idx] ^ qb, axis=1).sum(1)
                s1 = -ham.astype(np.float32)
            else:
                s1 = self.fpf[idx] @ first_pass(qf, self.fp_dim)
            m = min(len(idx), max(self.M, k))
            short = idx[np.argpartition(-s1, m - 1)[:m]] if m < len(idx) else idx
            dense = self.full[short].astype(np.float32) @ qf
            lex = self.bm25(q_terms or [], short) if w_bm25 > 0 and q_terms else np.zeros(len(short), np.float32)
            score = dense + w_bm25 * lex
            order = np.argsort(-score)
            out = []
            for j in order:
                c = self.cells[int(short[j])]
                if not c.fresh(now) or (allow is not None and not allow(c)):
                    continue
                out.append((c, float(score[j]), {"dense": float(dense[j]), "bm25": float(lex[j])}))
                if len(out) >= k:
                    break
            return out

    def question_nn(self, qvec: np.ndarray, k: int = 3, now: float | None = None) -> list[tuple[Cell, float]]:
        """Value cells whose QUESTION vector is nearest this question's (plain embedding NN baseline)."""
        now = time.time() if now is None else now
        with self.lock:
            idx = np.flatnonzero(self.alive & self.has_q)
            if not len(idx):
                return []
            s = self.qv[idx] @ np.asarray(qvec, np.float32)[:FULL_DIM]
            out = []
            for j in np.argsort(-s)[: k * 3]:
                c = self.cells[int(idx[j])]
                if c.fresh(now):
                    out.append((c, float(s[j])))
                if len(out) >= k:
                    break
            return out

    # ------------------------------------------------------------------ size + persistence
    def size_report(self) -> dict:
        n = len(self.cells)
        vec_b = n * (self.fp_dim * 4 + self.fp_dim // 8 + FULL_DIM * 2)
        qv_b = int(self.has_q.sum()) * FULL_DIM * 4
        return {"cells": n, "value_cells": sum(1 for c in self.cells.values() if c.kind == "value"),
                "text_bytes": self.text_bytes, "vector_bytes": vec_b + qv_b,
                "bytes_per_cell": round((self.text_bytes + vec_b + qv_b) / n) if n else 0,
                "evicted": self.evicted, "full_dropped": self.full_dropped}

    def snapshot(self) -> tuple:
        """A consistent copy for writing elsewhere (cheap: dict copies + array slices)."""
        with self.lock:
            ids = sorted(self.cells)
            meta = [dict(self.cells[i].__dict__, ents=list(self.cells[i].ents), links=list(self.cells[i].links),
                         arc=dict(self.cells[i].arc)) for i in ids]
            return meta, self.full[ids].copy(), self.qv[ids].astype(np.float16), self.has_q[ids].copy()

    @staticmethod
    def write(path: Path, snap: tuple):
        meta, full, qv, has_q = snap
        path.mkdir(parents=True, exist_ok=True)
        tmp = path / "cells.jsonl.tmp"
        tmp.write_text("\n".join(json.dumps(m) for m in meta) + ("\n" if meta else ""))
        np.savez(path / "cells_vec.tmp.npz", full=full, qv=qv, has_q=has_q)
        tmp.replace(path / "cells.jsonl")
        (path / "cells_vec.tmp.npz").replace(path / "cells_vec.npz")

    def save(self, path: Path):
        self.write(path, self.snapshot())

    def load(self, path: Path) -> int:
        try:
            meta = [json.loads(x) for x in (path / "cells.jsonl").read_text().splitlines() if x.strip()]
            z = np.load(path / "cells_vec.npz")
        except (OSError, ValueError):
            return 0
        now = time.time()
        n = 0
        self.remap: dict = {}
        for m, v, q, hq in zip(meta, z["full"], z["qv"], z["has_q"]):
            c = Cell(**m)
            old = c.id
            if not c.fresh(now):
                continue
            self.remap[old] = self.add(c, v.astype(np.float32), q.astype(np.float32) if hq else None)
            n += 1
        for c in self.cells.values():          # ids are reassigned on load: links follow
            c.links = [self.remap[i] for i in c.links if i in self.remap]
        return n
