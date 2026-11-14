"""Keystone: one structure for Onset's index, cache and memory.

    write(q, docs, vecs)   after every retrieval: passages -> memory cells, Wikidata facts -> typed edges, the question
                           -> a value cell keyed by its typed relation (subject -[relation]-> value) and by its learned
                           answer-equivalence code; the question's entities get activation (session + user)
    lookup(q)              before paying: (1) typed key O(1) on (subject, relation); (2) answer-equivalence hash over
                           value cells; (3) question-vector NN; (4) dense + BM25 passage search with activation bias.
                           Returns the best local candidate and the features the value-of-information gate reads.
    decide(cands, feats)   heuristic gate or the calibrated VOI model (keystone/voi.py): serve locally or pay.

Components are switchable (flags) so every one of them can be ablated on the workload stream.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import ahash as ah
from . import verify_eq
from .cells import Cell, CellStore, key_sentences, toks
from .graph import Graph
from .relkey import FROM_PROP, RELATIONS, TTL_S, RelKey, canon

SRC_TTL = {"wikipedia": 30 * 86400.0, "local wikipedia": 30 * 86400.0, "wikidata": 30 * 86400.0,
           "wiktionary": 90 * 86400.0}


@dataclass
class Flags:
    cells: bool = True          # A: memory cells (dense + BM25 passage search)
    key: bool = True            # B: Arc-keyed typed edges
    hash: bool = True           # D: learned answer-equivalence hash
    act: bool = True            # E: activation (memory) bias + unified memory/cache
    qnn: bool = True            # question-vector NN over value cells (plain embedding baseline for D)
    verify: bool = True         # D: learned answer-equivalence verifier on value-cell candidates
    hash_r: int = 4             # Hamming radius for a hash hit
    hash_cos: float = 0.80      # ... and the matched question's cosine
    qnn_cos: float = 0.92
    dense_sim: float = 0.62     # heuristic gate: passage cosine
    w_act: float = 0.03


@dataclass
class Q:
    text: str
    words: list
    slots: tuple = ()
    intent: str = ""
    acts: tuple = ()
    live: bool = False
    qvec: np.ndarray | None = None
    key: RelKey | None = None
    arc: dict = field(default_factory=dict)
    user: str = ""
    session: str = ""

    @property
    def scopes(self) -> list[str]:
        return [s for s in (f"s:{self.session}" if self.session else "", f"u:{self.user}" if self.user else "") if s]


@dataclass
class Cand:
    path: str                   # key | hash | qnn | dense
    cells: list                 # [(Cell, score)]
    score: float
    as_of: float
    eq_p: float = -1.0


def _vclass_ttl(tier: str, live: bool, key: RelKey | None, source: str) -> tuple[str, float]:
    if tier == "card" or live:
        return "live", TTL_S["live"]
    if key is not None:
        return key.vclass, key.ttl_s
    return "general", SRC_TTL.get((source or "").lower(), 86400.0)


class Keystone:
    def __init__(self, flags: Flags | None = None, model: ah.AHash | None = None, max_cells: int = 20000,
                 fp_dim: int = 256, fp_quant: str = "int8"):
        self.f = flags or Flags()
        self.cells = CellStore(fp_dim=fp_dim, fp_quant=fp_quant, max_cells=max_cells)
        self.g = Graph()
        self.model = model if model is not None else (ah.AHash.load() if self.f.hash else None)
        self.verifier = verify_eq.load()
        self.codes: dict[int, np.ndarray] = {}           # value cell id -> code
        self._cmat = None
        self.writes = 0

    # ------------------------------------------------------------------ write
    def write(self, q: Q, docs: list, vecs: dict, tier: str = "web", now: float | None = None,
              answer: str | None = None) -> list[int]:
        """docs: lanes.Doc-like (id, title, text, url, source, meta). vecs: doc id -> vector (the reranker's)."""
        now = time.time() if now is None else now
        self.writes += 1
        qt = toks(q.text)
        key = q.key if (self.f.key and q.key is not None) else None
        subj_node = self.g.node(key.subject) if key is not None and key.complete and \
            not key.subject.startswith("@") else None
        q_nodes = [self.g.node(v, kind=t) for t, v in q.slots if t in ah.ENT]
        q_nodes = [n for n in q_nodes if n]
        if subj_node and subj_node not in q_nodes:
            q_nodes.append(subj_node)
        ids = []
        fact_vals = []
        for d in docs:
            v = vecs.get(d.id)
            if v is None:
                continue
            meta = getattr(d, "meta", None) or {}
            src = (d.source or "").lower()
            vclass, ttl = _vclass_ttl(tier, q.live, None, src)
            title = re.sub(r" · .*$", "", d.title or "")
            t_node = self.g.node(title) if title and src in ("wikipedia", "local wikipedia") else None
            fact = meta.get("fact")
            ents = [n for n in [t_node, *q_nodes] if n]
            c = Cell(id=-1, kind="passage", title=d.title or "", payload=key_sentences(d.text, qt), source=d.source,
                     url=d.url, full=(d.text or "")[:1200], question=q.text, arc=dict(q.arc), ents=ents, t=now,
                     ttl=ttl, vclass=vclass, doc_id=d.id)
            i = self.cells.add(c, v)
            ids.append(i)
            for n in q_nodes:
                if t_node and n != t_node:
                    self.g.link(n, t_node)
            m = re.match(r"wikidata:(Q\d+):(P\d+)", d.id or "")
            if fact and m and self.f.key:
                item = self.g.node(fact.get("item", ""), qid=m.group(1))
                rel = FROM_PROP.get(fact.get("prop", ""))
                if item and rel:
                    if key is not None and key.complete and not key.subject.startswith("@"):
                        self.g.alias.setdefault(canon(key.subject), item)     # the spoken name -> the QID node
                        subj_node = item
                    for val in fact.get("values") or []:
                        vn = re.sub(r"\s*\(.*\)$", "", val)
                        vnode = self.g.node(vn) if RELATIONS[rel][0] in ("person", "place", "org", "thing") else None
                        self.g.add_edge(item, rel, val, cell=i, t=now, ttl=TTL_S[RELATIONS[rel][1]], prov="Wikidata",
                                        value_node=vnode)
                    fact_vals.append(d.text)
        # the value cell: this question's answer (typed when the question has a key)
        if ids:
            top = self.cells.cells[ids[0]]
            payload = (answer or (fact_vals[0] if fact_vals else top.payload))[:320]
            vclass, ttl = _vclass_ttl(tier, q.live, key, top.source)
            kt = key.text() if key is not None and key.complete else ""
            arc = dict(q.arc)
            if key is not None:
                arc.update(subject=key.subject, vclass=key.vclass)
            vc = Cell(id=-1, kind="value", title=(kt or q.text)[:120], payload=payload, source=top.source, url=top.url,
                      key=kt, question=q.text, arc=arc, ents=q_nodes, t=now, ttl=ttl, vclass=vclass,
                      doc_id=("value:" + kt) if kt else ("q:" + " ".join(sorted(set(qt)))), links=ids[:3])
            vv = vecs.get(docs[0].id)
            vid = self.cells.add(vc, vv if vv is not None else q.qvec, q.qvec)
            if q.qvec is not None and self.model is not None and self.f.hash:
                self.codes[vid] = self.model.code(q.qvec, q.arc)
                self._cmat = None
            if kt and subj_node and self.f.key:
                self.g.add_edge(subj_node, key.relation, payload, cell=vid, t=now, ttl=ttl, prov=top.source)
        if self.f.act:
            for n in q_nodes:
                for sc in q.scopes:
                    self.g.activate(sc, n, 1.0 if sc.startswith("s:") else 0.5, now)
        return ids

    # ------------------------------------------------------------------ read
    def _code_matrix(self):
        if self._cmat is None:
            live = [i for i in self.codes if i in self.cells.cells]
            self.codes = {i: self.codes[i] for i in live}
            self._cmat = (np.array(live, int), np.stack([self.codes[i] for i in live]) if live else None)
        return self._cmat

    def lookup(self, q: Q, now: float | None = None) -> tuple[list[Cand], dict]:
        now = time.time() if now is None else now
        t0 = time.perf_counter()
        feats: dict = {"live": float(q.live), "n_cells": math.log1p(len(self.cells)), "key_hit": 0.0,
                       "key_known": float(q.key is not None and q.key.complete), "hash_ham": 64.0,
                       "hash_cos": 0.0, "qnn_cos": 0.0, "dense_top": 0.0, "dense_gap": 0.0, "bm25_top": 0.0,
                       "ent_cov": 0.0, "act": 0.0, "age_frac": 1.0, "subj_known": 0.0}
        cands: list[Cand] = []
        key = q.key if self.f.key else None
        # (1) typed key
        if key is not None and key.complete:
            node = self.g.resolve(key.subject) if not key.subject.startswith("@") else None
            feats["subj_known"] = float(node is not None or key.subject.startswith("@"))
            n2, vals = self.g.lookup(key.subject, key.relation, now)
            if vals:
                cells = []
                for v in vals:
                    c = self.cells.cells.get(v.get("cell")) if v.get("cell") is not None else None
                    if c is not None and c.fresh(now):
                        cells.append((c, 1.0))
                        cells += [(self.cells.cells[j], 0.9) for j in c.links if j in self.cells.cells]
                if cells:
                    best_t = max(v["t"] for v in vals)
                    ttl = max(v["ttl"] for v in vals) or 1.0
                    feats["key_hit"] = 1.0
                    feats["age_frac"] = min(1.0, (now - best_t) / ttl)
                    cands.append(Cand("key", _dedup(cells), 1.0, best_t))
        # (2) learned answer-equivalence hash / (3) question NN
        if q.qvec is not None:
            if self.f.hash and self.model is not None and self.codes:
                ids, M = self._code_matrix()
                if M is not None:
                    code = self.model.code(q.qvec, q.arc)
                    ham = ah.hamming(M, code).astype(np.float32)
                    qv = self.cells.qv[ids] @ np.asarray(q.qvec, np.float32)[:768]
                    order = np.lexsort((-qv, ham))
                    for j in order[:5]:
                        c = self.cells.cells.get(int(ids[j]))
                        if c is None or not c.fresh(now):
                            continue
                        feats["hash_ham"], feats["hash_cos"] = float(ham[j]), float(qv[j])
                        if ham[j] <= self.f.hash_r + 8:
                            cells = [(c, 1.0)] + [(self.cells.cells[x], 0.9) for x in c.links if x in self.cells.cells]
                            cands.append(Cand("hash", _dedup(cells), float(qv[j]) - 0.02 * float(ham[j]), c.t))
                        break
            if self.f.qnn:
                nn = self.cells.question_nn(q.qvec, 1, now)
                if nn:
                    c, s = nn[0]
                    feats["qnn_cos"] = s
                    cells = [(c, s)] + [(self.cells.cells[x], s) for x in c.links if x in self.cells.cells]
                    cands.append(Cand("qnn", _dedup(cells), s, c.t))
            # (4) dense passages (+ BM25, + activation bias)
            if self.f.cells and not q.live:      # volatile questions: never from old passages (value cells in TTL only)
                res = self.cells.search(q.qvec, k=8, q_terms=toks(q.text), now=now,
                                        allow=lambda c: c.kind == "passage")
                if res:
                    act = self.g.activation(q.scopes, now) if (self.f.act and q.scopes) else {}
                    scored = []
                    for c, s, parts in res:
                        a = max((act.get(n, 0.0) for n in c.ents), default=0.0)
                        scored.append((c, s + self.f.w_act * math.log1p(a), parts, a))
                    scored.sort(key=lambda x: -x[1])
                    feats["dense_top"] = scored[0][2]["dense"]
                    feats["dense_gap"] = scored[0][2]["dense"] - (scored[1][2]["dense"] if len(scored) > 1 else 0)
                    feats["bm25_top"] = max(p["bm25"] for _, _, p, _ in scored)
                    feats["act"] = max(a for *_, a in scored)
                    subj = (key.subject if key is not None and key.subject and not key.subject.startswith("@")
                            else " ".join(v for t, v in q.slots if t in ah.ENT))
                    st = [w for w in canon(subj).split() if len(w) > 1]
                    if st:
                        words = set()
                        for c, *_ in scored[:5]:
                            words |= set(toks(c.title + " " + c.payload + " " + c.full[:600]))
                        feats["ent_cov"] = sum(1 for w in st if w in words) / len(st)
                    else:
                        feats["ent_cov"] = 0.5
                    cands.append(Cand("dense", [(c, s) for c, s, *_ in scored[:5]], scored[0][2]["dense"],
                                      max(c.t for c, *_ in scored[:5])))
        # answer-equivalence verifier: the stored question of each value-cell candidate vs this question
        feats["eq_p"] = 0.0
        if self.verifier is not None and self.f.verify and q.qvec is not None:
            for c in cands:
                if c.path in ("key", "hash", "qnn"):
                    vc = c.cells[0][0]
                    cos = float(self.cells.qv[vc.id] @ np.asarray(q.qvec, np.float32)[:768]) \
                        if self.cells.has_q[vc.id] else 0.0
                    c.eq_p = self.verifier.p(verify_eq.pair_features(cos, q.arc, vc.arc, q.key, _key_of_arc(vc)))
                    feats["eq_p"] = max(feats["eq_p"], c.eq_p)
        if self.f.act and q.scopes:
            nodes = [self.g.resolve(v) for t, v in q.slots if t in ah.ENT]
            act = self.g.activation(q.scopes, now, spread=0.0)
            feats["act"] = max([feats["act"]] + [act.get(n, 0.0) for n in nodes if n])
        feats["lookup_ms"] = (time.perf_counter() - t0) * 1000
        return cands, feats

    def plausible(self, q: Q) -> bool:
        """Cheap pre-filter before paying for a query embedding: the subject is a known node, or some cell holds at
        least half of the question's content terms (posting-list counts, no vectors)."""
        if q.key is not None and q.key.complete and (q.key.subject.startswith("@") or self.g.resolve(q.key.subject)):
            return True
        if any(self.g.resolve(v) for t, v in q.slots if t in ah.ENT):
            return True
        terms = [w for w in dict.fromkeys(toks(q.text))]
        if not terms:
            return False
        cnt: dict = {}
        for w in terms:
            for i in self.cells.post.get(w, ()):
                cnt[i] = cnt.get(i, 0) + 1
        return bool(cnt) and max(cnt.values()) >= max(1, (len(terms) + 1) // 2)

    def heuristic(self, cands: list[Cand], feats: dict) -> Cand | None:
        """The hand gate: a typed key hit; a hash hit within radius whose question is close; a near-identical
        question; or a strong passage match that names the subject."""
        by = {c.path: c for c in cands}

        def eq_ok(c):
            return c.eq_p < 0 or c.eq_p >= 0.5
        if "key" in by and eq_ok(by["key"]):
            return by["key"]
        if "hash" in by and feats["hash_ham"] <= self.f.hash_r and feats["hash_cos"] >= self.f.hash_cos and \
                eq_ok(by["hash"]):
            return by["hash"]
        if "qnn" in by and feats["qnn_cos"] >= self.f.qnn_cos and eq_ok(by["qnn"]):
            return by["qnn"]
        if "dense" in by and feats["dense_top"] >= self.f.dense_sim and feats["ent_cov"] >= 1.0 and not feats["live"]:
            return by["dense"]
        return None

    def served(self, cand: Cand, now: float | None = None):
        for c, _ in cand.cells[:1]:
            self.cells.hit(c.id)

    def touch(self, q: Q, now: float):
        """A question served locally still activates its entities (memory)."""
        if self.f.act:
            for t, v in q.slots:
                if t in ah.ENT:
                    n = self.g.resolve(v)
                    for sc in q.scopes:
                        self.g.activate(sc, n, 1.0 if sc.startswith("s:") else 0.5, now)

    def stats(self) -> dict:
        return {**self.cells.size_report(), **self.g.stats(), "codes": len(self.codes), "writes": self.writes}


def _key_of_arc(c: Cell):
    if not c.key or not c.arc.get("relation"):
        return None
    rel = c.arc["relation"]
    return RelKey(rel, c.arc.get("subject", ""), RELATIONS.get(rel, ("thing", "slow"))[0], c.arc.get("vclass", "slow"))


def _dedup(cells):
    seen, out = set(), []
    for c, s in cells:
        if c.id not in seen:
            seen.add(c.id)
            out.append((c, s))
    return out
