"""The answer graph: entities, typed edges to value cells, co-occurrence, and activation (memory).

  nodes       entity keys: a Wikidata QID when one is known ("Q182477"), else "n:<canonical name>"; aliases map every
              canonical name seen for it (the question's span, the doc title, the Wikidata label) to the node
  typed edges (subject node, relation) -> values [{value, node?, cell, t, ttl, prov}]: written by every answered
              question with a typed key (keystone/relkey.py) and by every Wikidata fact the facts lane returned.
              Lookup is O(1) on (subject, relation): paraphrases collapse to one key
  co-occurrence  node <-> node weights from entities named together in a retrieved passage / question
  activation  memory as activation on nodes: per session (half-life minutes) and per user (half-life days), bumped
              by every question that names a node and by the answer's value node; read with one hop of spreading
              (recency-weighted spreading activation). The "you" node holds the user's facts (home -> denver, ...).
              Personalisation = activation-biased ranking; a follow-up's "it" resolves to the most active node of
              the kind it asks for.
"""
from __future__ import annotations

import math
import time
from collections import Counter

from .relkey import canon, learn_name

YOU = "you"


class Graph:
    def __init__(self, session_half_s: float = 900.0, user_half_s: float = 7 * 86400.0):
        self.nodes: dict[str, dict] = {}
        self.alias: dict[str, str] = {}
        self.edges: dict[tuple, list] = {}
        self.cooc: dict[str, Counter] = {}
        self.act: dict[str, dict[str, tuple]] = {}         # scope ("s:<id>" | "u:<id>") -> node -> (level, t)
        self.half = {"s": session_half_s, "u": user_half_s}
        self.lookups = 0
        self.edge_hits = 0

    # ------------------------------------------------------------------ nodes
    def node(self, name: str, qid: str | None = None, kind: str = "") -> str | None:
        c = canon(name)
        if not c or len(c) < 2:
            return None
        key = self.alias.get(c)
        if qid:
            if key is None or not key.startswith("Q"):
                old = key
                key = qid
                n = self.nodes.setdefault(key, {"name": name, "qid": qid, "kind": kind, "aliases": set()})
                if old and old != key and old in self.nodes:            # merge the name node into the QID node
                    self._merge(old, key)
        if key is None:
            key = "n:" + c
        n = self.nodes.setdefault(key, {"name": name, "qid": qid, "kind": kind, "aliases": set()})
        n["aliases"].add(c)
        if kind and not n["kind"]:
            n["kind"] = kind
        self.alias[c] = key
        learn_name(c)
        return key

    def _merge(self, old: str, new: str):
        o = self.nodes.pop(old)
        self.nodes[new]["aliases"] |= o["aliases"]
        for a in o["aliases"]:
            self.alias[a] = new
        for (s, r), vals in list(self.edges.items()):
            if s == old:
                self.edges.setdefault((new, r), []).extend(vals)
                del self.edges[(s, r)]
            for v in vals:
                if v.get("node") == old:
                    v["node"] = new
        if old in self.cooc:
            self.cooc.setdefault(new, Counter()).update(self.cooc.pop(old))
        for c in self.cooc.values():
            if old in c:
                c[new] += c.pop(old)

    def resolve(self, name: str) -> str | None:
        """Node for a spoken name: exact alias, the possessive / plural form ("nvidias"), or a unique alias that the
        name is a word-prefix of ("nvidia" -> "nvidia corporation")."""
        c = canon(name)
        if not c:
            return None
        if c in self.alias:
            return self.alias[c]
        if c.endswith("s") and c[:-1] in self.alias:
            return self.alias[c[:-1]]
        cands = {k for a, k in self.alias.items() if a.startswith(c + " ") or c.startswith(a + " ")}
        return next(iter(cands)) if len(cands) == 1 else None

    # ------------------------------------------------------------------ typed edges
    def add_edge(self, subj: str, relation: str, value: str, *, cell: int | None = None, t: float | None = None,
                 ttl: float = 0.0, prov: str = "", value_node: str | None = None):
        vals = self.edges.setdefault((subj, relation), [])
        t = time.time() if t is None else t
        for v in vals:
            if v["value"].lower() == value.lower():
                v.update(t=max(v["t"], t), ttl=max(v["ttl"], ttl), cell=v["cell"] if cell is None else cell)
                return
        vals.append({"value": value, "node": value_node, "cell": cell, "t": t, "ttl": ttl, "prov": prov})
        if value_node:
            self.link(subj, value_node, 2.0)

    def lookup(self, subj_name: str, relation: str, now: float | None = None) -> tuple[str | None, list]:
        """(subject node, fresh values) for (subject, relation). A subject "@rel(x)" is a hop: resolve rel(x) first
        and look the relation up on its value's node."""
        now = time.time() if now is None else now
        self.lookups += 1
        if subj_name.startswith("@") and "(" in subj_name:
            inner, x = subj_name[1:-1].split("(", 1)
            node, vals = self.lookup(x, inner, now)
            for v in vals:
                n2 = v.get("node") or self.resolve(v["value"])
                if n2:
                    _, out = self.lookup_node(n2, relation, now)
                    if out:
                        return n2, out
            return None, []
        node = self.resolve(subj_name)
        if node is None:
            return None, []
        return self.lookup_node(node, relation, now)

    def lookup_node(self, node: str, relation: str, now: float) -> tuple[str, list]:
        vals = [v for v in self.edges.get((node, relation), []) if v["ttl"] <= 0 or now - v["t"] <= v["ttl"]]
        if vals:
            self.edge_hits += 1
        return node, vals

    def has_edge(self, subj_name: str, relation: str, now: float | None = None) -> bool:
        return bool(self.lookup(subj_name, relation, now)[1])

    def relations_of(self, node: str) -> list[str]:
        return [r for (s, r) in self.edges if s == node]

    # ------------------------------------------------------------------ co-occurrence
    def link(self, a: str, b: str, w: float = 1.0):
        if not a or not b or a == b:
            return
        self.cooc.setdefault(a, Counter())[b] += w
        self.cooc.setdefault(b, Counter())[a] += w

    def neighbours(self, node: str, k: int = 8) -> list[tuple[str, float]]:
        return self.cooc.get(node, Counter()).most_common(k)

    # ------------------------------------------------------------------ activation (memory)
    def activate(self, scope: str, node: str | None, amount: float = 1.0, now: float | None = None):
        if not node:
            return
        now = time.time() if now is None else now
        a = self.act.setdefault(scope, {})
        lvl = self.level(scope, node, now)
        a[node] = (lvl + amount, now)

    def level(self, scope: str, node: str, now: float) -> float:
        x = self.act.get(scope, {}).get(node)
        if x is None:
            return 0.0
        half = self.half.get(scope[:1], 900.0)
        return x[0] * math.pow(0.5, max(0.0, now - x[1]) / half)

    def activation(self, scopes: list[str], now: float | None = None, spread: float = 0.3) -> dict[str, float]:
        """node -> activation summed over scopes, plus one hop of spreading over co-occurrence (normalised weights)."""
        now = time.time() if now is None else now
        base: dict[str, float] = {}
        for sc in scopes:
            for n in self.act.get(sc, {}):
                v = self.level(sc, n, now)
                if v > 1e-3:
                    base[n] = base.get(n, 0.0) + v
        out = dict(base)
        if spread > 0:
            for n, v in base.items():
                nb = self.cooc.get(n)
                if not nb:
                    continue
                tot = sum(nb.values())
                for m, w in nb.most_common(10):
                    out[m] = out.get(m, 0.0) + spread * v * w / tot
        return out

    def most_active(self, scopes: list[str], kind: str | None = None, now: float | None = None) -> str | None:
        """Follow-up resolution by activation: the most active node (of `kind` when given)."""
        act = self.activation(scopes, now, spread=0.0)
        cands = [(v, n) for n, v in act.items() if n != YOU and (kind is None or self.nodes.get(n, {}).get("kind")
                                                                 in (kind, ""))]
        return max(cands)[1] if cands else None

    def user_fact(self, user: str, key: str, value: str, now: float | None = None):
        vn = self.node(value, kind="place" if key == "home" else "")
        self.nodes.setdefault(YOU, {"name": "you", "qid": None, "kind": "you", "aliases": set()})
        self.add_edge(f"{YOU}:{user}", key, value, t=now, ttl=0.0, prov="user", value_node=vn)

    def stats(self) -> dict:
        return {"nodes": len(self.nodes), "typed_edges": sum(len(v) for v in self.edges.values()),
                "edge_keys": len(self.edges), "cooc_pairs": sum(len(c) for c in self.cooc.values()) // 2,
                "lookups": self.lookups, "edge_hits": self.edge_hits}

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> dict:
        return {"nodes": {k: {**v, "aliases": sorted(v["aliases"])} for k, v in self.nodes.items()},
                "edges": [[s, r, vals] for (s, r), vals in self.edges.items()],
                "cooc": {k: dict(v) for k, v in self.cooc.items()}, "act": {k: {n: list(x) for n, x in v.items()}
                                                                           for k, v in self.act.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "Graph":
        g = cls()
        for k, v in (d.get("nodes") or {}).items():
            g.nodes[k] = {**v, "aliases": set(v.get("aliases") or [])}
            for a in g.nodes[k]["aliases"]:
                g.alias[a] = k
                learn_name(a)
        for s, r, vals in d.get("edges") or []:
            g.edges[(s, r)] = vals
        g.cooc = {k: Counter(v) for k, v in (d.get("cooc") or {}).items()}
        g.act = {k: {n: tuple(x) for n, x in v.items()} for k, v in (d.get("act") or {}).items()}
        return g
