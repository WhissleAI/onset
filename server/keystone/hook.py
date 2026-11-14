"""Keystone in the live app: the local tier before paid web, and ingestion of everything retrieved.

    local_first(frame, query, meter) -> LocalResult | None
        Called by router.route (through wrap_web) before the paid web lane, for speculative and final calls alike.
        Volatile questions can only be served from cells still inside their minutes-long TTL. Order: the typed key
        (O(1), no embedding: a hit costs microseconds); then, only if the store could plausibly answer (a known
        subject node or any value cells), one query embedding and the hash / question-NN / dense lookups; the VOI
        gate (or the hand gate) decides. Bounded by INDEX_TIMEOUT_MS; any error = None (pay as before).
    ingest(frame, query, docs, vecs, meter)
        Called by the speculator after a fire's docs got their reranker vectors: passages -> cells, facts -> typed
        edges, the question -> a value cell + activation. Docs the index itself served are not re-written.

Everything is labelled: docs served locally carry tier "index", source "Onset index · <original source>" and a
snapshot "from your Onset index · as of <time it was fetched>".
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np

from .. import config
from . import ahash as ah
from . import voi as voimod
from .core import Flags, Keystone, Q
from .relkey import key_of_frame

_ks: Keystone | None = None
_lock = threading.Lock()
_last_save = [0.0]
STATS = {"lookups": 0, "hits": 0, "key_hits": 0, "misses": 0, "errors": 0, "timeouts": 0, "ingested": 0,
         "skipped_live": 0, "ms": []}
SCOPE = {"user": "local", "session": "s0"}


@dataclass
class LocalResult:
    docs: list
    path: str
    p: float
    as_of: float
    label: str


def store() -> Keystone:
    global _ks
    with _lock:
        if _ks is None:
            _ks = Keystone(Flags(key=config.GRAPH_KEY, act=config.GRAPH_ACT, hash=config.INDEX_HASH),
                           max_cells=config.INDEX_MAX_CELLS, fp_dim=config.INDEX_FP_DIM)
            if config.INDEX_PERSIST:
                _load(_ks)
        return _ks


def reset(ks: Keystone | None = None):
    """Tests / A/B: a fresh, empty store (nothing loaded from disk)."""
    global _ks
    _ks = ks if ks is not None else Keystone(Flags(key=config.GRAPH_KEY, act=config.GRAPH_ACT,
                                                   hash=config.INDEX_HASH), max_cells=config.INDEX_MAX_CELLS)
    for k in STATS:
        STATS[k] = [] if k == "ms" else 0
    return _ks


def _load(ks: Keystone):
    import json
    from .graph import Graph
    d = config.KEYSTONE_DIR
    try:
        ks.cells.load(d)
        rm = getattr(ks.cells, "remap", {})
        ks.g = Graph.from_json(json.loads((d / "graph.json").read_text()))
        for vals in ks.g.edges.values():          # cell ids were reassigned on load
            for v in vals:
                if v.get("cell") is not None:
                    v["cell"] = rm.get(int(v["cell"]))
        z = np.load(d / "codes.npz")
        ks.codes = {rm[int(i)]: c for i, c in zip(z["ids"], z["codes"]) if int(i) in rm}
    except (OSError, ValueError, KeyError):
        pass


def save(force: bool = False):
    """Persist (cells + vectors, graph, codes) at most every 30 s. Never raises."""
    if not config.INDEX_PERSIST or _ks is None:
        return
    now = time.time()
    if not force and now - _last_save[0] < 30:
        return
    _last_save[0] = now
    import json
    try:
        d = config.KEYSTONE_DIR
        d.mkdir(parents=True, exist_ok=True)
        with _lock:
            _ks.cells.save(d)
            # cell ids change on reload: rebuild codes on load from the saved order
            (d / "graph.tmp").write_text(json.dumps(_ks.g.to_json(), default=list))
            (d / "graph.tmp").replace(d / "graph.json")
            ids = sorted(_ks.codes)
            np.savez(d / "codes.npz", ids=np.array(ids, int), codes=np.array([_ks.codes[i] for i in ids], np.uint8)
                     if ids else np.zeros((0, 8), np.uint8))
    except Exception:  # noqa: BLE001
        pass


def is_live(f) -> bool:
    from .. import searchcache
    try:
        return searchcache.ttl_class(searchcache.qkey(f)) == "live" or bool(f.when)
    except Exception:  # noqa: BLE001
        return bool(getattr(f, "when", ""))


def q_of(f, query: str, qvec=None) -> Q:
    k = key_of_frame(f) if config.GRAPH_KEY else None
    live = is_live(f)
    text = getattr(f, "inline", "") or getattr(f, "resolved_text", "") or query
    return Q(text=text, words=list(f.words), slots=tuple(f.slots), intent=f.intent, acts=tuple(f.acts), live=live,
             qvec=qvec, key=k, arc=ah.arc_of(f.words, f.intent, f.slots, f.acts, live, k), user=SCOPE["user"],
             session=SCOPE["session"])


def _docs(cand, now: float) -> list:
    from ..lanes import Doc
    out = []
    as_of = time.strftime("%Y-%m-%d %H:%M", time.localtime(cand.as_of))
    for c, s in cand.cells[:5]:
        text = c.full or c.payload
        if c.kind == "value" and c.links:
            text = c.payload                       # the answer itself; its source passages follow
        src = c.source or "web"
        out.append(Doc(id=f"index:{c.doc_id or c.id}", title=c.title, text=text, url=c.url, source=f"Onset index · {src}",
                       score=float(s), meta={"tier": "index", "snapshot": f"from your Onset index · as of {as_of}",
                                             "as_of": as_of, "index_path": cand.path, "index_cell": c.id}))
    return out


async def local_first(f, query: str, meter: dict | None = None) -> LocalResult | None:
    """See module doc. Never raises."""
    meter = meter if meter is not None else {}
    if not config.INDEX or f is None:
        return None
    t0 = time.perf_counter()
    STATS["lookups"] += 1
    try:
        ks = store()
        if not len(ks.cells):
            STATS["misses"] += 1
            return None
        q = q_of(f, query)
        now = time.time()
        # (1) the typed key alone: no embedding
        cands, feats = ks.lookup(q, now)
        best = ks.heuristic([c for c in cands if c.path == "key"], feats) if cands else None
        if best is None:
            if not ks.plausible(q):
                STATS["misses"] += 1
                return None
            from .. import embed
            budget = max(0.005, config.INDEX_TIMEOUT_MS / 1000 - (time.perf_counter() - t0))
            try:
                qv = (await asyncio.wait_for(embed.arun(embed.encode_queries, [q.text]), budget))[0]
            except asyncio.TimeoutError:
                STATS["timeouts"] += 1
                meter["index_why"] = "timeout"
                return None
            q.qvec = qv
            cands, feats = ks.lookup(q, now)
            model = voimod.load() if config.VOI else None
            if model is not None:
                best = voimod.choose(model, cands, feats, config.VOI_THETA)
            else:
                best = ks.heuristic(cands, feats)
        ms = (time.perf_counter() - t0) * 1000
        STATS["ms"] = (STATS["ms"] + [ms])[-200:]
        meter["index_ms"] = round(ms, 1)
        if best is None:
            STATS["misses"] += 1
            return None
        if q.live and best.path == "dense":
            STATS["skipped_live"] += 1
            return None
        STATS["hits"] += 1
        STATS["key_hits"] += best.path == "key"
        if not config.INDEX_SERVE:
            meter["index_shadow"] = best.path
            return None
        ks.served(best, now)
        ks.touch(q, now)
        p = float(getattr(best, "p", 1.0 if best.path == "key" else 0.0))
        return LocalResult(_docs(best, now), best.path, p, best.as_of, "Onset index")
    except Exception as e:  # noqa: BLE001
        STATS["errors"] += 1
        meter["index_error"] = f"{type(e).__name__}: {e}"[:120]
        return None


def wrap_web(web_fn, f, query: str, meter: dict):
    """router.route's paid web lane with the local tier in front of it."""
    if not config.INDEX:
        return web_fn

    async def go():
        lr = await local_first(f, query, meter)
        if lr is None:
            return await web_fn()
        from ..lanes import WebResult
        res = WebResult(lr.docs)
        res.backend = "index"
        meter.update(backend="index", credits=0.0, index_path=lr.path, index_p=round(lr.p, 3),
                     index_as_of=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(lr.as_of)))
        return res
    return go


def ingest(f, query: str, docs, vecs: dict, meter: dict | None = None):
    """After a fire's docs were embedded (speculator). Cheap (no network, no embedding); never raises."""
    if not config.INDEX or f is None or not docs:
        return
    meter = meter or {}
    try:
        fresh = [d for d in docs if (d.meta or {}).get("tier") != "index" and not (d.meta or {}).get("cached")
                 and d.id in vecs]
        if not fresh:
            return
        tier = meter.get("tier") or ("card" if any((d.meta or {}).get("card") for d in fresh) else "web")
        ks = store()
        q = q_of(f, query)
        from .. import embed  # noqa: F401  (query vector: reuse the speculator's when it exists)
        q.qvec = None
        ks.write(q, fresh, vecs, tier=tier)
        STATS["ingested"] += 1
        save()
    except Exception:  # noqa: BLE001
        STATS["errors"] += 1


def note_qvec(f, query: str, qvec):
    """The endpoint's query vector for this question (speculator finalize): attach it to the value cell the question
    wrote so later paraphrases find it (the hash / question NN need it). Cheap; never raises."""
    if not config.INDEX or f is None or qvec is None:
        return
    try:
        ks = store()
        q = q_of(f, query, qvec)
        k = q.key.text() if q.key is not None and q.key.complete else ""
        from .cells import toks
        doc_id = ("value:" + k) if k else ("q:" + " ".join(sorted(set(toks(q.text)))))
        i = ks.cells.by_doc.get(doc_id)
        if i is None:
            return
        ks.cells.qv[i] = np.asarray(qvec, np.float32)[:768]
        ks.cells.has_q[i] = True
        if ks.model is not None:
            ks.codes[i] = ks.model.code(qvec, q.arc)
            ks._cmat = None
    except Exception:  # noqa: BLE001
        STATS["errors"] += 1


def health() -> dict:
    ms = sorted(STATS["ms"])
    out = {k: v for k, v in STATS.items() if k != "ms"}
    out["p50_ms"] = round(ms[len(ms) // 2], 1) if ms else None
    if _ks is not None:
        out.update(_ks.stats())
    return {"on": config.INDEX, "serve": config.INDEX_SERVE, **out}
