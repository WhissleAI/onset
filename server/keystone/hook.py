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
                           max_cells=config.INDEX_MAX_CELLS, fp_dim=config.INDEX_FP_DIM, shared=shared_store())
            if config.INDEX_PERSIST:
                _load(_ks)
        return _ks


_shared = None


def shared_store():
    """The cross-user store (keystone/shared.py): a local SQLite file behind the SharedKeystone API, or None."""
    global _shared
    if not config.INDEX_SHARED:
        return None
    if _shared is None:
        from .shared import SQLiteShared
        try:
            _shared = SQLiteShared(config.KEYSTONE_SHARED if config.INDEX_PERSIST else ":memory:")
        except Exception:  # noqa: BLE001
            return None
    return _shared


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


_saving = threading.Lock()


def save(force: bool = False):
    """Persist (cells + vectors, graph, codes) at most every INDEX_SAVE_S. A cheap snapshot is taken here; the
    serialisation and disk writes run on a background thread (a synchronous save stalled the event loop ~75 ms with
    1.4k cells). force=True writes synchronously (tests, shutdown). Never raises."""
    if not config.INDEX_PERSIST or _ks is None:
        return
    now = time.time()
    if not force and now - _last_save[0] < config.INDEX_SAVE_S:
        return
    _last_save[0] = now
    import copy
    try:
        snap = _ks.cells.snapshot()
        graph = copy.deepcopy(_ks.g.to_json())
        ids = sorted(_ks.codes)
        codes = (np.array(ids, int), np.array([_ks.codes[i] for i in ids], np.uint8) if ids else np.zeros((0, 8), np.uint8))
    except Exception:  # noqa: BLE001
        return
    if force:
        _write(config.KEYSTONE_DIR, snap, graph, codes)
    else:
        threading.Thread(target=_write, args=(config.KEYSTONE_DIR, snap, graph, codes), daemon=True).start()


def _write(d, snap, graph, codes):
    import json
    if not _saving.acquire(blocking=False):
        return                                   # a write is already in progress: the next save catches up
    try:
        from .cells import CellStore
        CellStore.write(d, snap)
        (d / "graph.tmp").write_text(json.dumps(graph, default=list))
        (d / "graph.tmp").replace(d / "graph.json")
        np.savez(d / "codes.npz", ids=codes[0], codes=codes[1])
    except Exception:  # noqa: BLE001
        pass
    finally:
        _saving.release()


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
    slots = live_slots(f)
    return Q(text=text, words=list(f.words), slots=slots, intent=f.intent, acts=tuple(f.acts), live=live,
             qvec=qvec, key=k, arc=ah.arc_of(f.words, f.intent, slots, f.acts, live, k), user=SCOPE["user"],
             session=SCOPE["session"])


def live_slots(f) -> tuple:
    """The frame's typed spans minus any the speaker corrected away ("when was Google founded, no wait, Apple": the
    Google span is dropped), so an abandoned subject never gets a node, an edge, activation or a cell."""
    dropped = getattr(f, "dropped", frozenset())
    if not dropped:
        return tuple(f.slots)
    kept = {w.lower().strip(".,?!'") for i, w in enumerate(f.words) if i not in dropped}
    gone = {w.lower().strip(".,?!'") for i, w in enumerate(f.words) if i in dropped} - kept
    return tuple((t, v) for t, v in f.slots if not (set(str(v).lower().split()) & gone))


def addressed(f) -> bool:
    """Arc's addressedness (discourse.assistant.device_directed) and dialogue act: only questions / requests to the
    assistant write into the store (side talk, TV audio and statements never do)."""
    d = getattr(f, "directed", None)
    if d is not None and d < 0.3:
        return False
    acts = set(getattr(f, "acts", ()) or ())
    return not acts or bool(acts & {"question", "command_request"}) or getattr(f, "intent", "").startswith(
        ("discourse.intent.", "intent.request"))


def _docs(cand, now: float) -> list:
    from ..lanes import Doc
    out = []
    as_of = time.strftime("%Y-%m-%d %H:%M", time.localtime(cand.as_of))
    for c, s in cand.cells[:5]:
        text = c.full or c.payload
        if c.kind == "value" and c.links:
            text = c.payload                       # the answer itself; its source passages follow
        src = c.source or "web"
        shared = cand.path == "shared"
        label = (f"verified by Onset users · as of {as_of}" if shared else f"from your Onset index · as of {as_of}")
        out.append(Doc(id=f"index:{c.doc_id or c.id}", title=c.title, text=text, url=c.url,
                       source=f"{'Onset users' if shared else 'Onset index'} · {src}", score=float(s),
                       meta={"tier": "index", "snapshot": label, "as_of": as_of, "index_path": cand.path,
                             "index_cell": c.id, "index_scope": "shared" if shared else "personal"}))
    return out


async def local_first(f, query: str, meter: dict | None = None, final: bool = True) -> LocalResult | None:
    """See module doc. Never raises. A speculative call (final=False) only tries the typed key: no query embedding
    mid-sentence (the embedding pool is shared with the endpoint's query vector; INDEX_SPEC_EMBED=1 allows it)."""
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
        best = ks.heuristic([c for c in cands if c.path in ("key", "shared")], feats) if cands else None
        if best is None:
            if not final and not config.INDEX_SPEC_EMBED:
                STATS["misses"] += 1
                meter["index_why"] = "speculative: key only"
                return None
            if not ks.plausible(q):
                STATS["misses"] += 1
                return None
            from .. import embed
            budget = max(0.005, config.INDEX_TIMEOUT_MS / 1000 - (time.perf_counter() - t0))
            shared = _QV.get(q.text) or _QV.get(getattr(f, "resolved_text", ""))
            if shared is not None and shared.done() and (shared.cancelled() or shared.exception() is not None):
                shared = None
            try:
                qv = None
                if shared is not None:      # the speculator is already embedding this question: wait for it
                    done, _ = await asyncio.wait({shared}, timeout=budget)
                    if shared in done and not shared.cancelled() and shared.exception() is None:
                        qv = shared.result()[0]
                        meter["index_qvec"] = "shared"
                    elif shared not in done:
                        raise asyncio.TimeoutError
                if qv is None:
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


_QV: dict = {}


def share_qvec(text: str, task):
    """The speculator's endpoint query-embedding task for `text` (finalize): the local lookup awaits it instead of
    embedding the same question a second time on the same single-worker pool."""
    if len(_QV) > 32:
        for k in list(_QV)[:16]:
            del _QV[k]
    _QV[text] = task


def wrap_web(web_fn, f, query: str, meter: dict, final: bool = True):
    """router.route's paid web lane with the local tier in front of it."""
    if not config.INDEX:
        return web_fn

    async def go():
        lr = await local_first(f, query, meter, final)
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
    if not config.INDEX or f is None or not docs or not addressed(f):
        return
    meter = meter or {}
    try:
        fresh = [d for d in docs if (d.meta or {}).get("tier") != "index" and not (d.meta or {}).get("cached")
                 and d.source != "local" and d.id in vecs]     # local moments have their own index: never copied
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


def _value_cell(ks, q):
    from .cells import toks
    k = q.key.text() if q.key is not None and q.key.complete else ""
    return ks.cells.by_doc.get(("value:" + k) if k else ("q:" + " ".join(sorted(set(toks(q.text))))))


def note_answer(f, query: str, answer: str, verify: str | None = None):
    """The answer model's answer becomes the question's value-cell payload (the answer span, not the source's key
    sentences); its typed edge value follows. When server/verify.py found the answer in >= 1 cited source
    (status supported / agree), the typed edge is published to the shared store. Cheap; never raises."""
    if not config.INDEX or f is None or not answer:
        return
    try:
        ks = store()
        q = q_of(f, query)
        i = _value_cell(ks, q)
        if i is None or i not in ks.cells.cells:
            return
        c = ks.cells.cells[i]
        c.payload = answer.strip()[:320]
        if q.key is not None and q.key.complete:
            node = ks.g.resolve(q.key.subject)
            for v in ks.g.edges.get((node, q.key.relation), []) if node else []:
                if v.get("cell") == i:
                    v["value"] = c.payload
            if ks.shared is not None and verify in ("supported", "agree") and not q.key.subject.startswith("@"):
                from .shared import SharedEdge
                ks.shared.publish(SharedEdge(q.key.subject, q.key.relation, c.payload, "verified", verify, time.time(),
                                             q.key.ttl_s, source=c.source, url=c.url))
    except Exception:  # noqa: BLE001
        STATS["errors"] += 1


_PHRASE_REL = {"founded by": "founded_by", "created by": "creator_of", "written by": "author_of",
               "directed by": "director_of", "capital": "capital_of", "headquarters": "hq_of", "born in": "born_place",
               "made by": "manufacturer_of", "chief executive officer": "ceo_of", "spouse": "spouse_of",
               "discovered / invented by": "discoverer_of", "country": "country_of"}


def absorb_memory(mem):
    """Conversation memory (server/memory.py) into the one structure: its entities become nodes (QIDs kept), its
    Wikidata / answer edges typed edges, co-mentions co-occurrence, the turns' entities session activation (newest
    turn strongest), user facts edges from the "you" node. memory.py stays the fallback path. Never raises."""
    if not config.INDEX or mem is None:
        return
    try:
        ks = store()
        g = mem.graph()
        now = time.time()
        keys = {}
        for n in g.get("nodes") or []:
            keys[n["id"]] = ks.g.node(n["label"], qid=n.get("qid"), kind=n.get("kind", ""))
        last = max((max(n.get("turns") or [0]) for n in g.get("nodes") or []), default=0)
        for n in g.get("nodes") or []:
            age = last - max(n.get("turns") or [0])
            ks.g.activate(f"s:{SCOPE['session']}", keys.get(n["id"]), 0.5 ** age, now)
        for e in g.get("edges") or []:
            a, b = keys.get(e.get("a")), keys.get(e.get("b"))
            rel = _PHRASE_REL.get(e.get("rel") or "")
            if a and b and rel:
                ks.g.add_edge(a, rel, ks.g.nodes[b]["name"], t=now, ttl=0.0, prov="memory", value_node=b)
            elif a and b:
                ks.g.link(a, b)
        # user facts ("I live in Denver") stay in memory.py's per-browser store: the process-wide graph is shared by
        # every client of this server, so they are never copied here (and never reach the shared store)
    except Exception:  # noqa: BLE001
        STATS["errors"] += 1


def absorb_wikidata(subject: str, ks: Keystone | None = None) -> int:
    """A subject's cached Wikidata 1-hop neighbourhood (knowledge.prefetch / the facts lane filled knowledge._cache:
    entity search + one SPARQL query for every property the facts lane knows) -> typed edges on its QID node, with
    value nodes for entity-valued properties: a later relation or two-hop question about it is a graph lookup.
    Returns the number of edges written. No network; never raises."""
    try:
        from .. import knowledge
        from .relkey import FROM_PROP, RELATIONS, TTL_S
        ks = ks or store()
        cands = knowledge._cache.get(("s", subject))
        if not cands:
            return 0
        top = cands[0]
        ids = tuple(c["id"] for c in cands[:5])
        node = ks.g.node(top["label"], qid=top["id"])
        ks.g.alias.setdefault(knowledge.norm(subject), node)
        n, now = 0, time.time()
        for prop, (pids, _kind, _ph) in knowledge.PROPS.items():
            rel = FROM_PROP.get(prop)
            rows = knowledge._cache.get(("q", ids, tuple(pids)))
            if not rel or not rows:
                continue
            for pid in pids:
                vals = rows.get((top["id"], pid)) or []
                pref = [v for r, v in vals if r == "PreferredRank"] or [v for _, v in vals]
                for v in list(dict.fromkeys(pref))[:3]:
                    name = v.split(" (")[0]
                    vn = ks.g.node(name) if RELATIONS[rel][0] in ("person", "place", "org", "thing") else None
                    ks.g.add_edge(node, rel, v, t=now, ttl=TTL_S[RELATIONS[rel][1]], prov="Wikidata", value_node=vn)
                    n += 1
                if pref:
                    break
        return n
    except Exception:  # noqa: BLE001
        STATS["errors"] += 1
        return 0
