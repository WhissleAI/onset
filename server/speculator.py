"""The speculator: decides when to search, what to search, when to cancel, and when an early answer can be reused.

Inputs (from Session): a Frame on every ear update (about every 100 ms), the utterance audio so far, and finally the
endpoint (final frame). Output: events for the UI and the answer stream.

mode="headstart": speculative retrieval + early drafts.
mode="baseline" : wait for the endpoint, then search, rerank, answer. Same lanes, same reranker, same model.
"""
from __future__ import annotations

import asyncio
import re
import itertools
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import answer, confidence, config, cost, director, embed, lanes, meter, narrow, planner, recheck, rerank, searchcache, verify
from .frame import NEWS_INTENTS, Frame

_ids = itertools.count(1)

# --- tiered router (server/router.py, ROUTER=on) runs inside lanes.web; these only pass its meter through ---------
ROUTER_KEYS = ("route", "route_vertical", "route_conf", "route_why", "route_ms", "route_fallback", "route_error",
               "tier", "tier_src", "paid", "free_calls", "index_path", "index_p", "index_as_of", "index_ms",
               "index_shadow", "index_error")


def router_counts(web_fires) -> dict:
    """Per question: web-lane calls that reached the paid backend (not cached / replayed / answered by the router),
    free calls the router made (card APIs, Wikidata, ...), and how many fires each tier served."""
    paid = sum(1 for x in web_fires if x.meter.get("paid", 0 if x.meter.get("cached") or x.meter.get("replayed")
                                                   or x.meter.get("backend") in ("cache", None) else 1))
    tiers: dict = {}
    for x in web_fires:
        t = x.meter.get("tier")
        if t:
            tiers[t] = tiers.get(t, 0) + 1
    return {"web_paid": paid, "free_calls": sum(x.meter.get("free_calls", 0) for x in web_fires), "tiers": tiers}


@dataclass
class Fire:
    id: str
    lane: str                  # web | local | audio | final-web | final-local
    query: str
    frame_sig: tuple
    content: tuple
    t_fire: float
    task: asyncio.Task | None = None
    state: str = "fired"       # fired | done | stale | error | cancelled
    t_done: float | None = None
    docs: list = field(default_factory=list)
    used: bool = False
    n_words: int = 0
    meter: dict = field(default_factory=dict)   # web lane: backend, depth, topic, credits, cached, fallback
    key: object = None         # web lanes: searchcache.QKey of the frame it searched (SPEC_EFF policy + cache)
    part: str | None = None    # PLAN: the sub-question (planner part text) this search is for
    raw: list | None = None    # PLAN: the lane's docs as they landed (before their vectors: the plan ranks by lane order)
    landed: asyncio.Event | None = None


@dataclass
class Draft:
    id: str
    meaning_sig: tuple
    question: str
    qvec: np.ndarray
    stream: answer.Stream
    t_start: float
    doc_ids: tuple
    content: tuple = ()
    cover: dict | None = None     # SPEC_EFF: coverage of the borrowed pool it was drafted from


_NON_ANSWER = re.compile(r"\b(don't|didn't|did not|do not|doesn't|does not|no (?:information|mention)|not (?:mention|cover|"
                         r"include|say|state)|can't|cannot|isn't in|aren't in|none of)\b", re.I)


def _non_answer(text: str) -> bool:
    """A draft that says the sources lack the answer (looking at its first sentence)."""
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    about_sources = re.search(r"\b(sources?|results?|provided|information|passages?|these)\b", first, re.I)
    return bool(about_sources and _NON_ANSWER.search(first))


def PlanStream(*a, **kw):
    """PLAN: a multi-part answer is never a single card's one-liner ("weather in Paris and in London" has two): the
    answer stream (answer.Stream, looked up at call time) with its card shortcut off."""
    cls = answer.Stream
    if hasattr(cls, "_card_answer"):
        cls = type("PlanStream", (cls,), {"_card_answer": lambda self: False})
    return cls(*a, **kw)


class Speculator:
    def __init__(self, emit, mode: str = "headstart", t0: float | None = None, ledger: lanes.Ledger | None = None,
                 cost_mode: str | None = None):
        self.emit = emit
        self.prof = cost.profile(cost_mode)       # COST_MODE: speculation budget (server/cost.py)
        self.answering: set = set()               # meanings a finalize (soft or final) is answering: no drafts
        self.eot_seen = False                     # the ear gives an end-of-turn score (Flux / tagger gate)
        self.ledger = ledger if ledger is not None else lanes.Ledger()   # Tavily credits + cache (per session)
        self.budget_noted = False
        self.answer_meter: dict = {}
        self.mode = mode
        self.t0 = t0 or time.perf_counter()
        self.t_wall0 = time.time()                 # search cache: only entries from before this question count
        self.fires: list[Fire] = []
        self.fired_queries: dict[tuple, Fire] = {}
        self.pool: dict[str, lanes.Doc] = {}
        self.vecs: dict[str, np.ndarray] = {}
        self.embedding: dict[str, asyncio.Future] = {}   # doc id -> in-flight embedding (shared by concurrent fires)
        self.carried_ids: set[str] = set()
        self.gen = 0                                     # finalize generation: a newer finalize supersedes older ones
        self.drafts: list[Draft] = []
        self.web_used = 0
        self.local_used = 0
        self.audio_used = 0
        self.last_sig = None
        self.stable = 0
        self.sig_since = 0.0
        self.last_audio_at = 0.0
        self.draft_busy = False
        self.closed = False
        self.bg: list = []                     # after-answer work (VERIFY_RESEARCH re-check); never awaited by finalize
        self.answer_stream = None
        self.last_qvec: np.ndarray | None = None
        self.used_ids: list[str] = []
        self.context = ""                      # previous turn (question + answer), for follow-ups
        self.audio_qvec: np.ndarray | None = None   # the audio lane's latest (longest) spoken-question vector
        self.audio_qdur = 0.0
        self.last_frame: Frame | None = None
        self.held = 0                           # fires held back on a shaky key word (SPEC_CONF)
        # SPEC_EFF: web searches skipped because an earlier search already covers the meaning (reason -> count), and
        # for each skipped query the search whose results stand in for it (drafts + the final coverage check use them)
        self.skips: Counter = Counter()
        self.skip_why: dict[str, str] = {}
        self.eot_p: float | None = None          # the session's fused end-of-turn score for the current words
        self.eot_src = "fused"
        self.final_q: set = set()                # meanings an endpoint already answered (or searched) for
        self.uncovered: dict = {}                # meaning_sig -> Coverage of a pool that failed the draft-time check
        self.parent_of: dict[str, Fire] = {}
        self.cheap_sigs: set = set()             # router plans (vertical + args) already looked up speculatively
        self.sig_fire: dict = {}                 # plan sig -> the speculative cheap lookup for it
        self.cheap_parent: dict[str, Fire] = {}  # query -> the cheap lookup of the same plan (drafts borrow it)
        self.streams: list = []                  # meter: every answer stream of this question (drafts, soft, final)
        self.meter_used = None                   # meter: the stream that answered (last finalize)
        self.meter_metric: dict | None = None
        self.usage_sent = False
        # PLAN (server/planner.py): the multi-part plan of the current words, the search of each part (by part text),
        # when each part's text was first seen (stability), and the entity each multi-hop part found
        self.mplan: planner.MPlan | None = None
        self.part_fire: dict[str, Fire] = {}
        self.part_since: dict[str, float] = {}
        self.bridge: dict[str, str | None] = {}
        self.plan_sig = None

    # ------------------------------------------------------------------ helpers
    def ms(self, t=None):
        return round(((t or time.perf_counter()) - self.t0) * 1000)

    def _ev(self, **kw):
        self.emit({"t": "spec", **kw})

    def _fire(self, lane: str, query: str, f: Frame | None, coro_fn, meter: dict | None = None, part: str | None = None):
        fire = Fire(id=f"f{next(_ids)}", lane=lane, query=query, frame_sig=f.sig if f else (),
                    content=f.content if f else (), t_fire=time.perf_counter(), n_words=f.n if f else 0,
                    meter=meter if meter is not None else {},
                    key=searchcache.qkey(f) if f is not None and lane in ("web", "final-web") else None, part=part)
        self.fires.append(fire)
        self.fired_queries[(lane.replace("final-", ""), query)] = fire
        self._ev(id=fire.id, lane=lane, query=query, state="fired", ms=self.ms(fire.t_fire), word=fire.n_words,
                 **({"key": fire.key.text()} if fire.key is not None else {}), **({"part": part} if part else {}))

        async def run():
            try:
                docs = await coro_fn()
                for d in docs:
                    d.lane = lane
                fire.t_done = time.perf_counter()
                if fire.part is not None:
                    fire.raw = list(docs)
                    fire.landed.set()                       # PLAN: the endpoint / a bridge part needn't wait for vectors
                    self._part_raw(fire)
                await self._add_docs(docs)                  # vectors first, then "done": drafts rank on vectors
                if config.INDEX and f is not None and not f.plain:
                    from .keystone import hook as _ks       # Keystone ingest (server/keystone/): passages, typed
                    _ks.ingest(f, query, docs, self.vecs, fire.meter)   # edges, value cell; no network, no embedding
                fire.docs = docs                            # (a fire cancelled mid-embed keeps none: never revived
                fire.state = "done" if fire.state == "fired" else fire.state     # with vector-less docs)
                if lane == "web" and fire.meter.get("paid") == 0 and fire.meter.get("tier"):
                    self.web_used = max(0, self.web_used - 1)   # ROUTER: a free card / fact / wait fire spends no budget
                self._ev(id=fire.id, lane=lane, query=query, state=fire.state, ms=self.ms(fire.t_done),
                         took=round((fire.t_done - fire.t_fire) * 1000), n=len(docs),
                         top=[d.title for d in docs[:3]], ids=[d.id for d in docs],
                         docs=[self._brief(d, 160) for d in docs[:5]], **self._meter_ev(fire))
                # only paid web results go into the search cache: router cards / facts (meta tier / card / snapshot)
                # would come back as plain docs (searchcache keeps id/title/text/url/source), losing their labels
                web_docs = [d for d in docs if not (d.meta.get("tier") or d.meta.get("card") or d.meta.get("snapshot"))]
                if (fire.key is not None and web_docs and fire.meter.get("tier") in (None, "web")
                        and not (fire.meter.get("cached") or fire.meter.get("replayed") or fire.meter.get("fallback"))):
                    searchcache.cache().put(fire.key, query, web_docs, fire.meter.get("backend") or config.WEB_BACKEND,
                                            vecs={d.id: self.vecs[d.id] for d in web_docs if d.id in self.vecs})
                if (config.SPEC_EFF and lane == "web" and self.mode == "headstart" and not self.closed
                        and self.last_frame is not None and self.skip_why.get(self.last_frame.query()) == "hold"):
                    self._spec_web(self.last_frame)        # a "hold" waiting for these results can decide now
                if fire.part is not None:
                    self._plan_emit()
                if ((config.DRAFT_ON_LAND or self.prof.final_eot is not None) and self.mode == "headstart" and not self.closed and not self.draft_busy
                        and lane in ("web", "local") and self.last_frame is not None
                        and self.last_frame.query() == query and self._conf_ok(self.last_frame)):
                    self._maybe_draft(self.last_frame)     # draft now, not at the next ear update
            except asyncio.CancelledError:
                if fire.landed is not None:
                    fire.landed.set()
                fire.t_done = time.perf_counter()
                if fire.state == "fired":               # a stale fire stays "stale" (revivable; counted as stale)
                    fire.state = "cancelled"
                    self._ev(id=fire.id, lane=lane, query=query, state="cancelled", ms=self.ms(fire.t_done))
                raise
            except lanes.Skip as e:                         # ROUTER: nothing to search yet (free, not an error)
                fire.state = "skipped"
                fire.t_done = time.perf_counter()
                self.web_used = max(0, self.web_used - 1)
                self._ev(id=fire.id, lane=lane, query=query, state="skipped", ms=self.ms(fire.t_done),
                         reason=str(e)[:120], **self._meter_ev(fire))
                if fire.part is not None:
                    fire.landed.set()
                    self._plan_emit()
            except Exception as e:
                fire.state = "error"
                fire.t_done = time.perf_counter()
                self._ev(id=fire.id, lane=lane, query=query, state="error", ms=self.ms(fire.t_done),
                         error=f"{type(e).__name__}: {e}"[:200], **self._meter_ev(fire))
                if fire.part is not None:
                    fire.landed.set()
                    self._plan_emit()
            return fire

        if part is not None:
            fire.landed = asyncio.Event()
        fire.task = asyncio.create_task(run())
        return fire

    @staticmethod
    def _meter_ev(fire: Fire) -> dict:
        m = fire.meter
        if not m:
            return {}
        return {k: m[k] for k in ("backend", "depth", "topic", "time_range", "credits", "cached", "fallback", "replayed",
                                  "cache_match", "cache_miss", "narrow", "semcache", "as_of", "age_s", "cache_query")
                + ROUTER_KEYS if k in m}

    @staticmethod
    def _brief(d: lanes.Doc, n: int = 260) -> dict:
        """Doc.brief plus the "as of" time of a result served from the search cache (the UI labels it)."""
        b = d.brief(n)
        if d.meta.get("as_of") and not d.meta.get("card") and not d.meta.get("tier"):
            # (router cards / facts carry their own "as of" inside meta["card"] / meta["snapshot"]: never "cached")
            b["as_of"] = d.meta["as_of"]
            b["cached"] = True
        return b

    def _web(self, q: str, f: Frame, final: bool, qvec_task: asyncio.Task | None = None):
        """(coroutine factory, meter) for one web-lane call of this question. With CACHE_SEM, the cross-question
        search cache answers first (exact key; at the endpoint also the embedding nearest neighbour, `qvec_task`)."""
        meter: dict = {}
        hints = dict(f.web_hints())
        topic = "news" if f.intent in NEWS_INTENTS else "general"
        ent = f.entity_query() if q == f.query() else ""
        if config.WEB_BACKEND == "tavily":
            # Arc read-out narrowing (server/narrow.py): query shaped from the slots, sources for the question's family
            q = narrow.shape_query(f, q)
            doms, mode = narrow.domains(f)
            if doms:
                hints["include_domains"], hints["domains_mode"] = doms, mode
                meter["narrow"] = narrow.family(f)
        key = searchcache.qkey(f)

        async def go():
            if config.CACHE_SEM and self._cache_ok(f, final):
                c = searchcache.cache()
                hit = c.get(key, before=self.t_wall0)
                if hit is None and qvec_task is not None:
                    cands = c.nn_candidates(key, before=self.t_wall0)
                    if cands:
                        hit = c.nearest(key, (await asyncio.shield(qvec_task))[0], cands)
                if hit is not None:
                    return self._from_cache(hit, meter)
            return await lanes.web(q, topic=topic, intent=f.intent, text=f.resolved_text, final=final,
                                   ledger=self.ledger, meter=meter, entity=ent, hints=hints, frame=f)
        return go, meter

    def _cache_ok(self, f: Frame, final: bool) -> bool:
        """The search cache stands in for the PAID web lane only: with the router on, a frame whose plan goes to a
        card / local facts first is never answered from cached web docs (the cheap lane is live and free)."""
        p = self._plan(f, final)
        return p is None or list(p) == ["web"]

    def _from_cache(self, hit: searchcache.Hit, meter: dict) -> list:
        """A search-cache hit as lane docs: labelled cached + "as of" (meter, events, every doc), no credit spent."""
        e = hit.entry
        meter.update(backend=e.backend, cached=True, semcache=hit.match, as_of=hit.as_of, age_s=round(hit.age_s),
                     credits=0.0, cache_query=e.query)
        docs = []
        for x in e.docs:
            d = lanes.Doc(id=x["id"], title=x["title"], text=x["text"], url=x.get("url", ""), source=x.get("source", ""),
                          meta={"cached": True, "as_of": hit.as_of, "semcache": hit.match})
            if e.vecs and d.id in e.vecs:
                self.vecs.setdefault(d.id, e.vecs[d.id])       # no re-embedding
            docs.append(d)
        return docs

    def _web_affordable(self) -> bool:
        """Speculation stops (quietly) when the session's Tavily credits are down to the final-query reserve."""
        if config.WEB_BACKEND != "tavily":
            return True
        why = self.ledger.allow(lanes.COST.get(config.TAVILY_SPEC_DEPTH, 2), final=False)
        if why and not self.budget_noted:
            self.budget_noted = True
            self._ev(id="budget", lane="web", query="", state="skipped", ms=self.ms(), reason=why,
                     spent=self.ledger.spent)
        return why is None

    async def _add_docs(self, docs):
        """Give every doc a vector, then put it in the pool (the pool only ever holds ranked-ready docs).

        A doc another fire is already embedding is waited for, not re-embedded; if that fire is cancelled mid-embed
        (it went stale), the doc is embedded here instead, so a fresh fire never inherits a vector-less doc."""
        loop = asyncio.get_running_loop()
        ix = lanes.index()
        todo = list(docs)
        while todo:
            need, waits = [], set()
            for d in todo:
                if d.id in self.vecs:
                    self.pool.setdefault(d.id, d)
                    continue
                if d.id in self.embedding:
                    waits.add(self.embedding[d.id])
                    continue
                row = ix.row_of(d.id) if d.source == "local" else None
                if row is not None:
                    self.vecs[d.id] = ix.tv[row]      # the index's own text vector: no re-embedding on the GPU
                    self.pool.setdefault(d.id, d)
                    continue
                if d.id not in {x.id for x in need}:
                    need.append(d)
            if need:
                fut = loop.create_future()
                for d in need:
                    self.embedding[d.id] = fut
                try:
                    vec = await embed.arun(embed.encode_docs, [d.text[:600] for d in need], [d.title for d in need])
                    for d, v in zip(need, vec):
                        self.vecs[d.id] = v
                        self.pool.setdefault(d.id, d)
                finally:
                    for d in need:
                        if self.embedding.get(d.id) is fut:
                            del self.embedding[d.id]
                    if not fut.done():
                        fut.set_result(None)
            if waits:
                await asyncio.wait(waits)             # (wait, not await: our cancel must not cancel their future)
            todo = [d for d in todo if d.id not in self.vecs]
            if todo and not waits and not need:
                break                                  # unreachable; guards against a spin

    def seed(self, pool: dict):
        """Start with the previous question's documents (already embedded): follow-ups rank them instantly."""
        for d, v in pool.values():
            self.vecs.setdefault(d.id, v)
            self.pool.setdefault(d.id, d)
        self.carried_ids |= set(pool)

    def export_pool(self, k: int = 12, qvec: np.ndarray | None = None) -> dict:
        """The docs that fed or nearly fed this answer, for the next question: the answer's own sources first, then the
        rest of the fresh results by similarity to the question (not simply the first k found)."""
        cand: dict[str, lanes.Doc] = {}
        for x in self.fires:
            if x.used or x.state == "done":
                for d in x.docs:
                    if d.id in self.vecs:
                        cand.setdefault(d.id, d)
        qvec = qvec if qvec is not None else self.last_qvec
        first = [i for i in self.used_ids if i in cand]
        rest = [i for i in cand if i not in set(first)]
        if qvec is not None and rest:
            s = np.stack([self.vecs[i] for i in rest]) @ qvec
            rest = [rest[j] for j in np.argsort(-s, kind="stable")]
        return {i: (cand[i], self.vecs[i]) for i in (first + rest)[:k]}

    def rank(self, qvec: np.ndarray, k: int = 5, fresh_only: bool = True):
        """Top-k pool docs by cosine. Docs that only came from stale searches (a misheard or corrected-away query) are
        left out when enough fresh ones exist."""
        ids = [i for i in self.pool if i in self.vecs]
        if fresh_only:
            fresh = set()
            for x in self.fires:
                if x.state in ("done", "fired"):
                    fresh |= {d.id for d in x.docs}
            fresh |= self.carried_ids
            keep = [i for i in ids if i in fresh]
            if len(keep) >= 3:
                ids = keep
        if not ids:
            return []
        M = np.stack([self.vecs[i] for i in ids])
        s = M @ qvec
        order = np.argsort(-s)[:k]
        return [(self.pool[ids[i]], float(s[i])) for i in order]

    def rank_local(self, question: str, qvec: np.ndarray, k: int = 5):
        """Top-k moments of the local index for a question routed to it (server/rerank.py): reciprocal-rank fusion of
        the question's text cosine, the audio lane's spoken-question cosine (when it has landed) and a lexical + tag
        score, over the whole index (a few hundred moments: < 1 ms). The moments join the pool with the index's own
        vectors, so drafts, reuse checks and the next question's carry-over see them like any lane result."""
        ix = lanes.index()
        if not len(ix):
            return self.rank(qvec, k)
        ids = [d.id for d in ix.docs]
        text_s = ix.tv @ qvec
        audio_s = None
        if self.audio_qvec is not None and config.RERANK_AUDIO:
            audio_s = ix.tv @ self.audio_qvec
            if ix.av is not None:
                sa = ix.av @ self.audio_qvec
                for row, v in zip(ix.av_rows, sa):
                    audio_s[row] = max(audio_s[row], v)
        fused = rerank.fuse(ids, text_s, question, rerank.lexicon(ix), audio_s=audio_s)[:k]
        out = []
        for i, _, parts in fused:
            row = ix.row_of(i)
            if i not in self.pool:
                d = ix.docs[row]
                self.pool[i] = lanes.Doc(**{**d.__dict__, "score": parts["text"], "lane": "local"})
            self.vecs.setdefault(i, ix.tv[row])
            out.append((self.pool[i], parts["text"]))
        return out

    def _rank_for(self, f: Frame, qvec: np.ndarray, k: int = 5):
        """The final / draft ranking: the fused local ranker for local-routed questions, else pool cosine."""
        if f.route == "local" and config.RERANK_LOCAL and len(lanes.index()):
            return self.rank_local(f.resolved_text, qvec, k)
        return self.rank(qvec, k)

    # ------------------------------------------------------------------ live updates
    @staticmethod
    def key_conf(f: Frame) -> float | None:
        if not config.SPEC_CONF or not f.wconf:
            return None
        return confidence.key_conf(f, list(f.wconf))[0]

    @staticmethod
    def _director_thr(f: Frame):
        """(fire, draft) thresholds when the director drives speculation for this frame, else None."""
        if not (config.DIRECTOR and config.DIRECTOR_SPEC and f.director):
            return None
        m = director.model()
        t = (m.thr if m else {}) or {}
        return (t.get("director_fire", 0.5), t.get("director_draft", 0.5))

    def _conf_ok(self, f: Frame) -> bool:
        kc = self.key_conf(f)
        return kc is None or kc >= config.SPEC_CONF_DRAFT

    def on_frame(self, f: Frame, now: float | None = None):
        if self.mode != "headstart" or self.closed:
            return
        self.last_frame = f
        now = now or time.perf_counter()
        if f.sig != self.last_sig:
            self.last_sig, self.stable, self.sig_since = f.sig, 1, now
            self.eot_p = None
        else:
            self.stable += 1
        self._mark_stale(f)
        if f.jailbreak:
            return              # Arc flags a prompt-injection attempt: spend no speculative searches / drafts on it
        if self._plan_step(f, now):
            return              # PLAN: a multi-part question: its parts are searched instead of the whole sentence
        need = 1 if f.slots else config.STABLE_UPDATES        # typed spans are confident: fire on first sight
        dthr = self._director_thr(f)
        if dthr is not None:
            # the director's fire head replaces the stability / min-words rules (budget + narrowing gate still apply)
            if f.director["fire"] >= dthr[0] and f.has_content():
                q = f.query()
                self._spec_web(f)
                if q and f.route in ("local", "both") and len(lanes.index()) and ("local", q) not in self.fired_queries \
                        and self.local_used < config.LOCAL_BUDGET:
                    self.local_used += 1
                    self._fire("local", q, f, lambda q=q: lanes.local_text(q))
            if f.director["draft"] >= dthr[1] and not self.draft_busy:
                self._maybe_draft(f)
            return
        kc = self.key_conf(f)
        if kc is not None and kc < config.SPEC_CONF_LO:
            need = max(need, 2)                               # shaky decisive word: wait for one more ear update
        if f.n < config.MIN_WORDS or not f.has_content() or self.stable < need:
            if kc is not None and self.stable < need and f.n >= config.MIN_WORDS and f.has_content():
                self.held += 1
            return
        q = f.query()
        if not q:
            return
        self._spec_web(f)
        if (f.route in ("local", "both") and len(lanes.index()) and ("local", q) not in self.fired_queries
                and self.local_used < config.LOCAL_BUDGET):
            self.local_used += 1
            self._fire("local", q, f, lambda q=q: lanes.local_text(q))
        if (now - self.sig_since) * 1000 >= config.DRAFT_STABLE_MS and not self.draft_busy and self._conf_ok(f):
            self._maybe_draft(f)

    # ------------------------------------------------------------------ PLAN: multi-part questions (server/planner.py)
    def _plan_step(self, f: Frame | None, now: float | None = None) -> bool:
        """A multi-part question: keep the plan in step with the words and fire each part once it is stable (closed
        parts at once, the open one after SPEC_SETTLE_MS or a likely end of turn). True = the plan handled this frame
        (no whole-sentence speculation, no drafts). Single-part questions: planner.decompose -> None, False."""
        if not config.PLAN or f is None or self.mode != "headstart" or self.closed:
            return False
        try:
            mp = planner.decompose(f)
        except Exception:
            mp = None
        now = now or time.perf_counter()
        self._plan_set(mp, now)
        if mp is None:
            return False
        for part in mp.parts:
            text = self._part_text(part)
            if text is None:
                continue                                  # a bridge part waiting for its hop's entity
            since = self.part_since.setdefault(text, now)
            settled = (part.closed or (now - since) * 1000 >= config.SPEC_SETTLE_MS
                       or (self.eot_p is not None and self.eot_p >= config.SPEC_EOT_FIRE))
            if self.prof.final_eot is not None:
                settled = self._meaning_final(f, now)    # parity: the parts are searched once the question is final
            if settled:
                self._part_fire(part, text, f, final=False)
        self._plan_emit()
        return True

    def _plan_set(self, mp, now: float):
        """A new / changed plan: searches of parts the plan no longer has go stale (a correction re-plans)."""
        sig = mp.sig if mp is not None else None
        if sig == self.plan_sig:
            if mp is not None:
                self.mplan = mp
            return
        old = self.mplan
        self.mplan, self.plan_sig = mp, sig
        keep = set()
        if mp is not None:
            for p in mp.parts:
                t = self._part_text(p)
                if t:
                    keep.add(t)
        for text, fire in list(self.part_fire.items()):
            if text not in keep and fire.state in ("fired", "done"):
                was = fire.state
                fire.state = "stale"
                self._ev(id=fire.id, lane=fire.lane, query=fire.query, state="stale", ms=self.ms(), cancelled=False,
                         part=text, replanned=was)
        if old is not None or mp is not None:
            self._plan_emit(force=True)

    def _part_text(self, part) -> str | None:
        if not part.needs:
            return part.text
        hop = self.mplan.part(part.needs) if self.mplan is not None else None
        if hop is None:
            return part.fallback
        if hop.text not in self.bridge:
            return None
        return part.resolve(self.bridge[hop.text])

    def _part_cap_s(self) -> float:
        """Endpoint: how long a plan part's speculative search still on the wire is waited for before a final search
        races it. Parity waits longer (COST_PARITY_PART_WAIT_MS) instead of re-sending the same query."""
        if self.prof.name == "parity" and config.COST_PARITY_PART_WAIT_MS >= 0:
            return max(config.PLAN_PART_CAP_S, config.COST_PARITY_PART_WAIT_MS / 1000)
        return config.PLAN_PART_CAP_S

    def _plan_cap(self) -> int:
        n = len(self.mplan.parts) if self.mplan is not None else 1
        return self.prof.paid_max + config.PLAN_EXTRA_PAID * max(0, n - 1)

    def _part_fire(self, part, text: str, f: Frame, final: bool) -> Fire | None:
        """One search for one part (speculative or final), through the router like any web-lane call."""
        cur = self.part_fire.get(text)
        if cur is not None and (cur.state in ("fired", "done") or (final and cur.lane == "final-web")):
            return cur
        pf = planner.part_frame(text, f, self._bridge_type(part))
        q = pf.query()
        if not q:
            return None
        same = next((x for x in reversed(self.fires) if x.lane in ("web", "final-web") and x.query == q
                     and (x.state in ("fired", "done") or (x.state == "stale" and (x.docs or not x.task.done())))),
                    None)
        if same is not None:
            # the same search already ran: an earlier whole-sentence search of exactly this part, or this part under
            # a wording the plan has since refined ("mozart or" -> "mozart"): reuse it, don't pay twice
            if same.state == "stale":
                same.state = "done" if same.task is None or same.task.done() else "fired"
                self._ev(id=same.id, lane=same.lane, query=same.query, state=same.state, ms=self.ms(), revived=True)
            same.part = text
            if same.landed is None:
                # a whole-sentence fire adopted as a part has no landed event: its run() (still in flight) would raise
                # on landed.set() and never add its docs (seen in the arcvalue conversation run)
                same.landed = asyncio.Event()
                if same.task is None or same.task.done():
                    same.landed.set()
            self.part_fire[text] = same
            return same
        plan = self._plan(pf, final)
        cheap = plan is not None and plan.cheap_only
        if not final:
            if not cheap and (self._paid() >= self._plan_cap() or self.web_used >= config.WEB_BUDGET + len(
                    self.mplan.parts if self.mplan else ())):
                if self.skip_why.get(q) != "plan-budget":
                    self._skip(pf, "plan-budget", None)
                return None
            if not cheap and not self._web_affordable():
                return None
        self.web_used += 1
        go, meter = self._web(q, pf, final=final)
        if cheap and not final:
            meter["no_paid"] = True                       # a free lookup mid-sentence never falls back to paid
        fire = self._fire("final-web" if final else "web", q, pf, go, meter, part=text)
        self.part_fire[text] = fire
        return fire

    def _part_raw(self, fire: Fire):
        """A part's results landed: a hop part's entity (a Wikidata fact value) unlocks its bridge part at once."""
        mp = self.mplan
        if mp is None:
            return
        hit = False
        for p in mp.parts:
            if p.kind == "hop" and p.text == fire.part and p.text not in self.bridge and fire.state == "fired":
                self.bridge[p.text] = planner.bridge_of(fire.raw, p.role)
                hit = True
        if not hit:
            return
        if self.mode == "headstart" and not self.closed and self.last_frame is not None:
            self._plan_step(self.last_frame)
        else:
            self._plan_emit()

    def _plan_emit(self, force: bool = False, final: bool = False):
        mp = self.mplan
        parts = []
        if mp is not None:
            for p in mp.parts:
                t = self._part_text(p)
                fire = self.part_fire.get(t) if t else None
                parts.append({"id": p.id, "text": t or p.text.replace("{B}", "…"), "kind": p.kind, "closed": p.closed,
                              "state": fire.state if fire else ("waiting" if t else "needs " + p.needs),
                              "fire": fire.id if fire else None,
                              "tier": (fire.meter.get("tier") or fire.meter.get("route")) if fire else None,
                              "n": len(fire.docs) if fire else 0})
        ev = {"kind": mp.kind, "why": mp.why, "parts": parts} if mp is not None else {"kind": None, "parts": []}
        sig = repr(ev)
        if not force and not final and sig == getattr(self, "_plan_ev_sig", None):
            return
        self._plan_ev_sig = sig
        self.emit({"t": "plan", "ms": self.ms(), "final": final, **ev})

    def _bridge_type(self, part) -> str:
        if not part.needs or self.mplan is None:
            return ""
        hop = self.mplan.part(part.needs)
        return "place" if hop is not None and hop.role == "capital" else "person"

    @staticmethod
    def _is_landed(fire) -> bool:
        if fire.landed is not None:
            return fire.landed.is_set()
        return fire.task is None or fire.task.done()      # a whole-sentence search reused as a part

    @classmethod
    async def _landed(cls, fire, deadline: float):
        if fire is None or cls._is_landed(fire):
            return
        left = max(0.0, deadline - time.perf_counter())
        if fire.landed is not None:
            try:
                await asyncio.wait_for(fire.landed.wait(), left)
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.wait([fire.task], timeout=left)

    @staticmethod
    def _docs_of(fire) -> list:
        if fire is None or fire.state in ("error", "skipped", "cancelled"):
            return []
        return list(fire.docs or fire.raw or [])

    def _planned(self, f: Frame) -> bool:
        return config.PLAN and self.mode == "headstart" and self.mplan is not None and planner.decompose(f) is not None

    async def _final_plan(self, f: Frame, mp, m: dict, live) -> list | None:
        """Endpoint of a multi-part question (both modes): every part searched (speculative searches reused, missing
        parts fired now, in parallel; a bridge part after its hop), then the sources ranked per part. Returns
        [(doc, score, part index)] or None when superseded."""
        self.mplan = mp
        self.plan_sig = mp.sig
        self.final_q.add(f.query())
        texts = [p.text if not p.needs else p.fallback for p in mp.parts]
        deadline = time.perf_counter() + config.PLAN_WAIT_S
        over = []

        def ensure(part, text):
            fire = self.part_fire.get(text)
            if fire is None or fire.state in ("stale", "error", "cancelled") or (fire.state == "skipped"):
                if fire is not None and fire.lane == "final-web":
                    return fire
                if self._paid() >= self._plan_cap():
                    over.append(part.id)
                self.part_fire.pop(text, None)
                fire = self._part_fire(part, text, f, final=True)
            return fire

        async def run_part(part):
            if part.needs:
                hop = mp.part(part.needs)
                hf = ensure(hop, hop.text) if hop is not None else None
                await self._landed(hf, deadline)
                if hop is not None and hop.text not in self.bridge:
                    self.bridge[hop.text] = planner.bridge_of(self._docs_of(hf), hop.role)
            text = self._part_text(part) or part.fallback
            fire = ensure(part, text)
            if fire is not None and fire.lane == "web" and not self._is_landed(fire):
                # a speculative lookup still out at the endpoint (a slow Wikidata query): give it PLAN_PART_CAP_S
                # (parity: COST_PARITY_PART_WAIT_MS, it is already the question's paid call), then race one final
                # search (router hedge: fact vs web) against it; the first with docs wins
                t_w = time.perf_counter()
                await self._landed(fire, min(deadline, t_w + self._part_cap_s()))
                m["plan_part_wait_ms"] = max(m.get("plan_part_wait_ms", 0), round((time.perf_counter() - t_w) * 1000))
                if not self._is_landed(fire):
                    m["plan_hedged"] = m.get("plan_hedged", 0) + 1
                    spec_fire = fire
                    self.part_fire.pop(text, None)
                    pf = planner.part_frame(text, f, self._bridge_type(part))
                    go, meter = self._web(spec_fire.query, pf, final=True)
                    meter["hedge_ms"] = 0
                    fin = self._fire("final-web", spec_fire.query, pf, go, meter, part=text)
                    self.part_fire[text] = fin
                    waits = {asyncio.ensure_future(self._landed(spec_fire, deadline)),
                             asyncio.ensure_future(self._landed(fin, deadline))}
                    while waits and time.perf_counter() < deadline:
                        done, waits = await asyncio.wait(waits, timeout=deadline - time.perf_counter(),
                                                         return_when=asyncio.FIRST_COMPLETED)
                        if self._docs_of(spec_fire) or self._docs_of(fin):
                            break
                    for w in waits:
                        w.cancel()
                    fire = spec_fire if self._docs_of(spec_fire) else fin
                    self.part_fire[text] = fire
            await self._landed(fire, deadline)
            if fire is not None and not self._docs_of(fire) and fire.lane != "final-web" and fire.state != "fired":
                self.part_fire.pop(text, None)            # an empty / skipped speculative lookup: one final search
                fire = self._part_fire(part, text, f, final=True)
                await self._landed(fire, deadline)
            return text, fire

        res = await asyncio.gather(*[run_part(p) for p in mp.parts], return_exceptions=True)
        if not live():
            return None
        if over:
            m["budget_override"] = "plan-part"
        k = config.PLAN_K_PART or (3 if len(mp.parts) <= 2 else 2)
        out, seen, info = [], set(), []
        for i, (part, r) in enumerate(zip(mp.parts, res)):
            text, fire = r if not isinstance(r, BaseException) else (texts[i], None)
            docs = self._docs_of(fire)
            # lane order (the router puts the card / fact first; Wikipedia's own relevance after): no embedding on the
            # critical path (query vectors cost ~0.1-0.5 s on a busy laptop)
            sc = sorted(((d, float(getattr(d, "score", 0.0) or 0.0)) for d in docs),
                        key=lambda x: -(1 if (x[0].meta.get("strong") or x[0].meta.get("card")) else 0))
            n = 0
            for d, s_ in sc:
                if d.id in seen or n >= k:
                    continue
                seen.add(d.id)
                out.append((d, s_, i))
                n += 1
            info.append({"id": part.id, "text": text, "kind": part.kind, "fire": fire.id if fire else None,
                         "state": fire.state if fire else None, "tier": fire.meter.get("tier") if fire else None,
                         "landed": self._is_landed(fire) if fire else None,   # "fired" + landed: vectors pending
                         "paid": (fire.meter.get("paid", 1) if fire and fire.state == "done" else 0), "n": n})
        m["plan"] = {"kind": mp.kind, "why": mp.why, "parts": info}
        self._plan_emit(final=True)
        return out

    # ------------------------------------------------------------------ web fire policy
    @staticmethod
    def _plan(f: Frame, final: bool = False):
        """Agent R's router (lanes.plan(frame) -> lane names, when it exists): the paid web policy below applies only
        when the plan includes "web"; a cheap-only plan (card / facts) fires free lookups instead. None = no router:
        the frame's own route decides."""
        plan = getattr(lanes, "plan", None)
        if plan is None or getattr(f, "plain", False):
            return None                     # Arc-off arm: no router plan (the frame has no read-out to plan on)
        try:
            return plan(f, final=final)
        except Exception:
            return None

    def _paid(self) -> int:
        """Paid web calls this question so far (in flight or landed): not cache hits, not failures (not billed), not
        router skips, not calls the router answered for free (meter paid == 0: a card / local fact). A call still in
        flight counts as paid (its cheap lane may fall back to the web)."""
        return sum(1 for x in self.fires if x.lane in ("web", "final-web") and x.state not in ("error", "skipped")
                   and not x.meter.get("cached") and not x.meter.get("replayed") and x.meter.get("paid", 1) != 0
                   and not x.meter.get("no_paid"))      # a speculative cheap lookup can never reach the paid web

    def _decide(self, f: Frame) -> tuple[str, Fire | None]:
        """SPEC_EFF: does this frame's meaning need a new web search? ("fire", None) or (why not, the search whose
        results stand in for it). Why not: dup (same key) | subsumed | extends (more words, same subject: re-rank the
        pool) | grows (the entity phrase got longer and the earlier results already name it) | hold (it may be such a
        completion: wait for the earlier search to land)."""
        k = searchcache.qkey(f)
        cands = [x for x in self.fires if x.lane in ("web", "final-web") and x.key is not None
                 and (x.state in ("fired", "done") or (x.state == "stale" and x.docs))]
        best = None
        for x in reversed(cands):
            rel = searchcache.relation(k, x.key)
            if rel == "same":
                return "dup", x
            if rel == "subsumed":
                return "subsumed", x
            if best is not None:
                continue
            if rel == "extends":
                if x.state == "fired":
                    best = ("hold", x)
                elif config.SPEC_DRIFT and searchcache.lex_missing(k, x.key, x.docs):
                    best = ("fire", None)       # the added words aren't in its results: the meaning drifted
                else:
                    best = ("extends", x)
            elif rel == "grows" and config.SPEC_GROW_CHECK:
                added = [w for w in k.subject if not searchcache._tok_in(w, x.key.subject)]
                if x.state == "fired":
                    best = ("hold", x)
                elif searchcache.mentions(x.docs, added):
                    best = ("grows", x)
        return best if best is not None else ("fire", None)

    def _skip(self, f: Frame, why: str, parent: Fire | None):
        q = f.query()
        if parent is not None:
            self.parent_of[q] = parent
        if self.skip_why.get(q) != why:
            self.skips[why] += 1
            self.emit({"t": "spec_skip", "lane": "web", "query": q, "why": why, "ms": self.ms(),
                       "parent": parent.id if parent is not None else None})
        self.skip_why[q] = why

    def _coverage_fire(self, f: Frame, cov):
        """The pool doesn't cover the current partial (draft-time coverage check): that meaning gets its own search
        (the next paid call), now or once it has settled (see _settled)."""
        q = f.query()
        if f.meaning_sig not in self.uncovered:
            self.emit({"t": "spec_skip", "lane": "web", "query": q, "why": "uncovered", "ms": self.ms(),
                       "cover": cov.to_json()})
        self.uncovered[f.meaning_sig] = cov
        self.skip_why[q] = "uncovered"
        self._spec_web(f)

    def _settled(self, f: Frame, k, now: float | None = None) -> bool:
        """Has this meaning held long enough to spend a paid call on it? (time since its words last changed)
          - SPEC_SETTLE_MS for a first call: a partial's last word is often half-heard ("kansas city" -> "... chiefs",
            "switzer" -> "switzerland"), and a prefix that is still growing would spend the call on nothing;
          - SPEC_LAST_SETTLE_MS for the budget's LAST call, and for a first call with no entity and few words
            ("speed light", "when's best"): it must go to the final meaning (the endpoint searches it anyway);
          - at once when the fused end-of-turn score says the turn is probably over (SPEC_EOT_FIRE; Flux ear)."""
        if self.prof.final_eot is not None:
            return self._meaning_final(f)
        if self.last_sig != f.sig:
            return config.SPEC_SETTLE_MS <= 0
        if self.eot_p is not None and self.eot_p >= config.SPEC_EOT_FIRE and (self.eot_src == "fused" or (
                k.subject and not any(x.lane in ("web", "final-web") for x in self.fires))):
            return True                 # the turn is probably over: these are the final words (the tagger's gate,
            #                             a weaker hint than Flux's fused score, only releases a FIRST search that
            #                             names its subject)
        need = config.SPEC_SETTLE_MS
        if config.SPEC_TAIL_SETTLE_MS > need and k.subject and f.content and \
                searchcache.stem(f.content[-1]) in k.subject:
            need = config.SPEC_TAIL_SETTLE_MS       # the entity is the last thing said: it may still be growing
        if self.prof.paid_max > 1 and self._paid() >= self.prof.paid_max - 1:
            need = max(need, config.SPEC_LAST_SETTLE_MS)
        elif not k.subject and len([w for w in k.terms if w not in narrow.GENERIC]) < config.SPEC_NOSUBJ_MIN:
            need = max(need, config.SPEC_LAST_SETTLE_MS)
        return ((now or time.perf_counter()) - self.sig_since) * 1000 >= need

    def _meaning_final(self, f: Frame, now: float | None = None, thr: float | None = None) -> bool:
        """COST_MODE=parity: are these the question's final words? The end-of-turn score of exactly these words >=
        COST_PARITY_EOT (drafts: COST_PARITY_DRAFT_EOT); an ear without a turn score: the words held
        SPEC_LAST_SETTLE_MS."""
        if self.last_sig != f.sig:
            return False
        if self.eot_seen:
            return self.eot_p is not None and self.eot_p >= (self.prof.final_eot if thr is None else thr)
        return ((now or time.perf_counter()) - self.sig_since) * 1000 >= config.SPEC_LAST_SETTLE_MS

    def on_eot(self, p: float, src: str = "fused"):
        """Session: the end-of-turn score of the current words (Flux ear, FLUX_SOFT=fused: every update; other ears:
        1.0 when the tagger's semantic-endpoint gate is open, src="gate"). A high one lets a meaning that was waiting
        to settle spend its call now, a tick or two before the soft endpoint."""
        self.eot_p, self.eot_src = p, src
        self.eot_seen = True
        if config.COST_TRACE_EOT and self.last_frame is not None:
            self._ev(id="eot", lane="eot", query=self.last_frame.query(), state="eot", ms=self.ms(), p=round(p, 3))
        if (self.prof.draft_eot is not None and self.mode == "headstart" and not self.closed and not self.draft_busy
                and self.last_frame is not None and p >= self.prof.draft_eot and self._conf_ok(self.last_frame)):
            self._maybe_draft(self.last_frame)      # parity: drafts start once the meaning is final
        if (config.SPEC_EFF and self.mode == "headstart" and not self.closed and self.last_frame is not None
                and p >= config.SPEC_EOT_FIRE):
            if self._plan_step(self.last_frame):
                return
            self._spec_web(self.last_frame)

    def _spec_web(self, f: Frame):
        """One speculative web search for this frame, if the policy wants one."""
        q = f.query()
        if not q or f.route not in ("web", "both") or ("web", q) in self.fired_queries or q in self.final_q:
            return
        if self._planned(f):
            return                      # PLAN: the parts are searched, not the whole sentence
        plan = self._plan(f)
        if plan is not None and "web" not in plan:
            if plan.cheap_only:
                self._spec_cheap(f, plan)
            return
        if not config.SPEC_EFF:
            if self.web_used < config.WEB_BUDGET and narrow.specific(f) and self._web_affordable():
                self.web_used += 1
                self._fire("web", q, f, *self._web(q, f, final=False))
            return
        if not narrow.specific(f):
            return
        why, parent = self._decide(f)
        if why in ("extends", "grows", "hold") and f.meaning_sig in self.uncovered:
            why = "fire"                # its results were re-ranked for this partial and didn't cover it
        if why == "fire":
            if not self._settled(f, searchcache.qkey(f)):
                return
            if self._paid() >= self.prof.paid_max or self.web_used >= config.WEB_BUDGET:
                self._skip(f, "budget", None)
                return
            if self._route_hold(f, plan):
                self._skip(f, "route-hold", None)
                return
            if not self._web_affordable():
                return
            self.skip_why.pop(q, None)
            self.web_used += 1
            self._fire("web", q, f, *self._web(q, f, final=False))
            return
        self._skip(f, why, parent)     # "hold": decided again on the next update, or when the earlier search lands

    def _route_hold(self, f: Frame, plan) -> bool:
        """Router + efficiency: no PAID speculative call while the router is unsure what kind of question this is
        (plan conf < ROUTER_SPEC_HOLD_CONF: no card / fact / fresh / open-ended cue yet) and the frame is still
        growing (no likely end of turn yet): the next words usually decide the lane ("what's the ..." -> weather card).
        Corrections are never held (q27, sc27: the replacement is the final meaning)."""
        if plan is None:
            return False
        d = getattr(f, "director", None)
        if config.DIRECTOR and config.DIRECTOR_ROUTE and d and "route" in d and not (f.dropped or f.cue):
            # the route head calls the final lane before the rules can: a cheap lane (card / facts / local) or "not a
            # request" -> hold the paid call; the web -> release it now (no waiting for the lane to show itself)
            k = int(max(range(len(d["route"])), key=lambda i: d["route"][i]))
            if d["route"][k] >= config.DIRECTOR_ROUTE_THR:
                return director.ROUTES[k] != "web"
        weak = plan.lane == "web" and plan.conf < config.ROUTER_SPEC_HOLD_CONF
        # the generic "stable factoid" plan (any wh- question: local leads first, paid web as their fallback) is the
        # router's no-signal default while a question starts ("what's a hundred US dollars" -> currency card two
        # words later): the same hold applies to its paid fallback
        generic = config.ROUTER_SPEC_HOLD_LEADS and plan.lane == "facts" and plan.vertical == "leads"
        if not (weak or generic):
            return False
        if f.dropped or f.cue:
            return False
        return not (self.eot_p is not None and self.eot_p >= config.SPEC_EOT_FIRE)

    def _spec_cheap(self, f: Frame, plan):
        """A cheap-only plan (live-data card / local facts): look it up now, free, once per distinct plan (vertical +
        arguments). The call goes through lanes.web -> router.route, so it returns the card / facts (or falls back to
        the paid web itself when the cheap lane comes back empty: counted then)."""
        q = f.query()
        sig = getattr(plan, "sig", None) or (plan.lane, plan.vertical, q)
        if sig in self.cheap_sigs:
            if sig in self.sig_fire:
                self.cheap_parent[q] = self.sig_fire[sig]   # same lookup, more words: drafts borrow its result
            return
        if len(self.cheap_sigs) >= config.ROUTER_SPEC_CHEAP_MAX:
            return
        self.cheap_sigs.add(sig)
        self.web_used += 1                     # refunded when the router answers for free (see _fire.run)
        go, meter = self._web(q, f, final=False)
        meter["no_paid"] = True                # cheap lane only: an empty / weak lookup is skipped, not sent to the web
        self.sig_fire[sig] = self._fire("web", q, f, go, meter)

    def on_audio(self, wav: np.ndarray, route: str = "both", n_words: int = 0):
        """Audio lane: embed the spoken question itself (no words needed) and search the local moments index.

        It shares the GPU with document embedding, so it runs only where it pays: questions routed to the local
        corpus (up to 3 times as more speech arrives), or once early, before the words have settled."""
        if self.mode != "headstart" or not config.AUDIO_LANE or not len(lanes.index()) or route == "web":
            return
        if route != "local":
            return          # index/data/retrieval_eval.json: <1.5 s of audio is near useless (r@1 0.0-0.1)
        if self.audio_used >= 3:
            return
        dur = len(wav) / 16000
        if dur < 1.8 or dur - self.last_audio_at < 0.8:
            return
        self.last_audio_at = dur
        self.audio_used += 1
        w = wav.copy()

        async def go():
            docs, qa = await lanes.local_audio(w)
            if qa is not None and np.any(qa) and dur >= self.audio_qdur:
                self.audio_qvec, self.audio_qdur = qa, dur      # the local ranker fuses it (rank_local)
            return docs

        self._fire("audio", f"<{dur:.1f}s of speech>", None, go)

    def _mark_stale(self, f: Frame):
        cur = set(f.content)
        for fire in self.fires:
            if fire.part is not None:
                continue                            # PLAN: a part's search goes stale when the plan drops the part
            if fire.state == "stale" and fire.content and set(fire.content) <= cur and fire.docs:
                fire.state = "done"                 # the speaker came back to it (ASR flip-flop)
                self._ev(id=fire.id, lane=fire.lane, query=fire.query, state="done", ms=self.ms(), revived=True)
                continue
            if (fire.state == "stale" and not fire.docs and fire.task is not None and fire.task.done()
                    and fire.content and set(fire.content) <= cur
                    and self.fired_queries.get((fire.lane, fire.query)) is fire):
                # back to a query whose search was cancelled before its results were in: forget it so the next frame
                # fires it again (the web lane's cache usually holds the paid-for, finished request: free + instant)
                del self.fired_queries[(fire.lane, fire.query)]
                continue
            if fire.state not in ("fired", "done") or fire.lane.startswith("final") or fire.lane == "audio":
                continue
            if fire.content and not set(fire.content) <= cur:
                was = fire.state
                fire.state = "stale"
                cancel = config.STALE_CANCEL and was == "fired" and fire.task and not fire.task.done()
                if cancel:
                    fire.task.cancel()
                # default: let it finish. The search is already paid for, its results show (struck) in the live
                # results list, it can be revived if the ASR flips back, and ranking ignores stale results anyway.
                self._ev(id=fire.id, lane=fire.lane, query=fire.query, state="stale", ms=self.ms(),
                         cancelled=bool(cancel))

    def _maybe_draft(self, f: Frame):
        """Draft an answer for the current meaning as soon as its search results are in.

        Drafts whose meaning the question has since moved past are cancelled (their tokens are cheap; the time saved
        when the meaning holds is not)."""
        if any(d.meaning_sig == f.meaning_sig for d in self.drafts) or f.meaning_sig in self.uncovered:
            return
        if config.COST_DRAFT_DEDUP and f.meaning_sig in self.answering:
            return                                  # an answer for this meaning is already being generated
        if self.prof.draft_eot is not None and not self._meaning_final(f, thr=self.prof.draft_eot):
            return                                  # parity: draft only the final meaning (the draft is the answer)
        live = [d for d in self.drafts if not d.stream.done and d.meaning_sig != f.meaning_sig]
        for d in live:
            d.stream.cancel()
            self._ev(id=d.id, lane="draft", query=d.question, state="stale", ms=self.ms(), cancelled=True)
        if len(self.drafts) >= self.prof.draft_budget:
            return
        q = f.query()
        mine = [x for x in self.fires if x.query == q and x.lane in ("web", "local")]
        borrowed = False
        if not any(x.lane == "web" for x in mine) and q in self.cheap_parent:
            mine.append(self.cheap_parent[q])       # router: the same card / fact lookup (same plan args): no
            #                                         coverage check needed, it is the answer's own source
        elif config.SPEC_EFF and not any(x.lane == "web" for x in mine) and q in self.parent_of:
            mine.append(self.parent_of[q])          # no search of its own: the search that covers its meaning
            borrowed = True
        ready = [x for x in mine if x.state == "done"]
        if not ready or any(x.state == "fired" for x in mine):
            return                                  # wait for every lane of this query (web is the slow one)
        self.draft_busy = True

        async def go():
            try:
                qvec = (await embed.arun(embed.encode_queries, [f.resolved_text]))[0]
                ranked = self._rank_for(f, qvec)
                top = [d for d, _ in ranked]
                if not top or self.closed or self.last_sig != f.sig:
                    return                          # the question moved on while we embedded
                if borrowed:
                    # pool reuse: the earlier search's results, re-ranked for the longer partial. Draft from them only
                    # if they cover it; else that is a coverage failure: search this meaning now (budget permitting)
                    cov = searchcache.coverage(searchcache.qkey(f), ranked, self._searched_keys())
                    if not cov.ok:
                        self._coverage_fire(f, cov)
                        return
                else:
                    cov = None
                own = {d.id for x in ready for d in x.docs}
                if not own & {d.id for d in top}:
                    return                          # don't draft from other questions' results
                st = answer.Stream(f.resolved_text, top, tag="draft", context=self.context, hint=f.answer_hint())
                self.streams.append(st)
                d = Draft(id=f"d{next(_ids)}", meaning_sig=f.meaning_sig, question=f.text, qvec=qvec, stream=st,
                          t_start=time.perf_counter(), doc_ids=tuple(x.id for x in top), content=f.content,
                          cover=cov.to_json() if cov is not None else None)
                self.drafts.append(d)
                self._ev(id=d.id, lane="draft", query=f.text, state="fired", ms=self.ms(d.t_start))
            finally:
                self.draft_busy = False

        asyncio.create_task(go())

    # ------------------------------------------------------------------ endpoint
    async def finalize(self, f: Frame, t_speech_end: float, close: bool = True, by: str = "ear") -> dict:
        """Called when the question is over (close=True), or at a soft semantic endpoint (close=False: keep
        speculating on later words; a later finalize supersedes this answer). Streams the answer; returns metrics.

        A finalize that is superseded (a newer finalize started, or the speculator was closed) stops at its next step
        and emits nothing more: no sources, answer tokens or metric. Its metric comes back with superseded=True."""
        t_end = time.perf_counter()
        self.gen += 1
        gen = self.gen

        def live():
            return gen == self.gen

        def emit(e):
            if live():
                self.emit(e)

        self._mark_stale(f)                       # the final words can still revise away earlier searches
        self.answering.add(f.meaning_sig)
        if self.answer_stream is not None:
            self.answer_stream.cancel()           # superseded soft answer
            self.answer_stream = None
        self.closed = close
        m = {"mode": self.mode, "question": f.text, "query": f.query(), "intent": f.intent, "endpoint_by": by,
             "endpoint_lag_ms": round((t_end - t_speech_end) * 1000), "web_hints": f.web_hints(),
             "redacted": list(f.redacted), "jailbreak": f.jailbreak,
             **({"heard": list(f.resolved)} if f.resolved else {})}      # LOOP_ASR_FIX rewrites (heard.py)

        def superseded():
            m["superseded"] = True
            return m

        q = f.query() or " ".join(f.words)
        qtext = f.resolved_text
        fplan = self._plan(f, final=True)
        if fplan is not None:
            m["route"], m["route_conf"], m["route_why"] = fplan.lane, fplan.conf, fplan.why[:80]
        qvec_task = asyncio.create_task(embed.arun(embed.encode_queries, [qtext]))
        if config.INDEX and not f.plain:     # (Arc-off: Keystone is Arc-keyed; it neither learns from nor serves it)
            from .keystone import hook as _ks
            _ks.share_qvec(qtext, qvec_task)      # Keystone's endpoint lookup reuses this embedding (no second one)
        reuse_draft = None
        alt = None
        mp = None
        if config.PLAN and not f.plain:
            try:
                mp = planner.decompose(f)
            except Exception:
                mp = None
            if mp is not None and not mp.complete:
                mp = None
        plan_ranked = None
        try:
            if mp is not None:
                # PLAN: a multi-part question: every part searched (in parallel), sources grouped by part
                m["path"] = "plan"
                plan_ranked = await self._final_plan(f, mp, m, live)
                if plan_ranked is None or not live():
                    return superseded()
                qvec = (qvec_task.result()[0] if qvec_task.done() and not qvec_task.cancelled()
                        and qvec_task.exception() is None else None)     # not on the critical path
            elif self.mode == "plain":
                # Arc-off arm: one web search of the raw transcript after the endpoint. No router (cards / facts),
                # no topic / freshness / domain hints, no entity query, no search cache, no local lane.
                pm: dict = {}
                await asyncio.gather(self._fire("final-web", q, f, lambda: lanes.web(
                    q, final=True, ledger=self.ledger, meter=pm), meter=pm).task, return_exceptions=True)
                qvec = (await qvec_task)[0]
                m["path"] = "plain-search"
            elif self.mode == "baseline":
                alt = self._answer_race(q, f)
                waits = []
                if f.route in ("web", "both"):
                    waits.append(self._fire("final-web", q, f, *self._web(q, f, final=True, qvec_task=qvec_task)).task)
                if f.route in ("local", "both") and len(lanes.index()):
                    waits.append(self._fire("final-local", q, f, lambda: lanes.local_text(q)).task)
                await asyncio.gather(*waits, return_exceptions=True)
                qvec = (await qvec_task)[0]
                m["path"] = "search-after-endpoint"
            else:
                qvec = (await qvec_task)[0]
                if not live():
                    return superseded()
                # 1) a draft that answers this question? exact meaning, or a shorter question whose answer already
                #    contains every word the speaker added after it ("how tall is the Eiffel" -> "... Eiffel Tower")
                reuse_draft = await self._pick_draft(f, qvec, m)
                if not live():
                    return superseded()
                if (reuse_draft is not None and getattr(reuse_draft.stream, "by", None) == "card"
                        and not self._card_fits(reuse_draft.stream.card, qtext)):
                    m["draft_rejected_card"] = True     # a card for the partial ("weather in san") isn't the answer
                    reuse_draft = None
                if reuse_draft is None:
                    # 2) the final query already fired (in flight or done)? wait for it, don't re-search
                    # (a stale fire cancelled before its results landed doesn't count: it brought nothing)
                    same = [x for x in self.fires if x.query == q and x.lane in ("web", "local")
                            and (x.state in ("fired", "done") or (x.state == "stale" and x.docs))]
                    pending = [x.task for x in same if x.state == "fired" and x.task]
                    top = self.rank(qvec, 1)
                    best = top[0][1] if top else 0.0
                    m["pool_best"] = round(best, 3)
                    if same:
                        m["path"] = "speculative-hit"
                        if pending and fplan is not None and fplan.lane == "facts":
                            await self._wait_or_hedge([x for x in same if x.state == "fired"], f, q, m)
                        elif pending:
                            # don't let a slow lane hold the answer hostage: answer with what has landed after the cap
                            await asyncio.wait(pending, timeout=config.PENDING_CAP_S)
                    elif fplan is not None and fplan.cheap_only and f.route in ("web", "both"):
                        # router: a live-data card / local fact question. A card already in the pool that fits the
                        # final words answers it; else one cheap lookup now (the router itself falls back to the paid
                        # web if the cheap lane comes back empty or weak). No paid budget involved.
                        sf = self.sig_fire.get(fplan.sig)
                        if sf is not None and sf.state == "fired" and sf.task:
                            if fplan.lane == "facts":
                                await self._wait_or_hedge([sf], f, q, m)
                            else:
                                await asyncio.wait([sf.task], timeout=config.PENDING_CAP_S)   # same lookup in flight
                            if any(x.lane == "final-web" for x in self.fires):
                                sf = None                   # hedged: the final call below already answered
                        if fplan.lane == "card" and self._fitting_card(qtext):
                            m["path"] = "card-hit"
                        elif sf is not None and sf.state == "done" and sf.docs and fplan.lane == "facts":
                            m["path"] = "route-hit"             # the same fact lookup (same plan args) already landed
                        elif any(x.lane == "final-web" for x in self.fires):
                            pass                                # the hedge's final call answered (path set there)
                        else:
                            m["path"] = f"route-{fplan.lane}"
                            await asyncio.gather(self._fire("final-web", q, f, *self._web(q, f, final=True)).task,
                                                 return_exceptions=True)
                    elif config.SPEC_EFF and f.route == "web" and (fplan is None or "web" in fplan):
                        alt = await self._final_eff(f, q, qvec, qvec_task, m)
                    elif best >= config.COMMIT_TAU and self._nothing_new(f):
                        m["path"] = "pool-commit"
                    else:
                        m["path"] = "speculative-miss"
                        alt = self._answer_race(q, f)
                        waits = []
                        if f.route in ("web", "both") and self.web_used < config.WEB_BUDGET + 1 and \
                                (fplan is None or "web" in fplan):
                            waits.append(self._fire("final-web", q, f, *self._web(q, f, final=True)).task)
                        if f.route in ("local", "both") and len(lanes.index()):
                            waits.append(self._fire("final-local", q, f, lambda: lanes.local_text(q)).task)
                        await asyncio.gather(*waits, return_exceptions=True)
                else:
                    m["path"] = "draft-reuse"
                if reuse_draft is None and not self.rank(qvec, 1) and live():
                    # nothing to answer from yet (a slow search past PENDING_CAP_S): an answer with no sources is
                    # worse than a late one, so wait for whatever is still on the wire
                    # (stale searches too: an ASR revision marks a search stale while it is still on the wire)
                    pending = [x.task for x in self.fires if x.task and not x.task.done() and x.lane != "audio"]
                    if pending:
                        m["waited_empty"] = True
                        await asyncio.wait(pending, timeout=config.PENDING_EMPTY_S)
                    if (not self.rank(qvec, 1) and live() and f.route in ("web", "both")
                            and not any(x.lane == "final-web" for x in self.fires)
                            and (fplan is None or "web" in fplan)):
                        # every search so far came back empty: search the final question (over the budget if it
                        # must be: the calls that were spent bought nothing to answer from)
                        if self._paid() >= self.prof.final_max:
                            m["budget_override"] = "empty-pool"
                        m["path"] = (m.get("path") or "") + "+empty-retry"
                        await asyncio.gather(self._fire("final-web", q, f, *self._web(q, f, final=True)).task,
                                             return_exceptions=True)
        finally:
            if not qvec_task.done():
                qvec_task.cancel()
            if alt is not None and not live():
                alt.cancel()                      # superseded: nobody will race it
        if not live():
            return superseded()
        if qvec is not None:
            self.last_qvec = qvec
        ranked = self._rank_for(f, qvec, 5) if plan_ranked is None else [(d, sc) for d, sc, _ in plan_ranked]
        if fplan is not None and fplan.lane == "card" and plan_ranked is None:
            ranked = self._card_first(ranked, qtext, qvec)
        if config.INDEX and qvec is not None and not f.plain:
            from .keystone import hook as _ks
            _ks.note_qvec(f, q, qvec)                    # Keystone: this question's vector on the value cell it wrote
        if config.CACHE_SEM and qvec is not None:
            c = searchcache.cache()
            rids = {d.id for d, _ in ranked}
            for x in self.fires:
                if x.key is not None and x.docs and rids & {d.id for d in x.docs}:
                    c.note(x.key, qvec, {d.id: self.vecs[d.id] for d in x.docs if d.id in self.vecs})
        if reuse_draft is not None:
            st = reuse_draft.stream
            docs = [self.pool[i] for i in reuse_draft.doc_ids if i in self.pool]
            dv = {i: float(self.vecs[i] @ qvec) for i in reuse_draft.doc_ids if i in self.vecs}
            ranked = [(d, dv.get(d.id, 0.0)) for d in docs]      # the draft's own numbering = its citations
            self._ev(id=reuse_draft.id, lane="draft", query=reuse_draft.question, state="reused", ms=self.ms())
            if reuse_draft.cover is not None:
                m["cover"] = reuse_draft.cover
        else:
            docs = [d for d, _ in ranked]
            if plan_ranked is not None:
                groups = [(self._part_text(p) or p.fallback, [j + 1 for j, (_, _, pi) in enumerate(plan_ranked) if pi == i])
                          for i, p in enumerate(mp.parts)]
                hint = planner.answer_hint(mp, groups, qtext) + (" " + f.answer_hint() if f.answer_hint() else "")
                st = PlanStream(qtext, docs, tag="final", context=self.context, hint=hint)
            else:
                st = answer.Stream(qtext, docs, tag="final", context=self.context, hint=f.answer_hint())
            self.streams.append(st)
            if alt is not None:
                st = answer.Race(st, alt)
        t_sources = time.perf_counter()
        m["n_sources"] = len(docs)
        emit({"t": "sources", "ms": self.ms(t_sources),
              "docs": [{**self._brief(d), "score": round(sc, 3)} for d, sc in ranked]})
        self.answer_stream = st
        self.used_ids = [d.id for d in docs]
        used_ids = set(self.used_ids)
        for fire in self.fires:
            fire.used = any(d.id in used_ids for d in fire.docs)
        emit({"t": "used", "fires": [x.id for x in self.fires if x.used], "ms": self.ms()})
        if close:
            for d in self.drafts:
                if d is not reuse_draft:
                    d.stream.cancel()
        t_first = None
        async for tok in st.follow():
            if not live():
                break
            if t_first is None:
                t_first = time.perf_counter()
            emit({"t": "answer", "delta": tok, "ms": self.ms()})
        if not live():
            return superseded()
        t_done = time.perf_counter()
        if st.error:
            emit({"t": "answer_error", "error": st.error})
        v = None
        if config.VERIFY and docs and st.text() and not st.error:
            # VERIFY: after the last token (ready / done times are already taken): the answer's numbers / years /
            # names against the cited sources, ~1 ms, no network
            try:
                v = verify.check(st.text(), docs, qtext)
                m["verify"] = {k: v[k] for k in ("status", "label", "n_claims", "found", "ms")}
                emit({"t": "verify", "ms": self.ms(), **v})
            except Exception as e:
                m["verify"] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:120]}
        spec = [x for x in self.fires if not x.lane.startswith("final")]
        used_spec = [x for x in spec if x.used]
        if spec:
            m["first_fire_lead_ms"] = round((t_speech_end - min(x.t_fire for x in spec)) * 1000)
            m["first_fire_words"] = min((x.n_words for x in spec), default=None)
        if used_spec:
            m["used_fire_lead_ms"] = round((t_speech_end - min(x.t_fire for x in used_spec)) * 1000)
        m.update({
            "sources_ms": round((t_sources - t_speech_end) * 1000),
            "ready_ms": round(((t_first or t_done) - t_speech_end) * 1000),
            "done_ms": round((t_done - t_speech_end) * 1000),
            "web_calls": sum(1 for x in self.fires if x.lane in ("web", "final-web")),
            "paid_calls": self._paid(),
            "cost_mode": self.prof.name,
            "web_skipped": dict(self.skips),
            "search_cached": sum(1 for x in self.fires if x.meter.get("semcache")),
            "search_cached_as_of": min((x.meter["as_of"] for x in self.fires if x.meter.get("semcache")), default=None),
            "spec_fires": len(spec),
            "spec_used": sum(1 for x in spec if x.used),
            "stale": sum(1 for x in self.fires if x.state in ("stale", "cancelled")),
            "drafts": len(self.drafts),
            "draft_reused": reuse_draft is not None,
            "hedged": getattr(st, "hedged", False),
            "answer_by": getattr(st, "by", None) or ("draft" if reuse_draft is not None else "llm"),
            "answer_fallback": getattr(st, "fallback", None),          # "extractive": answer model unreachable
            "answer_fallback_reason": getattr(st, "fallback_reason", None),
            **self._credits(),
            "answer": st.text(),
            "sources": [d.title for d in docs],
            "error": st.error,
        })
        self.meter_used, self.meter_metric = st, m
        m.update(meter.fields(self.streams, st, m))     # answer-model tokens (used + wasted) and est. $
        emit({"t": "metric", **{k: v for k, v in m.items() if k != "answer"}})
        if v is not None and recheck.needs(v):
            # VERIFY_RESEARCH: after the metric (ready / done are taken): one targeted re-check of the key claim
            t = asyncio.create_task(self._recheck(st.text(), docs, qtext, v, m))
            self.bg.append(t)
        return m

    async def _recheck(self, text: str, docs: list, qtext: str, v: dict, m: dict):
        """Verify-triggered re-search (server/recheck.py): updates the metric in place and emits a "recheck" event
        (the verify badge becomes "re-checked ✓" / "corrected"). The paid lane only within this question's budget."""
        # the paid web lane: only for "sources disagree", one call at most, within this question's budget (parity: the
        # endpoint's spare call); free lanes (Wikidata, local leads) always go first
        paid_ok = (config.VERIFY_RESEARCH_PAID and v.get("status") == "disagree"
                   and self._paid() < self.prof.final_max)

        async def web(q):
            meter_ = {}
            res = await lanes.web(q, intent="", text=qtext, final=True, ledger=self.ledger, meter=meter_)
            return list(res)
        try:
            r = await recheck.run(text, docs, qtext, v, web=web, paid_ok=paid_ok)
        except Exception as e:
            r = {"status": "error", "error": f"{type(e).__name__}: {e}"[:120]}
        m["recheck"] = {k: r.get(k) for k in ("status", "label", "claim", "lanes", "paid", "ms", "source", "value")}
        if r.get("patched"):
            m["answer_patched"] = r["patched"]
        if r.get("paid"):
            m["paid_calls"] = (m.get("paid_calls") or 0) + r["paid"]
            m["recheck_paid"] = r["paid"]
        self.emit({"t": "recheck", "ms": self.ms(), **r})
        return r

    async def _final_eff(self, f: Frame, q: str, qvec, qvec_task, m: dict):
        """SPEC_EFF endpoint without a search for the final query itself: answer from the pool when it covers the
        question (coverage check: an earlier search of the same meaning, or any pool with a strong match), else ONE
        final search if the per-question budget (SPEC_PAID_MAX paid calls) still allows it. Returns the Tavily
        answer race or None."""
        self.final_q.add(q)                      # no speculative search for this meaning after this point
        why, parent = self._decide(f)
        inflight = [x.task for x in self.fires if x.lane in ("web", "final-web") and x.task and not x.task.done()]
        if why == "hold" or (parent is not None and parent.state == "fired") or (inflight and not self.rank(qvec, 1)):
            inflight = [x.task for x in self.fires if x.lane in ("web", "final-web") and x.task and not x.task.done()]
            if inflight:                         # an earlier search of this meaning is still on the wire
                await asyncio.wait(inflight, timeout=config.PENDING_CAP_S)
            why, parent = self._decide(f)
        m["spec_verdict"] = why
        if why in ("dup", "subsumed"):           # the same meaning was searched (other word order, or a soft final)
            m["path"] = "speculative-hit"
            return None
        k = searchcache.qkey(f)
        cov = searchcache.coverage(k, self.rank(qvec, 5), self._searched_keys())
        m["cover"] = cov.to_json()
        related = parent is not None and why in ("extends", "grows")
        if cov.ok and (related or cov.sim >= config.SPEC_COVER_SIM_ANY):
            m["path"] = "pool-reuse"
            return None
        if self._paid() >= self.prof.final_max:
            if self.rank(qvec, 1):
                m["path"] = "budget-pool"        # hard cap: answer from what was fetched
                return None
            # the budget went on searches that brought nothing (ASR-garbled partials): answering with no sources at
            # all is worse than one more call; flagged so it shows up in the metric
            m["budget_override"] = "empty-pool"
        m["path"] = "coverage-miss" if related else "speculative-miss"
        alt = self._answer_race(q, f)
        await asyncio.gather(self._fire("final-web", q, f, *self._web(q, f, final=True, qvec_task=qvec_task)).task,
                             return_exceptions=True)
        return alt

    async def _wait_or_hedge(self, fires: list, f: Frame, q: str, m: dict):
        """A speculative FACT lookup still in flight at the endpoint (Wikidata's SPARQL tail): wait for it until
        ROUTER_FACTS_HEDGE_MS after it was sent; if it is still out, one final call races the fact lookup against the
        paid web (router hedge_ms=0: both at once, the first good result wins), so a slow SPARQL query costs one paid
        call instead of up to PENDING_CAP_S of waiting. Free mid-sentence lookups stay free."""
        tasks = [x.task for x in fires if x.task and not x.task.done()]
        if not tasks:
            return
        hedge = config.ROUTER_FACTS_HEDGE_MS if self.prof.final_eot is None else config.COST_PARITY_FACTS_HEDGE_MS
        if hedge is None or hedge < 0:
            await asyncio.wait(tasks, timeout=config.PENDING_CAP_S)
            return
        left = min(config.PENDING_CAP_S, max(0.0, min(x.t_fire for x in fires) + hedge / 1000 - time.perf_counter()))
        if left > 0:
            await asyncio.wait(tasks, timeout=left)
        if all(t.done() for t in tasks) and any(x.docs for x in fires):
            return
        m["path"] = (m.get("path") or "") + "+hedge"
        go, meter = self._web(q, f, final=True)
        meter["hedge_ms"] = 0
        final = self._fire("final-web", q, f, go, meter).task
        # whichever lands first with docs: the spec lookup (free) or the hedge (fact or web)
        waits = {final} | {t for t in tasks if not t.done()}
        deadline = time.perf_counter() + config.PENDING_EMPTY_S
        while waits and time.perf_counter() < deadline:
            done, waits = await asyncio.wait(waits, timeout=deadline - time.perf_counter(),
                                             return_when=asyncio.FIRST_COMPLETED)
            if any(x.docs for x in fires) or (final.done() and not final.cancelled()
                                               and any(x.lane == "final-web" and x.docs for x in self.fires)):
                break

    @staticmethod
    def _card_fits(card, question: str) -> bool:
        from . import router
        try:
            return bool(card) and router.card_fits(card, question)
        except Exception:
            return False

    def _fitting_card(self, question: str) -> bool:
        """A live-data card from a speculative lookup (not stale) that answers the final question as asked."""
        for x in self.fires:
            if x.lane in ("web", "final-web") and x.state == "done":
                for d in x.docs:
                    if d.meta.get("card") and self._card_fits(d.meta["card"], question):
                        return True
        return False

    def _card_first(self, ranked, question: str, qvec):
        """A card question: the fitting card (if any landed) leads the sources, whatever its cosine (the answer's
        card path and the UI read it from there)."""
        for x in self.fires:
            if x.lane in ("web", "final-web") and x.state == "done":
                for d in x.docs:
                    if d.meta.get("card") and self._card_fits(d.meta["card"], question) and d.id in self.vecs:
                        # other cards (an earlier partial's: "weather in chicago" for "... tomorrow") don't fit:
                        # leave them out (two near-identical cards would read as one source twice)
                        rest = [(y, sc) for y, sc in ranked if y.id != d.id and not y.meta.get("card")]
                        return [(d, float(self.vecs[d.id] @ qvec))] + rest[:4]
        return ranked

    def _searched_keys(self) -> list:
        return [x.key for x in self.fires if x.key is not None and x.docs and x.state in ("done", "stale")]

    def _answer_race(self, q: str, f: Frame):
        """TAVILY_ANSWER=1: on the after-endpoint path, also ask Tavily for its own answer (include_answer) in a
        parallel request; whichever of it and Haiku produces text first is spoken. Returns the task or None."""
        if not (config.TAVILY_ANSWER and config.WEB_BACKEND == "tavily" and f.route in ("web", "both")):
            return None
        plan = lanes.web_plan(f.intent, f.resolved_text, final=True, answer=config.TAVILY_ANSWER_DEPTH,
                              hints=f.web_hints())
        return asyncio.create_task(lanes.tavily(q, plan, self.ledger, self.answer_meter, final=True))

    def _credits(self) -> dict:
        """Tavily credits this question cost (speculative vs final), cache hits, and the session's running total."""
        web = [x for x in self.fires if x.lane in ("web", "final-web")]
        spec = sum(x.meter.get("credits", 0.0) for x in web if x.lane == "web")
        final = sum(x.meter.get("credits", 0.0) for x in web if x.lane == "final-web")
        final += self.answer_meter.get("credits", 0.0)
        return {"credits": spec + final, "credits_spec": spec, "credits_final": final,
                "web_cached": sum(1 for x in web if x.meter.get("cached")),
                "web_fallbacks": sum(1 for x in web if x.meter.get("fallback")),
                "web_replayed": sum(1 for x in web if x.meter.get("replayed")),     # served from the recorded cache
                "web_backend": next((x.meter.get("backend") for x in reversed(web) if x.meter.get("backend")),
                                    config.WEB_BACKEND),
                "session_credits": self.ledger.spent,
                **router_counts(web)}

    def _nothing_new(self, f: Frame) -> bool:
        """True if every content word of the final question was already covered by some non-stale search."""
        seen = set()
        for x in self.fires:
            if x.state in ("done", "fired") and x.lane in ("web", "local"):
                seen |= set(x.content)
        return set(f.content) <= seen

    async def _pick_draft(self, f: Frame, qvec, m):
        from .frame import norm
        best = None
        for d in reversed(self.drafts):
            if d.stream.error is not None or getattr(d.stream, "fallback", None):
                continue                    # an extractive draft answered a partial question: re-extract at the end
            if d.stream.done and _non_answer(d.stream.text()):
                m["draft_rejected_nonanswer"] = True
                continue                    # "the sources don't mention X": answer again from the final ranking
            cos = float(d.qvec @ qvec)
            if d.meaning_sig == f.meaning_sig and cos >= 0.9:
                m["draft_cos"], m["draft_match"] = round(cos, 3), "exact"
                return d
            if cos < 0.85 or best is not None:
                continue
            missing = [w for w in f.content if w not in set(d.content)]
            gone = [w for w in d.content if w not in set(f.content)]
            if gone or not missing or len(missing) > 2:
                continue
            best = (d, cos, missing)
        if best is None:
            return None
        d, cos, missing = best
        # verify against the draft's own answer: wait for its first sentence (bounded)
        deadline = time.perf_counter() + 0.6
        while not d.stream.done and time.perf_counter() < deadline:
            txt = d.stream.text()
            if any(p in txt for p in ".!?") and len(txt) > 30:
                break
            await asyncio.sleep(0.02)
        if _non_answer(d.stream.text()):
            m["draft_rejected_nonanswer"] = True
            return None
        ans = set(norm(d.stream.text()).replace("-", " ").split())
        if all(w in ans or any(a.startswith(w) for a in ans) for w in missing):
            m["draft_cos"], m["draft_match"], m["draft_covered"] = round(cos, 3), "completion", missing
            return d
        m["draft_rejected"] = missing
        return None

    def close(self):
        """End of this question: cancel searches, drafts and any answer still streaming (an unconfirmed soft answer
        must not keep generating, or emitting, after the session moved on)."""
        self.closed = True
        self.gen += 1                              # a finalize still running emits nothing more
        if self.answer_stream is not None:
            self.answer_stream.cancel()
        for x in self.fires:
            if x.task and not x.task.done() and x.raw is None:
                x.task.cancel()             # (a PLAN part whose docs landed is only embedding them: let it finish)
        for d in self.drafts:
            d.stream.cancel()
        self._usage_final()

    def _usage_final(self):
        """Meter: the question's final token tally once every stream is stopped (drafts still running when the metric
        went out, a superseded soft answer). Emitted once; the client replaces the metric's token / $ fields with it.
        A question that never got a metric (superseded) still reports its tokens: they were spent."""
        if self.usage_sent or not self.streams:
            return
        self.usage_sent = True
        try:
            spec = [x for x in self.fires if not x.lane.startswith("final")]
            f = meter.fields(self.streams, self.meter_used, self.meter_metric or {
                "paid_calls": self._paid(), "spec_fires": len(spec), "spec_used": 0})
            if self.meter_metric is not None:
                self.meter_metric.update(f)            # session.history / eval rows carry the final tally too
            self.emit({"t": "usage", **f, "answered": self.meter_metric is not None, "mode": self.mode})
        except Exception:
            pass
