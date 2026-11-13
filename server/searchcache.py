"""Search efficiency for the web lane: search the subject, not the sentence (agent S).

Three pieces, all used by server/speculator.py:

  qkey(frame)          canonical query key from Arc Tagger's frame: the search topic (Tavily topic / freshness), the
                       subject (entity mentions + the follow-up carry), the content terms, the typed numbers / dates
                       and the relative time. Words are lightly stemmed and Flux's half-heard last word ("celt" of
                       "celtics", "ubiqu" of "ubiquitous") matches its completion by prefix.
  relation(new, old)   how a new partial's key relates to an earlier search: "same", "subsumed" (the earlier search
                       already asked for everything the new one asks), "extends" (same subject, more words: re-rank
                       the pool instead of searching), "grows" (the subject phrase got longer: "eiffel" -> "eiffel
                       tower"; a search only if the earlier results don't already name it), or None (meaning changed:
                       new entity, a correction, a different number / date / topic -> search now).
  coverage(...)        does the pool already answer the final question: embedding similarity of the final query to
                       the pool's best passages + entity and term coverage in their text.
  SearchCache          cross-question semantic cache of web results: exact key, then an embedding nearest neighbour
                       among entries with the same topic + subject + numbers. TTL per topic family (live data:
                       minutes; general: hours; stable facts: days). In memory, optional JSON persistence
                       (CACHE_PERSIST=1 -> runs/search_cache.json, git-ignored, docs only, capped at CACHE_MAX).
                       Every hit is labelled (meter semcache / as_of, doc meta cached + as_of): never shown as live.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import config
from .frame import NUMBER_TYPES, STOP, norm

# ------------------------------------------------------------------------------------------------------ query keys
_IRREG = {"children": "child", "people": "person", "men": "man", "women": "woman", "feet": "foot", "teeth": "tooth"}


def stem(w: str) -> str:
    """Light, deterministic stemming: case, possessive, plural ("chiefs" == "chief", "yolks" == "yolk")."""
    w = norm(w).strip("'-")
    if w.endswith("'s"):
        w = w[:-2]
    w = _IRREG.get(w, w)
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _tok_in(a: str, pool) -> bool:
    """`a` (possibly a half-heard word) matches some token of `pool`: equal, or a >= 3-letter prefix of it."""
    if a in pool:
        return True
    return len(a) >= 3 and any(b.startswith(a) or (len(b) >= 3 and a.startswith(b) and len(a) - len(b) <= 2)
                               for b in pool)


@dataclass(frozen=True)
class QKey:
    topic: str              # general | news | finance | weather (+ "/day" "/week" when a relative time is said)
    subject: frozenset      # stemmed entity tokens (mentions, entity phrases, the carried referent)
    phrases: tuple          # entity phrases (stemmed, space-joined), for "the phrase got longer"
    terms: frozenset        # stemmed content words (the subject's words included)
    nums: frozenset         # typed numbers / dates said: ("date", "tomorrow"), ("money", "$100")
    family: str = ""        # narrow.family (TTL class only; NOT part of the meaning: it flips as words arrive)

    def text(self) -> str:
        """Stable string form (cache key / logs)."""
        n = ",".join(f"{t}={v}" for t, v in sorted(self.nums))
        return f"{self.topic}|{' '.join(sorted(self.terms))}|{n}"


def _topic(f) -> str:
    h = f.web_hints()
    t = h.get("topic") or ("weather" if f.intent == "discourse.intent.weather_query" and not h.get("weather_off")
                           else "general")
    tr = h.get("time_range") or ""
    return f"{t}/{tr}" if tr else t


def qkey(f) -> QKey:
    from . import narrow
    ents = [v for t, v in f.slots if t not in NUMBER_TYPES]
    phrases = list(f.phrases) + ents + ([f.carried] if f.carried else [])
    ph = tuple(dict.fromkeys(" ".join(stem(w) for w in p.split() if norm(w) and norm(w) not in STOP) for p in phrases))
    ph = tuple(p for p in ph if p)
    subject = frozenset(w for p in ph for w in p.split())
    terms = frozenset(stem(w) for w in f.content if stem(w)) | subject
    nums = frozenset((t, norm(v)) for t, v in f.slots if t in NUMBER_TYPES)
    try:
        fam = narrow.family(f)
    except Exception:
        fam = ""
    return QKey(topic=_topic(f), subject=subject, phrases=ph, terms=terms, nums=nums, family=fam)


def _covers(old: frozenset, new: frozenset) -> bool:
    """Every token of `old` is matched by some token of `new` (prefix-aware)."""
    return all(_tok_in(a, new) for a in old)


def _topics_ok(a: str, b: str) -> bool:
    """Same search topic. Arc's news intent flickers on ordinary questions ("Amazon the company" = news_query for a
    word or two), so plain news vs general doesn't count as a change; a time range, finance or weather does."""
    return a == b or {a, b} <= {"general", "news"}


def relation(new: QKey, old: QKey) -> str | None:
    """How a new partial's key relates to an earlier search's key (see module doc). None = meaning changed."""
    if new == old or (_topics_ok(new.topic, old.topic) and new.terms == old.terms and new.nums == old.nums):
        return "same"
    if not _topics_ok(new.topic, old.topic) or not old.nums <= new.nums:
        return None                       # different freshness / topic, or a number / date changed or corrected
    if new.nums == old.nums and _covers(new.terms, old.terms) and _covers(new.subject, old.subject):
        return "subsumed"                 # the speaker (or the ASR) took words back: the old search asked for more
    if not _covers(old.terms, new.terms):
        return None                       # a word was replaced (a correction, or the ASR revised it): search again
    added = [w for w in new.subject if not _tok_in(w, old.subject)]
    if not added:
        return "extends"                  # same subject, more words (or a date / number added): re-rank what we have
    # a NEW entity token. Only "the same phrase got longer" ("eiffel" -> "eiffel tower") can avoid a search.
    for p in new.phrases:
        pw = p.split()
        if not set(added) & set(pw):
            continue
        for q in old.phrases:
            qw = q.split()
            if qw and all(_tok_in(w, pw) for w in qw):
                return "grows"
    return None


def mentions(docs, tokens) -> bool:
    """Every token is named in the title or the lead (first 400 chars) of some doc."""
    if not docs:
        return False
    heads = set()
    for d in docs:
        heads |= {stem(w) for w in norm(f"{d.title} {d.text[:400]}").replace("-", " ").split()}
    return all(_tok_in(t, heads) for t in tokens)


def lex_missing(new: QKey, old: QKey, docs) -> list[str]:
    """Words the new partial added (beyond the old search's terms, minus spoken filler) that none of the old search's
    results contain (title + text): a sign the question has moved past what was searched."""
    from .narrow import GENERIC
    added = [w for w in new.terms if not _tok_in(w, old.terms) and w not in GENERIC and len(w) > 2]
    if not added:
        return []
    words = set()
    for d in docs:
        words |= {stem(w) for w in norm(f"{d.title} {d.text}").replace("-", " ").split()}
    return [w for w in added if not _tok_in(w, words)]


@dataclass
class Coverage:
    ok: bool
    sim: float
    ent: float
    terms: float

    def to_json(self):
        return {"ok": self.ok, "sim": round(self.sim, 3), "ent": round(self.ent, 2), "terms": round(self.terms, 2)}


def coverage(key: QKey, ranked, searched=(), k: int = 5) -> Coverage:
    """Does the pool answer the final question? `ranked` = [(doc, cosine to the final query)], best first.
    ok = best cosine >= SPEC_COVER_SIM, every subject token named in the top-k docs, >= SPEC_COVER_TERMS of the
    content terms found in them (prefix-aware, stemmed), and - for live questions (news / sports / weather / prices)
    - some search actually asked for its topic and every date / number said (`searched` = keys of the searches
    whose results are in the pool): "who won the game" results don't answer "... on Sunday"."""
    top = ranked[:k]
    if not top:
        return Coverage(False, 0.0, 0.0, 0.0)
    if (key.topic != "general" or key.family in LIVE_FAMILIES) and not any(
            s.nums >= key.nums and _topics_ok(s.topic, key.topic) and s.topic.split("/")[0] == key.topic.split("/")[0]
            for s in searched):
        return Coverage(False, float(top[0][1]), 0.0, 0.0)
    sim = float(top[0][1])
    words = set()
    for d, _ in top:
        words |= {stem(w) for w in norm(f"{d.title} {d.text}").replace("-", " ").split()}
    ent = (sum(_tok_in(t, words) for t in key.subject) / len(key.subject)) if key.subject else 1.0
    terms = (sum(_tok_in(t, words) for t in key.terms) / len(key.terms)) if key.terms else 1.0
    ok = sim >= config.SPEC_COVER_SIM and ent >= config.SPEC_COVER_ENT and terms >= config.SPEC_COVER_TERMS
    return Coverage(ok, sim, ent, terms)


# ------------------------------------------------------------------------------------------- cross-question cache
LIVE_FAMILIES = {"weather", "finance", "sports", "news"}
STABLE_FAMILIES = {"definition", "convert", "recipe", "home", "reference"}


def ttl_class(key: QKey) -> str:
    """live (news / sports / weather / prices: minutes) | stable (definitions, conversions, recipes, reference facts
    with no relative time: days) | general (hours)."""
    if key.topic != "general" or key.family in LIVE_FAMILIES:
        return "live"
    if key.family in STABLE_FAMILIES and not any(t in ("date", "time", "time_of_day") for t, _ in key.nums):
        return "stable"
    return "general"


def ttl_s(key: QKey) -> float:
    return {"live": config.CACHE_TTL_LIVE_S, "stable": config.CACHE_TTL_STABLE_S}.get(ttl_class(key),
                                                                                     config.CACHE_TTL_GENERAL_S)


@dataclass
class Entry:
    key: QKey
    query: str
    backend: str
    docs: list                    # [{"id","title","text","url","source"}]
    t: float                      # wall time fetched (time.time())
    expires: float
    vecs: dict | None = None      # doc id -> vector (memory only)
    qvecs: list | None = None     # query vectors of questions this entry answered (memory only: the NN fallback)


@dataclass
class Hit:
    entry: Entry
    match: str                    # exact | nn
    sim: float = 1.0

    @property
    def as_of(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.entry.t))

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.entry.t)


class SearchCache:
    def __init__(self, path=None):
        self.path = path
        self.entries: dict[str, Entry] = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        if path is not None:
            self._load()

    def __len__(self):
        return len(self.entries)

    def clear(self):
        with self._lock:
            self.entries.clear()
            self.hits = self.misses = 0

    def _live(self, e: Entry, now: float) -> bool:
        return e.expires > now

    def get(self, key: QKey, before: float | None = None) -> Hit | None:
        """Exact hit. `before`: only entries fetched before this wall time (the asking question's start): a Head
        Start search of the SAME question never answers the baseline's (the A/B stays honest), earlier questions'
        searches answer both modes alike."""
        now = time.time()
        e = self.entries.get(key.text())
        if e is not None and self._live(e, now) and (before is None or e.t < before):
            self.hits += 1
            return Hit(e, "exact")
        return None

    def nn_candidates(self, key: QKey, before: float | None = None) -> list[Entry]:
        """Entries that may answer the same question with other words: same topic, numbers and subject."""
        now = time.time()
        return [e for e in self.entries.values() if self._live(e, now) and e.qvecs and e.key.topic == key.topic
                and (before is None or e.t < before)
                and e.key.nums == key.nums and e.key.subject == key.subject and e.key.text() != key.text()]

    def nearest(self, key: QKey, qvec: np.ndarray, cands: list[Entry] | None = None) -> Hit | None:
        best = None
        for e in (cands if cands is not None else self.nn_candidates(key)):
            s = max(float(v @ qvec) for v in e.qvecs)
            if s >= config.CACHE_NN_SIM and (best is None or s > best[1]):
                best = (e, s)
        if best is None:
            self.misses += 1
            return None
        self.hits += 1
        return Hit(best[0], "nn", best[1])

    def put(self, key: QKey, query: str, docs, backend: str, vecs: dict | None = None):
        if not docs or not config.CACHE_SEM:
            return
        now = time.time()
        e = Entry(key=key, query=query, backend=backend, t=now, expires=now + ttl_s(key),
                  docs=[{"id": d.id, "title": d.title, "text": d.text, "url": d.url, "source": d.source}
                        for d in docs], vecs=dict(vecs or {}), qvecs=[])
        with self._lock:
            self.entries[key.text()] = e
            if len(self.entries) > config.CACHE_MAX:
                for k in sorted(self.entries, key=lambda k: self.entries[k].t)[: len(self.entries) - config.CACHE_MAX]:
                    del self.entries[k]
        self._save()

    def note(self, key: QKey, qvec: np.ndarray | None = None, vecs: dict | None = None):
        """Remember the final question's vector (NN fallback) and doc vectors for an entry this question used."""
        e = self.entries.get(key.text())
        if e is None:
            return
        if qvec is not None and not any(float(v @ qvec) > 0.999 for v in e.qvecs or []):
            e.qvecs = (e.qvecs or [])[-7:] + [np.asarray(qvec, np.float32)]
        if vecs:
            e.vecs = {**(e.vecs or {}), **{k: v for k, v in vecs.items() if any(d["id"] == k for d in e.docs)}}

    # ------------------------------------------------------------------- persistence (docs only, small)
    def _save(self):
        if self.path is None or not config.CACHE_PERSIST:
            return
        try:
            data = {"version": 1, "entries": [
                {"key": {"topic": e.key.topic, "subject": sorted(e.key.subject), "phrases": list(e.key.phrases),
                         "terms": sorted(e.key.terms), "nums": sorted(map(list, e.key.nums)), "family": e.key.family},
                 "query": e.query, "backend": e.backend, "t": e.t, "expires": e.expires, "docs": e.docs}
                for e in self.entries.values() if e.expires > time.time()]}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self.path)
        except OSError:
            pass

    def _load(self):
        if not config.CACHE_PERSIST:
            return
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        now = time.time()
        for x in data.get("entries") or []:
            if x.get("expires", 0) <= now:
                continue
            k = x["key"]
            key = QKey(topic=k["topic"], subject=frozenset(k["subject"]), phrases=tuple(k["phrases"]),
                       terms=frozenset(k["terms"]), nums=frozenset(tuple(n) for n in k["nums"]),
                       family=k.get("family", ""))
            self.entries[key.text()] = Entry(key=key, query=x["query"], backend=x.get("backend", ""), docs=x["docs"],
                                             t=x["t"], expires=x["expires"], vecs={}, qvecs=[])


_cache: SearchCache | None = None


def cache() -> SearchCache:
    """The process-wide cache (both modes share it, like the recorded web cache)."""
    global _cache
    if _cache is None:
        _cache = SearchCache(config.CACHE_PATH)
    return _cache


def reset(path=None):
    """Tests / A/B: a fresh, empty cache (and forget the persisted file's contents for this process)."""
    global _cache
    _cache = SearchCache(None) if path is None else SearchCache(path)
    return _cache

