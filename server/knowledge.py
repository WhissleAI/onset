"""Local-knowledge tier: stable facts without a paid web search.

  1. Wikidata properties (free, ~0.3-0.6 s): "who founded Nvidia" -> P112 of the best-matching item that HAS a P112.
     Entity search (wbsearchentities) + one SPARQL query over the top candidates.
  2. Local Wikipedia lead index (no network, ~30 ms): lead paragraphs of ~10k popular / vital articles, embedded with
     EmbeddingGemma 2 (index/build_knowledge.py -> index/data/knowledge/). "who/what is X", superlatives, definitions.
  3. Wiktionary definitions (free) for "what does <word> mean"; the Wikipedia REST summary (free) for "who/what is X"
     when the local index has no confident hit.

Each returns lanes.Doc with meta tier="facts" and meta strong=True when it answers the question by itself (the router
then skips the paid web lane)."""
from __future__ import annotations

import asyncio
import datetime as dt
import html
import json
import re
import time
from dataclasses import dataclass

import numpy as np

from . import config
from .verticals import TTLCache, get_json

_cache = TTLCache(6 * 3600, 2048)

# property -> (Wikidata ids tried in order, value kind, phrase)
PROPS = {
    "founder": (["P112"], "item", "founded by"),
    "inception": (["P571"], "time", "founded / created"),
    "capital": (["P36"], "item", "capital"),
    "height": (["P2048", "P2044"], "quantity", "height / elevation"),
    "born_date": (["P569"], "time", "born"),
    "born_place": (["P19"], "item", "born in"),
    "died_date": (["P570"], "time", "died"),
    "creator": (["P170", "P50", "P86", "P57", "P287", "P178", "P61"], "item", "created by"),
    "author": (["P50", "P86", "P676"], "item", "written by"),
    "director": (["P57"], "item", "directed by"),
    "discoverer": (["P61", "P170", "P178", "P287"], "item", "discovered / invented by"),
    "language": (["P37"], "item", "official language"),
    "currency": (["P38"], "item", "currency"),
    "country": (["P17"], "item", "country"),
    "hq": (["P159"], "item", "headquarters"),
    "symbol": (["P246"], "string", "element symbol"),
    "manufacturer": (["P176", "P178"], "item", "made by"),
    "ceo": (["P169"], "item", "chief executive officer"),
    "population": (["P1082"], "quantity", "population"),
    "river": (["P206"], "item", "located next to body of water"),
    "head_state": (["P35", "P6"], "item", "head of state / government"),
    "area": (["P2046"], "quantity", "area"),
    "spouse": (["P26"], "item", "spouse"),
}
# properties whose value changes over time: medium confidence (web fallback when the lookup is weak)
CHANGING = {"ceo", "population", "head_state", "spouse"}

_S = r"(?P<s>[a-z0-9][a-z0-9 .'&-]*?)"
_SC = r"(?P<s>[a-z0-9][a-z0-9 .,'&-]*?)"          # a subject said first ("Python, the programming language, who...")
# (pattern on the normalized question, property). First match wins; the subject group is the entity.
PATTERNS = [
    (rf"\bwho (?:founded|co-?founded|started|established) (?:the )?(?:company )?{_S}$", "founder"),
    (rf"^{_SC},? who (?:founded|started|co-?founded) (?:that|the|this|it)?(?: company)?$", "founder"),
    (rf"\b(?:who is|who's) the founder of {_S}$", "founder"),
    (rf"\b(?:when|what year) (?:was|did) {_S} (?:get )?(?:founded|established|created|start(?:ed)?)$", "inception"),
    (rf"^{_SC},? (?:when|what year) was (?:it|that|this) (?:founded|established|created|started)$", "inception"),
    (rf"\bcapital (?:city )?of {_S}$", "capital"),
    (rf"\bhow (?:tall|high) is {_S}$", "height"),
    (rf"\bwhat is the (?:height|elevation) of {_S}$", "height"),
    (rf"\b(?:when|what year) was {_S} born$", "born_date"),
    (rf"^{_SC},? (?:when|what year) was (?:he|she) born$", "born_date"),
    (rf"\bwhere was {_S} born$", "born_place"),
    (rf"^{_SC},? where was (?:he|she) born$", "born_place"),
    (rf"\b(?:when|what year) did {_S} die$", "died_date"),
    (rf"\bwho (?:painted|sculpted|drew) {_S}$", "creator"),
    (rf"\bwho wrote {_S}$", "author"),
    (rf"\bwho (?:is|was) the author of {_S}$", "author"),
    (rf"\bwho directed {_S}$", "director"),
    (rf"\bwho (?:created|designed|developed|invented|made) {_S}$", "discoverer"),
    (rf"^{_SC},? who (?:created|designed|developed|invented|made) (?:it|that|this)$", "discoverer"),
    (rf"\bwho discovered {_S}$", "discoverer"),
    (rf"\bwhat languages? (?:is|are) (?:spoken|used) in {_S}$", "language"),
    (rf"\bwhat (?:is the )?(?:official )?languages? (?:of|in) {_S}$", "language"),
    (rf"\bwhat currency does {_S} use$", "currency"),
    (rf"\bwhat is the currency (?:of|in) {_S}$", "currency"),
    (rf"^{_SC},? (?:which|what) country is (?:it|that|this) in$", "country"),
    (rf"\b(?:which|what) country is {_S} in$", "country"),
    (rf"\bwhere is (?:the )?headquarters of {_S}$", "hq"),
    (rf"\bwhere (?:is|are) {_S} headquartered$", "hq"),
    (rf"\b(?:the )?headquarters of {_S}$", "hq"),
    (rf"\bchemical symbol (?:for|of) {_S}$", "symbol"),
    (rf"\b(?:which|what) company (?:makes|made|manufactures|produces) {_S}$", "manufacturer"),
    (rf"\bwho (?:makes|made|manufactures) {_S}$", "manufacturer"),
    (rf"\bwho is the ceo of {_S}$", "ceo"),
    (rf"\bwhat is the population (?:of|in) {_S}$", "population"),
    (rf"\bhow many people live in {_S}$", "population"),
    (rf"\bwhat river (?:runs|flows|goes) through {_S}$", "river"),
    (rf"\bwho is the (?:current )?(?:president|prime minister|leader) of {_S}$", "head_state"),
    (rf"\bwho is {_S} married to$", "spouse"),
]
PATTERNS = [(re.compile(p), k) for p, k in PATTERNS]
# question heads with the subject still to come ("who founded the company"): wait for it rather than search
HEADS = re.compile(r"\b(who (?:founded|started|painted|wrote|directed|discovered|invented|created)|capital of|how tall is|"
                   r"(?:when|what year) was|where was|chemical symbol for|what currency does|headquarters of|"
                   r"which company makes|who is the ceo of|what is the population of|what does the word)"
                   r"(?: the| a| an)?(?: company| novel| book| movie| film| painting)?$")
_TYPE_HINT = {"company": "company", "novel": "novel", "book": "book", "movie": "film", "film": "film",
              "painting": "painting", "programming language": "programming language", "band": "band", "song": "song",
              "play": "play", "the company": "company", "language": "language", "city": "city", "country": "country"}
# a superlative / ordinal question is not a property of one item
_NOT_PROP = re.compile(r"\b(first|second|third|last|most|least|largest|biggest|tallest|longest|highest|oldest|"
                       r"youngest|best|worst|next|latest|current|recent|newest|smallest|fastest)\b")
FILLER = re.compile(r"^(?:(?:um|uh|so|okay|ok|hey|well|oh|like|hmm|and|quick question|can you tell me|"
                    r"do you know|i was wondering|i forget|i've always wanted to know|tell me)[ ,]+)+")


def norm(text: str) -> str:
    t = text.lower().replace("’", "'").replace("?", " ").replace("!", " ")
    t = re.sub(r"[^a-z0-9 .,'&-]+", " ", t)
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    return FILLER.sub("", t).strip(" ,")


@dataclass
class FactQ:
    prop: str
    subject: str
    hint: str = ""


_PRON = {"it", "he", "she", "that", "this", "they", "him", "her", "them", "those", "these", "there"}


def _clean_subject(s: str) -> tuple[str, str]:
    s = s.strip(" ,.'")
    s = re.sub(r"^(?:the|a|an) ", "", s)
    hint = ""
    for w, h in sorted(_TYPE_HINT.items(), key=lambda x: -len(x[0])):
        m = re.search(rf"(?:,? (?:the|a|an))? {re.escape(w)}$", s)
        if m and m.start() > 0:
            s, hint = s[: m.start()], h
            break
        if s.startswith(w + " "):
            s, hint = s[len(w) + 1:], h
            break
    s = re.sub(r"^(?:the|a|an) ", "", s.strip(" ,"))
    s = re.sub(r",? (?:the|a|an)$", "", s)
    return s.strip(" ,"), hint


def parse_fact(text: str) -> FactQ | None:
    t = norm(text)
    for rx, prop in PATTERNS:
        m = rx.search(t)
        if not m:
            continue
        subj, hint = _clean_subject(m.group("s"))
        if not subj or subj in _PRON or subj in _TYPE_HINT or len(subj.split()) > 6:
            continue
        if _NOT_PROP.search(subj) and prop not in ("ceo", "head_state"):
            return None
        if prop == "head_state" and re.search(r"\b(first|second|third|last)\b", t):
            return None
        return FactQ(prop, subj, hint)
    return None


def waiting_head(text: str) -> bool:
    """The question so far is a property head whose subject hasn't been said yet."""
    return bool(HEADS.search(norm(text)))


_DEF = re.compile(r"\b(?:what does (?:the word |the term )?(?P<a>[a-z-]+) mean|(?:define|definition of|meaning of) "
                  r"(?:the word )?(?P<b>[a-z-]+))\b")
_WHO = re.compile(r"^(?:who|what) (?:is|was|are|were|'s) (?:a |an |the )?(?P<s>[a-z0-9][a-z0-9 .'&-]{1,40})$")


def parse_define(text: str) -> str | None:
    m = _DEF.search(re.sub(r"\s+", " ", re.sub(r"\b(?:actually|exactly|really|even)\b", "", norm(text))))
    if not m:
        return None
    w = m.group("a") or m.group("b")
    return None if w in ("it", "that", "this") else w


def parse_whois(text: str) -> str | None:
    t = re.sub(r"\b(?:actually|exactly|really|anyway)\b", "", norm(text))
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r" and how does it work$| and what does it do$", "", t)
    m = _WHO.search(t)
    if not m:
        return None
    s = m.group("s").strip()
    if _NOT_PROP.search(s) or re.search(r"\b(weather|time|score|price|news|best|difference|point|way)\b", s):
        return None
    return s if len(s.split()) <= 5 else None


# ------------------------------------------------------------------------------------------------- Wikidata
WD_API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"


async def _search(subject: str, k: int = 6) -> list[dict]:
    hit = _cache.get(("s", subject))
    if hit is not None:
        return hit
    data = await get_json(WD_API, {"action": "wbsearchentities", "search": subject, "language": "en",
                                   "uselang": "en", "format": "json", "limit": k, "type": "item"},
                          timeout=max(config.ROUTER_TIMEOUT_S, 0.8), breaker_s=20)
    out = [{"id": x["id"], "label": (x.get("display", {}).get("label") or {}).get("value") or x.get("label", ""),
            "desc": (x.get("display", {}).get("description") or {}).get("value") or x.get("description", "")}
           for x in data.get("search") or []]
    _cache.put(("s", subject), out)
    return out


def _sparql(ids: list[str], pids: list[str]) -> str:
    """One query for every candidate item x every property tried (Wikidata's endpoint allows ~5 concurrent queries per
    client: one query per lookup, not one per property)."""
    vals = " ".join(f"wd:{i}" for i in ids)
    props = " ".join(f"(p:{p} ps:{p} psv:{p} \"{p}\")" for p in pids)
    return f"""SELECT ?item ?pid ?rank ?v ?vLabel ?vd ?amt ?unitLabel ?t ?prec WHERE {{
 VALUES ?item {{{vals}}}
 VALUES (?p ?ps ?psv ?pid) {{{props}}}
 ?item ?p ?st . ?st wikibase:rank ?rank . FILTER(?rank != wikibase:DeprecatedRank)
 ?st ?ps ?v .
 OPTIONAL {{ ?st ?psv ?vn . OPTIONAL {{ ?vn wikibase:quantityAmount ?amt ; wikibase:quantityUnit ?unit . }}
            OPTIONAL {{ ?vn wikibase:timeValue ?t ; wikibase:timePrecision ?prec . }} }}
 FILTER NOT EXISTS {{ ?st pq:P582 ?end . }}
 OPTIONAL {{ ?v schema:description ?vd . FILTER(lang(?vd) = "en") }}
 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,mul". }}
}} LIMIT 300"""


_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def _fmt_time(t: str, prec: int) -> str:
    m = re.match(r"([+-]?\d+)-(\d\d)-(\d\d)", t)
    if not m:
        return t
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    era = " BC" if y <= 0 else ""
    y = abs(y) if y > 0 else abs(y) + 1
    if prec >= 11 and mo and d:
        return f"{_MONTHS[mo - 1]} {d}, {y}{era}"
    if prec == 10 and mo:
        return f"{_MONTHS[mo - 1]} {y}{era}"
    if prec <= 7:
        return f"{(y - 1) // 100 + 1}th century{era}"
    return f"{y}{era}"


def _fmt_qty(amt: str, unit: str) -> str:
    v = float(amt)
    u = "" if unit in ("1", "") else unit
    s = f"{v:,.2f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v):,}"
    if u == "metre":
        ft = v / 0.3048
        return f"{s} metres ({ft:,.0f} feet)"
    if u == "square kilometre":
        return f"{s} km²"
    return f"{s} {u}".strip()


_AGENT = {"founder", "creator", "author", "director", "discoverer", "manufacturer", "ceo"}
_DESCRIBE = {"P159", "P19", "P36", "P17", "P206", "P176"}      # place-like values: say which one


async def _wd_values(ids: list[str], pids: list[str]) -> dict[tuple, list[tuple]]:
    """{(item, pid): [(rank, value), ...]}"""
    key = ("q", tuple(ids), tuple(pids))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    data = await get_json(SPARQL, {"query": _sparql(ids, pids), "format": "json"},
                          headers={"Accept": "application/sparql-results+json"}, timeout=config.ROUTER_SLOW_TIMEOUT_S,
                          breaker_s=20)
    out: dict[tuple, list[tuple]] = {}
    for b in data["results"]["bindings"]:
        item = b["item"]["value"].rsplit("/", 1)[-1]
        pid = b["pid"]["value"]
        rank = b["rank"]["value"].rsplit("#", 1)[-1]
        if "amt" in b:
            val = _fmt_qty(b["amt"]["value"], (b.get("unitLabel") or {}).get("value", ""))
        elif "t" in b:
            val = _fmt_time(b["t"]["value"], int((b.get("prec") or {}).get("value", 11)))
        else:
            val = (b.get("vLabel") or b["v"])["value"]
            if re.fullmatch(r"Q\d+", val) or val.startswith("http"):
                continue                                  # no English label
            vd = (b.get("vd") or {}).get("value", "")
            if vd and pid in _DESCRIBE:
                val = f"{val} ({vd})"                     # "Toyota (city in Aichi Prefecture, Japan)"
        out.setdefault((item, pid), []).append((rank, val))
    _cache.put(key, out)
    return out


async def wikidata(q: FactQ) -> list:
    """Docs answering q from Wikidata (strong), or []. Raises on network failure (the router logs it)."""
    from .lanes import Doc
    cands = await _search(q.subject)
    if not cands:
        return []
    if q.hint:
        cands = sorted(cands, key=lambda c: 0 if q.hint in (c["desc"] or "").lower() else 1)
    cands = cands[:5]
    ids = [c["id"] for c in cands]
    pids, kind, phrase = PROPS[q.prop]
    await _await_prefetch(ids)
    res = await _wd_values(ids, pids)
    for c in cands:
        for pid in pids:
            if (c["id"], pid) not in res:
                continue
            vals = res[(c["id"], pid)]
            pref = [v for r, v in vals if r == "PreferredRank"]
            vals = pref or [v for _, v in vals]
            # a value naming the item itself ("Netflix" founded by "Netflix, Inc.") is the wrong item: try the next one
            sub = q.subject.lower()
            vals = [v for v in dict.fromkeys(vals)
                    if q.prop not in _AGENT or sub not in v.lower().split(" (")[0]][:6]
            if not vals:
                continue
            desc = f" ({c['desc']})" if c["desc"] else ""
            text = f"{c['label']}{desc}: {phrase}: {', '.join(vals)}."
            return [Doc(id=f"wikidata:{c['id']}:{pid}", title=f"{c['label']} · Wikidata", text=text,
                        url=f"https://www.wikidata.org/wiki/{c['id']}#{pid}", source="Wikidata", score=1.0,
                        meta={"tier": "facts", "strong": True, "fact": {"prop": q.prop, "pid": pid, "values": vals,
                                                                        "item": c["label"]},
                              "snapshot": "Wikidata, live"})]
    return []


_inflight: dict[tuple, asyncio.Future] = {}       # candidate ids -> a prefetch's SPARQL query on the wire


async def _await_prefetch(ids: list[str], budget_s: float = 1.5):
    """A lookup whose candidates a pre-warm is fetching right now waits for it (bounded) instead of sending a second
    query (Wikidata allows ~5 concurrent queries per client; two at once for the same items only slows both)."""
    fut = _inflight.get(tuple(ids))
    if fut is not None and not fut.done():
        try:
            await asyncio.wait_for(asyncio.shield(fut), budget_s)
        except Exception:
            pass


async def prefetch(subject: str) -> bool:
    """Warm the cache for follow-up questions about `subject` (server/memory.py pre-warm, after an answer): its entity
    search and ONE SPARQL query for every property in PROPS, split into the per-property entries wikidata() reads.
    Free (Wikidata); returns True when the cache was filled. Never raises."""
    try:
        cands = (await _search(subject))[:5]
        if not cands:
            return False
        ids = [c["id"] for c in cands]
        allp = sorted({p for pids, _, _ in PROPS.values() for p in pids})
        if _cache.get(("q", tuple(ids), tuple(PROPS["founder"][0]))) is not None or tuple(ids) in _inflight:
            return True                                   # already warm (or warming)
        fut = _inflight[tuple(ids)] = asyncio.get_running_loop().create_future()
        data = await get_json(SPARQL, {"query": _sparql(ids, allp).replace("LIMIT 300", "LIMIT 3000"),
                                       "format": "json"},
                              headers={"Accept": "application/sparql-results+json"},
                              timeout=max(config.ROUTER_SLOW_TIMEOUT_S, 3.0), breaker_s=20)
        rows: dict[tuple, list[tuple]] = {}
        for b in data["results"]["bindings"]:
            item = b["item"]["value"].rsplit("/", 1)[-1]
            pid = b["pid"]["value"]
            rank = b["rank"]["value"].rsplit("#", 1)[-1]
            if "amt" in b:
                val = _fmt_qty(b["amt"]["value"], (b.get("unitLabel") or {}).get("value", ""))
            elif "t" in b:
                val = _fmt_time(b["t"]["value"], int((b.get("prec") or {}).get("value", 11)))
            else:
                val = (b.get("vLabel") or b["v"])["value"]
                if re.fullmatch(r"Q\d+", val) or val.startswith("http"):
                    continue
                vd = (b.get("vd") or {}).get("value", "")
                if vd and pid in _DESCRIBE:
                    val = f"{val} ({vd})"
            rows.setdefault((item, pid), []).append((rank, val))
        for pids, _, _ in PROPS.values():
            key = ("q", tuple(ids), tuple(pids))
            if _cache.get(key) is None:
                _cache.put(key, {k: v for k, v in rows.items() if k[1] in pids})
        return True
    except Exception:
        return False
    finally:
        f = _inflight.pop(tuple(ids), None) if "ids" in locals() else None
        if f is not None and not f.done():
            f.set_result(None)


# ------------------------------------------------------------------------------------------------- local leads
class Leads:
    """index/data/knowledge/{leads.jsonl, vectors.npy}: one Wikipedia lead paragraph per article."""

    def __init__(self, path=None):
        path = path or config.KNOWLEDGE_DIR
        self.docs: list[dict] = []
        self.vecs = np.zeros((0, config.EMBED_DIM), np.float32)
        self.built = None
        meta = path / "leads.jsonl"
        if meta.exists() and (path / "vectors.npy").exists():
            self.docs = [json.loads(x) for x in meta.read_text().splitlines() if x.strip()]
            self.vecs = np.load(path / "vectors.npy").astype(np.float32)[:, : config.EMBED_DIM]
            try:
                self.built = json.loads((path / "stats.json").read_text()).get("built")
            except (OSError, ValueError):
                pass
            self._titles = {d["title"].lower(): i for i, d in enumerate(self.docs)}

    def __len__(self):
        return len(self.docs)

    def search(self, qvec: np.ndarray, k: int = 3, boost_title: str = "") -> list[tuple[int, float]]:
        if not len(self.docs):
            return []
        s = self.vecs @ qvec
        if boost_title:
            i = self._titles.get(boost_title.lower())
            if i is not None:
                s[i] += 0.08                                # the subject's own article, said by name
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]


_leads: Leads | None = None


def leads_index() -> Leads:
    global _leads
    if _leads is None:
        _leads = Leads()
    return _leads


async def leads(query: str, k: int = 3, subject: str = "") -> list:
    from . import embed
    from .lanes import Doc
    ix = leads_index()
    if not config.ROUTER_LEADS or not len(ix):
        return []
    qv = (await embed.arun(embed.encode_queries, [query]))[0]
    out = []
    for i, s in ix.search(qv, k, boost_title=subject):
        d = ix.docs[i]
        out.append(Doc(id=d["url"] or f"lead:{d['title']}", title=f"{d['title']} · local Wikipedia", text=d["text"],
                       url=d.get("url", ""), source="local Wikipedia", score=s,
                       meta={"tier": "facts", "strong": s >= config.ROUTER_LEADS_STRONG, "cos": round(s, 3),
                             "snapshot": f"Wikipedia snapshot {ix.built or ''}".strip()}))
    return out


# ------------------------------------------------------------------------------------------------- free REST
async def wiktionary(word: str) -> list:
    from .lanes import Doc
    hit = _cache.get(("wt", word))
    if hit is None:
        data = await get_json(f"https://en.wiktionary.org/api/rest_v1/page/definition/{word}", breaker_s=20)
        defs = []
        for sense in (data.get("en") or [])[:3]:
            for d in (sense.get("definitions") or [])[:2]:
                txt = re.sub(r"<[^>]+>", "", html.unescape(d.get("definition") or "")).strip()
                if txt:
                    defs.append(f"({sense.get('partOfSpeech', '').lower()}) {txt}")
        hit = defs
        _cache.put(("wt", word), hit)
    if not hit:
        return []
    return [Doc(id=f"wiktionary:{word}", title=f"{word} · Wiktionary", text=f"{word}: " + " ; ".join(hit[:4]),
                url=f"https://en.wiktionary.org/wiki/{word}", source="Wiktionary", score=1.0,
                meta={"tier": "facts", "strong": True, "snapshot": "Wiktionary, live"})]


async def wiki_summary(subject: str) -> list:
    from .lanes import Doc
    title = subject.strip().replace(" ", "_")
    hit = _cache.get(("ws", title.lower()))
    if hit is None:
        try:
            data = await get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", {"redirect": "true"},
                                  breaker_s=10)
        except Exception as e:
            if "404" in str(e) or "400" in str(e):
                _cache.put(("ws", title.lower()), {}, ttl=600)
                return []
            raise
        hit = data if data.get("type") == "standard" and data.get("extract") else {}
        _cache.put(("ws", title.lower()), hit)
    if not hit:
        return []
    return [Doc(id=(hit.get("content_urls") or {}).get("desktop", {}).get("page") or f"wiki:{hit.get('title')}",
                title=f"{hit.get('title')} · Wikipedia summary", text=hit["extract"],
                url=(hit.get("content_urls") or {}).get("desktop", {}).get("page", ""), source="Wikipedia summary",
                score=1.0, meta={"tier": "facts", "strong": True,
                                 "snapshot": f"Wikipedia, revised {hit.get('timestamp', '')[:10]}"})]


def stats() -> dict:
    ix = leads_index()
    return {"leads": len(ix), "leads_built": ix.built}


def _now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


_t = time.perf_counter
