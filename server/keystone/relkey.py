"""Arc-keyed relation keys: the question's MEANING as (relation, subject), decided by the Arc Tagger's read-out.

A question like "who founded the company Nvidia", "Nvidia's founder, who was it" or "who started Nvidia" has one
answer. Its words differ; its read-out doesn't: a wh- word that names the ANSWER TYPE (who -> person), a relation cue
("founded" / "founder" / "started"), and an Arc entity span for the SUBJECT ("nvidia", typed organization). Keystone
stores and looks up answers by that typed key:

    RelKey(relation="founded_by", subject="nvidia", answer_type="person", vclass="stable")

Arc doesn't emit a relation label by itself; the relation is read off three Arc signals plus a small cue lexicon:
  - the intent (discourse.intent.qa_factoid / qa_definition / weather_query / qa_stock / news_query ...): which
    family of relations is possible at all (a news_query never keys to a stable fact)
  - the entity span and its type (person / organization / place / work_product / event_name / term): the subject,
    and a type check (born_place wants a person, capital_of a place, author_of a work)
  - the wh- word + head ("who" / "when" / "how tall"): the answer type, which disambiguates the cue ("founded" +
    who -> founded_by, "founded" + when -> inception)
  - typed number spans (date / money / ...): part of the key for relations that take an argument (weather on a day)

The relation is usually known BEFORE the subject ("who founded ..." -> founded_by at word 2): streaming key
completion (keystone/anytime.py) narrows to that relation's edge table early, and the subject span completes it.

Every relation carries a volatility class -> TTL: founded_by / born / capital never go stale (stable), population /
CEO / head of state change over months (slow), prices / weather / scores / news change in minutes (live).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# relation -> (answer type, volatility class)
RELATIONS = {
    "founded_by": ("person", "stable"), "inception": ("date", "stable"), "capital_of": ("place", "stable"),
    "height_of": ("measure", "stable"), "born_date": ("date", "stable"), "born_place": ("place", "stable"),
    "died_date": ("date", "stable"), "creator_of": ("person", "stable"), "author_of": ("person", "stable"),
    "director_of": ("person", "stable"), "discoverer_of": ("person", "stable"), "language_of": ("thing", "stable"),
    "currency_of": ("thing", "stable"), "country_of": ("place", "stable"), "hq_of": ("place", "stable"),
    "symbol_of": ("thing", "stable"), "manufacturer_of": ("org", "stable"), "date_of": ("date", "stable"),
    "definition_of": ("definition", "stable"), "length_of": ("measure", "stable"), "area_of": ("measure", "stable"),
    "inventor_of": ("person", "stable"), "river_of": ("place", "stable"), "located_in": ("place", "stable"),
    "ceo_of": ("person", "slow"), "population_of": ("count", "slow"), "head_state_of": ("person", "slow"),
    "spouse_of": ("person", "slow"), "age_of": ("count", "slow"), "net_worth_of": ("money", "slow"),
    "weather_in": ("forecast", "live"), "climate_of": ("thing", "stable"), "price_of": ("money", "live"), "score_of": ("score", "live"),
    "news_about": ("news", "live"), "schedule_of": ("date", "live"), "hours_of": ("time", "live"),
}
# knowledge.PROPS name <-> relation (Wikidata facts write typed triples directly)
FROM_PROP = {"founder": "founded_by", "inception": "inception", "capital": "capital_of", "height": "height_of",
             "born_date": "born_date", "born_place": "born_place", "died_date": "died_date", "creator": "creator_of",
             "author": "author_of", "director": "director_of", "discoverer": "discoverer_of",
             "language": "language_of", "currency": "currency_of", "country": "country_of", "hq": "hq_of",
             "symbol": "symbol_of", "manufacturer": "manufacturer_of", "ceo": "ceo_of",
             "population": "population_of", "river": "river_of", "head_state": "head_state_of", "area": "area_of",
             "spouse": "spouse_of"}
TTL_S = {"stable": 90 * 86400.0, "slow": 30 * 86400.0, "live": 600.0}

# wh- head -> answer type
_WH = [
    (re.compile(r"\b(?:how (?:tall|high)|height|elevation)\b"), "measure_h"),
    (re.compile(r"\bhow (?:long|far|big|wide|deep|large)\b"), "measure"),
    (re.compile(r"\bhow (?:many|much)\b"), "count"),
    (re.compile(r"\bhow old\b"), "age"),
    (re.compile(r"\b(?:when|what year|which year|what date|what day|what time)\b"), "date"),
    (re.compile(r"\bwhere\b"), "place"),
    (re.compile(r"\b(?:who|whom|whose)\b"), "person"),
    (re.compile(r"\b(?:what|which)\b"), "thing"),
]
# (cue regex, {answer type -> relation}, default relation or None). Order matters: the first cue that fires wins.
_CUES = [
    (r"\b(?:found(?:ed|er|ers|ing)?|co-?found(?:ed|er|ers)?|start(?:ed)?|establish(?:ed)?|set up)\b",
     {"person": "founded_by", "date": "inception", "age": "inception"}, None),
    (r"\bcapital\b", {}, "capital_of"),
    (r"\b(?:headquarter(?:s|ed)?|hq|based)\b", {}, "hq_of"),
    (r"\bbirthplace\b|\bborn where\b|\bwhat (?:city|town|country) was \w+(?: \w+)? born\b", {}, "born_place"),
    (r"\bborn\b|\bbirth(?:day|date| year)?\b|\bcome into the world\b", {"place": "born_place", "date": "born_date",
                                                                       "person": None}, "born_date"),
    (r"\b(?:die|died|death|pass(?:ed)? away)\b", {"date": "died_date"}, "died_date"),
    (r"\b(?:wrote|written|author|writer)\b", {"person": "author_of", "date": "inception"}, "author_of"),
    (r"\b(?:directed|director)\b", {"person": "director_of", "thing": "director_of"}, "director_of"),
    (r"\b(?:painted|painter|sculpted|drew)\b", {}, "creator_of"),
    (r"\b(?:invent(?:ed|or)?)\b", {"date": "inception"}, "inventor_of"),
    (r"\b(?:discover(?:ed|er|y)?)\b", {"date": "date_of"}, "discoverer_of"),
    (r"\b(?:creat(?:ed|or)|design(?:ed|er)|develop(?:ed|er))\b", {"date": "inception"}, "creator_of"),
    (r"\b(?:ceo|chief executive|runs)\b", {}, "ceo_of"),
    (r"\b(?:president|prime minister|leader|king|queen|head of state)\b", {"date": None}, "head_state_of"),
    (r"\bpopulation\b|\bhow many people (?:live|are there)\b", {}, "population_of"),
    (r"\b(?:languages?|spoken|speak)\b", {}, "language_of"),
    (r"\bcurrency\b|\bmoney do they use\b", {}, "currency_of"),
    (r"\bchemical symbol\b|\bsymbol for\b", {}, "symbol_of"),
    (r"\b(?:makes|made|manufactur(?:es|er|ed)|produces)\b", {"org": "manufacturer_of", "thing": "manufacturer_of",
                                                            "person": "manufacturer_of"}, None),
    (r"\b(?:married|wife|husband|spouse)\b", {}, "spouse_of"),
    (r"\bnet worth\b|\bhow rich\b", {}, "net_worth_of"),
    (r"\briver\b", {"thing": "river_of", "place": "river_of"}, None),
    (r"\b(?:mean|meaning|definition|define)\b", {}, "definition_of"),
    (r"\b(?:tall|high|height|elevation)\b", {}, "height_of"),
    (r"\b(?:long|length)\b", {"measure": "length_of"}, None),
    (r"\b(?:area|square (?:miles|kilometers))\b", {}, "area_of"),
    (r"\b(?:climate|average (?:low |high )?temperature|typical weather|average (?:rainfall|temperature)|"
     r"(?:coldest|hottest|wettest|driest) month)\b", {}, "climate_of"),
    (r"\b(?:weather|forecast|rain|snow|temperature)\b", {}, "weather_in"),
    (r"\b(?:price|stock|worth|cost|trading)\b", {"money": "price_of", "count": "price_of", "thing": "price_of"}, None),
    (r"\b(?:score|won|win|beat|lost|result)\b", {}, "score_of"),
    (r"\b(?:news|latest|headlines?)\b", {}, "news_about"),
    (r"\b(?:open|close|hours)\b", {"date": "hours_of", "thing": "hours_of"}, None),
    (r"\b(?:next (?:game|match|race|launch)|playing next|schedule)\b", {}, "schedule_of"),
    (r"\b(?:which|what) country\b|\bwhere is\b|\blocated\b", {}, "located_in"),
    (r"\bhow old\b|\bage\b", {}, "age_of"),
]
_CUES = [(re.compile(p), m, d) for p, m, d in _CUES]
_CLIMATE = re.compile(r"\b(?:climate|average (?:low |high |daily )?(?:temperature|rainfall|snowfall|low|high)|"
                      r"typical (?:weather|temperature)|(?:coldest|hottest|wettest|driest|rainiest) month|"
                      r"usually (?:rain|snow)|best time (?:of year )?to visit)\b")
_EVENT_VERBS = re.compile(r"\b(?:land(?:ed)?|fall|fell|happen(?:ed)?|end(?:ed)?|begin|began|start(?:ed)?|sink|sank|"
                          r"collapse(?:d)?|signed|launch(?:ed)?|release(?:d)?|come out|came out|open(?:ed)?|built|"
                          r"invented|discovered|win|won|take place|took place|first (?:flown|released|published))\b")
LIVE_INTENTS = {"discourse.intent.weather_query": "weather_in", "discourse.intent.qa_stock": "price_of",
                "discourse.intent.news_query": "news_about", "discourse.intent.qa_currency": "price_of"}
_STOP = set("""a an the is are was were be been do does did of in on at to for from by with about what what's whats who
who's whos which when where how why can could would should will i me my you your it its it's this that these those
there please tell hey ok okay so um uh like just and or but really very know wondering curious quick question year
company novel book movie film painting city country person thing one exactly actually anyway he she they him her them
his their""".split())
_ART = re.compile(r"^(?:the|a|an) ")
_SUFFIX = re.compile(r"\s+(?:inc|incorporated|corp|corporation|company|co|ltd|limited|plc|llc|group)$")
_CONNECT = {"and", "of", "the", "&", "de", "von", "van", "du", "la", "le", "on", "in"}
NOT_SUBJECT = {"weather", "today", "tomorrow", "tonight", "now", "right now", "this week", "the world", "world"}


def canon(name: str) -> str:
    """Normalised entity name: lowercase, no possessive / article / corporate suffix / punctuation."""
    s = name.lower().replace("’", "'")
    s = re.sub(r"'s\b|s'\b", "", s)
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _ART.sub("", s)
    s = _SUFFIX.sub("", s)
    if s.endswith("s") and len(s) > 4 and " " not in s and s[:-1] in _POSSESSIVE_FIX:
        s = s[:-1]
    return s.strip()


_POSSESSIVE_FIX: set = set()      # names seen without the trailing s ("nvidias" -> "nvidia" once nvidia is known)


def learn_name(name: str):
    c = canon(name)
    if c and " " not in c:
        _POSSESSIVE_FIX.add(c)


@dataclass(frozen=True)
class RelKey:
    relation: str
    subject: str            # canonical subject name ("" = not said yet)
    answer_type: str
    vclass: str
    arg: str = ""           # live relations: the day / time / number said ("tomorrow")

    @property
    def complete(self) -> bool:
        return bool(self.relation and self.subject)

    def text(self) -> str:
        return f"{self.relation}({self.subject}{', ' + self.arg if self.arg else ''})"

    @property
    def ttl_s(self) -> float:
        fixed = TTL_S.get(self.vclass, TTL_S["slow"])
        if LEARNED_TTL[0]:
            from . import kdir
            return kdir.learned_ttl(self.relation, self.vclass, fixed)
        return fixed


# KDIR_TTL=1 (with KDIR=1): relation TTLs learned from data (eval/keystone/fit_ttl.py -> kdir.TTL_PATH) replace the
# fixed volatility classes; set by server.config at import.
LEARNED_TTL = [False]


def _norm_words(words) -> list[str]:
    out = []
    for w in words:
        w = w.lower().replace("’", "'")
        w = re.sub(r"[^a-z0-9'&-]+", "", w).strip("'-")
        if w:
            out.append(w)
    return out


def answer_type(low: str) -> str:
    for rx, t in _WH:
        if rx.search(low):
            return t
    return "thing"


def relation_of(low: str, intent: str = "", subj_type: str = "") -> tuple[str, int] | tuple[None, int]:
    """(relation, character offset of its cue) or (None, -1)."""
    at = answer_type(low)
    if _CLIMATE.search(low):              # "average low in Atlanta in January" is climate (stable), not the forecast
        return "climate_of", _CLIMATE.search(low).start()
    if intent in LIVE_INTENTS and intent != "discourse.intent.qa_currency":
        m = None
        for rx, mp, d in _CUES:
            mm = rx.search(low)
            if mm and (mp.get(at) or d) in ("weather_in", "price_of", "news_about", "score_of", "schedule_of"):
                m = mm
                break
        return LIVE_INTENTS[intent], (m.start() if m else 0)
    if at == "date" and _EVENT_VERBS.search(low) and not re.search(r"\b(?:born|die|died|founded|start(?:ed)?|"
                                                                     r"establish(?:ed)?)\b", low):
        return "date_of", _EVENT_VERBS.search(low).start()
    at2 = "measure" if at == "measure_h" else at
    for rx, mp, d in _CUES:
        m = rx.search(low)
        if not m:
            continue
        rel = mp.get(at2, d) if mp else d
        if at == "measure_h" and rx.pattern.find("tall") >= 0:
            rel = "height_of"
        if rel is None:
            continue
        if rel == "born_date" and at2 == "person":
            continue
        return rel, m.start()
    if at == "measure_h":
        return "height_of", 0
    return None, -1


def _span_subject(words: list[str], slots) -> str:
    """The subject from the Arc entity spans: the longest run covering entity words, bridged over connectors ("pride
    and prejudice" from two spans "pride", "prejudice")."""
    ent_words = set()
    for t, v in slots or ():
        if t in ("date", "time", "time_of_day", "duration", "money", "percent", "cardinal", "ordinal", "quantity",
                 "number", "numeral", "age", "year"):
            continue
        for w in _norm_words(str(v).split()):
            if w not in _CONNECT:            # "of" / "and" inside a span are bridged below, never a run's start
                ent_words.add(w)
    if not ent_words:
        return ""
    idx = [i for i, w in enumerate(words) if w in ent_words or (w.endswith("s") and w[:-1] in ent_words)]
    if not idx:
        return ""
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        gap = words[cur[-1] + 1:i]
        if i == cur[-1] + 1 or (len(gap) <= 2 and all(g in _CONNECT for g in gap)):
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    best = max(runs, key=lambda r: (r[-1] - r[0], r[-1]))
    a, b = best[0], best[-1]
    while b + 1 < len(words) and re.fullmatch(r"\d+|[ivx]+", words[b + 1]):     # "apollo 11", "world war ii"
        b += 1
    return " ".join(words[a:b + 1])


def _fallback_subject(words: list[str], cue_end: int, low: str) -> str:
    """No Arc span: the content words after the relation cue ("capital of australia" -> australia), else all content
    words that aren't relation / wh- words."""
    m = re.search(r"\b(?:of|for|in|does|did|is|was|through)\s+(?:the\s+)?([a-z0-9 &'-]+)$", low[cue_end:]) \
        if cue_end >= 0 else None
    cand = m.group(1) if m else ""
    toks = [w for w in (cand or low).split() if w not in _STOP and not any(rx.fullmatch(w) for rx in _CUE_WORDS)]
    return " ".join(toks[:5])


_CUE_WORDS = [re.compile(x) for x in (r"found\w*", r"co-?found\w*", r"start\w*", r"establish\w*", r"born", r"capital",
                                     r"tall", r"high", r"height", r"wrote", r"written", r"author", r"direct\w*",
                                     r"paint\w*", r"invent\w*", r"discover\w*", r"creat\w*", r"design\w*",
                                     r"develop\w*", r"ceo", r"population", r"language\w*", r"spoken", r"currency",
                                     r"symbol", r"chemical", r"make\w*", r"made", r"manufactur\w*", r"married",
                                     r"headquarter\w*", r"based", r"die\w*", r"death", r"mean\w*", r"definition",
                                     r"define", r"use", r"live", r"people", r"many", r"much", r"old", r"long",
                                     r"weather", r"forecast", r"price", r"stock", r"news", r"latest", r"elevation",
                                     r"river", r"runs", r"flows", r"land\w*", r"fall", r"fell", r"happen\w*",
                                     r"located", r"behind", r"who", r"whose", r"set", r"up", r"name\w*", r"called", r"birth\w*")]


_HOP = re.compile(r"\b(?:the )?(founder|author|writer|director|ceo|creator|inventor|painter|capital|president|"
                  r"leader|wife|husband|spouse|manufacturer|maker) of (?:the )?(.+?)(?:\s+(?:born|founded|located|die|"
                  r"died|from|based|headquartered|built|written|made|start|started|now|today|live|lives))?(?:\s+(?:when|if)\b.*)?$")
_HOP_REL = {"founder": "founded_by", "author": "author_of", "writer": "author_of", "director": "director_of",
            "ceo": "ceo_of", "creator": "creator_of", "inventor": "inventor_of", "painter": "creator_of",
            "capital": "capital_of", "president": "head_state_of", "leader": "head_state_of", "wife": "spouse_of",
            "husband": "spouse_of", "spouse": "spouse_of", "manufacturer": "manufacturer_of", "maker": "manufacturer_of"}


def hop_of(low: str):
    """A two-hop question ("where was the author of pride and prejudice born"): (inner relation, its subject, the
    outer relation), else None. The outer relation is read off the question with the inner phrase removed."""
    m = _HOP.search(low)
    if not m:
        return None
    inner, subj = _HOP_REL[m.group(1)], canon(m.group(2))
    rest = (low[:m.start()] + " " + low[m.end(2):]).strip()
    outer, _ = relation_of(rest)
    if outer is None or outer == inner or not subj:
        return None
    return inner, subj, outer


_MULTI = re.compile(r"\b(?:and|plus|also|as well as)\b")
_MULTI_WH = re.compile(r"\b(?:and|plus|also)(?: also)? (?:what|what's|whats|when|where|who|how|which|is|the weather|"
                       r"weather)\b|\bflights?\b.*\b(?:and|plus)\b")
_COMPARE = re.compile(r"\b(?:versus|vs|compare[ds]?|comparison|which is (?:\w+er|more|less)|who is (?:\w+er)|"
                      r"\w+er than|more than|less than|which came out first|who was born first|which (?:one|state|"
                      r"country|city) (?:is|has) (?:\w+er|more|less))\b")


def _all_relations(low: str) -> list[str]:
    """Every relation whose cue appears (multi-part detection)."""
    at = answer_type(low)
    out = []
    for rx, mp, d in _CUES:
        if rx.search(low):
            r = (mp.get("measure" if at == "measure_h" else at, d) if mp else d) or d
            if r:
                out.append(r)
    return out


def key_of(words, slots=(), intent: str = "", num_slots=()) -> RelKey | None:
    """The typed key of a question (complete or partial: relation known, subject "" when not said yet). None: no
    relation cue (open-ended / how-to / advice: the learned hash and the embedding search handle those)."""
    ws = _norm_words(words)
    if not ws:
        return None
    low = " ".join(ws)
    hop = hop_of(low) if intent not in LIVE_INTENTS else None
    if hop is None and (_MULTI_WH.search(low) or (_MULTI.search(low) and len(set(_all_relations(low))) > 1)):
        return None                      # a multi-part question (the planner splits it): no single key
    if _COMPARE.search(low):
        return None
    if hop is not None:
        inner, isubj, outer = hop
        ans, vclass = RELATIONS[outer]
        vin = RELATIONS[inner][1]
        return RelKey(outer, f"@{inner}({isubj})", ans, "slow" if "slow" in (vclass, vin) else vclass)
    rel, at = relation_of(low, intent)
    if rel is None:
        return None
    ans, vclass = RELATIONS[rel]
    subj = _span_subject(ws, slots)
    if subj:
        # drop the relation's own words if the span swallowed them ("founder nvidia")
        subj = " ".join(w for w in subj.split() if not any(rx.fullmatch(w) for rx in _CUE_WORDS[:3]))
    if not subj:
        subj = _fallback_subject(ws, at, low)
    subj = canon(subj)
    if subj in NOT_SUBJECT:
        subj = ""
    arg = ""
    if vclass == "live":
        days = re.findall(r"\b(today|tonight|tomorrow|yesterday|this (?:week|weekend)|last night|right now|now|"
                          r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|"
                          r"may|june|july|august|september|october|november|december|next week|next month)\b", low)
        arg = days[-1] if days else "now"
    return RelKey(rel, subj, ans, vclass, arg)


def key_of_frame(f) -> RelKey | None:
    """From a live query frame (server/frame.py): the question as meant (correction spliced in place)."""
    try:
        from ..router import repaired_text
        text = repaired_text(f)
    except Exception:
        text = " ".join(f.words)
    words = text.split()
    slots = [(t, v) for t, v in f.slots]
    if getattr(f, "carried", "") and not any(t not in ("date", "time") for t, _ in slots):
        slots.append(("thing", f.carried))
        words = words + f.carried.split()
    return key_of(words, slots, f.intent)
