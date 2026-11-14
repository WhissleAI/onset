"""Query frames: what the partial question means right now, built from Arc Tagger read-outs.

A frame is (intent, slots, content words). Its signature changes only when the meaning changes, so a partial that
re-sends the same words, or adds a filler word, doesn't trigger another search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config

STOP = set("""a an the is are was were be been being am do does did of in on at to for from by with about into over
what what's whats who who's whos whom whose which when where where's why how how's hows can could would should will
shall may might must i me my you your we our us it its it's this that these those there here please tell show give find
search look up hey ok okay so um uh like just and or but if then than also really very some any get got let let's
lets know want need yeah yes no oh well right now wondering wonder curious quick question asked ask
always ever forget forgot remember mind sure guess think kind sort actually basically literally anyone somebody
someone thing things they them their he she his her him it's say said says tell told
mm mmm hmm hm mhm mm-hmm uh-huh er erm ah huh i've i'd i'm i'll you've you'd you're we've didn't don't doesn't
isn't wasn't can't couldn't won't wouldn't haven't hasn't wanted wanting many much have has had
ive im didnt dont doesnt isnt wasnt cant couldnt wont wouldnt""".split())
# ^ last three lines: spoken fillers and contractions that leaked into real queries ("i've wanted many bones adult human
# body", "mm invented telephone", "didn't sky blue"); Wikipedia ANDs every word, so one leaked filler sinks the search
# hey/ok wake fillers are already in STOP; these mark a spoken self-correction
CORRECTION = [("i", "mean"), ("sorry",), ("no", "wait"), ("actually",), ("rather",), ("scratch", "that"),
              ("not", "that"), ("make", "that"), ("i", "meant"),
              # how live ASR actually hears them (eval set): "Sari, I mean", "no weight", "I met"
              ("sari",), ("no", "weight"), ("i", "met"), ("i", "mean", "it"), ("no", "way")]
# A spoken repair that the ear split into its own turn ("what's the weather in Boston, sorry," | "I mean Denver"):
# session.py merges such a turn into the previous one, then the correction logic below handles it as one utterance.
# REPAIR_LEAD: a turn that STARTS with one of these (after fillers) is a correction of the turn just before it.
# REPAIR_TAIL: a turn that ENDS on one of these is cut mid-correction; "rather" / "not that" / "no way" end ordinary
# questions too ("which would you rather", "is it not that"), so they only count at the start.
REPAIR_LEAD = [("sorry",), ("i", "mean"), ("i", "meant"), ("actually",), ("no", "wait"), ("make", "that"),
               ("rather",), ("scratch", "that"), ("sari",), ("no", "weight"), ("i", "met")]
REPAIR_TAIL = [("sorry",), ("i", "mean"), ("i", "meant"), ("actually",), ("no", "wait"), ("make", "that"),
               ("scratch", "that"), ("sari",), ("no", "weight"), ("i", "met"), ("or", "rather")]
_LEAD_FILLERS = {"um", "uh", "oh", "er", "erm", "ah", "hmm", "mm"}
INTENT_HINT = {
    "discourse.intent.weather_query": "weather forecast",
    "discourse.intent.news_query": "news",
    "discourse.intent.qa_stock": "stock price",
    "discourse.intent.qa_currency": "exchange rate",
    "discourse.intent.recommendation_locations": "best places",
    "discourse.intent.recommendation_events": "events",
    "discourse.intent.recommendation_movies": "movies",
    "discourse.intent.cooking_recipe": "recipe",
    "discourse.intent.transport_traffic": "traffic",
    "discourse.intent.qa_definition": "definition",
}
WEB_INTENTS = {"discourse.intent.weather_query", "discourse.intent.news_query", "discourse.intent.qa_stock",
               "discourse.intent.qa_currency", "discourse.intent.qa_factoid", "discourse.intent.qa_definition",
               "discourse.intent.recommendation_locations", "discourse.intent.recommendation_events",
               "discourse.intent.recommendation_movies", "discourse.intent.cooking_recipe",
               "discourse.intent.cooking_query", "discourse.intent.transport_query",
               "discourse.intent.transport_traffic", "intent.request_information"}
# Phrases that point at the local corpus (the recorded moments) instead of the web.
LOCAL_CUES = ["in the game", "the announcer", "the commentator", "in the clip", "in the recording", "in the video",
              "in the meeting", "in the call", "in the hearing", "the judge", "did he say", "did she say",
              "did they say", "was said", "the moment", "in the replay", "on the call", "the justice",
              "the panel", "the teacher", "in the commentary", "on the podcast", "in the broadcast", "the lawyer",
              "the attorney", "the witness", "in the debate", "the commentators", "the host"]
NEWS_INTENTS = {"discourse.intent.news_query"}
# words after "actually" that continue the question instead of replacing a word ("what does the word ubiquitous
# actually mean exactly": "ubiquitous" was corrected away and the web query became "word mean exactly")
NOT_REPLACEMENT = {"mean", "means", "meant", "exactly", "work", "works", "is", "do", "does", "worth", "cost", "costs",
                   "happen", "happened", "happening", "go", "goes", "say", "says"}

PRONOUNS = {"it", "its", "it's", "he", "she", "his", "her", "him", "they", "them", "their", "there", "that", "this",
            "those", "these"}
PERSONAL = {"he", "she", "his", "her", "him", "hers", "himself", "herself"}
WEAK_PRONOUNS = {"there", "that", "this", "those", "these"}     # often not referential: only for short questions
NUMBER_TYPES = {"date", "time", "time_of_day", "duration", "money", "percent", "cardinal", "ordinal", "quantity",
                "number", "numeral", "age", "year"}

# --- the rest of Arc Tagger's per-word read-out (beyond intent + slots) -----------------------------------------------
# Personal data that must not leave the device in a web query: span types precise enough to redact. (private_name is
# NOT here: on this eval it fires on "Nvidia", "Mount", "Toyota", "Mozart"; the utterance-level pii.request / pii.leak
# flags fire on most ordinary questions, so neither is used.)
PII_REDACT = {"safety.pii.phone_number": "phone", "safety.pii.email": "email", "safety.pii.street_address": "address",
              "safety.pii.card_account_number": "card", "safety.pii.gov_id": "id", "safety.pii.date_of_birth": "birthday",
              "safety.pii.url_handle": "handle"}
# Redact the identifying part only, so the search keeps what it needs (Arc tags "colorado", "there" and
# "1600 pennsylvania avenue" all as street_address): a span is hidden only if its words have the shape of that kind of
# data, and an address / birthday loses just its house number / year, keeping the street, city or day for the search.
_STREET = {"street", "st", "avenue", "ave", "road", "rd", "lane", "ln", "drive", "dr", "boulevard", "blvd", "court",
           "ct", "way", "place", "pl", "apartment", "apt", "suite", "unit", "terrace", "circle", "highway", "parkway"}
_EMAILISH = {"at", "dot", "com", "gmail", "yahoo", "hotmail", "outlook", "underscore", "net", "org"}
PROFANITY = set("fuck fucking fucked fucker shit shitty bullshit bitch bastard asshole dick damn crap piss cunt".split())


def _pii_hide(kind: str, toks: list[str], a: int, b: int, raw: list[str] = ()) -> set:
    idx = range(a, b)
    nums = [i for i in idx if _numtok(toks[i])]
    digits = sum(sum(c.isdigit() for c in toks[i]) or 1 for i in nums)
    if kind in ("safety.pii.phone_number", "safety.pii.card_account_number", "safety.pii.gov_id"):
        return set(idx) if digits >= 5 else set()          # an actual number, not "one" or "2024"
    if kind in ("safety.pii.email", "safety.pii.url_handle"):
        raw = [str(raw[i]) if i < len(raw) else toks[i] for i in idx]
        return set(idx) if any("@" in w or "_" in w or w.lower().strip(".,") in _EMAILISH for w in raw) else set()
    if kind == "safety.pii.street_address":
        street = any(toks[i] in _STREET for i in idx)
        return set(nums) if nums and (street or b - a >= 3) else set()   # house number only; street + city stay
    if kind == "safety.pii.date_of_birth":                  # keep the day ("what happened on june fifth"), hide the year
        return {i for i in nums if re.fullmatch(r"(19|20)\d\d", toks[i])} | (spoken_years(toks, a, b) if config.PII_YEARS
                                                                               else set())
    return set(idx)


# Spoken years, as the live ear writes them: "nineteen eighty five", "twenty eleven", "nineteen oh five", "two thousand
# and eight". The tagger's date_of_birth span covers them, but only digit years ("1985") used to be hidden: spoken birth
# years reached the search 67 of 69 times (docs/ARC_VALUE.md, privacy).
_TEENS = {"ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
_TENS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
_UNITS_W = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
_CENTURY = {"eighteen", "nineteen", "twenty"}


def spoken_years(toks: list[str], a: int = 0, b: int | None = None) -> set:
    """Token indices of every spoken year that starts in toks[a:b] (it may run past b: a span cut mid-year)."""
    b = len(toks) if b is None else b
    out: set = set()
    i = a
    while i < b:
        t = toks[i]
        n = toks[i + 1] if i + 1 < len(toks) else ""
        j = -1
        if t in _CENTURY and (n in _TENS or n in _TEENS or n in ("oh", "hundred")):
            j = i + 2                                      # "nineteen eighty", "twenty eleven", "nineteen oh", ...
            if n in _TENS and j < len(toks) and toks[j] in _UNITS_W:
                j += 1                                     # "... five"
            elif n == "oh" and j < len(toks) and toks[j] in _UNITS_W:
                j += 1
            elif n == "oh":
                j = -1
        elif t == "two" and n == "thousand":
            j = i + 2                                      # "two thousand", "two thousand and eight", "... ten"
            k = j + (1 if j < len(toks) and toks[j] == "and" else 0)
            if k < len(toks) and (toks[k] in _UNITS_W or toks[k] in _TEENS):
                j = k + 1
            elif k < len(toks) and toks[k] in _TENS:
                j = k + 1 + (1 if k + 1 < len(toks) and toks[k + 1] in _UNITS_W else 0)
        if j > i:
            out.update(range(i, j))
            i = j
        else:
            i += 1
    return out


# A birth year said outside a date_of_birth span ("I was born in nineteen eighty five, what happened that year"): the
# year words near a birth cue are hidden too; the month, day and place stay (they are what the search needs).
_BIRTH_CUE = {"born", "birthday", "birth", "birthdate", "dob", "bday"}


def birth_year_tokens(toks: list[str], window: int = 8) -> set:
    cues = [i for i, t in enumerate(toks) if t in _BIRTH_CUE]
    if not cues:
        return set()
    ys = spoken_years(toks) | {i for i, t in enumerate(toks) if re.fullmatch(r"(19|20)\d\d", t)}
    return {i for i in ys if any(abs(i - c) <= window for c in cues)}


# Relative time said in the question (Arc's typed entity.number.date / duration / time_of_day span, or the words):
# drives Tavily freshness. Absolute dates ("in 2024", "March 3rd 2025") mean history: no freshness filter.
_WHEN_DAY = re.compile(r"\b(today|tonight|right now|now|this morning|this afternoon|this evening|currently|live|"
                       r"24 hours|twenty four hours|at the moment)\b")
_WHEN_WEEK = re.compile(r"\b(yesterday|last night|weekend|this week|past week|last week|sunday|saturday|monday|"
                        r"tuesday|wednesday|thursday|friday|recent|recently|latest|lately|few days|past few days)\b")
# only from an Arc typed date / duration span (entity.number.date ...): "this year", "last month", "past few weeks" are
# too common as plain words ("the year of the dragon") to set freshness on their own
_WHEN_MONTH = re.compile(r"\b(this month|last month|past month|past few weeks|last few weeks|few weeks ago|"
                         r"couple of weeks|past thirty days|last thirty days)\b")
_WHEN_YEAR = re.compile(r"\b(this year|past year|last twelve months|past twelve months|so far this year)\b")
# results of games / races / matches are news (dated pages), not encyclopedia pages
_RESULTS = {"game", "match", "score", "scores", "won", "win", "winner", "race", "prix", "beat", "lost", "playoff",
            "playoffs", "final", "finals", "result", "results", "standings", "fight", "tournament"}
_WEATHER_W = {"weather", "rain", "raining", "rainy", "snow", "snowing", "sunny", "temperature", "forecast", "cold", "hot",
              "warm", "wind", "windy", "humid", "humidity", "storm", "umbrella", "degrees", "cloudy", "fog", "heat"}
_MARKET = re.compile(r"\b(s ?(?:and|&)? ?p 500|s p 500|nasdaq|dow jones|the dow|stock market)\b")
_FINANCE_W = {"stock", "stocks", "price", "worth", "rate", "exchange", "bitcoin", "crypto", "ethereum", "market",
              "shares", "dollars", "euros", "dollar", "euro", "yen", "pounds"}

_PUNCT = re.compile(r"[^\w'\- ]+")


def norm(s: str) -> str:
    return _PUNCT.sub("", s.lower()).strip()


@dataclass
class Frame:
    words: list[str]
    intent: str = ""                 # best fine intent (discourse.intent.*) that is on, else coarse intent.*
    intent_p: float = 0.0
    coarse: str = ""
    slots: tuple = ()                # sorted ((type, text), ...)
    content: tuple = ()              # content words, in order, after correction handling
    corrected: bool = False
    carried: str = ""                # entity inherited from the previous question ("when was it founded")
    followup_of: str = ""            # previous question, when this one is an elliptical follow-up ("and when?")
    route: str = "web"               # web | local | both
    act_question: bool = False
    command_complete: bool = False
    spans: list = field(default_factory=list)
    dropped: frozenset = frozenset()  # token indices removed by a self-correction (cue words + corrected-away entity)
    phrases: tuple = ()              # entity phrases as spoken (internal stopwords kept): "pride and prejudice"
    # the rest of the read-out (shown live; freshness / topic / redaction use it)
    acts: tuple = ()                 # dialogue acts that are on: ("question",), ("command_request",) ...
    directed: float | None = None    # discourse.assistant.device_directed probability
    redacted: tuple = ()             # kinds of personal data kept out of the web query: ("phone", "email")
    redact_tok: frozenset = frozenset()
    cue: frozenset = frozenset()      # token indices of the self-correction cue ("actually make that")
    content_idx: tuple = ()          # token index of each content word (what goes into the search query)
    jailbreak: bool = False          # safety.jailbreak.attempt on the spoken question
    when: str = ""                   # "day" | "week" | "" : relative time the question is about
    when_text: str = ""              # the words that said it ("last night", "right now")
    when_typed: bool = False         # `when` came from an Arc typed time span (entity.number.date / duration / time)
    ent_types: tuple = ()            # entity mention types in spoken order (person, organization, place, team, ...)
    num_types: tuple = ()            # typed number spans (date, money, percent, duration, ...)
    wconf: tuple = ()                # fused per-word confidence (server/confidence.py), when the ear gives word conf
    director: dict | None = None     # the search director's latest outputs (server/director.py), DIRECTOR=on
    convert: tuple = ()              # (amount as said, from unit, to unit): "a hundred us dollars", "dollars", "euros"
    # conversation memory (server/memory.py): the question with its reference resolved IN PLACE ("when was it founded"
    # -> "when was nvidia founded"; the router parses it like a standalone question), the reference phrase memory
    # resolved ("that company", "the first one"), and the user facts the query used ("home": "denver")
    inline: str = ""
    mem_ref: str = ""
    user_used: tuple = ()
    # Arc-off ("plain voice search", mode="plain" / ARC_OFF=1): a frame built from the words alone, no tagger read-out.
    # Its query is the raw transcript cleaned of punctuation; no correction handling, carry, redaction, hints, lanes.
    plain: bool = False
    # retrieval-in-the-loop (server/heard.py, LOOP_ASR_FIX): misheard spans the session's known names re-resolved
    # ({"said": "mari guri", "as": "Marie Curie", "src": "leads", "score": .9, "a": 2, "b": 4}); words[a] holds the
    # name, words[a+1:b] are "" so token indices stay aligned
    resolved: tuple = ()
    sconf: tuple = ()                # per-word "is it right": min(ear confidence, fused), heard.suspect_conf
    persona_hint: str = ""           # how to answer this user (server/persona.py, PERSONA=1): one short line

    @property
    def n(self):
        return len(self.words)

    @property
    def text(self):
        return " ".join(w for w in self.words if w)

    @property
    def sig(self):
        return (self.intent, self.slots, self.content)

    @property
    def meaning_sig(self):
        """Coarser signature used to decide if a drafted answer still fits: intent + slots + content words."""
        return (self.intent, self.slots, tuple(sorted(set(self.content))))

    @property
    def clean_text(self) -> str:
        """The question as the speaker meant it: correction cue and the corrected-away entity removed ("who founded
        Microsoft, sorry, I meant Amazon" → "who founded Amazon"), so the answer model doesn't comment on Microsoft."""
        if not self.dropped and not self.redact_tok:
            return self.text
        out, masked = [], False
        for i, w in enumerate(self.words):
            if i in self.dropped or not w:
                continue
            if i in self.redact_tok:
                if not masked:
                    out.append("[private]")       # personal data never leaves the device
                masked = True
                continue
            masked = False
            out.append(w)
        return " ".join(out)

    @property
    def kept_text(self) -> str:
        """The words that still mean something after the turn: no corrected-away entity or cue, no personal data
        (what conversation memory keeps: "who founded Microsoft, sorry, Amazon" -> "who founded Amazon")."""
        return " ".join(w for i, w in enumerate(self.words) if w and i not in self.dropped and i not in self.redact_tok)

    @property
    def resolved_text(self) -> str:
        """The question with its carried referent spelled out, for ranking and for the answer model."""
        t = self.clean_text
        if self.user_used:
            t = f"{t} ({', '.join(v for _, v in self.user_used)})"
        if self.followup_of:
            return f"{t} (follow-up to: {self.followup_of})"
        return f"{t} ({self.carried})" if self.carried else t

    def entity_query(self) -> str:
        """A second, entity-only web query ("turkey", "ludwig beethoven" → the entity's own article, whose intro
        usually holds the capital / birthplace / founder / HQ that the content query's articles only mention in
        passing). The phrase said after a correction wins, then the last entity. Empty when it adds nothing."""
        if not self.phrases:
            return ""
        e = self.phrases[-1]
        return "" if e == self.query() or not e.strip() else e

    def query(self) -> str:
        """Search query: content words in spoken order + an intent hint word when it adds information."""
        if self.plain:
            return " ".join(w for w in (norm(x) for x in self.words) if w)
        q = list(self.content)
        hint = INTENT_HINT.get(self.intent, "") if (self.intent_p >= 0.5 and self.n >= 3) else ""
        if hint and not any(h in q for h in hint.split()):
            q.append(hint)
        return " ".join(q)

    def web_hints(self) -> dict:
        """Tavily topic / freshness from the read-out (lanes.web_plan applies them):
          - finance intent, or a money span / finance word with a price-like question -> topic finance (+ max age)
          - news intent -> topic news; a relative time said ("today", "last night") -> time_range day | week
          - a result question (game / race / score) said with a relative time ("last night", "this weekend",
            "right now") -> topic news + that time_range: results are dated news pages, not encyclopedia articles
        Weather keeps lanes.web_plan's own max_age. Returns {} when nothing applies (plain general search)."""
        if self.plain:
            return {}
        h: dict = {}
        toks = set(self.content)
        if self.intent == "discourse.intent.weather_query" and not toks & _WEATHER_W:
            # Arc's weather intent with no weather word ("is it bad to take ibuprofen on an empty stomach"): not a
            # forecast, so no 6-hour freshness filter (it would drop every evergreen health page)
            return {"weather_off": True, "why": "no weather word"}
        if self.intent in ("discourse.intent.qa_stock", "discourse.intent.qa_currency") and self.intent_p >= 0.35:
            h["topic"] = "finance"
        elif _MARKET.search(" ".join(norm(w) for w in self.words)):
            h["topic"] = "finance"
        elif ("money" in self.num_types or toks & {"bitcoin", "crypto", "ethereum", "stock", "stocks", "shares"}) and \
                toks & _FINANCE_W and self.intent != "discourse.intent.weather_query":
            h["topic"] = "finance"
        elif self.intent == "discourse.intent.news_query":
            h["topic"] = "news"
            if self.when:
                h["time_range"] = self.when
        elif self.when and (toks & _RESULTS or "team" in self.ent_types or "event_name" in self.ent_types) \
                and self.intent != "discourse.intent.weather_query":
            h["topic"] = "news"
            h["time_range"] = self.when
        elif self.when in ("week", "month", "year") and self.when_typed and \
                self.intent != "discourse.intent.weather_query":
            # Arc typed a past window ("what did Apple announce yesterday", "flu cases this month"): keep the general
            # topic, restrict to that window (Tavily time_range works on every topic). Not "today / now": "when does
            # the Louvre close today" wants the evergreen hours page, not today's news
            h["time_range"] = self.when
        if h:
            h["why"] = self.when_text or ""
        return h

    def readout(self) -> dict:
        """Compact live read-out for the UI frame card."""
        h = self.web_hints()
        fresh = ""
        if h.get("topic") and h.get("topic") != "general":
            fresh = h["topic"] + (f" · past {h['time_range']}" if h.get("time_range") else "")
        elif h.get("time_range"):
            fresh = f"past {h['time_range']}"
        elif self.intent == "discourse.intent.weather_query" and not h.get("weather_off"):
            fresh = "weather · last 6 h"
        return {"acts": list(self.acts), "directed": None if self.directed is None else round(self.directed, 2),
                "redacted": list(self.redacted), "jailbreak": self.jailbreak, "when": self.when,
                "when_text": self.when_text, "fresh": fresh, "ent_types": list(self.ent_types),
                "num_types": list(self.num_types), "convert": list(self.convert)}

    def answer_hint(self) -> str:
        """Extra instruction for the answer model, from the read-out. Today: an amount to convert ("what's a hundred US
        dollars in euros"): sources usually list a rate (often the inverse one, 1 EUR = 1.16 USD), and the answer must
        be the converted AMOUNT, in the direction asked."""
        if not self.convert:
            return self.persona_hint
        amount, src, dst = self.convert
        return (f"This asks to convert an amount: {amount} into {dst}. Start with the converted amount in {dst} "
                f"(from {src} to {dst}, not the reverse), then give the rate used and cite it. If a source only gives the "
                f"inverse rate ({dst} to {src}), divide by it." + (" " + self.persona_hint if self.persona_hint else ""))

    def has_content(self) -> bool:
        return len(self.content) >= 2 or bool(self.slots)

    def to_json(self):
        return {"intent": self.intent, "p": round(self.intent_p, 2), "coarse": self.coarse,
                "slots": [list(s) for s in self.slots], "content": list(self.content), "route": self.route,
                "corrected": self.corrected, "carried": self.carried, "query": self.query(), "n": self.n,
                "command_complete": self.command_complete, "readout": self.readout(),
                # per-token read-out for the annotated transcript (indices into words)
                "content_idx": list(self.content_idx), "dropped": sorted(self.dropped), "cue": sorted(self.cue),
                "redact": sorted(self.redact_tok), **self._mem_json(), **self._conf_json(),
                **({"heard": list(self.resolved)} if self.resolved else {})}

    def _mem_json(self) -> dict:
        if not (self.inline or self.mem_ref or self.user_used):
            return {}
        from .memory import user_label
        return {"memory": {"inline": self.inline, "ref": self.mem_ref,
                           "user": [user_label(k, v) for k, v in self.user_used]}}

    def _conf_json(self) -> dict:
        if not self.wconf:
            return {}
        from . import confidence
        k, j = confidence.key_conf(self, list(self.wconf))
        sc = confidence.slot_confs(self, list(self.wconf))
        return {"wconf": [round(c, 3) for c in self.wconf], "key_conf": round(k, 3), "key_word": j,
                "slot_conf": [[t, x, round(c, 3)] for (t, x), c in sc.items()]}


def plain(words: list[str]) -> Frame:
    """Arc-off frame ("plain voice search", what a typical voice assistant does): the raw final transcript, lowercased
    and cleaned of punctuation, IS the query. Nothing from Arc Tagger's read-out: no intent, slots, correction handling,
    entity carry / memory, PII redaction, freshness / topic hints, router lanes or planner. content = every word (no
    stopword list either: that is query shaping), only so the speculator's bookkeeping (stale checks) has something."""
    toks = [t for t in (norm(w) for w in words) if t]
    return Frame(words=list(words), content=tuple(dict.fromkeys(toks)), route="web", plain=True)


def repair_start(toks: list[str]) -> bool | None:
    """Does this turn open with a repair marker (after fillers)? None = can't tell yet: the words so far are the start
    of one ("i" of "i mean", "no" of "no wait"), so the caller waits for the next partial."""
    t = [x for x in toks if x]
    while t and t[0] in _LEAD_FILLERS:
        t = t[1:]
    if not t:
        return None
    undecided = False
    for pat in REPAIR_LEAD:
        if tuple(t[:len(pat)]) == pat:
            return True
        if len(t) < len(pat) and tuple(t) == pat[:len(t)]:
            undecided = True
    return None if undecided else False


def repair_end(toks: list[str]) -> bool:
    """Does this turn end on a repair marker ("what's the weather in Boston, sorry")? The speaker is mid-correction."""
    t = [x for x in toks if x]
    return any(len(t) >= len(pat) and tuple(t[-len(pat):]) == pat for pat in REPAIR_TAIL)


# A turn that ends on a function word is cut mid-phrase ("is it bad to take a" | "ibuprofen on an empty stomach":
# Flux's own EndOfTurn split showcase sc15 there). Endings that close ordinary questions are exempt: "what time is it"
# never ends on "is", but "where it is", "what's going on", "who do I talk to" do.
DANGLING = {"a", "an", "the", "to", "of", "for", "in", "on", "with", "and", "or", "my", "your", "is", "are"}
_DANGLE_OK = {("it", "is"), ("he", "is"), ("she", "is"), ("that", "is"), ("this", "is"), ("there", "is"),
              ("what", "is"), ("where", "is"), ("who", "is"), ("they", "are"), ("you", "are"), ("we", "are"),
              ("there", "are"), ("what", "are"), ("where", "are"), ("who", "are"), ("going", "on"), ("turn", "on"),
              ("turned", "on"), ("go", "on"), ("goes", "on"), ("carry", "on"), ("hold", "on"), ("talk", "to"),
              ("listen", "to"), ("go", "to"), ("looking", "for"), ("in", "on"), ("come", "on"), ("log", "in"),
              ("sign", "in"), ("plug", "in"), ("check", "in"), ("go", "with"), ("goes", "with"), ("made", "of"),
              ("think", "of"), ("afraid", "of"), ("used", "to"), ("born", "in"), ("live", "in"), ("located", "in"),
              ("based", "in")}
# a pronoun before a stranded preposition ends a question: "which museum is it in", "what team does he play for"
_DANGLE_OK |= {(p, q) for p in ("it", "he", "she", "they", "them", "him", "her", "that", "this", "one")
               for q in ("in", "on", "for", "with", "to", "of")}


def dangling_end(toks: list[str]) -> bool:
    """Does this turn end on a function word that needs a continuation ("is it bad to take a")?"""
    t = [x for x in toks if x]
    if len(t) < 2 or t[-1] not in DANGLING:
        return False
    return (t[-2], t[-1]) not in _DANGLE_OK


def unfinished(toks: list[str]) -> bool:
    """The turn was cut off: mid-correction ("..., sorry") or mid-phrase ("... to take a")."""
    return repair_end(toks) or dangling_end(toks)


# --- conversions: "what's a hundred US dollars in euros", "how many kilometers is 26 miles" ----------------------------
_CURRENCY = set("""dollar dollars usd buck bucks euro euros eur pound pounds sterling gbp quid yen jpy yuan rmb renminbi
rupee rupees inr peso pesos franc francs chf won ruble rubles rouble roubles rand real reais lira liras dirham dirhams
riyal riyals krona kronor krone kroner zloty zlotys baht ringgit dong shekel shekels bitcoin bitcoins btc ether eth
cad aud cent cents pence""".split())
_MEASURE = set("""mile miles km kms kilometer kilometers kilometre kilometres meter meters metre metres foot feet ft inch
inches cm centimeter centimeters centimetre centimetres mm millimeter millimeters millimetre millimetres yard yards lb lbs
kilogram kilograms kilo kilos kg kgs gram grams ounce ounces oz stone stones ton tons tonne tonnes liter liters litre
litres gallon gallons quart quarts pint pints cup cups ml milliliter milliliters millilitre millilitres tablespoon
tablespoons teaspoon teaspoons fahrenheit celsius kelvin degree degrees mph kph acre acres hectare hectares""".split())
UNITS = _CURRENCY | _MEASURE
_UNIT_MOD = {"us", "u.s", "american", "canadian", "australian", "british", "hong", "kong", "new", "zealand", "swiss",
             "japanese", "chinese", "indian", "mexican", "south", "african", "of", "square", "cubic", "fluid", "metric",
             "imperial", "the"}
_NUMW = set("""one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion
trillion half dozen grand k""".split())
_BIG = {"hundred", "thousand", "million", "billion", "trillion", "dozen", "grand"}


def _numtok(t: str) -> bool:
    return any(c.isdigit() for c in t) or t in _NUMW


def _unit_after(toks: list[str], i: int, span: int = 3) -> int:
    """Index of the first unit word in toks[i:i+span] (skipping modifiers: "US dollars", "square feet"), or -1."""
    for k in range(i, min(len(toks), i + span)):
        if toks[k] in UNITS:
            return k
        if toks[k] not in _UNIT_MOD and not _numtok(toks[k]):
            return -1
    return -1


def conversion(words: list[str], drop: set | frozenset = frozenset()) -> tuple:
    """(amount as said, from unit, to unit) when the question gives an amount and asks for it in another unit:
    "a hundred US dollars in euros", "convert 50 euros to dollars", "how many kilometers is 26 miles". () otherwise.
    Corrected-away words (drop) are skipped: "100 dollars in pounds, sorry, euros" converts to euros."""
    idx = [i for i, w in enumerate(words) if i not in drop and norm(w)]
    raw = [words[i] for i in idx]
    toks = [norm(w).replace("$", "") for w in raw]
    for a in range(len(toks)):
        if not _numtok(toks[a]) or toks[a] in ("k", "grand"):
            continue
        b = a
        while b + 1 < len(toks) and (_numtok(toks[b + 1]) or (toks[b + 1] in ("and", "point") and b + 2 < len(toks)
                                                               and _numtok(toks[b + 2]))):
            b += 1
        k = _unit_after(toks, b + 1)
        if k < 0 and "$" in raw[a]:
            k, src = b, "dollars"                    # "$100 in euros": the unit is the symbol
        elif k < 0:
            continue
        else:
            src = toks[k]
            while k + 2 < len(toks) and _numtok(toks[k + 1]) and toks[k + 2] in UNITS:
                k += 2                               # "5 feet 10 inches"
        dst = ""
        for c in range(k + 1, len(toks) - 1):       # "... in / to / into euros"
            if toks[c] in ("in", "to", "into", "as"):
                u = _unit_after(toks, c + 1)
                if u >= 0 and toks[u] != src:
                    dst = toks[u]
                    break
        if not dst:                                  # "how many kilometers is 26 miles"
            for c in range(0, a - 1):
                if toks[c] in ("many", "much") and toks[c + 1] in UNITS and toks[c + 1] != src:
                    dst = toks[c + 1]
                    break
        if not dst:
            return ()
        s = a - 1 if a > 0 and toks[a - 1] == "a" and toks[a] in _BIG else a
        return (" ".join(raw[s:k + 1]).lower(), src, dst)
    return ()


def _correction_at(toks: list[str]) -> int:
    """Index just after the last self-correction marker, or -1."""
    last = -1
    for i in range(len(toks)):
        for pat in CORRECTION:
            if tuple(toks[i:i + len(pat)]) == pat:
                last = max(last, i + len(pat))
    return last


def _cue_tokens(toks: list[str], cut: int) -> set:
    """Token indices of every correction cue that ends at or before `cut`."""
    out = set()
    if cut <= 0:
        return out
    for i in range(len(toks)):
        for pat in CORRECTION:
            if tuple(toks[i:i + len(pat)]) == pat and i + len(pat) <= cut:
                out.update(range(i, i + len(pat)))
    return out


def _cap(w: str) -> bool:
    return bool(w) and (w[0].isupper() or w[0].isdigit())


def _phrases(words: list[str], toks: list[str], slot_spans: list[dict], cut: int, drop_tok: set) -> list[str]:
    """Entity phrases for the entity-only query, in spoken order. Mention spans keep their inner stopwords ("united
    states", "pride and prejudice") and grow over the proper-noun run around them (the tagger's span is sometimes
    just "Mount" of "Mount Kilimanjaro", "Apollo" of "Apollo 11"); after a correction with no tagged entity, the words
    after the cue are the replacement ("sorry, I mean pride and prejudice"), trimmed of edge stopwords."""
    out = []
    for s in slot_spans:
        if _cls(s) != "mention":
            continue
        a, b = s["start"], s["end"]
        while b < len(words) and b not in drop_tok and _cap(words[b]) and toks[b] and toks[b] not in STOP:
            b += 1
        while a - 1 > 0 and a - 1 not in drop_tok and _cap(words[a - 1]) and toks[a - 1] and toks[a - 1] not in STOP:
            a -= 1
        ws = [t for t in toks[a:b] if t]
        while ws and ws[0] in STOP:
            ws = ws[1:]
        while ws and ws[-1] in STOP:
            ws = ws[:-1]
        if ws:
            out.append((s["start"], " ".join(ws)))
    if cut > 0 and not any(st >= cut for st, _ in out):
        ws = [toks[i] for i in range(cut, len(toks)) if toks[i] and i not in drop_tok]
        while ws and ws[0] in STOP:
            ws = ws[1:]
        while ws and ws[-1] in STOP:
            ws = ws[:-1]
        if ws and len(ws) <= 5:
            out.append((cut, " ".join(ws)))
    out.sort()
    return [p for _, p in out]


def _cls(s: dict) -> str:
    """Correction class: any entity mention replaces any entity mention; numbers replace the same number type."""
    return "mention" if s["type"].startswith("entity.mention") else s["type"]


MODIFIERS = set("""today tomorrow tonight yesterday weekend morning afternoon evening night winter summer spring fall
autumn monday tuesday wednesday thursday friday saturday sunday january february march april may june july august
september october november december week month year next last this""".split())
TIME_WORDS = MODIFIERS - {"next", "last", "this", "week", "month", "year"}
_POSS = {"its", "his", "their", "her"}
_REF_PRONOUN = ["it", "he", "she", "him", "they", "them", "its", "his", "her", "their", "it's", "there", "that", "this"]


def _inline(toks: list[str], name: str, span, prev, prev_q: list[str], only_ref: bool, skip=frozenset(),
            place: bool = False) -> str:
    """The question with its reference replaced in place: "when was it founded" -> "when was nvidia founded", "where
    is its headquarters" -> "where is the headquarters of sony", "what is the population there" -> "... in canberra",
    "who founded that company" -> "who founded nvidia". A reference with nothing else said ("go back to the first
    one") -> the previous question about the referent. "" when there is nothing to replace."""
    t = [w for i, w in enumerate(toks) if w and i not in skip]
    if only_ref and prev_q:
        olds = sorted((prev or {}).get("entities") or [], key=len, reverse=True)
        q = f" {' '.join(prev_q)} "
        for o in olds:
            if f" {o} " in q:
                return q.replace(f" {o} ", f" {name} ").strip()
        return ""
    if span is not None:
        a, b = span
        return " ".join([w for w in toks[:a] if w] + [name] + [w for w in toks[b:] if w])
    for p in (["there"] if place else []) + _REF_PRONOUN:
        if p in t:
            i = t.index(p)
            nxt = t[i + 1] if i + 1 < len(t) else ""
            if p in _POSS and nxt and nxt not in STOP:
                rep, j = f"the {nxt} of {name}", i + 2
            elif p == "there":
                rep, j = f"in {name}", i + 1
            elif p == "it's":
                rep, j = f"{name} is", i + 1
            else:
                rep, j = name, i + 1
            return " ".join(t[:i] + [rep] + t[j:])
    return ""


def _apply_user(f: Frame, toks: list[str], content: list, user: dict, ent) -> list:
    """User facts (server/memory.py) fill what the question leaves out: no place in a weather / time / "near me"
    question -> the user's home; a food question -> their diet. Only the resolved word reaches the query."""
    from .memory import _FOOD, _NEAR, _TIME_PLACE, _WEATHER
    used = []
    home, diet = user.get("home"), user.get("diet")
    low = " ".join(toks)
    has_place = any(t == "place" for t, _ in f.slots) or (ent is not None and ent[0] == "place")
    if home and not has_place and (set(toks) & _WEATHER or _NEAR.search(low) or _TIME_PLACE.search(low)):
        used.append(("home", home))
        f.slots = tuple(sorted(set(f.slots) | {("place", home)}))
        content = content + [w for w in home.split() if w not in content]
    if diet and set(toks) & _FOOD and diet not in toks:
        used.append(("diet", diet))
        content = content + [diet]
    if used:
        f.user_used = tuple(used)
    return content


def build(words: list[str], spans: list[dict], current: dict | None, events_seen: set | None = None,
          prev_intent: tuple | None = None, local_ok: bool = True, carry: dict | None = None) -> Frame:
    toks = [norm(w) for w in words]
    f = Frame(words=list(words), spans=spans)
    cur = current or {}
    intents = [x for x in (cur.get("intent") or []) if x.get("on")]
    fine = [x for x in intents if x["tag"].startswith("discourse.intent.")]
    coarse = [x for x in intents if x["tag"].startswith("intent.")]
    if fine:
        f.intent, f.intent_p = fine[0]["tag"], float(fine[0]["p"])
    elif coarse:
        f.intent, f.intent_p = coarse[0]["tag"], float(coarse[0]["p"])
    f.coarse = coarse[0]["tag"] if coarse else ""
    if (not fine or (f.intent.startswith("intent.") and len(words) > 2)) and prev_intent and prev_intent[0]:
        # a correction or filler word can drop the fine intent for a word or two; keep the last confident one
        if prev_intent[0].startswith("discourse.intent.") and not fine:
            f.intent, f.intent_p = prev_intent
    f.act_question = any(x["tag"] == "discourse.act.question" and x.get("on") for x in (cur.get("act") or []))
    f.acts = tuple(x["tag"].split(".")[-1] for x in (cur.get("act") or [])
                   if x.get("on") and x["tag"].startswith("discourse.act."))
    f.directed = next((float(x["p"]) for x in (cur.get("act") or [])
                       if x["tag"] == "discourse.assistant.device_directed"), None)
    f.jailbreak = any(x["tag"] == "safety.jailbreak.attempt" and x.get("on") for x in (cur.get("safety") or []))
    # personal data spans: their words stay out of the query, the slots and the entity phrases
    pii, redact_tok = [], set()
    for s in spans:
        if s["type"] in PII_REDACT and s["end"] <= len(toks):
            hide = _pii_hide(s["type"], toks, s["start"], s["end"], words)
            if hide:
                pii.append(s)
                redact_tok |= hide
    if config.PII_YEARS:
        by = birth_year_tokens(toks) - redact_tok
        if by:
            pii.append({"type": "safety.pii.date_of_birth"})
            redact_tok |= by
    redact_tok |= {i for i, t in enumerate(toks) if t in PROFANITY}   # never useful to search, never sent
    f.redact_tok = frozenset(redact_tok)
    f.redacted = tuple(dict.fromkeys(PII_REDACT[s["type"]] for s in pii))
    f.command_complete = bool(events_seen and "discourse.assistant.command_complete" in events_seen)

    # slots: entity mentions + typed numbers (drop the generic entity.numeral duplicate)
    slot_spans = [s for s in spans if s.get("family") in ("entity", "number") and s["type"] != "entity.numeral"
                  and s["end"] <= len(toks) and not set(range(s["start"], s["end"])) & redact_tok
                  # a time word the tagger read as a mention ("in winter" -> place) is not a subject
                  and not (s["type"].startswith("entity.mention") and toks[s["start"]:s["end"]]
                           and all(t in MODIFIERS for t in toks[s["start"]:s["end"]]))]
    cut = _correction_at(toks)
    drop_tok = set()
    cue_tok = _cue_tokens(toks, cut)
    f.cue = frozenset(cue_tok)
    if cue_tok:
        # the tagger sometimes tags the cue itself as an entity ("Sari," → place, "Scratch" → organization); such a
        # span would be "corrected away" instead of the real one (q31: Paris survived, "Sari" was dropped)
        slot_spans = [s for s in slot_spans if not set(range(s["start"], s["end"])) & cue_tok]
    if cut > 0:
        f.corrected = True
        after_types = {_cls(s) for s in slot_spans if s["start"] >= cut}
        # for each corrected class, drop the LAST span of that class said before the marker
        drop_ids = set()
        for c in after_types:
            before = [i for i, s in enumerate(slot_spans) if s["start"] < cut and _cls(s) == c]
            if before:
                drop_ids.add(before[-1])
        keep = []
        for i, s in enumerate(slot_spans):
            if i in drop_ids:
                drop_tok.update(range(s["start"], s["end"]))      # the corrected-away entity
                continue
            keep.append(s)
        slot_spans = keep
        # marker words themselves
        for i in range(len(toks)):
            for pat in CORRECTION:
                if tuple(toks[i:i + len(pat)]) == pat and i + len(pat) <= cut:
                    drop_tok.update(range(i, i + len(pat)))
        before_mentions = [s for s in slot_spans if s["start"] < cut and _cls(s) == "mention"]
        if not after_types and before_mentions and any(toks[i] and toks[i] not in STOP for i in range(cut, len(toks))):
            # "where was Mozart born, actually make that Beethoven" with the replacement untagged: the corrected thing
            # is the entity said before the cue, not the last content word ("born" was dropped, Mozart kept)
            s0 = before_mentions[-1]
            drop_tok.update(range(s0["start"], s0["end"]))
            slot_spans = [s for s in slot_spans if s is not s0]
        elif not after_types:
            # "the Lakers game, sorry, Warriors game": no typed span after the marker. Treat the word right after
            # the marker as replacing the content word right before it.
            after = [i for i in range(cut, len(toks)) if toks[i] and toks[i] not in STOP]
            before = [i for i in range(0, cut) if toks[i] and toks[i] not in STOP and i not in drop_tok]
            if after and before and not all(toks[i] in NOT_REPLACEMENT for i in after):
                # drop the before-token(s) that the after-phrase repeats around (e.g. "game")
                rep = {toks[i] for i in after}
                # the replacement repeats a head word of the before-phrase ("the BOILING point of water, no wait, the
                # FREEZING point"): the word right before that head is what changed, not the last content word
                # ("water"; cx11 kept "boiling", dropped "water" and answered 100 C)
                head = next((k for k, i in enumerate(after) if toks[i] in {toks[j] for j in before}), None)
                anchor = None
                if head and toks[after[head - 1]] not in {toks[j] for j in before}:
                    anchor = next((j for j in before if toks[j] == toks[after[head]]), None)
                prev = [j for j in before if anchor is not None and j < anchor]
                if prev and toks[prev[-1]] not in rep:
                    drop_tok.add(prev[-1])
                else:
                    for i in reversed(before):
                        if toks[i] in rep:
                            continue
                        drop_tok.add(i)
                        break
    if drop_tok:
        # a slot whose words were corrected away goes too ("capital of Canada, actually make that Turkey")
        slot_spans = [s for s in slot_spans if not set(range(s["start"], s["end"])) & drop_tok]
    f.slots = tuple(sorted({(s["type"].split(".")[-1], norm(s["text"])) for s in slot_spans}))
    f.dropped = frozenset(drop_tok)
    f.ent_types = tuple(dict.fromkeys(s["type"].split(".")[-1] for s in slot_spans
                                      if s["type"].startswith("entity.mention")))
    f.num_types = tuple(dict.fromkeys(s["type"].split(".")[-1] for s in slot_spans
                                      if s["type"].startswith("entity.number")))
    low_kept = " ".join(t for i, t in enumerate(toks) if t and i not in drop_tok)
    # each typed time span with the two words before it: Arc often tags just "month" of "this month"
    time_spans = [" ".join(t for t in toks[max(0, s["start"] - 2):s["end"]] if t) for s in slot_spans
                  if s["type"] in ("entity.number.date", "entity.number.duration", "entity.number.time_of_day")]
    for k, src in enumerate(time_spans + [low_kept]):
        typed = k < len(time_spans)
        hit = None
        for rx, val in ((_WHEN_DAY, "day"), (_WHEN_WEEK, "week")) + \
                (((_WHEN_MONTH, "month"), (_WHEN_YEAR, "year")) if typed else ()):
            m = rx.search(src)
            if m:
                hit = (val, m.group(0))
                break
        if hit:
            f.when, f.when_text = hit
            f.when_typed = typed
            break
    f.phrases = tuple(_phrases(f.words, toks, slot_spans, cut, drop_tok | redact_tok))
    f.convert = conversion(f.words, drop_tok | redact_tok)
    money_tok = {i for s in slot_spans if s["type"] in ("entity.number.money", "entity.number.percent")
                 for i in range(s["start"], s["end"])}
    seen, content, content_idx = set(), [], []
    for i, t in enumerate(toks):
        if i in money_tok and t and i not in drop_tok and i not in redact_tok:
            # keep a typed amount as said: "$100", "15%" (norm() would strip the unit symbol: "100")
            raw = re.sub(r"[^\w$%.,€£¥]", "", words[i]).strip(".,").lower()
            t = raw or t
        if not t or t in STOP or i in drop_tok or i in redact_tok or t in seen:
            continue
        seen.add(t)
        content.append(t)
        content_idx.append(i)
    f.content_idx = tuple(content_idx)
    # conversation memory: "who founded Nvidia?" → "when was it founded?" — a pronoun and no entity of its own
    # inherits the previous question's entity (carry = that frame's mention slots)
    has_mention = any(t not in NUMBER_TYPES for t, _ in f.slots)
    strong = any(t in PRONOUNS - WEAK_PRONOUNS for t in toks)
    weak = any(t in WEAK_PRONOUNS for t in toks) and len(content) <= 2
    ent = None
    pick = (carry or {}).get("pick") if not has_mention else None
    span = None
    if pick and pick.get("ent"):
        # an explicit reference memory resolved: "that company", "go back to the first one" (server/memory.py)
        ent = tuple(pick["ent"])
        a, b = pick.get("span") or (0, 0)
        if b > a:
            span = (a, b)
            keep = [(w, i) for w, i in zip(content, content_idx) if not a <= i < b]
            content, content_idx = [w for w, _ in keep], [i for _, i in keep]
            f.content_idx = tuple(content_idx)
        f.mem_ref = pick.get("why", "")
    elif carry and not has_mention and (strong or weak):
        # he/she → a person (often named only in the previous ANSWER: "relativity?" → "Einstein" → "where was he
        # born"); it/they/there → the thing the previous question was about, else any entity
        personal = any(t in PERSONAL for t in toks)
        order = ["person", "thing"] if personal else ["thing", "person"]
        for k in order:
            if carry.get(k):
                ent = tuple(carry[k][0])
                break
    # elliptical follow-ups with no pronoun: "and when?", "what about Nvidia?", "how about in 2020?", "and the CEO?"
    prev = (carry or {}).get("prev")
    head2 = toks[:2]
    elliptic = bool(prev) and (toks[:1] in (["and"], ["also"]) or head2 in (["what", "about"], ["how", "about"])
                               or (len(content) == 0 and len(toks) <= 4))
    prev_q = [norm(w) for w in ((prev or {}).get("inline") or (prev or {}).get("question", "")).split() if norm(w)]
    if elliptic and not ent:
        pc = list(prev.get("content") or [])
        pents = [w for e in prev.get("entities") or [] for w in e.split()]
        swap = head2 in (["what", "about"], ["how", "about"])
        wh = None if swap else next((t for t in toks if t in ("when", "where", "who", "why", "how", "which", "whom",
                                                               "whose")), None)
        modifier = bool(content) and all(w in MODIFIERS or w.isdigit() for w in content)
        if modifier:
            # "what about tomorrow?", "how about in winter?": the same question and subject, at another time
            # a new day / season replaces the one the previous question named ("today" -> "tomorrow")
            new_t = [w for w in content if w in TIME_WORDS]
            pc = [w for w in pc if not (new_t and w in TIME_WORDS)]
            content = pc + [w for w in content if w not in pc]
            if prev.get("entities"):
                f.carried = prev["entities"][0]
                f.phrases = (prev["entities"][0],)
            tail = [t for t in toks[2 if swap else 1:] if t]
            if prev_q and new_t and any(w in TIME_WORDS for w in prev_q):
                i = next(i for i, w in enumerate(prev_q) if w in TIME_WORDS)
                f.inline = " ".join(prev_q[:i] + [w for w in tail if w not in ("in", "on", "at")] + prev_q[i + 1:])
            else:
                f.inline = " ".join(prev_q + tail) if prev_q else ""
        elif has_mention or (swap and content):
            # "what about Nvidia": same question, new subject → swap the previous entity for the new one (the words
            # after "about" are the new subject even when the tagger doesn't mark them in so short a phrase)
            base = [w for w in pc if w not in pents]
            content = base + [w for w in content if w not in base]
            new = f.phrases[-1] if f.phrases else " ".join(w for w in content if w not in base)
            old = next((e for e in prev.get("entities") or [] if f" {e} " in f" {' '.join(prev_q)} "), "")
            if old and new:
                f.inline = f" {' '.join(prev_q)} ".replace(f" {old} ", f" {new} ").strip()
        else:
            # "and when?": the previous question's subject and verb + the new question word
            content = pc + [w for w in content if w not in pc]
            if prev.get("entities"):
                f.carried = prev["entities"][0]
                f.phrases = (prev["entities"][0],)
        if wh and wh not in content:
            content.append(wh)
        f.followup_of = prev.get("question", "")
        if not f.intent and prev.get("intent"):
            f.intent = prev["intent"]
    if ent:
        f.carried = ent[1]
        f.phrases = (ent[1],)
        f.slots = tuple(sorted(set(f.slots) | {ent}))
        only_ref = prev and not content
        if only_ref:
            # "and when was that?" / "go back to the first one": a reference AND nothing else said → the previous
            # question's meaning (its subject swapped for the referent) + question word
            pents = {w for e in prev.get("entities") or [] for w in e.split()} - set(ent[1].split())
            content = [w for w in (prev.get("content") or []) if w not in pents]
            wh = next((t for t in toks if t in ("when", "where", "who", "why", "how", "which")), None)
            if wh and wh not in content:
                content.append(wh)
            f.followup_of = prev.get("question", "")
        for w in reversed(ent[1].split()):
            if w not in content:
                content.insert(0, w)
        f.inline = _inline(toks, ent[1], span, prev, prev_q, bool(only_ref), drop_tok | redact_tok,
                           place=ent[0] == "place")
    user = (carry or {}).get("user")
    if user:
        content = _apply_user(f, toks, content, user, ent)
    f.content = tuple(content)

    low = " " + " ".join(toks) + " "
    local = any(" " + c + " " in low for c in LOCAL_CUES)
    # local corpus only when the question points at it ("in the game", "the judge", ...): searching it for every
    # web question put unrelated moments into the sources (eval smoke run, q26)
    f.route = "local" if (local and local_ok) else "web"
    return f
