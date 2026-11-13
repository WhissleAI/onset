"""Tiered router: the live query frame -> the cheapest lane that can answer it.

  card  : live-data verticals (server/verticals/): weather, currency / crypto, time / date, sunrise / sunset, units.
          Free, no key, ~0.1-0.6 s (units / time: 0 network). A card that fits answers by itself (answer.py card path).
  facts : local knowledge (server/knowledge.py): Wikidata properties, the local Wikipedia lead index, Wiktionary,
          the Wikipedia REST summary. Free.
  web   : the paid lane (Tavily | Wikipedia), for fresh or open-ended questions (news, scores, reviews, how-tos,
          health, recommendations) and anything the router is unsure about.
  wait  : (speculative calls only) the frame names a card / fact question whose key argument hasn't been said yet
          ("what's the weather in", "who founded the company"): spend nothing, the next frame will have it.

Every plan has a confidence. Cheap lanes run first; paid web is skipped when the cheap lane came back strong (a card,
a Wikidata property, a confident lead), and stays the fallback otherwise. The router runs inside lanes.web(), so both
race modes route identically: the A/B still differs only in when things start.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from . import config, knowledge
from .frame import Frame, norm
from .verticals import clock, fx, geo, units, weather

# ------------------------------------------------------------------------------------------------- text helpers
_WEATHER = {"weather", "rain", "raining", "rainy", "snow", "snowing", "sunny", "temperature", "forecast", "cold",
            "hot", "warm", "windy", "humid", "humidity", "storm", "stormy", "umbrella", "degrees", "cloudy", "foggy",
            "freezing", "chilly", "drizzle", "thunderstorm", "thunderstorms"}
_FRESH = re.compile(r"\b(latest|news|headlines?|today'?s?|tonight|right now|currently|current|this (?:week|weekend|"
                    r"morning|afternoon|evening|year|season)|last (?:night|week|weekend)|yesterday|score|scores|stocks?|"
                    r"price|prices|next|upcoming|recent|recently|lately|live|so far|open now|close today|hours|"
                    r"(?:who|which team|did \w+(?: \w+){0,3}) (?:won|win|beat|lose|lost)\b.*\b(?:game|match|race|"
                    r"grand prix|election|super bowl|final|finals|tournament|championship|playoffs?|series|cup))\b")
_OPEN = re.compile(r"\b(how (?:do|can|should|to|would) (?:i|you|we)|how to|what should i|should i|can i|is it (?:bad|"
                   r"safe|ok|okay|good)|best|recommend|recommendations?|review|reviews|vs|versus|better|worse|"
                   r"symptoms?|treat|cure|near me|places|restaurants?|tips|ideas|instead of|substitute|why)\b")
_WH = re.compile(r"^(?:who|what|whats|what's|when|where|which|how (?:many|much|tall|long|old|far|big|high|deep|fast|"
                 r"hot|cold|heavy))\b")
_TIME_Q = re.compile(r"\b(what time is it|what(?:'s| is) the (?:current |local )?time|(?:current|local) time|"
                     r"time (?:is it )?(?:right )?now|time zone is)\b")
_DATE_Q = re.compile(r"\b(what(?:'s| is) (?:the )?(?:date|day)(?: today)?|what day is (?:it|today)|today'?s date)\b")
_SUN_Q = re.compile(r"\b(sunrise|sunset|sun ?rise|sun ?set|sun (?:come up|go down|comes up|goes down)|"
                    r"dawn|dusk)\b")
_IN_PLACE = re.compile(r"\b(?:in|for|at) (?P<p>[a-z][a-z .'-]{1,40}?)(?:\s+(?:today|tonight|tomorrow|this|next|on|"
                       r"right|now|over|at|during|later|gonna|going|be|like|for|the day)\b|$)")
_NOT_PLACE = {"the morning", "the evening", "the afternoon", "the world", "the us", "general", "fahrenheit",
              "celsius", "degrees", "march", "april", "may", "june", "july", "august", "the summer", "the winter",
              "the spring", "the fall", "a row", "the day", "the week", "the weekend", "the past"}
_CRYPTO_Q = re.compile(r"\b(bitcoin|btc|ethereum|ether|eth)\b")
_RATE_Q = re.compile(r"\b(exchange rate|rate|worth|convert|conversion|how much is|how many|in|to)\b")


def repaired_text(f: Frame) -> str:
    """The question as meant, with a spoken correction put back IN PLACE: "When was Google founded, no wait, I mean
    Apple" -> "When was Apple founded" (Frame.clean_text drops Google but leaves Apple at the end, which breaks
    "when was X founded"-style patterns). Falls back to clean_text when there is nothing to splice."""
    if not f.dropped:
        return f.clean_text
    cue = set(f.cue)
    ent = sorted(set(f.dropped) - cue)
    if not ent or not cue:
        return f.clean_text
    last_cue = max(cue)
    a, b = ent[0], ent[-1]
    if b > last_cue:
        return f.clean_text
    repl = [w for i, w in enumerate(f.words) if i > last_cue and i not in f.dropped]
    if not repl:
        return f.clean_text
    before = [w for i, w in enumerate(f.words[:a]) if i not in f.dropped]
    mid = [w for i, w in enumerate(f.words) if b < i < min(cue) and i not in f.dropped]
    for j in range(min(3, len(before), len(repl)), 0, -1):          # "president of France, no wait, of the US"
        if [norm(x) for x in before[-j:]] == [norm(x) for x in repl[:j]]:
            repl = repl[j:]
            break
    return " ".join(before + repl + mid)


def _low(text: str) -> str:
    return " ".join(norm(w) for w in text.replace("?", " ").split() if norm(w))


def place_of(f: Frame | None, text: str) -> str:
    """The place said: the tagger's last place span, else 'in <place>' from the words."""
    if f is not None:
        ps = [s for t, s in f.slots if t == "place"]
        if ps:
            return ps[-1]
    low = _low(text)
    ms = list(_IN_PLACE.finditer(low))
    for m in reversed(ms):
        p = m.group("p").strip()
        p = re.sub(r"^(?:the city of|downtown) ", "", p)
        if p and p not in _NOT_PLACE and not re.search(r"\d", p):
            return p
    return ""


def fx_args(text: str, convert: tuple = ()) -> dict | None:
    """{"amount", "src", "dst"} for a currency / crypto question, else None."""
    from .answer import _amount_value
    low = _low(text)
    if convert:
        amount_said, su, du = convert
        ctx_s = low[: low.find(su)] if su in low else ""
        ctx_d = low[: low.rfind(du)] if du in low else low
        s, d = fx.code(su, amount_said + " " + ctx_s[-20:]), fx.code(du, ctx_d[-20:])
        if s and d and s != d:
            return {"amount": _amount_value(amount_said), "src": s, "dst": d}
    toks = low.split()
    cur = []
    for i, w in enumerate(toks):
        if fx.is_currency(w) and not (w == "won" and i + 1 < len(toks) and toks[i + 1] in ("the", "a")):
            c = fx.code(w, " ".join(toks[max(0, i - 3):i]))
            if c and (not cur or cur[-1][1] != c):
                cur.append((i, c, w))
    if not cur:
        return None
    if not _RATE_Q.search(low) and not _CRYPTO_Q.search(low):
        return None
    if re.search(r"\b(stock|shares?|market cap)\b", low):
        return None
    m = re.search(r"\bhow many (\w+)", low)
    if len(cur) >= 2:
        src, dst = cur[0][1], cur[1][1]
        if m and fx.is_currency(m.group(1)) and fx.code(m.group(1)) == cur[0][1]:
            src, dst = cur[1][1], cur[0][1]           # "how many yen (are) in a dollar"
        amt = _amount_value(" ".join(toks[max(0, cur[0][0] - 3):cur[0][0] + 1]))
        if src != cur[0][1]:
            amt = None
        return {"amount": amt, "src": src, "dst": dst}
    one = cur[0][1]
    if one in fx.CRYPTO:
        return {"amount": _amount_value(" ".join(toks[max(0, cur[0][0] - 3):cur[0][0] + 1])), "src": one, "dst": "USD"}
    if re.search(r"\b(exchange rate|rate)\b", low) and one != "USD":
        return {"amount": None, "src": one, "dst": "USD"}
    return None


# ------------------------------------------------------------------------------------------------- plan
@dataclass
class Plan:
    lane: str                          # card | facts | web | wait
    vertical: str = ""
    conf: float = 0.0
    args: dict = field(default_factory=dict)
    why: str = ""

    @property
    def skip_web(self) -> bool:
        return self.lane in ("card", "facts") and self.conf >= config.ROUTER_SKIP_CONF

    def to_json(self) -> dict:
        return {"lane": self.lane, "vertical": self.vertical, "conf": round(self.conf, 2), "why": self.why,
                "args": {k: v for k, v in self.args.items() if isinstance(v, (str, int, float, type(None)))}}


def plan(f: Frame) -> Plan:
    text = repaired_text(f)
    low = _low(text)
    toks = set(low.split())
    # --- units (no network): "how many ounces are in one cup", "26 miles in km"
    u = units.parse(text)
    if u:
        return Plan("card", "units", 0.95, {"parse": u}, "unit conversion")
    # --- currency / crypto
    fa = fx_args(text, f.convert)
    if fa:
        return Plan("card", "fx", 0.9, fa, "currency" + (" (crypto)" if {fa["src"], fa["dst"]} & fx.CRYPTO else ""))
    if f.intent == "discourse.intent.qa_currency" and f.intent_p >= 0.5 and re.search(r"\b(exchange|rate)\b", low):
        return Plan("wait", "fx", 0.5, {}, "currency question, second currency not said yet")
    # --- time / date
    if _TIME_Q.search(low) or _DATE_Q.search(low):
        what = "date" if _DATE_Q.search(low) and not _TIME_Q.search(low) else "time"
        return Plan("card", "time", 0.9, {"place": place_of(f, text), "what": what}, f"local {what}")
    # --- sunrise / sunset
    if _SUN_Q.search(low):
        p = place_of(f, text)
        which = "sunrise" if re.search(r"\b(sunrise|sun ?rise|come up|comes up|dawn)\b", low) else "sunset"
        when = weather.parse_day(text)
        off = when[1] if when and when[0] == "day" else 0
        if p:
            return Plan("card", "sun", 0.9, {"place": p, "which": which, "day": off}, which)
        return Plan("wait", "sun", 0.4, {}, f"{which}: no place yet")
    # --- weather
    wx = toks & _WEATHER
    wx_intent = f.intent == "discourse.intent.weather_query" and f.intent_p >= 0.5
    if wx and (wx_intent or toks & {"weather", "forecast", "rain", "snow", "temperature", "umbrella", "raining",
                                    "snowing", "degrees"}):
        when = weather.parse_day(text)
        if when == ("far",):
            return Plan("web", "weather", 0.4, {}, "weather outside the 7-day forecast")
        p = place_of(f, text)
        if p:
            rain_q = bool(re.search(r"\b(rain|raining|umbrella|snow|snowing|wet)\b", low)) and \
                bool(re.search(r"^(?:will|is|does|do|should|gonna|is it going)\b|\bwill it\b|\bgoing to\b", low))
            return Plan("card", "weather", 0.92, {"place": p, "when": when, "rain_q": rain_q}, "weather forecast")
        return Plan("wait", "weather", 0.5, {}, "weather: no place yet")
    # --- local knowledge
    fresh = bool(f.when) or bool(_FRESH.search(low))
    fq = knowledge.parse_fact(text)
    if fq and not fresh:
        c = 0.6 if fq.prop in knowledge.CHANGING else 0.85
        return Plan("facts", "wikidata", c, {"fact": fq, "subject": fq.subject}, f"Wikidata {fq.prop}")
    w = knowledge.parse_define(text)
    if w and not fresh:
        return Plan("facts", "define", 0.85, {"word": w}, "definition")
    if knowledge.waiting_head(text) and not fresh:
        return Plan("wait", "facts", 0.5, {}, "fact question, subject not said yet")
    s = knowledge.parse_whois(text)
    if s and not fresh and not _OPEN.search(low):
        return Plan("facts", "whois", 0.8, {"subject": s}, "who/what is")
    if fresh:
        return Plan("web", "", 0.3, {}, "fresh: " + (f.when_text or (_FRESH.search(low).group(0) if _FRESH.search(low)
                                                                      else "")))
    if _OPEN.search(low):
        return Plan("web", "", 0.3, {}, "open-ended: " + _OPEN.search(low).group(0))
    factoid = f.intent in ("discourse.intent.qa_factoid", "discourse.intent.qa_definition") or \
        bool(_WH.search(knowledge.norm(text)))
    if factoid:
        return Plan("facts", "leads", 0.5, {"query": text}, "stable factoid")
    return Plan("web", "", 0.2, {}, "unsure")


# ------------------------------------------------------------------------------------------------- run a plan
async def _cheap(p: Plan, f: Frame, meter: dict) -> list:
    v = p.vertical
    calls = [0]

    def n(k=1):
        calls[0] += k

    try:
        if v == "units":
            return [units.card(p.args["parse"]).doc()]
        if v == "fx":
            n()
            return [(await fx.card(p.args["amount"], p.args["src"], p.args["dst"])).doc()]
        if v == "time":
            place = None
            if p.args.get("place"):
                n()
                place = await geo.geocode(p.args["place"])
                if place is None:
                    return []
            return [clock.time_card(place, p.args.get("what", "time")).doc()]
        if v == "sun":
            n()
            place = await geo.geocode(p.args["place"])
            return [clock.sun_card(place, p.args["which"], p.args.get("day", 0)).doc()] if place else []
        if v == "weather":
            n(2)
            place = await geo.geocode(p.args["place"])
            if place is None:
                return []
            c = await weather.card(place, p.args.get("when"), p.args.get("rain_q", False))
            return [c.doc()] if c else []
        if v == "wikidata":
            n(2)
            fq = p.args["fact"]
            wd, ld = await asyncio.gather(knowledge.wikidata(fq), knowledge.leads(text_of(f), 2, fq.subject),
                                          return_exceptions=True)
            wd = [] if isinstance(wd, BaseException) else wd
            ld = [] if isinstance(ld, BaseException) else ld
            if wd:
                return wd + [d for d in ld[:1] if d.score >= 0.72]     # the fact + its article's lead for context
            return ld
        if v == "define":
            n()
            wt, ld = await asyncio.gather(knowledge.wiktionary(p.args["word"]),
                                          knowledge.leads(text_of(f), 2, p.args["word"]), return_exceptions=True)
            wt = [] if isinstance(wt, BaseException) else wt
            ld = [] if isinstance(ld, BaseException) else ld
            return wt + [d for d in ld[:1] if d.score >= 0.72]
        if v == "whois":
            n()
            ld, ws = await asyncio.gather(knowledge.leads(text_of(f), 3, p.args["subject"]),
                                          knowledge.wiki_summary(p.args["subject"]), return_exceptions=True)
            ld = [] if isinstance(ld, BaseException) else ld
            ws = [] if isinstance(ws, BaseException) else ws
            seen = {d.url for d in ws}
            return ws + [d for d in ld if d.url not in seen and (d.score >= 0.72 or not ws)]
        if v == "leads":
            return await knowledge.leads(text_of(f), 3)
    finally:
        meter["free_calls"] = meter.get("free_calls", 0) + calls[0]
    return []


def text_of(f: Frame) -> str:
    t = repaired_text(f)
    return f"{t} ({f.carried})" if f.carried else t


def _strong(docs: list, p: Plan) -> bool:
    if not docs:
        return False
    if p.lane == "card":
        return True
    return any(d.meta.get("strong") for d in docs)


async def route(query: str, f: Frame, *, final: bool, meter: dict, web_fn):
    """Run the plan for this frame. `web_fn()` is the paid web lane (lanes.web without the router). Returns a
    lanes.WebResult; meter gets route / tier / paid / free_calls for the metric and the UI chip."""
    from .lanes import WebResult
    t = time.perf_counter()
    p = plan(f) if f.route != "local" else Plan("web", "", 0.0, {}, "local corpus question")
    meter.update(route=p.lane, route_vertical=p.vertical, route_conf=round(p.conf, 2), route_why=p.why[:80])
    if p.lane == "wait":
        if not final:
            from .lanes import Skip
            meter.update(tier="wait", backend="router", credits=0.0, paid=0)
            raise Skip(p.why)
        p = Plan("web", p.vertical, 0.3, {}, p.why + " (final: web)")
    cheap: list = []
    if p.lane in ("card", "facts"):
        try:
            cheap = await asyncio.wait_for(_cheap(p, f, meter), config.ROUTER_LANE_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            meter["route_error"] = f"{type(e).__name__}: {e}"[:120]
            cheap = []
        meter["route_ms"] = round((time.perf_counter() - t) * 1000)
        strong = _strong(cheap, p)
        if cheap and (strong or (p.skip_web and p.lane == "card")):
            meter.update(tier=p.lane, tier_src=cheap[0].source, backend=p.lane, credits=0.0, paid=0)
            res = WebResult(cheap)
            res.backend = p.lane
            return res
        meter["route_fallback"] = "empty" if not cheap else "weak"
        if meter.get("no_paid") and not final:
            # a SPECULATIVE cheap lookup (the speculator's cheap-only plan): no paid fallback mid-sentence; the
            # endpoint decides (its final call does fall back to the web)
            from .lanes import Skip
            meter.update(tier="wait", backend="router", credits=0.0, paid=0)
            raise Skip(f"{p.lane} lookup {meter['route_fallback']}: no paid fallback while speculating")
    web = await web_fn()
    backend = getattr(web, "backend", "") or meter.get("backend", "")
    paid = 0 if (getattr(web, "cached", False) or backend in ("cache", "router")) else 1
    meter.update(tier="web" if len(web) or not cheap else p.lane, paid=paid)
    if web and len(web):
        meter["tier_src"] = web[0].source or backend
    elif cheap:
        meter["tier_src"] = cheap[0].source
    if not cheap:
        return web
    out = WebResult(list(cheap) + [d for d in web if d.id not in {c.id for c in cheap}])
    out.backend, out.credits, out.cached = backend, getattr(web, "credits", 0.0), getattr(web, "cached", False)
    out.answer = getattr(web, "answer", None)
    return out


# ------------------------------------------------------------------------------------------------- answer side
def card_fits(card: dict, question: str) -> bool:
    """Does this card answer `question` (the final question text)? Its key arguments must still be what was asked:
    the same place and day (weather), currencies and amount (fx), units and amount (units), place (time / sun)."""
    v, a = card.get("vertical"), card.get("args") or {}
    low = _low(question)
    if v == "units":
        u = units.parse(question)
        return bool(u) and u["src"] == a.get("src") and u["dst"] == a.get("dst") and \
            abs(u["amount"] - a.get("amount", -1)) < 1e-3
    if v == "fx":
        from .frame import conversion
        fa = fx_args(question, conversion(question.split()))
        return bool(fa) and fa["src"] == a.get("src") and fa["dst"] == a.get("dst") and \
            (fa["amount"] or 1.0) == (a.get("amount") or 1.0)
    place = (a.get("place") or "").strip()
    if place:
        said = place_of(None, question)
        # the place the final question names must BE the card's place, not contain it ("san" of "san francisco",
        # a partial the speculator fired on); when the words don't parse, the place must at least be said
        if (said and said != place) or f" {place} " not in f" {low} ":
            return False
    if v == "weather":
        return weather.when_key(weather.parse_day(question)) == a.get("when")
    if v == "sun":
        return bool(_SUN_Q.search(low))
    if v == "time":
        return bool(_TIME_Q.search(low) or _DATE_Q.search(low))
    return False


def pick_card(question: str, docs) -> tuple[int, dict] | None:
    """(index in docs, card) of the first card that fits the question, else None."""
    for i, d in enumerate(docs):
        c = (getattr(d, "meta", None) or {}).get("card")
        if c and card_fits(c, question):
            return i, c
    return None


def reset_caches():
    """Forget every cached card / fact (eval: each A/B run starts cold, so neither mode reads the other's lookups)."""
    from .verticals import fx as _fx, geo as _geo, weather as _wx
    for c in (_fx._cache, _fx._crypto, _geo._cache, _wx._cache, knowledge._cache):
        c.clear()


async def warm():
    from . import verticals
    await verticals.warm()


def health() -> dict:
    from . import verticals
    return {"on": config.ROUTER, "skip_conf": config.ROUTER_SKIP_CONF, **knowledge.stats(),
            "apis": {h: dict(s) for h, s in verticals.STATS.items()},
            "down": [h for h in verticals.STATS if verticals.host_down(h)]}
