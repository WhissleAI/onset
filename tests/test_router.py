"""Tiered router (server/router.py), verticals, knowledge patterns, card answers. No network: a mock transport plays
the free APIs (Open-Meteo, Frankfurter, Coinbase, Wikidata, Wiktionary, Wikipedia REST)."""
import asyncio
import json

import httpx
import pytest

from server import answer, config, knowledge, lanes, router, verticals
from server.frame import Frame, conversion
from server.verticals import fx, units, weather


@pytest.fixture(autouse=True)
def _router_on(monkeypatch):
    monkeypatch.setattr(config, "ROUTER", True)
    monkeypatch.setattr(config, "ROUTER_LEADS", False)          # the lead index is a built artifact: not in tests
    knowledge._cache.clear()
    verticals.STATS.clear()
    from server.verticals import fx as _fx, geo as _geo, weather as _wx
    _fx._cache.clear(), _fx._crypto.clear(), _geo._cache.clear(), _wx._cache.clear()
    calls: list[str] = []
    verticals.use_transport(httpx.MockTransport(lambda req: _fake_api(req, calls)))
    yield calls
    verticals.use_transport(None)


def _fake_api(req: httpx.Request, calls: list) -> httpx.Response:
    host, path = req.url.host, req.url.path
    calls.append(f"{host}{path}")
    q = dict(req.url.params)
    if host == "geocoding-api.open-meteo.com":
        name = q["name"]
        if name == "atlantis":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"results": [{"name": name.title(), "admin1": "Illinois", "country": "United States",
                                                      "country_code": "US", "latitude": 41.85, "longitude": -87.65,
                                                      "timezone": "America/Chicago", "population": 2_000_000}]})
    if host == "api.open-meteo.com":
        import datetime as dt
        from zoneinfo import ZoneInfo
        today = dt.datetime.now(ZoneInfo("America/Chicago")).date()
        days = [(today + dt.timedelta(days=i)).isoformat() for i in range(7)]
        return httpx.Response(200, json={
            "current": {"time": "2026-11-13T10:00", "temperature_2m": 5.0, "weather_code": 3, "wind_speed_10m": 10},
            "daily": {"time": days, "weather_code": [61] * 7, "temperature_2m_max": [10.0] * 7,
                      "temperature_2m_min": [2.0] * 7, "precipitation_probability_max": [70] * 7,
                      "precipitation_sum": [3.0] * 7, "sunrise": [""] * 7, "sunset": [""] * 7}})
    if host == "api.frankfurter.dev":
        return httpx.Response(200, json={"amount": 1.0, "base": q["base"], "date": "2026-11-12",
                                         "rates": {q["symbols"]: 0.9}})
    if host == "api.coinbase.com":
        return httpx.Response(200, json={"data": {"amount": "70000.00", "base": "BTC", "currency": "USD"}})
    if host == "www.wikidata.org":
        s = q.get("search", "")
        if s == "nowhere thing":
            return httpx.Response(200, json={"search": []})
        return httpx.Response(200, json={"search": [{"id": "Q1", "display": {"label": {"value": s.title()},
                                                                            "description": {"value": "a thing"}}}]})
    if host == "query.wikidata.org":
        return httpx.Response(200, json={"results": {"bindings": [
            {"item": {"value": "http://www.wikidata.org/entity/Q1"}, "pid": {"value": "P112"},
             "rank": {"value": "http://wikiba.se/ontology#NormalRank"}, "v": {"value": "http://x/Q2"},
             "vLabel": {"value": "Jensen Huang"}}]}})
    if host == "en.wiktionary.org":
        return httpx.Response(200, json={"en": [{"partOfSpeech": "Adjective",
                                                 "definitions": [{"definition": "Being <b>everywhere</b> at once."}]}]})
    if host == "en.wikipedia.org":
        return httpx.Response(404, json={})
    return httpx.Response(500)


def F(text: str, intent: str = "", p: float = 0.9, slots=(), **kw) -> Frame:
    words = text.split()
    f = Frame(words=words, intent=intent, intent_p=p, slots=tuple(slots), **kw)
    f.convert = conversion(words)
    return f


# ----------------------------------------------------------------------------------------------------- plans
def test_plan_lanes():
    assert router.plan(F("what's the weather gonna be like in chicago tomorrow", "discourse.intent.weather_query",
                         slots=[("place", "chicago")])).vertical == "weather"
    assert router.plan(F("what's a hundred US dollars in euros at today's rate")).vertical == "fx"
    assert router.plan(F("how much is one bitcoin worth right now in US dollars")).args["src"] == "BTC"
    assert router.plan(F("how many ounces are in one cup")).vertical == "units"
    assert router.plan(F("what time is it in tokyo")).vertical == "time"
    assert router.plan(F("when is sunset in paris today")).vertical == "sun"
    p = router.plan(F("who founded the company nvidia", "discourse.intent.qa_factoid"))
    assert (p.lane, p.vertical, p.args["subject"]) == ("facts", "wikidata", "nvidia") and p.skip_web
    assert router.plan(F("what does the word ubiquitous mean")).vertical == "define"
    for q in ("what's the latest news about spacex", "how do i get a red wine stain out of a carpet",
              "who won the lakers game last night", "what are some good ramen places in seattle",
              "is it bad to take ibuprofen on an empty stomach", "what is the nvidia stock price today"):
        assert router.plan(F(q)).lane == "web", q


def test_plan_waits_for_the_missing_argument():
    assert router.plan(F("what's the weather", "discourse.intent.weather_query")).lane == "wait"
    assert router.plan(F("who founded the company")).lane == "wait"
    # "is it bad to take ibuprofen": Arc's weather intent with no weather word is not a forecast
    assert router.plan(F("is it bad to take ibuprofen", "discourse.intent.weather_query")).lane == "web"


def test_repaired_text_puts_the_correction_in_place():
    words = "When was Google founded? No, wait. I mean, Apple.".split()
    f = Frame(words=words, dropped=frozenset({2, 4, 5, 6, 7}), cue=frozenset({4, 5, 6, 7}))
    assert router.repaired_text(f) == "When was Apple. founded?"
    assert router.plan(f).args["subject"] == "apple"
    words = "Who was the first president of France, no wait of the United States".split()
    f = Frame(words=words, dropped=frozenset({6, 7, 8}), cue=frozenset({7, 8}))
    assert router.repaired_text(f) == "Who was the first president of the United States"


# ----------------------------------------------------------------------------------------------------- verticals
def test_units():
    assert units.card(units.parse("how many ounces are in one cup")).spoken.startswith("1 cup is 8 fluid ounces")
    p = units.parse("how many teaspoons are in a table spoon actually make that a quarter cup")
    assert (p["src"], p["amount"]) == ("cup", 0.25) and units.card(p).spoken.startswith("A quarter cup is 12 teaspoons")
    assert round(units.convert(26.2, "miles", "km"), 2) == 42.16
    assert round(units.convert(212, "fahrenheit", "celsius")) == 100
    assert units.parse("how many miles is a marathon") is None


def test_fx_args():
    assert router.fx_args("what's $100 in euros", conversion("what's $100 in euros".split())) == \
        {"amount": 100.0, "src": "USD", "dst": "EUR"}
    assert router.fx_args("exchange rate canadian dollars to euros")["src"] == "CAD"
    assert router.fx_args("how many yen in a dollar") == {"amount": None, "src": "USD", "dst": "JPY"}
    assert router.fx_args("who won the game") is None
    assert router.fx_args("what is apple stock price in dollars") is None


def test_weather_days():
    import datetime as dt
    fri = dt.date(2026, 11, 13)
    assert weather.offsets(weather.parse_day("weather tomorrow"), fri) == [1]
    assert weather.offsets(weather.parse_day("this weekend"), fri) == [1, 2]
    assert weather.offsets(weather.parse_day("on sunday"), fri) == [2]
    assert weather.parse_day("weather in march") == ("far",)
    assert weather.parse_day("what's the weather in denver") is None


def test_fact_patterns():
    cases = {"who founded the company nvidia": ("founder", "nvidia"),
             "Netflix, who started that company": ("founder", "netflix"),
             "Amazon the company, what year was it founded": ("inception", "amazon"),
             "what is the capital of australia": ("capital", "australia"),
             "Python, the programming language, who created it": ("discoverer", "python"),
             "who wrote the novel one hundred years of solitude": ("author", "one hundred years of solitude"),
             "Mount Kilimanjaro, which country is it in": ("country", "mount kilimanjaro"),
             "um can you tell me who invented the telephone": ("discoverer", "telephone")}
    for q, want in cases.items():
        fq = knowledge.parse_fact(q)
        assert fq and (fq.prop, fq.subject) == want, q
    assert knowledge.parse_fact("who was the first woman to win a nobel prize") is None
    assert knowledge.parse_define("what does the word ubiquitous actually mean") == "ubiquitous"


# ----------------------------------------------------------------------------------------------------- route
def _web_spy():
    called = []

    async def web():
        called.append(1)
        r = lanes.WebResult([lanes.Doc(id="w1", title="Web", text="web text", source="wikipedia")])
        r.backend = "wikipedia"
        return r
    return web, called


def test_route_card_skips_web(_router_on):
    web, called = _web_spy()
    m: dict = {}
    f = F("what's the weather gonna be like in chicago tomorrow", "discourse.intent.weather_query",
          slots=[("place", "chicago")])
    res = asyncio.run(router.route("weather chicago tomorrow", f, final=True, meter=m, web_fn=web))
    assert not called and m["tier"] == "card" and m["paid"] == 0 and m["free_calls"] == 2
    c = res[0].meta["card"]
    assert c["source"] == "Open-Meteo" and c["as_of"] and "Tomorrow" in c["spoken"] and "°F" in c["spoken"]


def test_route_fx_and_crypto():
    web, called = _web_spy()
    m: dict = {}
    res = asyncio.run(router.route("", F("what's a hundred US dollars in euros"), final=True, meter=m, web_fn=web))
    assert not called and res[0].meta["card"]["spoken"].startswith("100 US dollars is about 90 euros")
    res = asyncio.run(router.route("", F("how much is one bitcoin worth in US dollars"), final=True, meter={},
                                   web_fn=web))
    assert "70,000" in res[0].meta["card"]["spoken"] and res[0].source == "Coinbase spot price"


def test_route_fact_and_fallback():
    web, called = _web_spy()
    m: dict = {}
    res = asyncio.run(router.route("", F("who founded the company nvidia"), final=True, meter=m, web_fn=web))
    assert not called and m["tier"] == "facts" and "Jensen Huang" in res[0].text and res[0].source == "Wikidata"
    # nothing found on Wikidata -> the paid web lane runs, its docs come back
    m = {}
    res = asyncio.run(router.route("", F("who founded nowhere thing"), final=True, meter=m, web_fn=web))
    assert called and m["tier"] == "web" and m["route_fallback"] == "empty" and res[0].id == "w1"


def test_route_api_down_falls_back_to_web(monkeypatch):
    verticals.use_transport(httpx.MockTransport(lambda req: httpx.Response(503)))
    web, called = _web_spy()
    m: dict = {}
    f = F("what's the weather in chicago", "discourse.intent.weather_query", slots=[("place", "chicago")])
    res = asyncio.run(router.route("", f, final=True, meter=m, web_fn=web))
    assert called and res[0].id == "w1" and m["tier"] == "web"


def test_route_wait_is_free_only_while_speculating():
    web, called = _web_spy()
    f = F("what's the weather", "discourse.intent.weather_query")
    m: dict = {}
    with pytest.raises(lanes.Skip):
        asyncio.run(router.route("", f, final=False, meter=m, web_fn=web))
    assert not called and m["tier"] == "wait"
    asyncio.run(router.route("", f, final=True, meter={}, web_fn=web))
    assert called                                # the final question still gets an answer


def test_lanes_web_routes_only_with_a_frame_and_router_on(monkeypatch):
    async def live(*a, **k):
        r = lanes.WebResult([lanes.Doc(id="live", title="t", text="x", source="wikipedia")])
        r.backend = "wikipedia"
        return r
    monkeypatch.setattr(lanes, "_live", live)
    f = F("how many ounces are in one cup")
    assert asyncio.run(lanes.web("ounces cup", frame=f))[0].meta["tier"] == "card"
    assert asyncio.run(lanes.web("ounces cup"))[0].id == "live"
    monkeypatch.setattr(config, "ROUTER", False)
    assert asyncio.run(lanes.web("ounces cup", frame=f))[0].id == "live"


# ----------------------------------------------------------------------------------------------------- answer
def test_card_answers_without_the_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the answer model must not be called for a fitting card")
    monkeypatch.setattr(answer, "client", boom)
    doc = units.card(units.parse("how many ounces are in one cup")).doc()
    other = lanes.Doc(id="w", title="w", text="unrelated", source="wikipedia")

    async def go():
        st = answer.Stream("I'm baking, how many ounces are in one cup", [other, doc])
        await st.task
        return st
    st = asyncio.run(go())
    assert st.by == "card" and st.text() == "1 cup is 8 fluid ounces (US measures). [2]" and st.error is None


def test_card_must_fit_the_final_question():
    c = {"vertical": "weather", "args": {"place": "boston", "when": "now"}}
    assert router.card_fits(c, "what's the weather in boston")
    assert not router.card_fits(c, "what's the weather in denver")           # corrected away
    assert not router.card_fits(c, "what's the weather in boston tomorrow")  # a different day
    c = {"vertical": "fx", "args": {"amount": 100.0, "src": "USD", "dst": "EUR"}}
    assert router.card_fits(c, "what's $100 in euros")
    assert not router.card_fits(c, "what's $100 in pounds")
