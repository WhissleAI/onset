"""Router (search/router) x speculation efficiency (search/spec-efficiency): the integration contract.

- lanes.plan(frame) gates the PAID web lane: a card / facts-only plan never spends a paid speculative or final call
  (cheap lookups still fire), a web plan does.
- the router's Skip (a "wait" plan) costs nothing: no SPEC_PAID_MAX budget, web_used refunded.
- the search cache stores paid web docs only: never a card / fact (it would come back as a plain doc).
- the "cached" badge never shows on a card (cards carry their own "as of" inside meta["card"]).
- the route-conf hold: no paid speculative call while the router is unsure and the turn is still going.

Run: .venv/bin/python -m pytest -q tests/test_integrate.py
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_speculation as ts  # noqa: E402  (fake lanes, fake answer model, speak())
from server import config, frame, lanes, searchcache, speculator  # noqa: E402

fakes = ts.fakes          # the same autouse fixture (fake web / embeddings / answer stream)

CALLS: list[dict] = []    # every lanes.web call: query, final, the plan, paid


def _plan_for(text: str, final: bool = False) -> lanes.RoutePlan:
    low = text.lower()
    if "weather" in low:
        p = lanes.RoutePlan(["card"])
        p.lane, p.vertical, p.conf, p.why = "card", "weather", 0.92, "weather forecast"
        p.sig = ("card", "weather", low.split(" in ")[-1] if " in " in low else "")
    elif "capital" in low:
        p = lanes.RoutePlan(["facts"])
        p.lane, p.vertical, p.conf, p.why = "facts", "wikidata", 0.85, "Wikidata capital"
        p.sig = ("facts", "wikidata", "capital")
    elif "wait" in low:
        p = lanes.RoutePlan(["facts", "web"])        # a fact question whose subject may not be said yet
        p.lane, p.conf, p.why = "facts", 0.5, "stable factoid"
    elif "unsure" in low:
        p = lanes.RoutePlan(["web"])
        p.lane, p.conf, p.why = "web", 0.2, "unsure"
    else:
        p = lanes.RoutePlan(["web"])
        p.lane, p.conf, p.why = "web", 0.3, "open-ended"
    return p


def _card_doc(query):
    return lanes.Doc(id=f"card:weather:{query}", title="Weather", text=f"Rain in {query}. (Source: Open-Meteo)",
                     url="", source="Open-Meteo", score=1.0,
                     meta={"tier": "card", "card": {"vertical": "weather", "title": "Weather", "spoken": "Rain.",
                                                    "source": "Open-Meteo", "as_of": "2026-11-13T10:00",
                                                    "args": {"place": "chicago"}, "data": {}, "live": True}})


async def fake_routed_web(query, topic="general", *, final=False, meter=None, frame=None, **kw):
    """lanes.web with the router in front of it, played by the fake plan: card plans answer for free, "wait" plans
    raise Skip while speculating, everything else is a paid web call."""
    meter = meter if meter is not None else {}
    text = frame.text if frame is not None else query
    p = _plan_for(text, final)
    meter.update(route=p.lane, route_conf=p.conf)
    if "wait" in text.lower() and "nvidia" not in text.lower() and not final:      # the router's "wait" plan
        meter.update(tier="wait", backend="router", paid=0)
        CALLS.append({"q": query, "final": final, "lane": "wait", "paid": 0})
        raise lanes.Skip("key word not said yet")
    await asyncio.sleep(0.05)
    if p.lane == "facts" and p.vertical == "wikidata":
        meter.update(tier="facts", backend="facts", paid=0, credits=0.0, free_calls=1)
        CALLS.append({"q": query, "final": final, "lane": "facts", "paid": 0})
        return [lanes.Doc(id="wd:capital", title="Australia · Wikidata", text="Australia: capital: Canberra.",
                          source="Wikidata", meta={"tier": "facts", "snapshot": "Wikidata"})]
    if p.lane == "card":
        meter.update(tier="card", backend="card", paid=0, credits=0.0, free_calls=2)
        CALLS.append({"q": query, "final": final, "lane": "card", "paid": 0})
        return [_card_doc(query)]
    meter.update(tier="web", backend="fake", paid=1)
    CALLS.append({"q": query, "final": final, "lane": "web", "paid": 1})
    return [lanes.Doc(id=f"web:{query}:{i}", title=query.title(), text=f"About {query}: fact {i}.", source="fake")
            for i in range(3)]


@pytest.fixture(autouse=True)
def _routed(fakes, monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(config, "ROUTER", True)
    monkeypatch.setattr(config, "SPEC_EFF", True)
    monkeypatch.setattr(config, "CACHE_SEM", True)
    monkeypatch.setattr(config, "CACHE_PERSIST", False)
    monkeypatch.setattr(lanes, "web", fake_routed_web)
    monkeypatch.setattr(lanes, "plan", lambda f, final=False: _plan_for(f.text, final))
    searchcache.reset()
    yield
    searchcache.reset()


def _speak(text, mode="headstart", gap=0.25, events=None):
    return asyncio.run(ts.speak(text, mode, gap=gap, events=events))[0]


# ----------------------------------------------------------------------------------------------- the plan gates web
def test_card_plan_spends_no_paid_call_but_still_looks_up():
    ev = []
    m = _speak("what's the weather gonna be like in Chicago tomorrow", events=ev)
    assert m["paid_calls"] == 0 and m["web_paid"] == 0
    assert CALLS and all(c["lane"] == "card" for c in CALLS)
    assert any(not c["final"] for c in CALLS), "the card should be looked up speculatively (free)"
    assert m["route"] == "card"


def test_card_plan_baseline_also_free():
    m = _speak("what's the weather gonna be like in Chicago tomorrow", mode="baseline")
    assert m["paid_calls"] == 0 and len(CALLS) == 1 and CALLS[0]["final"]


def test_web_plan_spends_paid_calls_within_budget():
    m = _speak("who founded the company Nvidia and when did it go public")
    assert 1 <= m["paid_calls"] <= config.SPEC_PAID_MAX
    assert m["paid_calls"] == sum(c["paid"] for c in CALLS)


def test_real_router_plan_lanes(monkeypatch):
    """lanes.plan on the real router: cheap-only for a confident card, web listed for open questions / fallbacks."""
    monkeypatch.undo()                       # the real lanes.plan / router (ROUTER on for this test)
    monkeypatch.setattr(config, "ROUTER", True)
    from server import router
    f = frame.Frame(words="what's the weather in Chicago tomorrow".split(), intent="discourse.intent.weather_query",
                    intent_p=0.9, slots=(("place", "chicago"),), content=("weather", "chicago", "tomorrow"))
    p = lanes.plan(f)
    assert list(p) == ["card"] and p.cheap_only and p.conf >= config.ROUTER_SKIP_CONF
    g = frame.Frame(words="what are some good ramen places in Seattle".split(), intent="discourse.intent.qa_factoid",
                    intent_p=0.9, content=("good", "ramen", "places", "seattle"))
    assert "web" in lanes.plan(g)
    h = frame.Frame(words="what's the weather in".split(), intent="discourse.intent.weather_query", intent_p=0.9,
                    content=("weather",))
    assert list(lanes.plan(h)) == ["wait"] and list(lanes.plan(h, final=True)) == ["web"]
    assert router.plan(f).lane == "card"
    monkeypatch.setattr(config, "ROUTER", False)
    assert lanes.plan(f) is None


# ----------------------------------------------------------------------------------------------- Skip + budget
def test_skip_refunds_budget_under_spec_paid_max(monkeypatch):
    monkeypatch.setattr(config, "SPEC_PAID_MAX", 1)
    ev = []
    m = _speak("wait who founded the famous graphics chip company called Nvidia", events=ev, gap=0.6)
    skipped = [e for e in ev if e.get("t") == "spec" and e.get("state") == "skipped"]
    assert skipped, "the 'wait' plan should have been skipped by the router"
    assert len(skipped) >= 2
    # the skips cost nothing: with SPEC_PAID_MAX=1 the one paid call still went to the real question (no budget
    # skip, no over-budget override), and web_paid / paid_calls agree
    paid = [c for c in CALLS if c["paid"]]
    assert m["paid_calls"] == 1 == m["web_paid"] and len(paid) == 1 and "nvidia" in paid[0]["q"]
    assert not m["web_skipped"].get("budget") and "budget_override" not in m


def test_skip_is_free_unit():
    async def go():
        sp = speculator.Speculator(lambda e: None)

        async def skip():
            raise lanes.Skip("weather: no place yet")
        sp.web_used = 1
        fire = sp._fire("web", "weather", None, skip, {"tier": "wait", "paid": 0})
        await fire.task
        assert fire.state == "skipped" and sp.web_used == 0 and sp._paid() == 0

        meter = {}

        async def card():
            await asyncio.sleep(0.05)
            meter.update(tier="card", paid=0)          # the router fills the meter when the card lands
            return [_card_doc("x")]
        sp.web_used = 1
        f2 = sp._fire("web", "weather x", None, card, meter)
        assert sp._paid() == 1                      # in flight: counted (its cheap lane may fall back to web)
        await f2.task
        assert sp._paid() == 0 and sp.web_used == 0
    asyncio.run(go())


# ----------------------------------------------------------------------------------------------- search cache
def test_card_results_are_never_cached_as_plain_docs():
    _speak("what's the weather gonna be like in Chicago tomorrow")
    assert len(searchcache.cache()) == 0
    CALLS.clear()
    ev = []
    m = _speak("what's the weather gonna be like in Chicago tomorrow", events=ev)
    assert m["search_cached"] == 0 and CALLS                  # looked up again (live), not served from the cache
    src = [e for e in ev if e.get("t") == "sources"][-1]["docs"]
    assert src and src[0].get("card") and not any(d.get("cached") for d in src)


def test_mixed_result_caches_only_the_web_docs():
    async def go():
        sp = speculator.Speculator(lambda e: None)
        f = frame.Frame(words="who founded nvidia".split(), intent="discourse.intent.qa_factoid", intent_p=0.9,
                        content=("founded", "nvidia"), slots=(("organization", "nvidia"),), phrases=("nvidia",))
        fact = lanes.Doc(id="wd:Q1", title="Nvidia", text="founded by Jensen Huang", source="Wikidata",
                         meta={"tier": "facts", "snapshot": "Wikidata 2026-11"})
        web = lanes.Doc(id="web:1", title="Nvidia", text="Nvidia was founded in 1993", source="fake")

        async def mixed():
            return [fact, web]
        fire = sp._fire("web", f.query(), f, mixed, {"tier": "web", "paid": 1, "backend": "fake"})
        await fire.task
        (e,) = searchcache.cache().entries.values()
        assert [d["id"] for d in e.docs] == ["web:1"]
    asyncio.run(go())


def test_web_results_still_cached_and_labelled():
    _speak("who founded the company Nvidia")
    assert len(searchcache.cache()) >= 1
    CALLS.clear()
    m = _speak("who founded the company Nvidia")
    assert m["search_cached"] >= 1 and m["paid_calls"] < 2


# ----------------------------------------------------------------------------------------------- cached badge
def test_cached_badge_never_on_cards():
    card = _card_doc("chicago")
    card.meta["as_of"] = "2026-11-13T10:00:00"            # even if something stamped an as_of on it
    b = speculator.Speculator._brief(card)
    assert "cached" not in b and b.get("card") and b.get("tier") == "card"
    web = lanes.Doc(id="w", title="t", text="x", meta={"cached": True, "as_of": "2026-11-13T10:00:00"})
    assert speculator.Speculator._brief(web)["cached"] is True


def test_cached_badge_js_skips_cards():
    js = (Path(__file__).resolve().parent.parent / "web/app.js").read_text()
    assert "d.cached && !d.card" in js and "${cachedDoc ? (asOf" in js


# ----------------------------------------------------------------------------------------------- route-conf hold
def test_unsure_plan_holds_paid_speculation_until_end_of_turn():
    ev = []
    m = _speak("unsure question about the history of jazz music in new orleans", events=ev)
    spec_paid = [c for c in CALLS if not c["final"] and c["paid"]]
    holds = [e for e in ev if e.get("t") == "spec_skip" and e.get("why") == "route-hold"]
    assert holds
    # held while the turn was still going; at most one paid speculative call, after the end-of-turn signal
    assert len(spec_paid) <= 1
    if spec_paid:
        t_hold = min(e["ms"] for e in holds)
        fired = [e for e in ev if e.get("t") == "spec" and e.get("state") == "fired" and e.get("lane") == "web"]
        assert fired and min(e["ms"] for e in fired) > t_hold
    assert m["paid_calls"] >= 1 and m.get("answer")


def test_correction_is_not_held(monkeypatch):
    sp = speculator.Speculator(lambda e: None)
    p = _plan_for("unsure")
    f = frame.Frame(words="capital of canada actually turkey".split(), cue=frozenset({3}), dropped=frozenset({2, 3}))
    assert not sp._route_hold(f, p)
    g = frame.Frame(words="capital of canada".split())
    assert sp._route_hold(g, p)
    sp.eot_p = 0.9
    assert not sp._route_hold(g, p)


def test_generic_factoid_plan_is_held_but_specific_facts_are_not():
    sp = speculator.Speculator(lambda e: None)
    g = frame.Frame(words="what's a hundred".split())
    leads = lanes.RoutePlan(["facts", "web"])
    leads.lane, leads.vertical, leads.conf = "facts", "leads", 0.5
    assert sp._route_hold(g, leads)
    wd = lanes.RoutePlan(["facts", "web"])
    wd.lane, wd.vertical, wd.conf = "facts", "wikidata", 0.6     # a changing Wikidata property: web may be needed
    assert not sp._route_hold(g, wd)
    sp.eot_p = 0.5
    assert not sp._route_hold(g, leads)


def test_facts_plan_one_lookup_and_a_draft_from_it():
    """A cheap-only plan is looked up once per plan (same args); later partials borrow that lookup for their draft
    instead of firing again, and the endpoint reuses it (no second lookup, no paid call)."""
    ev = []
    m = _speak("so tell me what is the capital city of the country of Australia", events=ev, gap=0.3)
    assert m["paid_calls"] == 0
    assert len([c for c in CALLS if c["lane"] == "facts"]) == 1
    assert m["drafts"] >= 1
    assert m["path"] in ("draft-reuse", "route-hit", "speculative-hit")
