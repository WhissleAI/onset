"""feat/polish: three measured regressions (PROGRESS.md "Polish").

1. Parity multi-part: a plan part's search still on the wire at the endpoint is waited for (COST_PARITY_PART_WAIT_MS)
   instead of being raced at PLAN_PART_CAP_S by a second call of the same query.
2. Wikidata one-holder properties (CEO, head of state, spouse): ambiguous or freshly changed statements fall through.
3. Open mic gate v2: terse search-style requests count as positive evidence of a request.
"""
import asyncio
import datetime as dt
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import test_speculation as ts  # noqa: E402
from server import config, cost, knowledge, lanes  # noqa: E402


# ------------------------------------------------------------------------------------ 1. parity: in-flight plan part
def _slow_parts(monkeypatch, delay):
    async def web(query, topic="general", final=False, frame=None, meter=None, **kw):
        ts.WEB_CALLS.append(query)
        await asyncio.sleep(delay if "everest" in query else 0.02)    # part 1 fires mid-sentence (~0.8 s early)
        return [lanes.Doc(id=f"web:{query}", title=query.title(), text=f"About {query}: 8,000 metres.", source="fake")]
    monkeypatch.setattr(lanes, "web", web)


def _final_web(ev):
    return [e for e in ts.specs(ev, state="fired") if e["lane"] == "final-web"]


def test_parity_waits_for_an_in_flight_part_without_a_second_call(monkeypatch):
    monkeypatch.setattr(config, "COST_MODE", "parity")
    _slow_parts(monkeypatch, 1.6)               # lands ~0.8 s after the endpoint: past PLAN_PART_CAP_S, inside 1.1 s
    m, ev = asyncio.run(ts.speak("which is taller Everest or K2", "headstart"))
    assert m["path"] == "plan" and m["cost_mode"] == "parity"
    assert not m.get("plan_hedged")                               # no duplicate search of a part
    assert not any(e["query"] == "tall everest" for e in _final_web(ev))
    assert all(p["n"] for p in m["plan"]["parts"])                # both parts' sources reached the answer
    assert m["plan_part_wait_ms"] > config.PLAN_PART_CAP_S * 1000
    assert m["paid_calls"] <= cost.profile("parity").final_max + config.PLAN_EXTRA_PAID


def test_parity_old_behaviour_switch_races_a_second_call(monkeypatch):
    monkeypatch.setattr(config, "COST_MODE", "parity")
    monkeypatch.setattr(config, "COST_PARITY_PART_WAIT_MS", -1.0)
    _slow_parts(monkeypatch, 1.6)
    m, ev = asyncio.run(ts.speak("which is taller Everest or K2", "headstart"))
    assert m.get("plan_hedged") and any(e["query"] == "tall everest" for e in _final_web(ev))


def test_parity_wait_is_bounded(monkeypatch):
    """A part slower than the parity wait still gets the race (the answer never waits past the bound for it)."""
    monkeypatch.setattr(config, "COST_MODE", "parity")
    monkeypatch.setattr(config, "COST_PARITY_PART_WAIT_MS", 600.0)
    _slow_parts(monkeypatch, 2.6)
    m, ev = asyncio.run(ts.speak("which is taller Everest or K2", "headstart"))
    assert m.get("plan_hedged") and m["plan_part_wait_ms"] < 2000          # raced, not waited out (2.6 s)


def test_speed_mode_keeps_the_part_cap(monkeypatch):
    from server import speculator
    sp = speculator.Speculator(lambda e: None, "headstart", cost_mode="speed")
    assert sp._part_cap_s() == config.PLAN_PART_CAP_S
    sp2 = speculator.Speculator(lambda e: None, "headstart", cost_mode="parity")
    assert sp2._part_cap_s() == config.COST_PARITY_PART_WAIT_MS / 1000


# ------------------------------------------------------------------------------------ 2. Wikidata one-holder facts
def _fake(monkeypatch, cands, stmts, sitelinks):
    """stmts: [(item, pid, value, rank, start or None)]."""
    knowledge._cache.clear()

    async def get_json(url, params, **kw):
        if "w/api.php" in url:
            return {"search": [{"id": i, "label": lab, "description": d} for i, lab, d in cands]}
        ent = lambda i: {"value": f"http://www.wikidata.org/entity/{i}"}  # noqa: E731
        rows = [{"item": ent(i), "sl": {"value": str(n)}} for i, n in sitelinks.items()]
        for i, pid, v, rank, start in stmts:
            r = {"item": ent(i), "pid": {"value": pid}, "rank": {"value": f"http://wikiba.se/ontology#{rank}"},
                 "vLabel": {"value": v}, "v": {"value": "x"}}
            if start:
                r["start"] = {"value": f"{start}T00:00:00Z"}
            rows.append(r)
        return {"results": {"bindings": rows}}
    monkeypatch.setattr(knowledge, "get_json", get_json)


def _ago(days):
    return (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=days)).isoformat()


NV = [("Q182477", "Nvidia", "American multinational technology company")]


def _ceo(monkeypatch, stmts):
    _fake(monkeypatch, NV, stmts, {"Q182477": 120})
    return asyncio.run(knowledge.wikidata(knowledge.parse_fact("who is the ceo of nvidia")))


def test_ceo_changed_days_ago_falls_through(monkeypatch):
    """The live Wikidata state on 2026-11-14: the CFO made preferred-rank CEO two days earlier (the old CEO's
    statement has an end time, so the SPARQL filter already drops it)."""
    assert _ceo(monkeypatch, [("Q182477", "P169", "Colette Kress", "PreferredRank", _ago(2))]) == []
    assert "not settled" in knowledge._UNSURE[-1]["why"]


def test_settled_ceo_answers(monkeypatch):
    d = _ceo(monkeypatch, [("Q182477", "P169", "Jensen Huang", "NormalRank", "1993-04-05")])
    assert d and d[0].meta["fact"]["values"] == ["Jensen Huang"]
    d = _ceo(monkeypatch, [("Q182477", "P169", "Jensen Huang", "PreferredRank", None)])      # no start date
    assert d and d[0].meta["fact"]["values"] == ["Jensen Huang"]


def test_preferred_rank_wins_over_normal(monkeypatch):
    d = _ceo(monkeypatch, [("Q182477", "P169", "Old Boss", "NormalRank", "1990-01-01"),
                           ("Q182477", "P169", "Jensen Huang", "PreferredRank", "1993-04-05")])
    assert d and d[0].meta["fact"]["values"] == ["Jensen Huang"]


def test_two_current_ceos_without_preferred_fall_through(monkeypatch):
    assert _ceo(monkeypatch, [("Q182477", "P169", "A Person", "NormalRank", "2001-01-01"),
                              ("Q182477", "P169", "B Person", "NormalRank", "2010-01-01")]) == []


def test_recent_rule_only_for_one_holder_props_and_switch(monkeypatch):
    _fake(monkeypatch, NV, [("Q182477", "P112", "Jensen Huang", "NormalRank", _ago(2))], {"Q182477": 120})
    d = asyncio.run(knowledge.wikidata(knowledge.parse_fact("who founded nvidia")))
    assert d                                                       # founder: not a one-holder property
    monkeypatch.setattr(config, "WD_RECENT_DAYS", 0.0)
    assert _ceo(monkeypatch, [("Q182477", "P169", "Colette Kress", "PreferredRank", _ago(2))])


# ------------------------------------------------------------------------------------ 3. open mic: search-style asks
from types import SimpleNamespace  # noqa: E402

from server import openmic  # noqa: E402


def _sf(raw, dd=0.95):
    return SimpleNamespace(directed=dd, words=raw.split(), intent="", intent_p=0.0)


def _shape(raw, dd=0.95):
    return openmic.search_like(_sf(raw, dd), openmic.sentences(raw.split()), raw.split())


@pytest.mark.parametrize("raw,shape", [
    ("Osprey or Gregory backpack for a week long trek.", "compare"),
    ("keto or mediterranean diet for someone with high blood pressure", "compare"),
    ("a summer job for a college student outdoors in colorado with housing included", "keyword"),
    ("uh like lah which hawker centre near bugis is good ah", "filler"),
    ("i'm trying to remember who directed inception", "embedded"),
    ("i'd love to find a good sushi place nearby", "embedded"),
    ("i'm at the hilton on two hundred river road where's the closest starbucks", "clause"),
    ("Born March ninth nineteen fifty five, am I eligible for the IRS benefit", "clause"),
])
def test_search_style_requests_have_evidence(raw, shape):
    assert _shape(raw) == shape


@pytest.mark.parametrize("raw", [
    "Seven or eight probably.",                       # the scenes' room talk each narrow rule was checked against
    "Fali or Thailand or something.",
    "I don't know if this is still good.",
    "I wonder if we need more garlic. Let me check the pantry.",
    "So who can take the lead on this? I can maybe take it.",
    "and that punch just lifts out. Oh, what a shame for you.",
    "you don't know what you're dealing with detective",
    "An Essay on the Civil War.",
    "Grilled salmon with rice and vegetables. Salmon sounds good.",
    "Mom, a sandwich for lunch near the park.",
])
def test_room_talk_has_no_search_evidence(raw):
    assert _shape(raw) is None


def test_search_evidence_needs_device_directed_and_switch(monkeypatch):
    raw = "Osprey or Gregory backpack for a week long trek."
    assert _shape(raw, dd=0.5) is None
    f = SimpleNamespace(directed=0.99, words=raw.split(), intent="", intent_p=0.0, acts=(), act_question=False,
                        slots=(), ent_types=(), has_content=lambda: True)
    monkeypatch.setattr(config, "OPENMIC_SEARCH", False)
    assert openmic.skip(f, {"raw": raw.split()}) == "no_request"
    monkeypatch.setattr(config, "OPENMIC_SEARCH", True)
    assert openmic.skip(f, {"raw": raw.split()}) is None
