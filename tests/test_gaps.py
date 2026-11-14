"""Arc-value gaps (docs/ARC_VALUE.md): live-data phrasings the router missed, the Wikidata item sanity check, spoken
birth years, and the open-mic gate v2. Frames come from the real native tagger (word by word)."""
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from server import config, frame as framemod, knowledge, openmic, router  # noqa: E402
import replay_frames  # noqa: E402


def fr(text):
    return replay_frames.frame_for(text)


# ----------------------------------------------------------------------------------------- 1. live-data phrasings
@pytest.mark.parametrize("text,vertical,place", [
    ("how hot is it in phoenix right now", "weather", "phoenix"),
    ("how cold is it in chicago today", "weather", "chicago"),
    ("what time was it in tokyo right now", "time", "tokyo"),       # Flux heard "was" for "is" (ld10)
    ("what time was it in tokyo", "time", "tokyo"),
])
def test_loose_live_data_phrasings_get_a_card(text, vertical, place):
    p = router.plan(fr(text))
    assert (p.lane, p.vertical, p.args.get("place")) == ("card", vertical, place)


@pytest.mark.parametrize("text", [
    "how hot does it get in phoenix in july",          # climate, not now
    "what time was it in tokyo when the bomb fell",    # history: the place doesn't end the question
    "how hot is the sun",
    "how cold is it in space",
])
def test_loose_phrasings_do_not_false_fire(text):
    assert router.plan(fr(text)).lane != "card"


def test_loose_phrasings_off_switch_and_card_fit(monkeypatch):
    card = {"vertical": "time", "args": {"place": "tokyo", "what": "time"}}
    assert router.card_fits(card, "what time was it in tokyo")
    assert not router.card_fits(card, "what time was it in tokyo when the bomb fell")
    monkeypatch.setattr(config, "ROUTER_LOOSE", False)
    assert router.plan(fr("how hot is it in phoenix right now")).lane != "card"
    assert not router.card_fits(card, "what time was it in tokyo")


# ----------------------------------------------------------------------------------------- 2. Wikidata sanity check
def _fake_wikidata(monkeypatch, cands, values, sitelinks):
    """cands: [(id, label, desc)]; values: {(id, pid): value}; sitelinks: {id: n}."""
    knowledge._cache.clear()

    async def get_json(url, params, **kw):
        if "w/api.php" in url:
            return {"search": [{"id": i, "label": lab, "description": d} for i, lab, d in cands]}
        ent = lambda i: {"value": f"http://www.wikidata.org/entity/{i}"}  # noqa: E731
        rows = [{"item": ent(i), "sl": {"value": str(n)}} for i, n in sitelinks.items()]
        rows += [{"item": ent(i), "pid": {"value": pid}, "rank": {"value": "http://wikiba.se/ontology#NormalRank"},
                  "vLabel": {"value": v}, "v": {"value": "x"}} for (i, pid), v in values.items()]
        return {"results": {"bindings": rows}}
    monkeypatch.setattr(knowledge, "get_json", get_json)


HAMLET = [("Q5084", "hamlet", "small settlement in a rural area"), ("Q27178", "Hamlet", "1948 film directed by Laurence Olivier"),
          ("Q43374750", "Hamlet", "male given name"), ("Q280104", "Hamlet", "2010 video game"),
          ("Q16276236", "Hamlet", "family name"), ("Q41567", "Hamlet", "tragedy by William Shakespeare")]


def test_wikidata_picks_the_play_not_the_films_composer(monkeypatch):
    # mp18: the play was the 6th search candidate (only 5 were looked at) and the film's P86 (composer) counted as
    # "written by"
    _fake_wikidata(monkeypatch, HAMLET, {("Q27178", "P86"): "William Walton", ("Q41567", "P50"): "William Shakespeare"},
                   {"Q5084": 74, "Q27178": 45, "Q41567": 143, "Q280104": 3})
    d = asyncio.run(knowledge.wikidata(knowledge.parse_fact("who wrote hamlet")))
    assert d and "William Shakespeare" in d[0].text and "Walton" not in d[0].text
    assert d[0].meta["fact"]["sitelinks"] == 143
    knowledge._cache.clear()


def test_wikidata_composer_only_counts_for_music(monkeypatch):
    _fake_wikidata(monkeypatch, HAMLET[:2], {("Q27178", "P86"): "William Walton"}, {"Q5084": 74, "Q27178": 45})
    assert asyncio.run(knowledge.wikidata(knowledge.parse_fact("who wrote hamlet"))) == []      # fall through
    _fake_wikidata(monkeypatch, [("Q1", "Macbeth", "opera by Giuseppe Verdi")], {("Q1", "P86"): "Giuseppe Verdi"},
                   {"Q1": 30})
    d = asyncio.run(knowledge.wikidata(knowledge.parse_fact("who wrote macbeth")))
    assert d and "Verdi" in d[0].text                  # an opera's composer did write it
    knowledge._cache.clear()


def test_wikidata_unsure_falls_through(monkeypatch):
    # two same-name items about as popular that disagree: no answer (the router falls through to the next lane)
    _fake_wikidata(monkeypatch, [("Q1", "Georgia", "state of the United States"), ("Q2", "Georgia", "country in the Caucasus")],
                   {("Q1", "P36"): "Atlanta", ("Q2", "P36"): "Tbilisi"}, {"Q1": 233, "Q2": 351})
    assert asyncio.run(knowledge.wikidata(knowledge.parse_fact("what is the capital of georgia"))) == []
    # an answer from a loosely matching item far down the search ("when was the founder born" -> Maria Montessori)
    _fake_wikidata(monkeypatch, [("Q1", "founder", "person who creates an institution"), ("Q2", "shipwrecking", "event"),
                                 ("Q3", "Maria Montessori", "Italian physician")],
                   {("Q3", "P569"): "August 31, 1870"}, {"Q1": 16, "Q2": 48, "Q3": 109})
    assert asyncio.run(knowledge.wikidata(knowledge.parse_fact("when was the founder born"))) == []
    # a clear case still answers
    _fake_wikidata(monkeypatch, [("Q1", "Nvidia", "company")], {("Q1", "P112"): "Jensen Huang"}, {"Q1": 90})
    assert "Jensen Huang" in asyncio.run(knowledge.wikidata(knowledge.parse_fact("who founded nvidia")))[0].text
    knowledge._cache.clear()


# ----------------------------------------------------------------------------------------- 3. spoken birth years
def test_spoken_years():
    t = "born on july twenty eighth nineteen seventy nine in two thousand and eight twenty eleven nineteen oh five".split()
    ys = framemod.spoken_years(t)
    assert [t[i] for i in sorted(ys)] == ["nineteen", "seventy", "nine", "two", "thousand", "and", "eight", "twenty",
                                         "eleven", "nineteen", "oh", "five"]
    assert framemod.spoken_years("twenty eight two thousand four".split()) == {2, 3, 4}   # the day stays


@pytest.mark.parametrize("text,hidden,kept", [
    ("but i was born on july twenty eighth nineteen seventy nine what's my zodiac sign",
     {"nineteen", "seventy"}, {"july", "twenty", "eighth", "zodiac"}),
    ("um born on february twenty second two thousand five how old am i today", {"thousand"}, {"february", "second"}),
    ("i was born in nineteen eighty five in boston what happened that year", {"nineteen", "eighty"}, {"boston"}),
])
def test_spoken_birth_year_stays_out_of_the_query(text, hidden, kept):
    f = fr(text)
    q = set(f.query().split())
    assert not (hidden & q) and kept <= q and "birthday" in f.redacted and "[private]" in f.clean_text


def test_history_year_is_not_hidden_and_switch(monkeypatch):
    assert {"nineteen", "eighty", "five"} <= set(fr("who won the world series in nineteen eighty five").query().split())
    monkeypatch.setattr(config, "PII_YEARS", False)
    assert "seventy" in fr("but i was born on july twenty eighth nineteen seventy nine what's my zodiac sign").query()


# ----------------------------------------------------------------------------------------- 4. open-mic gate v2
def test_followup_fragment_after_a_request_is_answered():
    f = fr("and chips")
    assert openmic.skip(f, {"raw": ["And", "chips."], "since_request_s": 4.0}) is None
    assert openmic.skip(f, {"raw": ["And", "chips."]}) is not None                       # no request before it
    assert openmic.skip(f, {"raw": ["And", "chips."], "since_request_s": 90.0}) is not None
    # Arc heard a different speaker: not the asker's follow-up
    assert openmic.skip(f, {"since_request_s": 4.0, "speaker": "B", "request_speaker": "A"}) is not None
    g = fr("how far is it")
    assert openmic.skip(g, {"since_request_s": 6.0}) is None


@pytest.mark.parametrize("raw", [
    "Mom, can I have more cereal?",
    "Did you pay the electric bill?",
    "Yeah. Tomorrow works for me.",
    "You always burn the first one.",
])
def test_room_talk_is_held(raw):
    f = fr(raw.replace(",", "").replace("?", "").replace(".", ""))
    assert openmic.skip(f, {"raw": raw.split()}) is not None


@pytest.mark.parametrize("raw", [
    "What's the weather in Denver tomorrow?",
    "Set a timer for four minutes.",
    "Okay. Sounds good. Talk later. What's the news today?",
    "sixty kilos in pounds",
])
def test_requests_still_answered(raw):
    f = fr(raw.replace(",", "").replace("?", "").replace(".", ""))
    assert openmic.skip(f, {"raw": raw.split()}) is None


def test_gate_version_switch(monkeypatch):
    from server.session import open_mic_skip
    f = fr("you always burn the first one")
    monkeypatch.setattr(config, "OPENMIC_GATE", 1)
    v1 = open_mic_skip(f)
    monkeypatch.setattr(config, "OPENMIC_GATE", 2)
    assert v1 is None and open_mic_skip(f, {"raw": "You always burn the first one.".split()}) is not None
