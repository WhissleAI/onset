"""Spoken self-corrections the ear split into two turns, and conversion answers (showcase sc27 / sc08).

sc27: Deepgram Flux ended the turn at "what's the weather in Boston, sorry," and heard "I mean Denver" as a new turn,
so both modes answered for Boston. The session now merges a repair turn into the turn before it (frame.py's correction
logic then asks about Denver) and supersedes any answer already given, in both modes alike.
sc08: "what's a hundred US dollars in euros" was answered with the rate (1 EUR = 1.1 USD), not the amount.

Run: .venv/bin/python -m pytest -q tests/test_repair.py
"""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import answer, config, frame  # noqa: E402
from server.session import Session  # noqa: E402
from test_speculation import FakeStream, fakes  # noqa: E402,F401  (fakes: autouse fake lanes + answer model)


# ----------------------------------------------------------------------------- frame: markers + conversions
def test_repair_start_and_end():
    assert frame.repair_start(["i", "mean", "denver"]) is True
    assert frame.repair_start(["sorry", "denver"]) is True
    assert frame.repair_start(["um", "actually", "make", "it", "denver"]) is True
    assert frame.repair_start(["scratch", "that"]) is True
    assert frame.repair_start(["i"]) is None            # might become "i mean": wait for the next partial
    assert frame.repair_start(["no"]) is None
    assert frame.repair_start(["when", "was", "it", "founded"]) is False
    assert frame.repair_start(["denver"]) is False
    assert frame.repair_end("what's the weather in boston sorry".split())
    assert frame.repair_end("capital of canada actually".split())
    assert frame.repair_end("boston i mean".split())
    assert not frame.repair_end("which would you rather".split())
    assert not frame.repair_end("what's the weather in boston".split())


@pytest.mark.parametrize("text,want", [
    ("what's a hundred us dollars in euros at today's rate", ("a hundred us dollars", "dollars", "euros")),
    ("how much is 100 dollars in euros", ("100 dollars", "dollars", "euros")),
    ("What's $100 in Euros at today's rate?", ("$100", "dollars", "euros")),
    ("convert 50 euros to dollars", ("50 euros", "euros", "dollars")),
    ("how many kilometers is 26 miles", ("26 miles", "miles", "kilometers")),
    ("what's 72 fahrenheit in celsius", ("72 fahrenheit", "fahrenheit", "celsius")),
    ("what is 5 feet 10 inches in centimeters", ("5 feet 10 inches", "feet", "centimeters")),
    ("how much is 1.5 million yen in dollars", ("1.5 million yen", "yen", "dollars")),
    ("what's the exchange rate for yen", ()),            # a rate question, no amount: no hint
    ("how much is a ticket to paris", ()),
    ("how many calories in 100 grams of rice", ()),
    ("what's 2 plus 2", ()),
])
def test_conversion_detection(text, want):
    assert frame.conversion(text.split()) == want


def test_conversion_skips_corrected_away_unit():
    words = "what's 100 dollars in pounds sorry euros".split()
    assert frame.conversion(words, drop={4, 5}) == ("100 dollars", "dollars", "euros")


def test_answer_hint_and_prompt():
    f = frame.build("what's a hundred us dollars in euros at today's rate".split(), [], None)
    h = f.answer_hint()
    assert "a hundred us dollars into euros" in h and "converted amount in euros" in h and "not the reverse" in h
    assert f.readout()["convert"] == ["a hundred us dollars", "dollars", "euros"]
    p = answer.build_prompt(f.text, [], hint=h)
    assert p.endswith("Note: " + h)
    assert answer.build_prompt("who founded nvidia", []).endswith("Spoken question: who founded nvidia")
    assert frame.build("who founded nvidia".split(), [], None).answer_hint() == ""
    # the rule holds for every conversion, hint or not: amount first, right direction, rate shown
    assert "converted amount first" in answer.SYSTEM and "inverse rate" in answer.SYSTEM


def test_hint_reaches_the_answer_model(monkeypatch):
    async def go():
        from test_speculation import speak
        await speak("how much is 100 dollars in euros", "baseline")
    asyncio.run(go())
    finals = [s for s in FakeStream.instances if s.tag == "final"]
    assert finals and "into euros" in finals[-1].hint


# ----------------------------------------------------------------------------- session: repairs across ear turns
def _times(words, start, gap=0.25):
    return [(start + i * gap, start + i * gap + gap * 0.9) for i in range(len(words))]


async def _turn(s, t0, text, start):
    """Feed one ear turn: partials as each word ends (+0.1 s), then return its final word list (not yet finalized)."""
    words = text.split()
    times = _times(words, start)
    for k in range(1, len(words) + 1):
        at = t0 + times[k - 1][1] + 0.1
        while time.perf_counter() < at:
            await asyncio.sleep(0.005)
        s.on_partial([dict(text=w, start=a, end=b) for w, (a, b) in zip(words[:k], times[:k])])
    return [dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)]


async def _two_turns(mode, first, second, gap_s, wait_first=False, events=None):
    """Two ear turns: `first` closes (ear final), then `second` starts gap_s (audio) after first's last word."""
    events = events if events is not None else []
    t0 = time.perf_counter()
    s = Session(events.append, mode=mode, t0=t0)
    ws1 = await _turn(s, t0, first, 0.2)
    t1 = asyncio.create_task(s.on_final(ws1))
    if wait_first:
        await t1
    ws2 = await _turn(s, t0, second, ws1[-1]["end"] + gap_s)
    t2 = asyncio.create_task(s.on_final(ws2))
    await asyncio.gather(t1, t2)
    return s, events


@pytest.mark.parametrize("mode", ["baseline", "headstart"])
def test_trailing_marker_turn_is_held_and_merged(mode):
    """sc27 as Flux splits it: "what's the weather in boston sorry" | "i mean denver". One answer, for Denver."""
    async def go():
        s, ev = await _two_turns(mode, "what's the weather in boston sorry", "i mean denver", 0.3)
        s.close()
        return s, ev
    s, ev = asyncio.run(go())
    assert any(e["t"] == "repair_hold" for e in ev) and any(e["t"] == "repair" for e in ev)
    assert len(s.history) == 1, [m.get("question") for m in s.history]
    m = s.history[0]
    assert "denver" in m["query"] and "boston" not in m["query"]
    assert m["question"].endswith("i mean denver")
    eps = [e for e in ev if e["t"] == "endpoint"]
    assert eps and "boston" not in eps[-1]["frame"]["query"]
    assert s.carry["prev"]["entities"] == ["denver"]           # a follow-up ("and tomorrow?") is about Denver


@pytest.mark.parametrize("mode", ["baseline", "headstart"])
def test_leading_marker_supersedes_finished_answer(mode):
    """The ear closed "what's the weather in boston" (answered) and then hears "sorry i mean denver": the Boston
    answer is withdrawn (history, conversation memory) and the corrected question is answered once."""
    async def go():
        ev = []
        s, ev = await _two_turns(mode, "what's the weather in boston", "sorry i mean denver", 0.4, wait_first=True,
                                 events=ev)
        s.close()
        return s, ev
    s, ev = asyncio.run(go())
    assert [m["query"] for m in s.history] == ["weather denver"]
    assert len(s.superseded) == 1 and "boston" in s.superseded[0]["query"] and s.superseded[0]["superseded"]
    ext = [e for e in ev if e["t"] == "extended" and e.get("correction")]
    assert ext and ext[0]["words"][-1] == "boston"
    # memory = the corrected turn only: a follow-up refers to Denver, not Boston
    assert s.carry["prev"]["entities"] == ["denver"]
    assert "Answer to what's the weather in denver" in s.context    # the corrected answer, not Boston's


def test_leading_marker_cancels_answer_in_flight(monkeypatch):
    monkeypatch.setattr(FakeStream, "TTFT", 1.5)          # the Boston answer is still streaming when "sorry" comes

    async def go():
        s, ev = await _two_turns("baseline", "what's the weather in boston", "sorry i mean denver", 0.3)
        s.close()
        return s, ev
    s, ev = asyncio.run(go())
    assert [m["query"] for m in s.history] == ["weather denver"]
    assert len(s.superseded) == 1 and s.superseded[0]["superseded"]
    boston = [x for x in FakeStream.instances if "boston" in x.question and "denver" not in x.question]
    assert all(x.cancelled or not x.tokens for x in boston)
    assert not any("boston" in (e.get("delta") or "").lower() and "denver" not in (e.get("delta") or "").lower()
                   for e in ev if e["t"] == "answer")


@pytest.mark.parametrize("second,gap", [
    ("when was it founded", 0.3),           # a follow-up right after, no marker: its own question
    ("sorry i mean amd", 2.5),             # a marker, but long after: a new question
])
def test_not_merged(second, gap):
    async def go():
        s, ev = await _two_turns("baseline", "who founded nvidia", second, gap, wait_first=True)
        s.close()
        return s, ev
    s, ev = asyncio.run(go())
    assert len(s.history) == 2 and not s.superseded
    assert not any(e["t"] == "extended" for e in ev)


def test_held_final_answers_after_hold_when_nothing_follows(monkeypatch):
    monkeypatch.setattr(config, "REPAIR_HOLD_S", 0.3)

    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="baseline", t0=t0)
        ws = await _turn(s, t0, "what does ubiquitous mean actually", 0.2)
        t_final = time.perf_counter()
        m = await s.on_final(ws)
        held = time.perf_counter() - t_final
        s.close()
        return s, ev, m, held
    s, ev, m, held = asyncio.run(go())
    assert held >= 0.3 and len(s.history) == 1 and m["query"].startswith("ubiquitous")
    assert any(e["t"] == "repair_hold" for e in ev)


def test_in_utterance_correction_unchanged():
    """q27 (one ear turn): "capital of canada actually make that turkey" still corrects inside the utterance."""
    async def go():
        from test_speculation import speak
        m, _ = await speak("what's the capital of canada actually make that turkey", "baseline")
        return m
    m = asyncio.run(go())
    assert "turkey" in m["query"] and "canada" not in m["query"]
