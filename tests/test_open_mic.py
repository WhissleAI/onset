"""Open mic (click the mic, it keeps listening): a plain statement is heard, not searched or answered; push-to-talk and
replays answer it as before. Every event carries the id of the question it belongs to ("q")."""
import asyncio
import time

import pytest

from server import config, speculator
from server.frame import Frame
from server.session import Session, open_mic_skip


def frame(words, acts=(), intent="", p=0.0, act_question=False):
    f = Frame(words=list(words))
    f.acts, f.intent, f.intent_p, f.act_question = tuple(acts), intent, p, act_question
    return f


@pytest.mark.parametrize("words,acts,intent,p,why", [
    ("i like to eat pancakes", ("statement",), "intent.report_inform", 0.62, "statement"),
    ("we went there last week", ("statement",), "", 0.0, "statement"),
    ("i had pasta yesterday", (), "intent.report_inform", 0.7, "statement"),
    ("what is the weather in boston", ("question",), "discourse.intent.weather_query", 0.93, None),
    ("tell me about nvidia", ("command_request",), "intent.request_information", 0.9, None),
    ("pancake recipes", (), "discourse.intent.cooking_recipe", 0.8, None),   # keyword query, web intent
    ("how tall is the eiffel tower", ("statement",), "", 0.0, None),        # question word up front wins
    ("nvidia stock", (), "", 0.0, None),                                       # nothing says statement: answer
])
def test_open_mic_gate(words, acts, intent, p, why, monkeypatch):
    monkeypatch.setattr(config, "OPENMIC_GATE", 1)          # the v1 statement rule (v2: tests/test_gaps.py)
    assert open_mic_skip(frame(words.split(), acts, intent, p, act_question="question" in acts)) == why


async def _speak(s, text):
    words = text.split()
    times = [(0.2 + i * 0.25, 0.2 + i * 0.25 + 0.22) for i in range(len(words))]
    for k in range(1, len(words) + 1):
        s.on_partial([dict(text=w, start=a, end=b) for w, (a, b) in zip(words[:k], times[:k])])
        await asyncio.sleep(0)
    return await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])


def _run(open_mic, text, monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", False)
    finalized = []

    async def finalize(self, f, t_end, **kw):
        finalized.append(f.text)
        return {"mode": self.mode, "question": f.text, "path": "stub"}
    monkeypatch.setattr(speculator.Speculator, "finalize", finalize)
    monkeypatch.setattr(speculator.Speculator, "on_frame", lambda self, f, now=None: None)
    events = []

    async def go():
        s = Session(events.append, mode="headstart", t0=time.perf_counter(), open_mic=open_mic)
        m = await _speak(s, text)
        s.close()
        return m
    return asyncio.run(go()), events, finalized


def test_open_mic_statement_is_heard_not_answered(monkeypatch):
    m, events, finalized = _run(True, "i like to eat pancakes", monkeypatch)
    kinds = [e["t"] for e in events]
    assert "heard" in kinds and "endpoint" not in kinds and not finalized
    heard = next(e for e in events if e["t"] == "heard")
    assert heard["why"] == "statement" and heard["words"] == ["i", "like", "to", "eat", "pancakes"]
    assert m["skipped"] == "statement"


def test_push_to_talk_statement_still_answered(monkeypatch):
    m, events, finalized = _run(False, "i like to eat pancakes", monkeypatch)
    kinds = [e["t"] for e in events]
    assert "heard" not in kinds and "endpoint" in kinds and finalized == ["i like to eat pancakes"]


def test_open_mic_question_answered(monkeypatch):
    m, events, finalized = _run(True, "who founded the company nvidia", monkeypatch)
    assert "heard" not in [e["t"] for e in events] and finalized


def test_events_carry_question_id(monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", False)

    async def finalize(self, f, t_end, **kw):
        self.emit({"t": "sources", "ms": 0})            # the speculator's own events carry its question's id
        return {"mode": self.mode, "question": f.text}
    monkeypatch.setattr(speculator.Speculator, "finalize", finalize)
    monkeypatch.setattr(speculator.Speculator, "on_frame", lambda self, f, now=None: None)
    events = []

    async def go():
        s = Session(events.append, mode="headstart", t0=time.perf_counter())
        await _speak(s, "who founded nvidia")
        words = "and when was it founded".split()
        times = [(5.0 + i * 0.25, 5.0 + i * 0.25 + 0.22) for i in range(len(words))]
        s.on_partial([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])
        s.close()
    asyncio.run(go())
    assert all("q" in e for e in events)
    first = [e for e in events if e["q"] == 1]
    second = [e for e in events if e["q"] == 2]
    assert {"words", "frame", "endpoint", "sources"} <= {e["t"] for e in first}
    assert second and second[0]["t"] == "words" and second[0]["words"][0]["w"] == "and"
    # nothing of the first question is tagged with the second's id
    assert not [e for e in second if e["t"] in ("endpoint", "sources")]
