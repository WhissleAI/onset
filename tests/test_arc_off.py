"""Arc-off arm ("plain voice search", mode="plain" / ARC_OFF=1; docs/ARC_VALUE.md).

What a typical voice assistant does: the raw final transcript (punctuation stripped) is the web query, searched once
after the ear's endpoint; no tagger read-out (no correction handling, carry / memory, redaction, hints, router lanes,
planner, open-mic gate). Same ear, web lane, reranker and answer model. The other modes must not change.

Run: .venv/bin/python -m pytest -q tests/test_arc_off.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config, frame, lanes, speculator  # noqa: E402
from server.session import Session  # noqa: E402
from test_speculation import WEB_CALLS, FakeStream, fakes, speak, specs  # noqa: E402,F401  (fakes: autouse)


import pytest  # noqa: E402

from server import answer  # noqa: E402


class CtxStream(FakeStream):
    """FakeStream that also keeps the previous-turn context the answer model was given."""

    def __init__(self, question, docs, tag="", context="", hint=""):
        self.context = context
        super().__init__(question, docs, tag=tag, context=context, hint=hint)


@pytest.fixture(autouse=True)
def ctx_stream(fakes, monkeypatch):
    monkeypatch.setattr(answer, "Stream", CtxStream)
    monkeypatch.setattr(speculator.answer, "Stream", CtxStream)
    yield


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------------- frame
def test_plain_frame_is_the_raw_transcript():
    f = frame.plain("What's the capital of Canada, actually make that Turkey?".split())
    assert f.plain and f.route == "web"
    assert f.query() == "what's the capital of canada actually make that turkey"
    assert f.web_hints() == {} and f.answer_hint() == "" and f.entity_query() == ""
    assert f.clean_text == f.text and f.resolved_text == f.text
    assert not f.corrected and not f.slots and not f.redacted and not f.carried and not f.dropped
    j = f.to_json()
    assert j["query"] == f.query() and j["readout"]["fresh"] == ""


def test_plain_frame_keeps_personal_data_and_time_words():
    # no redaction and no freshness: exactly what Arc adds
    f = frame.plain("call me at 555 123 4567 about the game last night".split())
    assert "555 123 4567" in f.query()
    assert f.web_hints() == {}
    assert frame.plain("convert 100 dollars to euros".split()).answer_hint() == ""


def test_plain_frame_has_no_router_plan(monkeypatch):
    monkeypatch.setattr(config, "ROUTER", True)
    f = frame.plain("what's the weather in denver".split())
    assert speculator.Speculator._plan(f) is None


# ----------------------------------------------------------------------------- session
def test_plain_searches_raw_transcript_once_after_endpoint():
    m, ev = run(speak("what's the capital of Canada actually make that Turkey", "plain"))
    assert m["mode"] == "plain" and m["path"] == "plain-search"
    assert WEB_CALLS == ["what's the capital of canada actually make that turkey"]
    assert specs(ev, state="fired", lane="web") == []            # nothing speculative
    assert m["query"] == WEB_CALLS[0]
    st = [s for s in FakeStream.instances if s.tag == "final"][-1]
    assert st.question == "what's the capital of Canada actually make that Turkey"   # not cleaned of the correction
    assert st.hint == "" and st.context == ""


def test_headstart_and_baseline_still_correct_away():
    for mode in ("baseline", "headstart"):
        WEB_CALLS.clear()
        m, _ = run(speak("what's the capital of Canada actually make that Turkey", mode))
        assert "canada" not in m["query"] and "turkey" in m["query"], (mode, m["query"])


def test_plain_has_no_conversation_memory():
    async def go():
        ev = []
        t0 = time.perf_counter()
        s = Session(ev.append, mode="plain", t0=t0)

        async def say(text, start):
            ws = [dict(text=w, start=start + i * 0.2, end=start + i * 0.2 + 0.18) for i, w in enumerate(text.split())]
            s.on_partial(ws)
            return await s.on_final(ws)
        m1 = await say("who founded the company Nvidia", 0.2)
        m2 = await say("when was it founded", 3.0)
        s.close()
        return m1, m2, s
    m1, m2, s = run(go())
    assert m2["query"] == "when was it founded"                  # no carried entity
    assert "memory" not in m2 and s.context == "" and s.carry_pool == {}
    finals = [x for x in FakeStream.instances if x.tag == "final"]
    assert finals[-1].context == ""
    assert len(s.history) == 2


def test_plain_context_flag_passes_previous_turn(monkeypatch):
    monkeypatch.setattr(config, "PLAIN_CONTEXT", True)

    async def go():
        s = Session(lambda e: None, mode="plain", t0=time.perf_counter())
        for text, start in (("who founded Nvidia", 0.2), ("when was it founded", 3.0)):
            ws = [dict(text=w, start=start + i * 0.2, end=start + i * 0.2 + 0.18) for i, w in enumerate(text.split())]
            await s.on_final(ws)
        s.close()
    run(go())
    finals = [x for x in FakeStream.instances if x.tag == "final"]
    assert finals[-1].context.startswith("Q: who founded Nvidia")
    assert WEB_CALLS[-1] == "when was it founded"                 # still no query rewriting


def test_plain_never_merges_a_split_repair():
    """sc27: the ear split "... Boston, sorry," | "I mean Denver". Arc modes merge the repair; plain answers each turn."""
    async def go():
        s = Session(lambda e: None, mode="plain", t0=time.perf_counter())
        a = [dict(text=w, start=0.2 + i * 0.2, end=0.38 + i * 0.2) for i, w in
             enumerate("what's the weather in Boston, sorry,".split())]
        b = [dict(text=w, start=2.0 + i * 0.2, end=2.18 + i * 0.2) for i, w in enumerate("I mean Denver".split())]
        await s.on_final(a)
        await s.on_final(b)
        s.close()
        return s
    s = run(go())
    assert [m["query"] for m in s.history] == ["what's the weather in boston sorry", "i mean denver"]


def test_plain_answers_statements_on_open_mic():
    async def go():
        s = Session(lambda e: None, mode="plain", t0=time.perf_counter(), open_mic=True)
        ws = [dict(text=w, start=0.2 + i * 0.2, end=0.38 + i * 0.2) for i, w in enumerate("i like to eat pancakes".split())]
        m = await s.on_final(ws)
        s.close()
        return m
    m = run(go())
    assert m.get("skipped") is None and m["path"] == "plain-search"


def test_plain_does_not_learn_user_facts():
    async def go():
        s = Session(lambda e: None, mode="plain", t0=time.perf_counter())
        ws = [dict(text=w, start=0.2 + i * 0.2, end=0.38 + i * 0.2) for i, w in enumerate("I live in Denver".split())]
        m = await s.on_final(ws)
        s.close()
        return m, s
    m, s = run(go())
    assert m["path"] == "plain-search" and not s.mem.user


def test_plain_skips_router_even_when_on(monkeypatch):
    monkeypatch.setattr(config, "ROUTER", True)
    seen = []

    async def web(query, topic="general", **kw):
        seen.append(kw)
        return [lanes.Doc(id="d", title="Denver", text="Denver weather.", source="fake")]
    monkeypatch.setattr(lanes, "web", web)
    m, _ = run(speak("what's the weather in Denver", "plain"))
    assert len(seen) == 1 and seen[0].get("frame") is None and not seen[0].get("hints") and not seen[0].get("entity")
    assert m["answer_by"] != "card"


def test_arc_off_env_forces_plain(monkeypatch):
    monkeypatch.setattr(config, "ARC_OFF", True)
    s = Session(lambda e: None, mode="headstart")
    assert s.mode == "plain" and s.plain and not s.semantic and s.dstream is None
    s.close()
