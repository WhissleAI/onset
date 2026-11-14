"""Typed questions (TYPED_INPUT) and streaming redaction with the Arc Tagger's safety heads (TAGGER_SAFETY)."""
import asyncio
import time

import pytest

from server import config, safety, speculator
from server import app as appmod
from server.app import TypedText
from server.session import Session


# ----------------------------------------------------------------------------- typed words + keystroke timings
def test_typed_text_only_finished_words():
    t = TypedText()
    assert t.update("who", 0.1) == []                       # a word being typed never reaches the tagger
    ws = t.update("who ", 0.3)
    assert [w["text"] for w in ws] == ["who"] and ws[0]["start"] == 0.1 and ws[0]["end"] == 0.3
    t.update("who fou", 0.5)
    ws = t.update("who founded nvi", 0.9)
    assert [w["text"] for w in ws] == ["who", "founded"] and ws[1]["start"] == 0.5 and ws[1]["end"] == 0.9
    ws = t.update("who founded nvidia", 1.2, final=True)    # Enter finishes the last word
    assert [w["text"] for w in ws] == ["who", "founded", "nvidia"] and ws[2]["start"] == 0.9
    ws = t.update("who made ", 1.5)                         # an edited word gets new times
    assert [w["text"] for w in ws] == ["who", "made"] and ws[1]["start"] == 1.5 and ws[0]["start"] == 0.1


def test_typed_text_paste():
    ws = TypedText().update("what is the capital of peru", 2.0, final=True)
    assert len(ws) == 6 and all(w["start"] == w["end"] == 2.0 for w in ws)


# ----------------------------------------------------------------------------- mask decisions
def _sp(typ, a, b, score=0.9, text=""):
    return {"type": typ, "start": a, "end": b, "score": score, "text": text,
            "family": "entity" if typ.startswith("entity") else "profanity"}


def test_mask_idx_tagger_heads(monkeypatch):
    monkeypatch.setattr(safety, "ENABLED", True)
    words = "what the fuck is the capital of france".split()
    m = safety.mask_idx(words, [_sp("safety.profanity.strong", 2, 3), _sp("entity.mention.place", 7, 8)], tagger=True)
    assert m == {2: "profanity"}
    # a toxicity span on a name the tagger also reads as an entity is not masked ("Dick Van Dyke", "Peru")
    words = "you are a moron what is the capital of peru".split()
    spans = [_sp("safety.toxicity.insult_harassment", 3, 4, 0.85), _sp("safety.toxicity.insult_harassment", 9, 10, 0.64),
             _sp("entity.mention.place", 9, 10)]
    assert safety.mask_idx(words, spans, tagger=True) == {3: "insult"}
    # below the toxicity threshold: kept ("black hole" read as an insult at 0.59)
    assert safety.mask_idx("what is a black hole".split(), [_sp("safety.toxicity.insult_harassment", 4, 5, 0.59)],
                           tagger=True) == {}


def test_mask_idx_fallback_word_list(monkeypatch):
    monkeypatch.setattr(safety, "ENABLED", False)
    assert safety.mask_idx("this shit is slow".split(), []) == {1: "profanity"}
    monkeypatch.setattr(safety, "ENABLED", True)
    assert safety.mask_idx("this shit is slow".split(), [], tagger=False) == {1: "profanity"}
    assert safety.mask_idx("this shit is slow".split(), []) == {1: "profanity"}     # a tagger without the heads


def test_turn_flags(monkeypatch):
    monkeypatch.setattr(safety, "ENABLED", True)
    assert safety.turn_flag("I want to kill myself".split(), [], None) == "self_harm"
    assert safety.turn_flag("how do I kill a python process".split(), [], None) is None
    assert safety.turn_flag("x".split(), [], {"safety": [{"tag": "safety.crisis.self_harm", "p": 0.4, "on": True}]},
                            tagger=True) == "self_harm"
    w = "go back to your country you filthy immigrants".split()
    assert safety.turn_flag(w, [_sp("safety.toxicity.hate_identity", 6, 8, 0.92)], None, tagger=True) == "hate"
    assert safety.no_search("hate") and safety.no_search("self_harm") and not safety.no_search("insult")


def test_shown():
    assert safety.shown("fuck") == "f***" and safety.shown("shit,") == "s***," and safety.shown("ass") == "a**"


def test_answer_masker_streams_whole_words():
    am = safety.AnswerMasker(None)                         # word-list fallback (no handle)
    out = am.feed("This is bull") + am.feed("shit, frankly") + am.feed(" the end.")
    out += am.flush()
    assert out == "This is b*******, frankly the end." and am.masked == 1


def test_answer_masker_tagger():
    from server.tagger import Tagger
    t = Tagger(domain="talk", emit="profanity", options={"redact": "profanity=mask"})
    am = safety.AnswerMasker(t)
    out = "".join(am.feed(d) for d in ["He said ", "\"what the f", "uck\" and left. ", "Dick Van Dyke ", "starred."])
    out += am.flush()
    assert "f***" in out and "Dick Van Dyke" in out and am.masked >= 1
    t.close()


def test_frame_masks_out_of_query(monkeypatch):
    from server import frame as fr
    monkeypatch.setattr(safety, "ENABLED", True)
    words = "what the fuck is the capital of australia".split()
    f = fr.build(words, [_sp("safety.profanity.strong", 2, 3), _sp("entity.mention.place", 7, 8, text="australia")],
                 {"intent": [], "act": []}, safety_tagger=True)
    assert "fuck" not in f.query() and "fuck" not in f.text and "fuck" not in f.clean_text and "fuck" not in f.kept_text
    j = f.to_json()
    assert j["mask"] == [2] and j["mask_kind"] == [[2, "profanity"]] and 2 not in j["redact"]
    assert "[private]" not in f.clean_text


# ----------------------------------------------------------------------------- sessions: typed + safety
def _stub(monkeypatch, finalized):
    async def finalize(self, f, t_end, **kw):
        finalized.append(f)
        self.emit({"t": "answer", "delta": "That is bull", "ms": 1})
        self.emit({"t": "answer", "delta": "shit, says the source.", "ms": 2})
        self.emit({"t": "metric", "ms": 3})
        return {"mode": self.mode, "question": f.text, "path": "stub", "answer": "That is bullshit, says the source."}
    monkeypatch.setattr(speculator.Speculator, "finalize", finalize)
    monkeypatch.setattr(speculator.Speculator, "on_frame", lambda self, f, now=None: None)


async def _type(s, text, cps=12.0):
    t = TypedText()
    now = 0.0
    for k in range(1, len(text) + 1):
        now += 1.0 / cps
        s.on_partial(t.update(text[:k], now))
        await asyncio.sleep(0)
    ws = t.update(text, now + 0.2, final=True)
    s.end_of_audio()
    return await s.on_final(ws)


def test_typed_session_turn(monkeypatch):
    finalized, events = [], []
    _stub(monkeypatch, finalized)

    class Spy:
        observed = 0

        def observe(self, *a, **k):
            Spy.observed += 1

    async def go():
        s = Session(events.append, mode="headstart", t0=time.perf_counter(), arcstate=Spy())
        s.typed, s.semantic = True, False
        m = await _type(s, "who founded the company nvidia")
        s.close()
        return m
    m = asyncio.run(go())
    assert m["modality"] == "typed" and finalized and "nvidia" in finalized[0].query()
    assert Spy.observed == 0                                 # no voice baselines from a typed turn
    words = [e for e in events if e["t"] == "words"]
    assert words and all(e.get("typed") for e in words)
    # the answer is masked as it streams; the stored answer is the masked one
    shown = "".join(e["delta"] for e in events if e["t"] == "answer")
    assert "bullshit" not in shown and "b*******" in shown
    assert "bullshit" not in m["answer"] and m.get("answer_masked") == 1


@pytest.mark.parametrize("text,kind", [("I want to kill myself tonight", "self_harm"), ("you are a stupid idiot", "masked")])
def test_no_search_turns(monkeypatch, text, kind):
    finalized, events = [], []
    _stub(monkeypatch, finalized)
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", False)

    async def go():
        s = Session(events.append, mode="headstart", t0=time.perf_counter())
        s.typed = True
        m = await _type(s, text)
        s.close()
        return m
    m = asyncio.run(go())
    assert not finalized and m["path"] == "safety" and m["web_calls"] == 0
    ans = "".join(e["delta"] for e in events if e["t"] == "answer")
    if kind == "self_harm":
        assert m["safety"] == "self_harm" and "988" in ans
    else:
        assert ans == safety.DECLINE_LINE and "idiot" not in m["question"]
    assert any(e["t"] == "spec_hold" and e["why"] == "safety" for e in events)


def test_spoken_profanity_kept_out_of_query(monkeypatch):
    finalized, events = [], []
    _stub(monkeypatch, finalized)
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", False)

    async def go():
        s = Session(events.append, mode="headstart", t0=time.perf_counter())
        words = "what the fuck is the capital of australia".split()
        ws = [dict(text=w, start=0.2 + i * 0.25, end=0.4 + i * 0.25) for i, w in enumerate(words)]
        for k in range(1, len(ws) + 1):
            s.on_partial(ws[:k])
        m = await s.on_final(ws)
        s.close()
        return m
    m = asyncio.run(go())
    f = finalized[0]
    assert "fuck" not in f.query() and "fuck" not in m["question"] and "australia" in f.query()
    fr = [e for e in events if e["t"] == "frame" and e.get("final")][-1]
    assert 2 in fr["mask"]


# ----------------------------------------------------------------------------- websocket: a typed run, race mode
def test_ws_typed_race(monkeypatch):
    from fastapi.testclient import TestClient
    finalized = []
    _stub(monkeypatch, finalized)
    monkeypatch.setattr(appmod.app.router, "on_startup", [])
    monkeypatch.setattr(appmod.ear, "make_ear", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no ear")))
    with TestClient(appmod.app) as c:
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["t"] == "ready"
            ws.send_json({"t": "start", "typed": True, "modes": ["headstart", "baseline"], "rid": 7})
            for k, txt in enumerate(["who ", "who founded ", "who founded nvidia"]):
                ws.send_json({"t": "type", "rid": 7, "text": txt, "final": k == 2})
            got = []
            while True:
                e = ws.receive_json()
                got.append(e)
                if e["t"] == "run_done":
                    break
    metrics = [e for e in got if e["t"] == "metric"]
    assert {e["mode"] for e in metrics} == {"headstart", "baseline"}
    assert all(e.get("rid") == 7 for e in got if e.get("mode") in ("headstart", "baseline"))
    assert len(finalized) == 2 and all(f.query().endswith("nvidia") for f in finalized)
    assert any(e["t"] == "ear" and e.get("ear") == "typed" for e in got)
