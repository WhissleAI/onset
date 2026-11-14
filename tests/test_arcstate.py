"""ArcState (server/arcstate.py): Arc's acoustic state carried across push-to-talk presses. No models are loaded here."""
import json
import time
import types
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from server import arcstate, ear
from server import memory as memmod


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(arcstate, "ENABLED", True)


def words_times(text, rate=0.3, start=0.5, gap=0.05):
    ws = text.split()
    t, times = start, []
    for w in ws:
        times.append((t, t + rate))
        t += rate + gap
    return ws, times


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.linalg.norm(x)


def frames_for(st, e, t0, t1, rng, noise=0.05):
    st.add_frames([(t, unit(e + noise * rng.standard_normal(len(e)))) for t in np.arange(t0, t1, 0.1)])


def test_relative_pace_and_hesitation_vs_your_usual():
    st = arcstate.ArcState()
    for i, q in enumerate(["who founded the company nvidia", "when was it founded", "where is the headquarters"]):
        w, t = words_times(q)
        st.observe(("q", i), w, t, [0.05] * 10)
    w, t = words_times("um who is the uh chief executive", rate=0.6, gap=0.5)     # slow, fillers, long pauses
    r = st.observe(("q", 9), w, t, [0.05] * 10)
    assert r["rel"]["pace_rel"] < 0.8 and r["rel"]["hes_rel"] > 1.8
    assert "slower than you usually speak" in r["notes"] and "more hesitant than usual" in r["notes"]
    # the first question has no baseline: nothing relative, nothing said
    st2 = arcstate.ArcState()
    r2 = st2.observe("k", *words_times("who founded nvidia"), [0.05])
    assert r2["rel"] == {} and r2["notes"] == []


def test_same_question_from_both_ab_sessions_is_observed_once():
    st = arcstate.ArcState()
    w, t = words_times("who wrote hamlet")
    st.observe((0.5, tuple(w)), w, t, None)
    st.observe((0.5, tuple(w)), w, t, None)
    assert st.turns == 1


def test_voice_enrollment_keeps_you_and_flags_another_voice():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal(64), rng.standard_normal(64)
    st = arcstate.ArcState()
    out = []
    for i, (e, q) in enumerate([(a, "who founded nvidia"), (a, "when was it founded"), (b, "what about amd"),
                                (a, "where is it")]):
        st.new_stream()                                  # every press: a new stream, the clock restarts
        w, t = words_times(q)
        frames_for(st, e, t[0][0], t[-1][1], rng)
        out.append(st.observe(i, w, t, None))
    assert [o["who"] for o in out] == ["you", "you", "other", "you"]
    assert [o["voice"] for o in out] == ["S1", "S1", "S2", "S1"]
    assert out[1]["sim"] > 0.9 and out[2]["notes"] == []          # no "you sound ..." notes about another voice
    assert st.style_hints()["speaker"] == "you" and st.features()["as_nvoices"] == 0.5


def test_too_little_speech_assumes_the_enrolled_voice():
    st = arcstate.ArcState()
    w, t = words_times("hi")
    r = st.observe(0, w, t, None)
    assert r["who"] == "you" and r["frames"] == 0


def test_resets_all_topic_and_idle(monkeypatch):
    rng = np.random.default_rng(1)
    st = arcstate.ArcState()
    for i in range(3):
        w, t = words_times("who founded the company nvidia")
        frames_for(st, rng.standard_normal(16) * 0 + 1, t[0][0], t[-1][1], rng)
        st.observe(i, w, t, [0.05], intent="discourse.intent.qa_factoid")
    assert st.intents and st.voices
    st.note_turn(None, False, "nvidia", key="a")
    assert not st.note_turn(None, True, "jensen huang", key="b")          # a follow-up is not a topic switch
    assert st.note_turn(None, False, "eiffel tower", key="c")             # a new subject, not a follow-up: switch
    assert not st.intents and st.resets["topic"] == 1 and st.voices      # the voice and baselines stay
    monkeypatch.setattr(arcstate, "IDLE_S", 60)
    st.last_active = time.time() - 120
    st.intents.append("x")
    assert st.touch() == "idle" and not st.intents and st.voices
    assert all(v.base.n.get("pace", 0) <= 2 for v in st.voices)           # softened, not gone
    st.reset("all")
    assert not st.voices and st.turns == 0 and st.last is None


def mem_with(*names_turns):
    m = memmod.Memory()
    for name, kind, turn in names_turns:
        m.ents[name.lower()] = {"key": name.lower(), "name": name, "type": kind, "turns": [turn]}
    return m


def test_keyterms_newest_names_first_and_skip_common_words():
    st = arcstate.ArcState()
    m = mem_with(("Rafael Nadal", "person", 1), ("Novak Djokovic", "person", 2), ("company", "thing", 2),
                 ("Schiphol", "place", 3), ("apple", "organization", 3))
    kt = st.keyterms(m)
    assert kt == ["Schiphol", "Novak Djokovic", "Rafael Nadal"]
    assert st.keyterms(None) == [] and st.keyterms(memmod.Memory()) == []


def test_flux_url_carries_the_keyterms_once():
    u = ear.flux_url(base="wss://example.invalid/v2/listen", keyterms=["Rafael Nadal", "Schiphol", "rafael nadal"])
    q = parse_qs(urlparse(u).query)
    assert q["keyterm"] == ["Rafael Nadal", "Schiphol"]
    assert "keyterm" not in parse_qs(urlparse(ear.flux_url(base="wss://example.invalid/v2/listen")).query)


def test_persistence_is_opt_in_and_in_the_data_dir(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    st = arcstate.ArcState()
    w, t = words_times("who founded nvidia today")
    frames_for(st, rng.standard_normal(32), t[0][0], t[-1][1], rng)
    st.observe(0, w, t, [0.05])
    monkeypatch.setattr(arcstate, "PERSIST", False)
    arcstate.save("client123abc", st, tmp_path)
    assert not any(tmp_path.rglob("*.json"))
    assert arcstate.load("client123abc", tmp_path).voices == []
    monkeypatch.setattr(arcstate, "PERSIST", True)
    arcstate.save("client123abc", st, tmp_path)
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1 and files[0].parent == arcstate.state_dir(tmp_path)
    assert "client123abc" not in files[0].name and "nvidia" not in files[0].read_text()   # no words, hashed id
    back = arcstate.load("client123abc", tmp_path)
    assert len(back.voices) == 1 and np.allclose(back.voices[0].c, st.voices[0].c, atol=1e-4)
    arcstate.forget("client123abc", tmp_path)
    assert not any(tmp_path.rglob("*.json"))


def test_tap_keeps_speech_frame_embeddings_on_the_audio_clock():
    calls = []
    eng = types.SimpleNamespace(bank=object(), _bank_step=lambda *a: calls.append(a), utt_p=None)
    tap = arcstate.Tap(types.SimpleNamespace(eng=eng), offset_s=2.0)
    assert tap.ok
    eng._bank_step(3, np.ones(4), np.zeros(9), True)
    eng._bank_step(4, np.ones(4), np.zeros(9), False)            # silence: not kept
    out = tap.drain(2.0)
    assert [round(t, 2) for t, _ in out] == [2.3] and not calls  # Arc's own bank step is skipped
    assert arcstate.Tap(types.SimpleNamespace(eng=types.SimpleNamespace(bank=None))).ok is False


def test_pitch_of_a_150_hz_voice():
    sr = 16000
    x = np.arange(sr) / sr
    tone = 0.2 * np.sign(np.sin(2 * np.pi * 150 * x)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * x) ** 2)
    st = arcstate.pitch_st(tone.astype(np.float32))
    assert st == pytest.approx(12 * np.log2(1.5), abs=0.5)
    assert arcstate.pitch_st(np.zeros(sr, np.float32)) is None


def test_module_getters_for_persona_and_director():
    st = arcstate.ArcState()
    for i in range(3):
        st.observe(i, *words_times("who founded the company nvidia"), [0.05])
    w, t = words_times("who founded the company nvidia", rate=0.15, gap=0.02)    # fast
    sess = types.SimpleNamespace(arcstate=st, words=w, times=t)
    rel = arcstate.relative(sess)
    assert rel["pace_rel"] > 1.25
    assert arcstate.relative(types.SimpleNamespace(arcstate=None, words=w, times=t)) is None
    from server import director
    r = {"words": ["a"], "ends": [0.1], "spans": [], "pause": 0.0, "arcstate": st.features()}
    v = director.step_feats2(r, None, groups=["state"])
    assert len(v) == len(director.ARCSTATE_F) and v[director.ARCSTATE_F.index("as_ok")] == 1.0
    assert json.dumps(st.style_hints())


def test_make_ear_sends_keyterms_and_state_to_flux(monkeypatch):
    import asyncio
    from server import config
    monkeypatch.setattr(config, "EAR", "flux")
    monkeypatch.setattr(config, "DEEPGRAM_API_KEY", "dg-key")
    monkeypatch.setattr(config, "FLUX_URL", "ws://127.0.0.1:9/v2/listen")      # never the real service in tests
    monkeypatch.setattr(config, "ARC_META", "off")
    st = arcstate.ArcState()

    async def go():
        e = ear.make_ear([], keyterms=["Rafael Nadal"], arcstate=st)
        u = e.url
        e.close()
        return e, u
    e, u = asyncio.run(go())
    assert isinstance(e, ear.FluxEar) and e.arcstate is st
    assert parse_qs(urlparse(u).query)["keyterm"] == ["Rafael Nadal"]


def test_ws_reset_voice_and_forget_clear_arc_state(monkeypatch):
    from fastapi.testclient import TestClient
    from server import app as appmod, config
    monkeypatch.setattr(config, "EAR", "arc")
    with TestClient(appmod.app).websocket_connect("/ws") as ws:
        assert ws.receive_json()["t"] == "ready"
        ws.send_json({"t": "arc_reset"})
        r = ws.receive_json()
        assert r["t"] == "arcstate" and r["reset"] == "all" and r["voices"] == 0
        ws.send_json({"t": "forget"})
        assert ws.receive_json()["forgot"] is True
