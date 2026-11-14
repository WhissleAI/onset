"""Persona (server/persona.py, PERSONA=1): per-question style from how it was asked, the per-user profile, the interest
rerank and the free sense search, the hooks in frame / speculator / session / lanes, and what is (not) stored."""
import asyncio
import json
import time

import numpy as np
import pytest

from server import frame as framemod
from server import lanes, persona, speculator
from server.session import Session


def _ws(text, gap=0.05, dur=0.22, t0=0.2, pause_after=()):
    out, t = [], t0
    for i, w in enumerate(text.split()):
        out.append({"text": w, "start": t, "end": t + dur, "conf": 0.95})
        t += dur + gap + (0.6 if i in pause_after else 0.0)
    return out


def _unit(*xs):
    v = np.array(xs, np.float32)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------- signals + style
def test_signals_read_fillers_hedges_pauses_and_pace():
    s = persona.signals("um what is uh i think photosynthesis".split(), _ws("um what is uh i think photosynthesis",
                                                                          pause_after=(0, 2)))
    assert s["fillers"] == 2 and s["hedges"] == 1 and s["pauses"] == 2 and s["pace"] > 0
    assert persona.style(s)["kind"] == "fuller"
    quick = persona.signals("boiling point of ethanol".split(), _ws("boiling point of ethanol", gap=0.0, dur=0.12))
    assert quick["pace"] >= persona.FAST_SPS
    st = persona.style(quick)
    assert st["kind"] == "short" and "quickly" in st["why"] and "15 words" in st["hint"]
    normal = persona.signals("boiling point of ethanol".split(), _ws("boiling point of ethanol", gap=0.1, dur=0.35))
    assert persona.style(normal)["kind"] == ""                     # nothing to change: no hint, no chip


def test_style_priorities_and_wrong_needs_a_previous_answer():
    w = "no that's not what i meant who wrote frankenstein".split()
    assert persona.style(persona.signals(w, prev_answered=True))["kind"] == "fix"
    assert persona.style(persona.signals(w, prev_answered=False))["kind"] != "fix"
    assert persona.style(persona.signals("tell me more about how they form".split()))["kind"] == "more"
    assert persona.style(persona.signals("just give me the short answer who painted it".split()))["kind"] == "short"
    s = persona.signals("how tall is everest in feet".split())
    assert s["units"] == "imperial"


def test_profile_learns_length_and_units_and_applies_them_later():
    p = persona.Profile()
    p.observe(persona.signals("just give me the short answer who painted the mona lisa".split()))
    p.observe(persona.signals("how tall is everest in feet".split()))
    later = persona.style(persona.signals("how tall is k2".split()), p)
    assert later["kind"] == "short" and "usually" in later["why"]
    assert "miles, feet" in later["hint"]                          # units carried from an earlier question
    assert p.summary()["units"] == "imperial"


# ----------------------------------------------------------------------------- profile: interests, store, privacy
def test_interests_sense_and_because():
    cats = np.stack([_unit(1, 0, 0, 0)] + [_unit(0, 0, 0, 1)] * (len(persona.CATEGORIES) - 1))
    cats[persona.CATEGORIES.index("planet")] = _unit(0, 1, 0, 0)
    p = persona.Profile()
    p.observe({}, [("Jupiter", _unit(0.1, 1, 0, 0))], cat_vecs=cats)
    p.observe({}, [("Venus", _unit(0, 1, 0.1, 0))], cat_vecs=cats)
    assert [x["cat"] for x in p.items] == ["planet", "planet"]
    assert p.sense() == "planet"
    assert p.because(_unit(0, 1, 0.12, 0))["label"] == "Venus"
    v = persona.View(p)

    class F:
        phrases = ("mercury",)
        content = ("hot", "mercury")

        def entity_query(self):
            return ""
    assert v.sense_query(F()) == "mercury planet"


def test_private_turn_adds_no_interest_and_export_has_no_vectors():
    p = persona.Profile()
    p.observe({}, [("Some Page", _unit(1, 0))], private=True)
    assert not p.items
    p.observe({}, [("Ford Mustang", _unit(1, 0))], cat_vecs=np.stack([_unit(1, 0)] * len(persona.CATEGORIES)))
    ex = json.dumps(p.export())
    assert "Ford Mustang" in ex and "vec" not in ex
    assert set(p.export()) >= {"turns", "topics", "length", "units"}


def test_store_roundtrip_ttl_and_forget(tmp_path):
    st = persona.Store(tmp_path)
    p = persona.Profile()
    p.observe(persona.signals("how tall is everest in feet".split()), [("Mount Everest", _unit(1, 2, 3))],
              cat_vecs=np.stack([_unit(1, 0, 0)] * len(persona.CATEGORIES)))
    st.save("client_abc123", p)
    q = st.load("client_abc123")
    assert q.turns == 1 and q.units == "imperial" and q.items[0]["label"] == "Mount Everest"
    assert abs(float(q.items[0]["vec"] @ _unit(1, 2, 3)) - 1) < 1e-2          # float16 on disk
    raw = st.path("client_abc123").read_text()
    assert "everest in feet" not in raw                                          # never the user's words
    d = p.to_json()
    d["updated"] = time.time() - 10 * persona.TTL_S
    assert not persona.Profile.from_json(d)                                      # expired
    st.forget("client_abc123")
    assert not st.path("client_abc123").exists()
    st.save("../etc", p)
    assert not persona.Store.valid("../etc") and len(list(tmp_path.glob("*.json"))) == 0


# ----------------------------------------------------------------------------- retrieval
def test_bias_rank_flips_an_ambiguous_top_and_skips_carried_docs():
    vecs = {"animal": _unit(1, 0.2, 0), "car": _unit(0.95, 0, 0.35), "old": _unit(0.5, 0, 1)}
    q = _unit(1, 0.1, 0.1)
    plain = persona.bias_rank(list(vecs), vecs, q, None)
    assert plain[0][0] == "animal"
    p = _unit(0, 0, 1)
    assert persona.bias_rank(list(vecs), vecs, q, p, beta=0.25)[0][0] == "car"
    assert persona.bias_rank(list(vecs), vecs, q, p, beta=2.0)[0][0] == "old"
    assert persona.bias_rank(list(vecs), vecs, q, p, beta=2.0, skip={"old"})[0][0] == "car"


def test_ambiguous():
    D = lanes.Doc
    assert persona.ambiguous([D("1", "Mercury (element)", "x")], "mercury")
    assert persona.ambiguous([D("1", "Mercury", "Mercury most commonly refers to:\n...")], "mercury")
    assert persona.ambiguous([D("1", "Apple (disambiguation)", "...")], "apple")
    assert not persona.ambiguous([D("1", "Google", "Google LLC is ..."), D("2", "Google Search", "...")], "google")


def test_wikipedia_sense_adds_docs_only_when_ambiguous(monkeypatch):
    D = lanes.Doc
    pages = {"hot mercury": [D("a", "Mercury (element)", "liquid metal")], "mercury planet": [D("b", "Mercury (planet)", "planet")],
             "who founded google": [D("c", "Google", "Google LLC")], "google car": [D("d", "Waymo", "self-driving")],
             "fast jaguar": [D("e", "Jaguar · local Wikipedia", "big cat")],
             "jaguar car": [D("f", "Jaguar Cars", "british car maker"), D("g", "Jaguar XJ220", "sports car")]}

    async def fake(q, k):
        return list(pages.get(q, []))
    monkeypatch.setattr(lanes, "_wiki_search", fake)
    got = asyncio.run(lanes.wikipedia_sense("hot mercury", "", "mercury planet"))
    assert [d.title for d in got] == ["Mercury (element)", "Mercury (planet)"]
    assert got[1].meta["persona_sense"] == "mercury planet"
    got = asyncio.run(lanes.wikipedia_sense("who founded google", "", "google car"))
    assert [d.title for d in got] == ["Google"]                  # not ambiguous: the sense search is dropped
    got = asyncio.run(lanes.wikipedia_sense("fast jaguar", "", "jaguar car"))
    assert [d.title for d in got] == ["Jaguar · local Wikipedia", "Jaguar Cars", "Jaguar XJ220"]   # named by the subject


def test_speculator_rank_hint_and_event(monkeypatch):
    monkeypatch.setattr(persona, "BIAS", 0.4)
    ev = []
    sp = speculator.Speculator(ev.append, "headstart", 0.0)
    D = lanes.Doc
    docs = {"animal": (D("animal", "Jaguar", "big cat"), _unit(1, 0.2, 0)),
            "car": (D("car", "Jaguar XJ220", "sports car"), _unit(0.8, 0, 0.6)),
            "x": (D("x", "Jaguar warrior", "aztec"), _unit(0.3, 1, 0))}
    for k, (d, v) in docs.items():
        sp.pool[k], sp.vecs[k] = d, v
    cats = np.stack([_unit(1, 0, 0)] * len(persona.CATEGORIES))
    cats[persona.CATEGORIES.index("car")] = _unit(0, 0, 1)
    p = persona.Profile()
    p.observe({}, [("Ford Mustang", _unit(0, 0, 1))], cat_vecs=cats)
    assert p.sense() == "car"
    monkeypatch.setattr(persona.View, "cat", lambda self, i, v: "car" if i == "car" else "animal")
    sp.persona = persona.View(p)
    f = framemod.build("how fast is a jaguar".split(), [], None, set(), None)
    out = sp._rank_for(f, _unit(1, 0.1, 0.1))
    assert out[0][0].title == "Jaguar XJ220"
    e = [x for x in ev if x["t"] == "persona"]
    assert e and e[0]["sense"] == "Jaguar XJ220: you asked about Ford Mustang earlier"
    assert "Answer about Jaguar XJ220" in sp._hint(f)
    sp.persona = None
    assert sp._rank_for(f, _unit(1, 0.1, 0.1))[0][0].title == "Jaguar"   # no persona: unchanged
    sp.close()


def test_frame_answer_hint_keeps_conversion_and_adds_persona():
    f = framemod.build("how tall is k2".split(), [], None, set(), None)
    assert f.answer_hint() == ""
    f.persona_hint = "Use miles, feet and °F."
    assert f.answer_hint() == "Use miles, feet and °F."
    f.convert = ("a hundred us dollars", "us dollars", "euros")
    assert f.answer_hint().startswith("This asks to convert") and f.answer_hint().endswith("°F.")


# ----------------------------------------------------------------------------- session wiring
@pytest.mark.parametrize("on", [False, True])
def test_session_flag_and_modes(monkeypatch, on):
    monkeypatch.setattr(persona, "ENABLED", on)

    async def go():
        a = Session(lambda e: None, mode="headstart")
        b = Session(lambda e: None, mode="baseline")
        a.words = "just the year when was hamlet written".split()
        f = a._build()
        r = (a.spec.persona is not None, b.spec.persona is not None, f.persona_hint)
        a.close(); b.close()
        return r
    hs, bl, hint = asyncio.run(go())
    assert bl is False                                  # the comparison arm is never personalised
    assert hs is on and (bool(hint) is on)


def test_session_learns_and_exports_profile(monkeypatch):
    monkeypatch.setattr(persona, "ENABLED", True)
    monkeypatch.setattr(persona, "_cat_vecs", np.stack([_unit(1, 0)] * len(persona.CATEGORIES)))
    ev = []

    async def finalize(self, f, t_end, **kw):
        d = lanes.Doc("p", "Porsche 911", "sports car", source="wikipedia")
        self.pool["p"], self.vecs["p"], self.used_ids = d, _unit(1, 0), ["p"]
        return {"mode": self.mode, "question": f.text, "path": "stub", "answer": "About 400 hp [1]."}
    monkeypatch.setattr(speculator.Speculator, "finalize", finalize)
    monkeypatch.setattr(speculator.Speculator, "on_frame", lambda self, f, now=None: None)

    async def go():
        s = Session(ev.append, mode="headstart", t0=time.perf_counter())
        ws = _ws("how much horsepower does a porsche 911 have")
        for k in range(1, len(ws) + 1):
            s.on_partial(ws[:k])
        m = await s.on_final(ws)
        mem = s.export_memory()
        s.close()
        b = Session(lambda e: None, mode="headstart")
        b.import_memory(json.loads(json.dumps(mem)))
        r = (m, b.persona.turns, b.persona.items[0]["label"], b.spec.persona is not None)
        b.close()
        return r
    m, turns, label, view = asyncio.run(go())
    assert m["persona"]["style"] == "" and "hint_tok" in m["persona"]
    assert (turns, label, view) == (1, "Porsche 911", True)
    assert any(e["t"] == "persona" and e.get("profile") for e in ev)


def test_interest_sources_cited_or_named_never_keystone_cells():
    D = lanes.Doc
    f = framemod.build("how much horsepower does a porsche 911 have".split(), [], None, set(), None)
    docs = [D("index:1", "how much horsepower does a porsche 911 have", "x"),      # Keystone cell: the user's words
            D("a", "Tax horsepower", "x", source="wikipedia"),
            D("b", "Porsche 911 · local Wikipedia", "x", source="local Wikipedia"),
            D("c", "Pony", "x", source="wikipedia")]
    vecs = {d.id: _unit(1, 0) for d in docs}
    got = [t for t, _ in persona.interest_sources(f, "It makes 379 hp [3].", docs, vecs)]
    assert got == ["Porsche 911"]                                     # the cited one, suffix stripped
    got = [t for t, _ in persona.interest_sources(f, "The sources don't say.", docs, vecs)]
    assert got == ["Tax horsepower", "Porsche 911"]                   # no citation: titles that share a word
    assert "Pony" not in got and not any("how much" in t for t in got)


def test_quick_turns_lean_the_profile_short():
    p = persona.Profile()
    q = persona.signals("speed of light in a vacuum".split(), _ws("speed of light in a vacuum", gap=0.0, dur=0.1))
    assert q["quick"]
    p.observe(q)
    p.observe(q)
    slow = persona.signals("atomic number of tungsten".split(), _ws("atomic number of tungsten", gap=0.1, dur=0.35))
    assert not slow["quick"] and persona.style(slow, p)["kind"] == "short"
