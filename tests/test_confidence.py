"""Confidence fusion (server/confidence.py), the fused end-of-turn policy (FLUX_SOFT=fused), the search director's
runtime fallbacks (server/director.py) and the simulator's Flux replay: deterministic, no network."""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import confidence, config, director  # noqa: E402
from server.session import Session  # noqa: E402
from test_speculation import fakes  # noqa: E402,F401  (autouse: fake lanes, embeddings, answer model)


def test_word_track_revisions_and_age():
    t = confidence.WordTrack()
    assert t.update(["who", "founded"]) == {"revs": [0, 0], "age": [0, 0]}
    assert t.update(["Who", "found"]) == {"revs": [0, 1], "age": [1, 0]}        # case ignored, word 1 revised
    assert t.update(["who", "found", "the"]) == {"revs": [0, 1, 0], "age": [2, 1, 0]}


def test_fused_word_conf_orders_shaky_words_lower():
    ws = ["who", "founded", "nvidia"]
    steady = confidence.word_confs(ws, [0.99, 0.99, 0.99], [0, 0, 0], [5, 5, 5], [])
    shaky = confidence.word_confs(ws, [0.99, 0.99, 0.45], [0, 0, 2], [5, 5, 0], [])
    assert shaky[2] < steady[2] - 0.2 and abs(shaky[0] - steady[0]) < 1e-9
    assert all(0.0 <= c <= 1.0 for c in steady + shaky)


def test_key_conf_is_the_weakest_content_word():
    class F:
        content_idx = (1, 3)
    k, j = confidence.key_conf(F(), [0.2, 0.9, 0.1, 0.6])
    assert (round(k, 3), j) == (0.6, 3)              # the stop word at index 2 doesn't count


def _feats(**kw):
    base = dict(eot=0.05, pause_s=0.0, n_words=6, cc_last=False, cc_score=0.0, gate=False, act_question=True,
                slot_end=False, has_content=True, last_content=False, arc_cc_p=0.05, arc_cc_recent=False,
                arc_for_me=0.9, is_eager=False, last_conf=0.9)
    base.update(kw)
    return confidence.eot_features(**base)


def test_eot_score_rises_with_the_evidence_of_a_finished_question():
    mid = confidence.eot_score(_feats())
    done = confidence.eot_score(_feats(eot=0.55, pause_s=0.4, cc_last=True, cc_score=0.9, gate=True, slot_end=True,
                                       last_content=True, arc_cc_p=0.6, arc_cc_recent=True, is_eager=True,
                                       last_conf=0.99))
    assert mid is not None and mid < 0.2 < 0.8 < done
    assert 0 < confidence.eot_threshold("eager") <= confidence.eot_threshold("update") < 1


def _ws(words, conf=0.99, last_end=None):
    out, t = [], 0.4
    for w in words:
        out.append({"text": w, "start": t, "end": t + 0.3, "conf": conf})
        t += 0.3
    if last_end is not None:
        out[-1]["end"] = last_end
    return out


def test_turn_tick_refreshes_timings_and_confidences_of_unchanged_words():
    """Flux re-sends the turn every update; the ear forwards a partial only when the TEXT changes. The fused features
    (pause since the last word, last-word confidence) must still see the newest timings / confidences."""
    async def run():
        s = Session(lambda e: None, mode="headstart")
        try:
            s.on_turn_info("Update", 0.1)
            s.on_partial(_ws(["who", "founded", "nvidia"], conf=0.6, last_end=1.2))
            s.on_turn_tick("Update", _ws(["who", "founded", "nvidia"], conf=0.6, last_end=1.2))
            s.on_turn_info("Update", 0.3)
            s.on_turn_tick("Update", _ws(["who", "founded", "nvidia"], conf=0.98, last_end=1.5))
            assert s.last_ws[-1]["end"] == 1.5 and s.confs[-1] == 0.98
            assert s.track.age[-1] == 1                  # aged on the unchanged update
        finally:
            s.close()
    asyncio.run(run())


def test_fused_soft_endpoint_fires_on_a_confident_update(monkeypatch):
    monkeypatch.setattr(config, "FLUX_SOFT", "fused")
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)     # (the fakes fixture turns it off)
    events = []

    async def run():
        s = Session(events.append, mode="headstart")
        try:
            ws = _ws(["who", "founded", "the", "company", "nvidia"])
            s.on_turn_info("Update", 0.05)
            s.on_partial(ws)
            s.on_turn_tick("Update", ws)
            assert s.soft is None                        # no pause yet, low Flux eot confidence
            s._audio_end = ws[-1]["end"] + 0.6
            s.arc = {"cc_p": 0.7, "for_me": 0.95, "cc_events": [{"t": ws[-1]["end"] + 0.1}]}
            s.on_turn_info("EagerEndOfTurn", 0.6)
            s.on_turn_tick("EagerEndOfTurn", ws)
            assert s.soft is not None and s.soft["by"] == "fused", [e for e in events if e.get("t") == "error"]
            await asyncio.sleep(0)
        finally:
            s.close()
    asyncio.run(run())
    assert any(e.get("t") == "eot_score" for e in events)


def test_director_without_a_model_falls_back_to_fused(monkeypatch, tmp_path):
    monkeypatch.setattr(director, "MODEL_DIR", tmp_path)          # no director.npz there
    monkeypatch.setattr(director, "_model", None)
    monkeypatch.setattr(director, "_failed", None)
    monkeypatch.setattr(director.Model.__init__, "__defaults__", (tmp_path,))
    monkeypatch.setattr(config, "DIRECTOR", True)
    monkeypatch.setattr(config, "FLUX_SOFT", "director")
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", True)
    assert director.model() is None

    async def run():
        s = Session(lambda e: None, mode="headstart")
        try:
            ws = _ws(["who", "founded", "the", "company", "nvidia"])
            s._audio_end = ws[-1]["end"] + 0.6
            s.arc = {"cc_p": 0.7, "for_me": 0.95, "cc_events": [{"t": ws[-1]["end"] + 0.1}]}
            s.on_turn_info("EagerEndOfTurn", 0.6)
            s.on_partial(ws)
            s.on_turn_tick("EagerEndOfTurn", ws)
            assert s.soft is not None and s.soft["by"] == "fused"
        finally:
            s.close()
    asyncio.run(run())


def test_director_features_match_the_feature_list():
    r = {"words": ["who", "founded", "nvidia"], "conf": [0.99, 0.9, 0.8], "ends": [0.5, 0.9, 1.3],
         "revs": [0, 0, 1], "age": [3, 2, 0], "spans": [{"type": "entity.mention.organization", "family": "entity",
                                                          "start": 2, "end": 3, "score": 0.9, "ver": 0.8,
                                                          "alt": {"p": 0.1}}],
         "intent": [{"tag": "discourse.intent.qa_factoid", "p": 0.7}], "act": [{"tag": "discourse.act.question", "p": 0.8}],
         "cc_at": 2, "cc_score": 0.8, "gate": True, "act_question": True, "has_content": True, "content_idx": [1, 2],
         "slots": [["organization", "nvidia"]], "query": "founded nvidia", "event": "Update", "eot": 0.2,
         "pause": 0.1, "arc_cc_p": None, "arc_cc_t": None, "arc_for_me": None}
    x = director.step_feats(r, None, [0.9, 0.9, 0.6])
    assert x.shape == (director.F,) and np.isfinite(x).all()
    i = {k: n for n, k in enumerate(director.FEATS)}
    assert x[i["cc_last"]] == 1 and x[i["slot_end"]] == 1 and x[i["sp_mention"]] == 1 and x[i["arc_cc_missing"]] == 1
    toks, wtok, wfeat = director.step_tokens(r, [0.9, 0.9, 0.6])
    assert (toks[-3:] > 0).all() and (toks[:-3] == 0).all() and wfeat[2, 2] == 1.0


def test_sim_flux_emulator_forwards_partials_only_on_changed_text():
    from sim.flux_traces import FluxEmu
    calls = []

    class S:
        def on_meta(self, m): calls.append(("meta",))
        def on_turn_info(self, ev, p): calls.append(("info", ev))
        def on_partial(self, ws, now=None): calls.append(("partial", len(ws)))
        def on_turn_tick(self, ev, ws=None): calls.append(("tick", ev))
        def on_eager(self, ws, now=None): calls.append(("eager",))
        async def on_final(self, ws): calls.append(("final", len(ws)))

    finals = []
    emu = FluxEmu(S(), finals, lambda c: (c.close(), finals.append(1))[1])
    w = [{"text": "hi", "start": 0.1, "end": 0.3}]
    emu.on({"ev": "Update", "turn": 0, "eot": 0.1, "words": w, "arc": None}, 0)
    emu.on({"ev": "EagerEndOfTurn", "turn": 0, "eot": 0.4, "words": w, "arc": None}, 0)
    assert [c[0] for c in calls].count("partial") == 1 and ("eager",) in calls and calls.count(("tick", "Update")) == 1
