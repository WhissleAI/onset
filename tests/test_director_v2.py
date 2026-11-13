"""Director v2 (server/director.py v2 streams + heads, server/arcmeta.ArcExtra, the session / speculator hooks):
every hook defaults OFF; with a hook on, the head's reading decides; without one, the rule stands. No network."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import arcmeta, config, director  # noqa: E402
from server.frame import Frame  # noqa: E402
from server.session import Session, open_mic_skip  # noqa: E402


def _rec(words, **kw):
    n = len(words)
    r = {"words": list(words), "conf": [0.95] * n, "ends": [0.3 + 0.3 * i for i in range(n)],
         "starts": [0.08 + 0.3 * i for i in range(n)], "revs": [0] * n, "age": [1] * n, "spans": [],
         "intent": [], "act": [], "cc_at": None, "cc_score": None, "gate": False, "act_question": False,
         "has_content": True, "content_idx": list(range(n)), "slots": [], "query": " ".join(words),
         "event": "Update", "eot": 0.1, "pause": 0.2, "t": 0.3 * n + 0.2, "arc_cc_p": None, "arc_cc_t": None,
         "arc_for_me": None}
    r.update(kw)
    return r


def test_v2_hooks_default_off():
    assert not (config.DIRECTOR_STREAMS or config.DIRECTOR_ARC_EAR or config.DIRECTOR_ADDRESSED or
                config.DIRECTOR_ROUTE or config.DIRECTOR_PII)


def test_step_feats2_groups_and_missing_streams():
    r = _rec(["what's", "the", "weather", "in", "denver"])
    x = director.step_feats2(r, None)
    names = [k for g in director.GROUP_ORDER for k in director.GROUPS[g]]
    assert x.shape == (len(names),) and np.isfinite(x).all()
    i = {k: n for n, k in enumerate(names)}
    assert x[i["x2_ok"]] == 0 and x[i["aw_ok"]] == 0           # no Arc extra / Arc ear stream: read as missing
    assert x[i["lane_card"]] == 1 and x[i["vert_weather"]] == 1   # the router's own plan for these words
    assert 0 < x[i["rate"]] <= 1 and x[i["last_dur"]] > 0
    only = director.step_feats2(r, None, groups=["pros", "ear"])
    assert only.shape == (len(director.GROUPS["pros"]) + len(director.GROUPS["ear"]),)


def test_step_feats2_reads_arc_streams():
    r = _rec(["is", "it", "going", "to", "rain"],
             x2={"speech": 0.9, "ovl": 0.4, "chg": 0.7, "nspk": 2, "spk": 1, "lvl": -20.0, "lvl_ref": -22.0,
                 "lvl3": -21.0, "utt": {"q": 0.8, "dd": 0.7}},
             aw=[["is", 0.08, 0.3], ["it", 0.38, 0.6], ["going", 0.68, 0.9], ["to", 0.98, 1.2], ["rain", 1.28, 1.5],
                 ["today", 1.6, 1.9]], af=[])
    names = [k for g in director.GROUP_ORDER for k in director.GROUPS[g]]
    i = {k: n for n, k in enumerate(names)}
    x = director.step_feats2(r, None)
    assert x[i["x2_ok"]] == 1 and x[i["x_ovl"]] == pytest.approx(0.4) and x[i["x_spk_other"]] == 1
    assert x[i["u_q"]] == pytest.approx(0.8)
    assert x[i["aw_ok"]] == 1 and x[i["aw_last_in"]] == 1 and x[i["aw_more"]] == 1 and x[i["aw_ahead"]] > 0


def test_word_feats2_marks_rule_hidden_house_number():
    words = ["weather", "at", "1420", "maple", "street", "boulder", "colorado"]
    r = _rec(words, spans=[{"type": "safety.pii.street_address", "family": "pii", "start": 2, "end": 7,
                            "score": 0.9}])
    wf = director.word_feats2(r)
    assert wf.shape == (director.W, director.WF2)
    hidden = [words[j] for j in range(len(words)) if wf[j, 1] > 0]
    assert hidden == ["1420"]                                    # the house number only; colorado stays


def test_arc_extra_state_from_engine_ring():
    class Tab:
        idx = {"meta.speech": 0, "meta.overlap": 1, "speaker.change": 2, "discourse.act.statement": 3}

    class R:
        cols = {"frame": [0, 1], "ev_at": [2], "utt": [3]}

    rr, tt = R(), Tab()

    class Eng:
        R, tab = rr, tt
        speakers = {0: {}, 1: {}}
        last_spk = 1
        utt_p = np.array([0.7], np.float32)
        last = {"frame": np.array([0.9, 0.3], np.float32), "ev_at": np.array([0.6], np.float32)}

    class S:
        eng = Eng()
    x = arcmeta.ArcExtra()
    for _ in range(3):
        x.step(S(), np.full(1600, 0.01, np.float32))
    st = x.state()
    assert st["speech"] == pytest.approx(0.9) and st["ovl"] == pytest.approx(0.3) and st["chg"] == pytest.approx(0.6)
    assert st["nspk"] == 2 and st["spk"] == 1 and st["utt"]["st"] == pytest.approx(0.7)
    assert st["lvl"] == pytest.approx(-40.0, abs=0.1)


class _M:
    thr = {"director_addr": 0.5, "director_pii": 0.5}


def _out(words, addr=0.9, route=(0.0, 0.0, 1.0, 0.0, 0.0), pii=None):
    n = len(words)
    p = np.zeros(director.W, np.float32)
    for j, v in enumerate(pii or []):
        p[director.W - n + j if n < director.W else j] = v
    k = min(n, director.W)
    pw = np.zeros(director.W, np.float32)
    pw[:k] = (pii or [0.0] * n)[-k:]
    return {"eot": 0.5, "fire": 0.5, "draft": 0.5, "keep": np.zeros(director.W), "addr": addr,
            "route": np.array(route, np.float32), "pii": pw, "n": n, "words": list(words)[-director.W:]}


def _frame(words, acts=(), intent="", p=0.0):
    f = Frame(words=list(words))
    f.acts, f.intent, f.intent_p, f.act_question = tuple(acts), intent, p, "question" in acts
    return f


def test_open_mic_gate_uses_addressed_head_only_when_on(monkeypatch):
    monkeypatch.setattr(director, "_model", _M())
    s = Session(lambda e: None, mode="headstart")
    try:
        stmt = _frame("i like to eat pancakes".split(), ("statement",), "intent.report_inform", 0.6)
        ask = _frame("what time does the game start".split(), ("question",), "discourse.intent.qa_factoid", 0.8)
        # off: the rule
        assert s._not_request(stmt, _out(stmt.words, addr=0.99)) == open_mic_skip(stmt) == "statement"
        monkeypatch.setattr(config, "DIRECTOR_ADDRESSED", True)
        # on: the head decides (a question asked of another person: not addressed; a statement it reads as a request)
        assert s._not_request(ask, _out(ask.words, addr=0.1)) == "not_addressed"
        assert s._not_request(stmt, _out(stmt.words, addr=0.9)) is None
        # the head read other words (stale): the rule stands
        assert s._not_request(stmt, _out(["something", "else"], addr=0.9)) == "statement"
        monkeypatch.setattr(config, "DIRECTOR_ADDR_THR", 0.95)
        assert s._not_request(stmt, _out(stmt.words, addr=0.9)) == "not_addressed"
    finally:
        s.close()


def test_apply_pii_replaces_rule_only_on_matching_words(monkeypatch):
    monkeypatch.setattr(director, "_model", _M())
    words = ["weather", "at", "1420", "maple", "street", "boulder", "colorado"]
    f = Frame(words=list(words))
    f.redact_tok = frozenset({2, 6})
    assert director.apply_pii(f, _out(words, pii=[0, 0, 0.9, 0.1, 0, 0, 0.2]))
    assert f.redact_tok == frozenset({2})
    c = Frame(words=list(words), spans=[{"type": "entity.mention.place", "family": "entity", "start": 5, "end": 7}])
    assert director.apply_pii(c, _out(words, pii=[0, 0, 0.9, 0, 0, 0.9, 0.9]))
    assert c.redact_tok == frozenset({2})                          # a tagged place is never hidden
    g = Frame(words=list(words))
    g.redact_tok = frozenset({2})
    assert not director.apply_pii(g, _out(words[:-1], pii=[0.9] * 6)) and g.redact_tok == frozenset({2})
    assert not director.apply_pii(g, None)


def test_route_hook_holds_or_releases_paid_speculation(monkeypatch):
    from server import lanes, speculator
    f = Frame(words="what's the weather in".split())
    f.director = _out(f.words, route=(0.95, 0.0, 0.05, 0.0, 0.0))
    plan = lanes.RoutePlan(["web"])
    plan.lane, plan.conf = "web", 0.9                              # the rules: a confident web question (no hold)
    sp = speculator.Speculator(lambda e: None, "headstart", 0.0)
    try:
        assert sp._route_hold(f, plan) is False                    # hook off: the rules
        monkeypatch.setattr(config, "DIRECTOR", True)
        monkeypatch.setattr(config, "DIRECTOR_ROUTE", True)
        assert sp._route_hold(f, plan) is True                     # head: a card is coming -> no paid call
        f.director = _out(f.words, route=(0.02, 0.03, 0.95, 0.0, 0.0))
        weak = lanes.RoutePlan(["web"])
        weak.lane, weak.conf = "web", 0.1
        assert sp._route_hold(f, weak) is False                    # head: web -> release now
        f.director = _out(f.words, route=(0.5, 0.0, 0.5, 0.0, 0.0))
        assert sp._route_hold(f, plan) is False                    # not confident: the rules
    finally:
        sp.close()


def test_v2_model_numpy_heads_if_shipped():
    p = director.MODEL_DIR / "director.json"
    if not p.exists() or not json.loads(p.read_text()).get("v2"):
        pytest.skip("no v2 model shipped")
    m = director.Model()
    st = None
    F = m.meta["F"]
    out, st = m.step(st, np.zeros(F, np.float32), np.zeros(director.K, np.int64), np.zeros(director.W, np.int64),
                     np.zeros((director.W, 4), np.float32), np.zeros((director.W, director.WF2), np.float32))
    assert 0 <= out["addr"] <= 1 and abs(float(out["route"].sum()) - 1) < 1e-5 and out["pii"].shape == (director.W,)
