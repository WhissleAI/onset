"""Answer calibration (server/calibrate.py): features, the fitted model, reasons, router actions, flags."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from server import calibrate, lanes


class F(SimpleNamespace):
    def entity_query(self):
        return self.ent


def frame(**kw):
    base = dict(words=["who", "is", "the", "ceo", "of", "nvidia"], content_idx=(3, 5), sconf=(), wconf=(),
                phrases=("nvidia",), ent="nvidia", intent="discourse.intent.qa_factoid", intent_p=0.9,
                corrected=False, followup_of="", carried="", inline="", resolved=(), resolved_text="who is the ceo of nvidia")
    base.update(kw)
    return F(**base)


def doc(i, text="Nvidia is led by its CEO.", source="wikipedia", score=0.0, **meta):
    return lanes.Doc(id=f"d{i}", title=f"T{i}", text=text, source=source, score=score, meta=meta)


@pytest.fixture(autouse=True)
def _model():
    yield
    calibrate.reload(None)


def tiny_model():
    return {"bands": {"low": 0.5, "high": 0.8},
            "pre": {"features": ["bias", "rr_top", "facts"], "w": [0.0, 4.0, 1.0], "mu": [0, 0.6, 0.2], "sd": [1, 0.1, 0.4],
                    "fill": [0, 0.6, 0.2]},
            "post": {"features": ["bias", "rr_top", "v_agree", "non_answer"], "w": [0.5, 2.0, 1.5, -3.0],
                     "mu": [0, 0.6, 0.3, 0.1], "sd": [1, 0.1, 0.45, 0.3], "fill": [0, 0.6, 0.3, 0.1],
                     "iso": {"x": [0.0, 0.5, 1.0], "y": [0.05, 0.5, 0.95]}}}


def test_features_keep_both_scores_and_signals():
    ranked = [(doc(1, source="tavily", score=0.91), 0.82), (doc(2, source="tavily", score=0.40), 0.70),
              (doc(3, text="unrelated"), 0.55)]
    x = calibrate.features(frame(), ranked, {"lane": "web", "conf": 0.6, "tier": "web"})
    assert x["src_top"] == pytest.approx(0.91)
    assert x["src_margin"] == pytest.approx(0.91 - 0.40)
    assert x["rr_top"] == pytest.approx(0.82) and x["rr_margin"] == pytest.approx(0.12)
    assert x["agree_subj"] == 2.0                       # two of the top three name the subject
    assert x["lane_web"] == 1.0 and x["facts"] == 0.0
    assert "v_agree" not in x                           # pre stage: no answer features


def test_features_wikipedia_has_no_source_score_and_keystone_flags():
    ranked = [(doc(1, tier="index", index_scope="shared", as_of="2026-11-14 09:00"), 0.8)]
    x = calibrate.features(frame(), ranked, {"lane": "facts", "conf": 0.85}, v={"status": "agree", "n_claims": 2,
                                                                                 "found": 2}, answer="Jensen Huang [1].")
    assert x["src_top"] is None
    assert x["ks_hit"] == 1.0 and x["ks_shared"] == 1.0 and x["ks_age_h"] is not None
    assert x["v_agree"] == 1.0 and x["v_found_frac"] == 1.0 and x["n_cites"] == 1.0 and x["non_answer"] == 0.0


def test_key_word_is_the_least_sure_content_word():
    f = frame(sconf=(0.99, 0.99, 0.99, 0.97, 0.99, 0.62))
    assert calibrate.key_word(f) == (0.62, "nvidia")
    assert calibrate.key_word(frame()) == (None, "")


def test_no_model_means_no_conf():
    calibrate.reload({})
    assert calibrate.assess({"rr_top": 0.9}, "post") == {}


def test_model_is_monotone_and_bounded():
    calibrate.reload(tiny_model())
    lo = calibrate.raw({"rr_top": 0.5}, "pre")
    hi = calibrate.raw({"rr_top": 0.8}, "pre")
    assert 0.01 <= lo < hi <= 0.99
    miss = calibrate.raw({}, "pre")                     # missing -> the training mean: neutral
    assert lo < miss < hi


def test_reasons_and_bands():
    calibrate.reload(tiny_model())
    good = {"rr_top": 0.9, "v_agree": 1.0, "agree_subj": 2.0, "non_answer": 0.0}
    a = calibrate.assess(good, "post")
    assert a["band"] == "high" and a["reason"] == "2 sources agree"
    bad = {"rr_top": 0.5, "v_agree": 0.0, "non_answer": 0.0, "key_conf": 0.4, "_key_word": "Kress", "v_unverified": 1.0}
    b = calibrate.assess(bad, "post")
    assert b["band"] == "low" and b["reason"] == "heard 'Kress' unclearly"
    assert calibrate.reason({"rr_top": 0.5}, 0.1) == "check this"
    assert calibrate.label(0.9) == "High confidence" and calibrate.label(0.2) == "Low confidence"


def test_route_actions():
    calibrate.reload(tiny_model())
    assert calibrate.route_action(0.9, True, True) == ""
    assert calibrate.route_action(0.6, True, True) == "free"
    assert calibrate.route_action(0.2, True, True) == "paid"
    assert calibrate.route_action(0.2, False, True) == "free"      # parity budget spent: never a paid call
    assert calibrate.route_action(0.2, True, False) == "free"


def test_alt_query_never_repeats_a_search():
    f = frame()
    assert calibrate.alt_query(f, {"ceo nvidia"}) == "nvidia"
    assert calibrate.alt_query(f, {"ceo nvidia", "Nvidia"}) == "who is the ceo of nvidia"
    assert calibrate.alt_query(f, {"nvidia", "who is the ceo of nvidia"}) == ""


def test_published_date_flag(monkeypatch):
    plan = {"search_depth": "basic", "topic": "news", "max_results": 5}
    monkeypatch.setattr(calibrate, "CALIB_PUBDATE", False)
    assert "include_published_date" not in lanes.tavily_body("q", plan)
    monkeypatch.setattr(calibrate, "CALIB_PUBDATE", True)
    assert lanes.tavily_body("q", plan)["include_published_date"] is True


def test_stale_live_source_is_a_doubt():
    x = {"i_live": 1.0, "src_age_d": 30.0, "rr_top": 0.8}
    assert calibrate.reason(x, 0.3) == "sources may be out of date"


def test_shipped_model_loads():
    calibrate.reload(None)
    m = calibrate.model()
    assert m.get("pre") and m.get("post")
    p = calibrate.raw({"rr_top": 0.8, "v_agree": 1.0, "n_cites": 2.0, "ans_words": 20.0}, "post")
    assert 0.0 < p < 1.0
    for st in ("pre", "post"):
        assert len(m[st]["features"]) == len(m[st]["w"]) == len(m[st]["mu"]) == len(m[st]["sd"])


def test_free_docs_never_raise(monkeypatch):
    from server import recheck

    async def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(recheck, "_facts", boom)
    monkeypatch.setattr(recheck, "_leads", boom)
    assert asyncio.run(calibrate.free_docs("who is the ceo of nvidia", frame())) == []
