"""Continuous training loop (scripts/train_cycle.sh, eval/train/): promotion gates, versioning + rollback, PII
redaction of collected rows, the quiet flag / swap guard, and the Results view's "Model versions" line.
No models load here.
Run: .venv/bin/python -m pytest -q tests/test_train_loop.py
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval/train"))

import collect  # noqa: E402
import common as C  # noqa: E402
import gates as G  # noqa: E402
import registry as RG  # noqa: E402
import results as RS  # noqa: E402
import synth  # noqa: E402
from server import config, scorecard, trainlog  # noqa: E402


# ----------------------------------------------------------------------------- gates
def test_calib_gate_needs_a_real_brier_gain_and_no_ece_or_auc_loss():
    cur = {"test": {"brier": 0.161, "ece": 0.034, "auc": 0.72}}
    assert G.calib(cur, {"test": {"brier": 0.158, "ece": 0.030, "auc": 0.73}})["promote"]
    assert not G.calib(cur, {"test": {"brier": 0.160, "ece": 0.030, "auc": 0.73}})["promote"]       # < 0.002 gain
    v = G.calib(cur, {"test": {"brier": 0.150, "ece": 0.060, "auc": 0.73}})                          # ECE worse
    assert not v["promote"] and "ECE" in v["fails"][0]
    assert not G.calib(cur, {"test": {"brier": 0.150, "ece": 0.03, "auc": 0.70}})["promote"]        # AUC worse
    # live set: Brier may not get worse even when test improves
    cur2 = {**cur, "live": {"brier": 0.037, "ece": 0.05, "auc": 0.8}}
    assert not G.calib(cur2, {"test": {"brier": 0.15, "ece": 0.03, "auc": 0.72},
                              "live": {"brier": 0.05, "ece": 0.05, "auc": 0.8}})["promote"]


def test_kdir_gate_paid_down_without_regret_accuracy_or_p50_loss():
    base = {"paid_per_q": 0.490, "regret_rate": 0.0075, "acc_all": 0.80, "p50_ms": 186.0}
    cur = {"test": dict(base), "old_test": {**base, "paid_per_q": 0.341}}
    good = {"test": {**base, "paid_per_q": 0.480}, "old_test": {**base, "paid_per_q": 0.335}}
    assert G.kdir(cur, good)["promote"]
    assert not G.kdir(cur, {"test": {**good["test"], "regret_rate": 0.01}, "old_test": good["old_test"]})["promote"]
    assert not G.kdir(cur, {"test": {**good["test"], "acc_all": 0.79}, "old_test": good["old_test"]})["promote"]
    assert not G.kdir(cur, {"test": {**good["test"], "p50_ms": 190.0}, "old_test": good["old_test"]})["promote"]
    tiny = {"test": {**base, "paid_per_q": 0.489}, "old_test": {**base, "paid_per_q": 0.340}}
    assert not G.kdir(cur, tiny)["promote"]                               # within noise: keep the current model
    assert not G.kdir(cur, {"test": good["test"]})["promote"]             # a stream not measured never promotes


def test_director_gate():
    s = {"p50": 293.0, "p50_at10": 525.5, "p50_at20": 413.0, "premature": 32.2}
    cur = {"test": {**s, "eot_bce": 0.2483}, "test_v2": {**s, "eot_bce": 0.30}, "heldout80": dict(s),
           "addr": {"auc": 0.90}}
    better = {"test": {**s, "eot_bce": 0.2480}, "test_v2": {**s, "eot_bce": 0.297}, "heldout80": dict(s),
              "addr": {"auc": 0.90}}
    assert G.director(cur, better)["promote"]
    slower = json.loads(json.dumps(better))
    slower["heldout80"]["p50_at10"] = 530.0
    assert not G.director(cur, slower)["promote"]
    worse_test = json.loads(json.dumps(better))
    worse_test["test"]["eot_bce"] = 0.2500
    assert not G.director(cur, worse_test)["promote"]
    addr = json.loads(json.dumps(better))
    addr["addr"]["auc"] = 0.88
    assert not G.director(cur, addr)["promote"]
    # end-to-end: an accuracy loss on the 58 set blocks promotion
    e2e = {**better, "e2e": {"q58": {"acc": 0.98, "p50_ms": 1400}}}
    assert not G.director({**cur, "e2e": {"q58": {"acc": 1.0, "p50_ms": 1400}}}, e2e)["promote"]


def test_ttl_gate_and_no_regression():
    cur = {"heldout": {"loglik_per_obs": -0.30}, "test": {"paid_per_q": 0.5, "regret_rate": 0.01, "acc_all": 0.8,
                                                        "p50_ms": 180}}
    assert G.ttl(cur, {"heldout": {"loglik_per_obs": -0.29}, "test": dict(cur["test"])})["promote"]
    assert not G.ttl(cur, {"heldout": {"loglik_per_obs": -0.29}, "test": {**cur["test"], "paid_per_q": 0.51}})["promote"]
    assert not G.ttl(cur, {"heldout": {"loglik_per_obs": -0.30}, "test": dict(cur["test"])})["promote"]
    assert G.no_regression({"a": {"acc": 1, "p50_ms": 100}}, {"a": {"acc": 1, "p50_ms": 120}}, ["a"], p50_tol_ms=25) == []
    assert G.regressions_ok({"pytest": {"ok": False, "why": "exit 1"}}) == ["regression pytest: exit 1"]


# ----------------------------------------------------------------------------- versions + rollback
def test_registry_promote_and_rollback(tmp_path):
    repo = tmp_path / "repo"
    (repo / "server").mkdir(parents=True)
    live = repo / "server/calib_model.json"
    live.write_text('{"v": "shipped"}')
    reg = RG.Registry(tmp_path / "models", repo)
    assert reg.bootstrap("calib") == "v000"
    cand = tmp_path / "cand"
    cand.mkdir()
    (cand / "calib_model.json").write_text('{"v": "c1"}')
    v = reg.promote("calib", cand, {"cycle": 1})
    assert v == "v001" and reg.current("calib") == "v001" and json.loads(live.read_text())["v"] == "c1"
    assert reg.meta("calib")["previous"] == "v000"
    (cand / "calib_model.json").write_text('{"v": "c2"}')
    assert reg.promote("calib", cand, {"cycle": 2}) == "v002"
    assert reg.rollback("calib") == "v001" and json.loads(live.read_text())["v"] == "c1"
    assert reg.rollback("calib", "v000") == "v000" and json.loads(live.read_text())["v"] == "shipped"
    hist = [json.loads(x)["event"] for x in (tmp_path / "models/calib/history.jsonl").read_text().splitlines()]
    assert hist == ["bootstrap", "promote", "promote", "rollback", "rollback"]
    # a candidate missing a file is refused and nothing changes
    with pytest.raises(FileNotFoundError):
        reg.promote("director", cand, {})
    assert reg.current("calib") == "v000"


def test_registry_missing_model_files_are_skipped(tmp_path):
    reg = RG.Registry(tmp_path / "models", tmp_path)
    assert reg.bootstrap("kdir") is None and reg.versions("kdir") == []


# ----------------------------------------------------------------------------- PII + collection
def test_redact_masks_numbers_emails_handles_and_birth_years(monkeypatch):
    monkeypatch.setattr(config, "PII_YEARS", True)
    r = trainlog.redact("call me at 415 555 0199 or mail jane.doe@gmail.com i was born in nineteen eighty five @janedoe")
    assert "415" not in r and "0199" not in r and "jane.doe" not in r and "eighty" not in r and "janedoe" not in r
    assert "<pii>" in r and "<email>" in r and "<handle>" in r and "born" in r
    r2 = trainlog.redact("my card is four one one one two two two two")
    assert "two" not in r2.split()[-1] and r2.count("<pii>") == 8
    # ordinary questions are untouched; the tagger's own redacted indices are honoured
    assert trainlog.redact("who founded nvidia in 1993") == "who founded nvidia in 1993"
    assert trainlog.redact("i live at 12 elm street", hide_idx=[3]) == "i live at <pii> elm street"


def test_row_keeps_metrics_drops_answer_and_hashes_client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "TRAIN_COLLECT", True)
    trainlog.bind(7, "client-abc", "mic")
    e = {"t": "metric", "question": "my phone is 4155550199 what time is it", "answer": "It is noon.", "ready_ms": 812,
         "path": "card", "conf_feats": {"rr_top": 0.7}, "verify": {"status": "supported", "label": "x", "found": 2},
         "audio": "AAAA", "q": 1}
    trainlog.record(e, "headstart", 7)
    rows = [json.loads(x) for x in (tmp_path / "train/rows.jsonl").read_text().splitlines()]
    r = rows[0]
    assert r["ready_ms"] == 812 and r["conf_feats"] == {"rr_top": 0.7} and r["answer_words"] == 3
    assert "4155550199" not in json.dumps(r) and "answer" not in r and "audio" not in r and "label" not in r["verify"]
    assert r["cid"] != "client-abc" and len(r["cid"]) == 16 and "client-abc" not in json.dumps(r)
    trainlog.record({"t": "partial", "words": "x"}, "headstart", 7)                 # only metrics are kept
    monkeypatch.setattr(config, "TRAIN_COLLECT", False)
    trainlog.record(e, "headstart", 7)
    assert len((tmp_path / "train/rows.jsonl").read_text().splitlines()) == 1


def test_collect_scrub_and_log_parse():
    s = collect.scrub({"text": "card 4111 1111 1111 1111", "answer": "ok sure", "file": "/x/y.wav",
                       "client": "abc", "ready_ms": 5, "trace": [{"t": "spec", "query": "zip 94110 12345"}]})
    assert "4111" not in s["text"] and s["answer_words"] == 2 and "file" not in s and s["cid"] != "abc"
    assert "94110" not in s["trace"][0]["query"]
    lines = ["[req 12:06:35.960] run 1 headstart q1 endpoint by=semantic speech_end=2000ms",
             "[req 12:06:37.001] run 1 headstart q1 done ready=1438ms path=speculative-hit lane=facts | "
             "'who founded the company nvidia'"]
    r = collect.parse_log(lines, "2026-11-14")[0]
    assert r["ready_ms"] == 1438 and r["lane"] == "facts" and r["endpoint_by"] == "semantic"
    assert r["question"] == "who founded the company nvidia"
    assert collect.row_hash(r) == collect.row_hash({**r, "harvested": "later"})


# ----------------------------------------------------------------------------- the user first
def test_quiet_flag_and_swap_block_heavy_steps(monkeypatch, tmp_path):
    flag = tmp_path / ".quiet"
    monkeypatch.setattr(C, "QUIET_FLAG", flag)
    monkeypatch.setattr(C, "STOP_FLAG", tmp_path / "STOP")
    monkeypatch.setattr(C, "TRAIN", tmp_path)
    assert C.may_start_heavy(swap_gb=1.0)[0]
    flag.write_text("")
    ok, why = C.may_start_heavy(swap_gb=1.0)
    assert not ok and "quiet" in why
    flag.unlink()
    ok, why = C.may_start_heavy(swap_gb=17.5)
    assert not ok and "swap" in why
    assert C.swap_used_gb("vm.swapusage: total = 21504.00M  used = 20480.00M  free = 1024.00M") == 20.0
    # run_heavy never starts a step while quiet (and gives up when stop is requested)
    flag.write_text("")
    (tmp_path / "STOP").write_text("")
    assert C.run_heavy(["true"], str(tmp_path / "x.log")) == -1
    assert not (tmp_path / "x.log").exists()


def test_wait_for_slot_resumes_when_quiet_ends(monkeypatch, tmp_path):
    flag = tmp_path / ".quiet"
    flag.write_text("")
    monkeypatch.setattr(C, "QUIET_FLAG", flag)
    monkeypatch.setattr(C, "STOP_FLAG", tmp_path / "STOP")
    monkeypatch.setattr(C, "TRAIN", tmp_path)
    monkeypatch.setattr(C, "swap_used_gb", lambda text=None: 1.0)
    calls = []

    def fake_sleep(s):
        calls.append(s)
        flag.unlink()                     # the lead removes the flag
    assert C.wait_for_slot(poll_s=60, sleep=fake_sleep) and calls == [60]


# ----------------------------------------------------------------------------- synthetic data
def test_synthetic_sessions_cover_the_failure_classes_with_gold():
    seed = json.loads(synth.SEEDS.read_text())
    cache = {}

    def fake(q):
        if "P582" in q and "ORDER BY" in q:
            return [{"aLabel": {"value": "Joe Biden"}}]
        return [{"aLabel": {"value": "Ada Lovelace"}}]
    import random
    ss = synth.build_sessions(seed, len(seed["sessions"]), random.Random(1), cache, None, fetch=fake)
    classes = {s["cls"] for s in ss}
    assert classes == {"topic_switch", "elliptical", "self_correction", "open_mic", "multi_part", "live_data"}
    ell = next(s for s in ss if s["cls"] == "elliptical" and s["turns"][1]["text"] == "you know the old one")
    assert ell["turns"][1]["gold"]["followup"] and ell["turns"][1]["gold"]["expect"] == ["biden"]
    sw = next(s for s in ss if s["cls"] == "topic_switch" and "restaurants" in s["turns"][1]["text"])
    assert sw["turns"][1]["gold"]["never"] and sw["turns"][1]["gold"]["never"][0] in sw["turns"][0]["text"]
    om = next(s for s in ss if s["cls"] == "open_mic")
    assert [t["gold"]["addressed"] for t in om["turns"]] == [False, True, False]
    clips = synth.clips_of(ss, 3)
    assert all(c["split"] == "train" for c in clips) and any(c["followup_of"] for c in clips)
    w = synth.flux_like_words("who is the president of france", 180)
    assert len(w) == 6 and all(a["end"] <= b["start"] for a, b in zip(w, w[1:]))
    raw = json.dumps(seed)
    assert "/Users/" not in raw and "@" not in raw


# ----------------------------------------------------------------------------- results
def test_models_line_is_quoted_from_its_doc():
    m = scorecard.load()["models"]
    src = m["source"]
    assert src["quote"] in (ROOT / src["doc"]).read_text()
    for k, v in m["by_model"].items():
        assert f"{v['name']} {v['version']}" in src["quote"]
    assert m["line"].startswith("Model versions · last trained ")


def test_with_models_rewrites_only_the_models_block():
    text = (ROOT / "server/scorecard.json").read_text()
    new = RS.with_models(text, {"line": "x", "by_model": {}, "source": {}})
    assert json.loads(new)["models"]["line"] == "x"
    i = text.find(',\n  "models": ')
    assert new[:i] == text[:i]
    assert RS.with_models(new, scorecard.load()["models"]) == text


def test_web_shows_model_versions():
    js = (ROOT / "web/scorecard.js").read_text()
    assert "modelsLine(d.models)" in js and "sc-models" in js
