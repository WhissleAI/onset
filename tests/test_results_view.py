"""Results view + notices (server/scorecard.py, server/scorecard.json, server/notices.py, web/scorecard.js).
- every number on the Results slide is quoted, verbatim, from the doc it cites (so the slide can't drift from the data);
- /api/scorecard serves it with what this server runs now; /api/results carries provenance and no machine paths;
- notices: hidden when all is well, one plain line + an action when something is wrong, never "degraded mode".
Run: .venv/bin/python -m pytest -q tests/test_results_view.py
"""
import json
import re

import pytest
from fastapi.testclient import TestClient

from server import answer, config, lanes, notices, scorecard
from server import app as appmod

ROOT = config.ROOT


def test_scorecard_shape_six_problems_in_order():
    d = scorecard.load()
    assert [p["n"] for p in d["problems"]] == ["01", "02", "03", "04", "05", "06"]
    for item in d["problems"] + d["headline"]:
        assert item["big"] and item["source"]["doc"] and item["source"]["section"] and item["source"]["date"]
    for p in d["problems"]:
        assert p["problem"] and p["unit"] and p["vs"] and p["answer"]


def _norm(s: str) -> str:
    return s.replace("**", "")


@pytest.mark.parametrize("item", scorecard.load()["problems"] + scorecard.load()["headline"], ids=lambda i: i.get("id") or i["n"])
def test_every_number_is_quoted_from_its_source(item):
    """The cited doc contains the quote verbatim, and the big number appears in that quote."""
    src = item["source"]
    text = (ROOT / src["doc"]).read_text()
    assert src["quote"] in text, f"{src['doc']} no longer says {src['quote']!r}"
    big = item["big"].replace("→", " ").replace("%", "").replace(" s", "")
    nums = re.findall(r"\d+(?:[./]\d+)?", big)
    q = _norm(src["quote"])
    for n in nums:
        # 0.61 s on the slide = 610 ms in the cost table
        alt = str(int(round(float(n) * 1000))) if item.get("id") == "speed" else n
        assert n in q or alt in q, f"{n} not in the quote {q!r}"
    # each comparison number on a card is in its source doc too
    for n in re.findall(r"\d+/\d+|\d+\.\d+%?", item.get("vs", "") + " " + item.get("detail", "")):
        assert n.rstrip("%") in text, f"{n} ({item.get('vs') or item.get('detail')}) not found in {src['doc']}"


def test_headline_is_real_tavily_with_sets():
    d = scorecard.load()
    assert d["headline_label"] == "real Tavily, 100 spoken questions, 2026-11-14"
    assert [h["id"] for h in d["headline"]] == ["speed", "paid", "correct"]
    assert all(h["source"]["section"] == "10. On real Tavily" for h in d["headline"])
    text = (ROOT / "docs" / "RETRIEVAL.md").read_text()
    for x in d["headline_sets"]:
        q = _norm(x["source"]["quote"])
        assert x["source"]["quote"] in text
        for v in (x["gain"].split()[0], x["paid"], x["paid_vs"], str(x["n"])):
            assert v in q, (v, q)
    f = next(h for h in d["headline"] if h["id"] == "paid")["fewer"]
    assert f"Total: {f['onset_credits']} credits for Onset" in text and f"{f['plain_credits']} for plain search" in text
    assert round((1 - f["onset_credits"] / f["plain_credits"]) * 100) == f["pct"] and f"{f['pct']}% fewer" in d["headline"][1]["detail"]


def test_no_number_twice_and_every_card_says_where():
    d = scorecard.load()
    heads = {h["big"] for h in d["headline"]}
    assert not heads & {p["big"] for p in d["problems"]}          # e.g. 0.369 was on both
    assert all(p.get("lane") for p in d["problems"])
    p4 = d["problems"][3]
    assert p4["lane"] == "PERSONA flag, small set" and p4["source"]["doc"] == "docs/PERSONALIZATION.md"
    assert "8/9" in p4["big"] and "6/9" in p4["vs"] and "11/12" in p4["vs"] and "7/12" in p4["vs"]
    css = (ROOT / "web" / "scorecard.css").read_text()
    vs = css[css.index(".sc-vs {"):].split("}")[0]
    assert "var(--hs)" not in vs                                    # the comparison number is muted, not the accent


def test_scorecard_json_has_no_paths_or_keys():
    raw = (ROOT / "server" / "scorecard.json").read_text()
    assert "/Users/" not in raw and "sk-" not in raw and "tvly-" not in raw


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod.app.router, "on_startup", [])       # no model warm-up in a unit test
    monkeypatch.setattr(lanes, "_index", lanes.LocalIndex.__new__(lanes.LocalIndex))
    lanes._index.docs = []
    with TestClient(appmod.app) as c:
        yield c


def test_api_scorecard(client):
    d = client.get("/api/scorecard").json()
    assert len(d["problems"]) == 6 and len(d["headline"]) == 3
    assert d["now"]["cost_mode"] in ("speed", "parity") and d["now"]["web_backend"] == config.WEB_BACKEND


def test_api_results_empty_without_a_local_run(client, monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "ROOT", tmp_path)
    assert client.get("/api/results").json() == {}


def test_api_results_provenance_and_no_config(client, monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    old = {"created": "2026-11-12T21:51:58-0800",
           "config": {"web_backend": "tavily", "arc_dir": "/opt/somewhere/models/arc"},
           "summary": {"headstart": {"p50_ready_ms": 1320}},
           "rows": [{"id": "q1", "mode": "headstart", "ready_ms": 1200, "trace": [1, 2]},
                    {"id": "q1", "mode": "baseline", "ready_ms": 1700}]}
    (tmp_path / "runs" / "ab_latest.json").write_text(json.dumps(old))
    monkeypatch.setattr(appmod, "ROOT", tmp_path)
    d = client.get("/api/results").json()
    assert "config" not in d and "/opt/somewhere" not in json.dumps(d)
    assert "trace" not in d["rows"][0]
    pv = d["provenance"]
    assert pv["created"].startswith("2026-11-12") and pv["web_backend"] == "tavily"
    assert pv["cost_mode"] is None and pv["current"] is False          # recorded before the Parity budget
    # a run on this server's settings is current
    rows = [{**r, "cost_mode": "parity", "web_backend": "wikipedia"} for r in old["rows"]]
    (tmp_path / "runs" / "ab_latest.json").write_text(json.dumps({**old, "config": {"web_backend": "wikipedia"}, "rows": rows}))
    monkeypatch.setattr(config, "WEB_BACKEND", "wikipedia")
    monkeypatch.setattr(config, "COST_MODE", "parity", raising=False)
    from server import cost
    if cost.profile().name == "parity":
        assert client.get("/api/results").json()["provenance"]["current"] is True


# ----------------------------------------------------------------------------- notices
LIVE_WEB = {"status": "live", "cache_entries": 3}
LIVE_ANS = {"status": "live", "last_error": None}


def test_notices_empty_when_all_is_well():
    assert notices.build(LIVE_WEB, LIVE_ANS) == []
    assert notices.build({"status": "unknown", "cache_entries": 0}, {"status": "unknown"}) == []


def test_notice_missing_key_has_settings_action(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    n = notices.build(LIVE_WEB, {"status": "down", "last_error": "TypeError: Could not resolve authentication method"})
    assert len(n) == 1 and n[0]["id"] == "answer-key"
    assert n[0]["text"] == "Add an Anthropic key for full answers"
    assert n[0]["action"] == "settings" and n[0]["action_label"] == "Open Settings" and ".env" in n[0]["hint"]


def test_notice_bad_key_vs_network(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    bad = notices.build(LIVE_WEB, {"status": "down", "last_error": "AuthenticationError: 401 invalid x-api-key"})
    assert bad[0]["id"] == "answer-key"
    net = notices.build(LIVE_WEB, {"status": "down", "last_error": "APIConnectionError: Connection error."})
    assert net[0]["id"] == "answer-down" and "action" not in net[0]


def test_notice_web():
    assert notices.build({"status": "down", "cache_entries": 5}, LIVE_ANS)[0]["text"] == "Live web unavailable · using saved results"
    assert "no sources" in notices.build({"status": "down", "cache_entries": 0}, LIVE_ANS)[0]["text"]
    assert notices.build({"status": "cache", "cache_entries": 5}, LIVE_ANS)[0]["text"] == "Live web off · using saved results"


def test_notice_copy_is_plain():
    allv = notices.build({"status": "down", "cache_entries": 1}, {"status": "down", "last_error": "api_key"})
    for n in allv:
        assert "degraded" not in n["text"].lower() and "mode" not in n["text"].lower() and len(n["text"]) <= 70


def test_health_and_ready_carry_notices(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    answer._down("APIConnectionError: offline")
    h = client.get("/api/health").json()
    assert [n["id"] for n in h["notices"]] == ["answer-down"] and "answer" in h["degraded"]
    with client.websocket_connect("/ws") as ws:
        r = ws.receive_json()
        assert r["t"] == "ready" and r["notices"][0]["id"] == "answer-down"
    answer.reset_status()
    assert client.get("/api/health").json()["notices"] == []


# ----------------------------------------------------------------------------- web wiring (static checks)
def test_web_wiring():
    html = (ROOT / "web" / "index.html").read_text()
    assert 'id="scOpen"' in html and "scorecard.js" in html and "scorecard.css" in html
    assert ">Sources</button>" in html                     # the race toggle is not also called "Results"
    assert html.count('data-k="route"') == 2
    assert "Search starts at the third word" not in html
    js = (ROOT / "web" / "app.js").read_text()
    assert "Degraded mode" not in js and "runs/ab_latest.json</code>" not in js
    sc = (ROOT / "web" / "scorecard.js").read_text()
    assert "/api/scorecard" in sc and "Escape" in sc and "savePNG" in sc
