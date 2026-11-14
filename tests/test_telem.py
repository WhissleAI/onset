"""Telem lane (server/telem.py), failover chain (lanes._live), page fetch (server/pagefetch.py) and monitors
(server/telem_monitor.py), all against an in-process fake of router.telem.ai (no network, no spend)."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config, lanes, pagefetch, telem, telem_monitor  # noqa: E402

UUID_S = "11111111-2222-3333-4444-555555555555"
UUID_I = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _results(provider, urls):
    return [{"url": u, "title": f"{provider} {u}", "rank": i + 1, "summary": f"{provider} says about {u}"}
            for i, u in enumerate(urls)]


class FakeTelem:
    def __init__(self):
        self.calls = []
        self.fail_providers = set()
        self.status = 200
        self.tavily_status = 500
        self.changes = []

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "api.tavily.com" in url:
            self.calls.append(("tavily", None))
            return httpx.Response(self.tavily_status, json={"detail": {"error": "boom"}})
        body = json.loads(req.content or b"{}") if req.method == "POST" else {}
        self.calls.append((req.url.path, body, dict(req.headers)))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"code": "x", "message": "nope"}},
                                  headers={"retry-after": "5"})
        if req.url.path == "/v1/search":
            s = body["search"]
            provs = (s.get("providers") or {}).get("include") or ["exa", "brave", "tavily"]
            routing = None
            if s.get("auto_routing"):
                provs = provs[:2] if (s.get("providers") or {}).get("include") else ["exa", "brave"]
                routing = [{"providers": provs}]
            runs = []
            for p in provs:
                if p in self.fail_providers:
                    runs.append({"preprocessor_name": p, "status": "failed", "latency_ms": 900,
                                 "error": {"code": "upstream_timeout"}, "output_payload": None})
                    continue
                urls = ["https://a.example/1", f"https://{p}.example/2"]
                runs.append({"preprocessor_name": p, "status": "succeeded", "latency_ms": 400 + 100 * len(runs),
                             "output_payload": {"results": _results(p, urls), "warnings": []}})
            return httpx.Response(200, json={"status": "succeeded", "preprocessor_runs": runs, "routing": routing,
                                             "session_id": body.get("session_id") or UUID_S,
                                             "interaction_id": UUID_I})
        if req.url.path == "/v1/fetch":
            out = []
            for i, u in enumerate(body["urls"]):
                if "dead" in u:
                    fr = {"url": u, "status": "failed", "error": {"code": "bot_blocked"}}
                elif "missing" in u:
                    fr = {"url": u, "status": "succeeded", "http_status": 404, "content": "Not found",
                          "provider": "jina"}
                else:
                    fr = {"url": u, "status": "succeeded", "content": "The Brewers won the 2026 World Series in "
                          "five games.", "provider": "tavily", "latency_ms": 300}
                out.append({"batch_index": i, "output_payload": {"fetched_results": [fr]}})
            return httpx.Response(200, json={"status": "succeeded", "preprocessor_runs": out,
                                             "interaction_id": UUID_I, "session_id": UUID_S})
        if req.url.path == "/v1/monitors" and req.method == "POST":
            return httpx.Response(201, json={"id": "mon-1", "query": body["query"], "status": "active",
                                             "session_id": UUID_S, "agent_state": {"status": "compiling"}})
        if req.url.path.endswith("/agent/changes"):
            cur = int(req.url.params.get("cursor", 0))
            ch = [c for c in self.changes if c["seq"] > cur]
            nxt = max([c["seq"] for c in ch], default=cur)
            return httpx.Response(200, json={"changes": ch, "next_cursor": nxt, "watermark": nxt})
        if req.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    fk = FakeTelem()
    lanes.use_transport(httpx.MockTransport(fk.handler))
    lanes.reset_process()
    telem.reset()
    monkeypatch.setattr(telem, "TELEM_API_KEY", "tlm_test")
    monkeypatch.setattr(telem, "TELEM_MODE", "family")
    monkeypatch.setattr(telem, "WEB_FAILOVER", "")
    monkeypatch.setattr(telem, "TELEM_PROCESS_USD", 3.0)
    monkeypatch.setattr(config, "WEB_BACKEND", "telem")
    monkeypatch.setattr(config, "WEB_CACHE_FALLBACK", False)
    monkeypatch.setattr(config, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("TELEM_MONITOR_PATH", str(tmp_path / "mon.json"))
    telem_monitor.set_store(None)
    yield fk
    lanes.use_transport(None)
    lanes.reset_process()
    telem.reset()
    telem_monitor.set_store(None)


# ------------------------------------------------------------------------------------------------- request shape
def test_plan_by_family_and_mode():
    p = telem.plan("discourse.intent.news_query", "what happened in the election today", hints={})
    assert p["family"] == "news" and p["providers"] == telem.FAMILY_PROVIDERS["news"]
    assert p["time_range"] == "day" and p["published_after"]
    assert telem.plan(hints={"family": "reference"})["providers"] == ["exa", "tinyfish"]
    assert telem.plan(hints={"family": "local"})["providers"][0] == "brave"
    b = telem.search_body("q", telem.plan(mode="auto"))
    assert b["search"]["auto_routing"] == "accuracy" and "providers" not in b["search"]
    b = telem.search_body("q", telem.plan(mode="fixed"), session_id=UUID_S)
    assert b["search"]["providers"] == {"include": telem.TELEM_PROVIDERS} and b["session_id"] == UUID_S
    # Telem domains only restrict: "prefer" narrowing is not sent
    assert "include_domains" not in telem.plan(hints={"include_domains": ["espn.com"], "domains_mode": "prefer"})
    assert telem.plan(hints={"include_domains": ["espn.com"], "domains_mode": "restrict"})["include_domains"]


def test_parse_merges_engines_and_prices_runs():
    data = {"status": "partially_succeeded", "session_id": UUID_S, "interaction_id": UUID_I, "routing": None,
            "preprocessor_runs": [
                {"preprocessor_name": "brave", "status": "succeeded", "latency_ms": 500,
                 "output_payload": {"results": _results("brave", ["https://a/1", "https://b/2"])}},
                {"preprocessor_name": "exa", "status": "succeeded", "latency_ms": 900,
                 "output_payload": {"results": _results("exa", ["https://b/2", "https://c/3"])}},
                {"preprocessor_name": "perplexity", "status": "failed", "error": {"code": "timeout"},
                 "output_payload": None}]}
    res, info = telem.parse_search(data)
    by = {d.url: d for d in res}
    assert by["https://b/2"].meta["agree"] == 2 and by["https://b/2"].meta["providers"] == ["brave", "exa"]
    assert res[0].url == "https://b/2"                   # found by both engines ranks first
    assert by["https://a/1"].meta["provider"] == "brave" and by["https://a/1"].meta["provider_ms"] == 500
    assert all(d.source == "telem" for d in res)
    assert info["usd"] == pytest.approx(0.010)          # brave + exa; the failed run is free
    assert info["usd_est"] is True                       # exa's price is not in the response
    assert [r["ok"] for r in info["runs"]] == [True, True, False]


# ---------------------------------------------------------------------------------------------------- web lane
def test_web_lane_on_telem_meters_cost_and_session(fake):
    led = lanes.Ledger()
    m1 = {}
    docs = run(lanes.web("who won the world series", intent="discourse.intent.qa_factoid", final=True,
                         ledger=led, meter=m1, hints={"family": "reference"}))
    assert docs and all(d.source == "telem" for d in docs)
    assert m1["backend"] == "telem"
    assert m1["credits"] == pytest.approx(0.005 / config.METER_TAVILY_CREDIT_USD)   # exa $0.005 + tinyfish free
    assert m1["telem"]["session_id"] == UUID_S and m1["telem"]["interaction_id"] == UUID_I
    body1 = fake.calls[-1][1]
    assert body1["search"]["providers"]["include"] == ["exa", "tinyfish"] and "session_id" not in body1
    assert fake.calls[-1][2]["authorization"] == "Bearer tlm_test"
    run(lanes.web("a different question", final=True, ledger=led, meter={}))
    assert fake.calls[-1][1]["session_id"] == UUID_S     # the conversation's calls share one console session
    # same question again: ledger cache, no call, no spend
    n = len(fake.calls)
    m3 = {}
    run(lanes.web("who won the world series", intent="discourse.intent.qa_factoid", final=True, ledger=led,
                  meter=m3, hints={"family": "reference"}))
    assert len(fake.calls) == n and m3["cached"] and m3["credits"] == 0.0
    s = telem.question_summary([m1, m3])
    assert s["telem_calls"] == 1 and s["telem_cached"] == 1 and s["telem_usd"] == pytest.approx(0.005)
    assert s["telem_trace"]["session_id"] == UUID_S


def test_auto_routing_reports_choice(fake, monkeypatch):
    monkeypatch.setattr(telem, "TELEM_MODE", "auto")
    m = {}
    run(lanes.web("speculative decoding papers", final=True, ledger=lanes.Ledger(), meter=m))
    assert fake.calls[-1][1]["search"]["auto_routing"] == "accuracy"
    assert m["telem"]["routing"] == ["exa", "brave"]


def test_errors_open_breaker_and_cap(fake, monkeypatch):
    fake.status = 401
    with pytest.raises(telem.TelemError):
        run(telem.search("q", telem.plan(), lanes.Ledger(), {}, final=False))
    assert telem.blocked()
    with pytest.raises(lanes.WebUnavailable):
        run(telem.search("q2", telem.plan(), lanes.Ledger(), {}, final=False))
    telem.reset()
    fake.status = 200
    monkeypatch.setattr(telem, "TELEM_PROCESS_USD", 0.004)
    with pytest.raises(lanes.WebUnavailable, match="cap"):
        run(telem.search("q3", telem.plan(hints={"family": "news"}), lanes.Ledger(), {}, final=False))


def test_failover_tavily_to_telem_on_final_only(fake, monkeypatch):
    monkeypatch.setattr(config, "WEB_BACKEND", "tavily")
    monkeypatch.setattr(telem, "WEB_FAILOVER", "telem")
    assert lanes.telem_chain() == ["tavily", "telem"]
    m = {}
    docs = run(lanes.web("who won the world series", final=True, ledger=lanes.Ledger(), meter=m))
    assert docs and docs[0].source == "telem" and m["failover"] == "telem" and "tavily" in m["fallback"]
    with pytest.raises(Exception):                       # speculation never pays a second backend
        run(lanes.web("who won the series again", final=False, ledger=lanes.Ledger(), meter={}))
    monkeypatch.setattr(telem, "TELEM_API_KEY", "")      # no Telem key: no failover leg
    assert lanes.telem_chain() == ["tavily"]


def test_failover_skips_primary_while_breaker_open(fake, monkeypatch):
    monkeypatch.setattr(config, "WEB_BACKEND", "tavily")
    monkeypatch.setattr(telem, "WEB_FAILOVER", "telem")
    lanes.WEB_STATUS["down_until"] = 1e18
    n_tav = sum(1 for c in fake.calls if c[0] == "tavily")
    m = {}
    run(lanes.web("q", final=True, ledger=lanes.Ledger(), meter=m))
    assert sum(1 for c in fake.calls if c[0] == "tavily") == n_tav and m["failover"] == "telem"
    assert lanes.web_down()                              # a failover answer doesn't close the primary's breaker


# -------------------------------------------------------------------------------------------------------- fetch
def test_fetch_per_url_failures_and_pagefetch(fake, monkeypatch):
    r = run(telem.fetch(["https://ok.example/a", "https://dead.example/b", "https://missing.example/c",
                         "https://ok.example/a"]))
    assert [p["url"] for p in r["pages"]] == ["https://ok.example/a"]
    assert {f["url"] for f in r["failed"]} == {"https://dead.example/b", "https://missing.example/c"}
    assert fake.calls[-1][1]["urls"] == ["https://ok.example/a", "https://dead.example/b",
                                         "https://missing.example/c"]   # de-duplicated (a repeat is a 400)
    assert r["usd"] >= 0.001 and r["credits"] == pytest.approx(r["usd"] / config.METER_TAVILY_CREDIT_USD)
    monkeypatch.setattr(telem, "VERIFY_FETCH", "telem")
    r2 = run(pagefetch.fetch(["https://ok.example/a"], query="who won"))
    assert r2["backend"] == "telem" and "Brewers" in r2["pages"][0]["text"]
    fake.status = 503
    with pytest.raises(pagefetch.FetchError):
        run(pagefetch.fetch(["https://ok.example/z"]))
    docs = telem.pages_as_docs(r, query="who won the world series")
    assert docs[0].source == "telem-fetch" and "Brewers" in docs[0].text


# ----------------------------------------------------------------------------------------------------- monitors
def test_monitor_offer_create_poll(fake, monkeypatch):
    monkeypatch.setattr(telem_monitor, "TELEM_MONITORS", True)
    assert telem_monitor.offer("lakers score", "sports", now=1000) is None       # first ask
    assert telem_monitor.offer("who is Ada Lovelace", "reference", now=1001) is None
    o = telem_monitor.offer("Lakers score?", "sports", now=2000)
    assert o and "est." in o["note"]
    m = run(telem_monitor.create("lakers score", "sports"))
    assert fake.calls[-1][1] == {"query": "lakers score", "options": {"frequency": "6h"}, "ttl": "7d"}
    assert telem_monitor.offer("lakers score", "sports", now=3000) is None        # already watched
    fake.changes = [{"seq": 1, "kind": "run", "after": {}, "at": "2026-11-14T10:00:00Z"},
                    {"seq": 2, "kind": "value_changed", "subject": {"name": "Lakers vs Celtics"},
                     "before": "98-97", "after": {"value": "110-104 final", "quote": "Lakers won 110-104",
                                                  "url": "https://espn.example/g"}, "at": "2026-11-14T11:00:00Z"}]
    docs = run(telem_monitor.poll(m["id"]))
    assert len(docs) == 1 and docs[0].source == "telem-monitor" and docs[0].meta["live"]
    assert "110-104" in docs[0].text and docs[0].url == "https://espn.example/g"
    assert telem_monitor.store().monitors[m["id"]]["cursor"] == 2
    assert run(telem_monitor.poll(m["id"])) == []                                  # no change read twice
    assert run(telem_monitor.end(m["id"])) and not telem_monitor.store().monitors
