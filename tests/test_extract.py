"""VERIFY_EXTRACT (server/extract.py): when verify + the free re-check can't confirm the key claim, read the cited page
with Tavily /extract (claim sentence as the query), after the answer. Off by default; failures never surface."""
import asyncio
import json

import httpx
import pytest

from server import config, extract, knowledge, lanes, recheck, verify


def D(title, text, url):
    return lanes.Doc(id=url, title=title, text=text, url=url, source="tavily")


@pytest.fixture
def fake(monkeypatch):
    calls = []
    pages = {"https://a.example/ws": "The Milwaukee Brewers won the 2026 World Series in five games over the Rays.",
             "https://b.example/other": "Nothing relevant here."}

    def handler(req: httpx.Request):
        assert req.url.path == "/extract"
        assert req.headers["authorization"] == "Bearer tvly-test"
        b = json.loads(req.content)
        calls.append(b)
        if b.get("urls") == ["https://slow.example/"]:
            raise httpx.ReadTimeout("slow")
        res = [{"url": u, "raw_content": pages[u]} for u in b["urls"] if u in pages]
        failed = [{"url": u, "error": "not found"} for u in b["urls"] if u not in pages]
        return httpx.Response(200, json={"results": res, "failed_results": failed, "response_time": 0.4,
                                         "usage": {"credits": 1 if res else 0}, "request_id": "r1"})

    monkeypatch.setattr(config, "TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(config, "TAVILY_BASE_URL", "https://api.tavily.test")
    monkeypatch.setattr(config, "VERIFY_RESEARCH", True)
    lanes.reset_process()
    lanes.use_transport(httpx.MockTransport(handler))

    async def none(*a, **k):
        return []
    monkeypatch.setattr(knowledge, "wikidata", none)
    monkeypatch.setattr(knowledge, "leads", none)
    yield calls
    lanes.use_transport(None)
    lanes.reset_process()


Q = "who won the world series this year"
ANS = "The Milwaukee Brewers won the 2026 World Series [1]."
SNIP = [D("2026 World Series", "Game 5 recap: champagne in the clubhouse.", "https://a.example/ws"),
        D("Other", "Unrelated.", "https://b.example/other")]


def test_body_is_documented_shape():
    b = extract.body(["https://x.example"], query="claim", chunks=9, timeout_s=99)
    assert b["urls"] == ["https://x.example"] and b["query"] == "claim" and b["chunks_per_source"] == 5
    assert b["extract_depth"] == "basic" and b["format"] == "text" and b["include_usage"] is True
    assert b["timeout"] == 60.0
    assert "chunks_per_source" not in extract.body(["https://x.example"])      # only valid with a query


def test_cited_url_prefers_the_claims_sentence():
    docs = [D("one", "", "https://one.example"), D("two", "", "https://two.example")]
    ans = "Paris is big [1]. It was founded in 250 BC [2]."
    assert extract.cited_url(ans, docs, "250") == "https://two.example"
    assert extract.cited_url(ans, docs, "") == "https://one.example"
    card = [lanes.Doc(id="card", title="Weather", text="", url="", source="card")]
    assert extract.cited_url("Sunny [1].", card) == ""


def test_off_by_default_no_call(fake, monkeypatch):
    monkeypatch.setattr(extract, "VERIFY_EXTRACT", False)
    v = verify.check(ANS, SNIP, Q)
    assert not recheck.needs(v)            # flag off: "The Milwaukee Brewers" isn't a re-check claim (as before)
    ans = "Nvidia was founded in 1993 [1]."
    v = verify.check(ans, SNIP, "when was nvidia founded")
    assert recheck.needs(v)
    r = asyncio.run(recheck.run(ans, SNIP, "when was nvidia founded", v))
    assert r["status"] == "unconfirmed" and not fake and "extract" not in r["lanes"]


def test_page_confirms_the_claim(fake, monkeypatch):
    monkeypatch.setattr(extract, "VERIFY_EXTRACT", True)
    v = verify.check(ANS, SNIP, Q)
    r = asyncio.run(recheck.run(ANS, SNIP, Q, v))
    assert r["status"] == "rechecked" and r["label"] == "re-checked ✓ page" and r["lanes"][-1] == "extract"
    assert r["extract_credits"] == 1 and r["paid"] == 0
    assert fake[0]["urls"] == ["https://a.example/ws"] and "Brewers" in fake[0]["query"] and "[1]" not in fake[0]["query"]


def test_page_without_the_claim_stays_unconfirmed(fake, monkeypatch):
    monkeypatch.setattr(extract, "VERIFY_EXTRACT", True)
    ans = "The Milwaukee Brewers won the 2026 World Series [2]."
    v = verify.check(ans, SNIP, Q)
    r = asyncio.run(recheck.run(ans, SNIP, Q, v))
    assert r["status"] == "unconfirmed" and r["extract"]["url"] == "https://b.example/other"


def test_timeout_and_cap_fall_back_quietly(fake, monkeypatch):
    monkeypatch.setattr(extract, "VERIFY_EXTRACT", True)
    slow = [D("2026 World Series", "recap", "https://slow.example/")]
    v = verify.check(ANS, slow, Q)
    r = asyncio.run(recheck.run(ANS, slow, Q, v))
    assert r["status"] == "unconfirmed" and r["extract"]["error"]
    monkeypatch.setattr(config, "TAVILY_PROCESS_CREDITS", 1)
    lanes._PROCESS["spent"] = 1.0
    with pytest.raises(extract.ExtractError, match="budget"):
        asyncio.run(extract.extract(["https://a.example/ws"], "x"))
