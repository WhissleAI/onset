"""DuckDuckGo lane (server/ddg.py): parsing, ads, redirects, bot check, cache, and the lane's fallback to Wikipedia."""
import asyncio

import httpx
import pytest

from server import config, ddg, lanes

PAGE = """
<div class="result results_links results_links_deep web-result result--ad ">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://ads.example.com/x">Buy now</a></h2>
  <a class="result__snippet" href="https://ads.example.com/x">An ad</a><div class="clear"></div>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://en.wikipedia.org/wiki/Ankara">Ankara - Wikipedia</a></h2>
  <a class="result__snippet" href="https://en.wikipedia.org/wiki/Ankara"><b>Ankara</b> is the capital city of Turkey&#x27;s republic.</a>
  <div class="clear"></div>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fturkey&amp;rut=abc">Turkey facts</a></h2>
  <a class="result__snippet" href="#">Facts about   Turkey.</a>
  <div class="clear"></div>
</div>
<div class="result results_links results_links_deep web-result ">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://en.wikipedia.org/wiki/Ankara">dupe</a></h2>
  <div class="clear"></div>
</div>
"""


def test_parse_skips_ads_unwraps_redirects_and_dedupes():
    rows = ddg.parse(PAGE)
    assert [r["url"] for r in rows] == ["https://en.wikipedia.org/wiki/Ankara", "https://example.org/turkey"]
    assert rows[0]["snippet"] == "Ankara is the capital city of Turkey's republic."
    assert rows[1]["snippet"] == "Facts about Turkey."


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_cache_and_bot_check_backoff():
    ddg.reset()
    calls = []

    def ok(req):
        calls.append(req)
        return httpx.Response(200, text=PAGE)

    async def run():
        c = _client(ok)
        a = await ddg.search("Capital of  Turkey", c)
        b = await ddg.search("capital of turkey", c)
        assert a == b and len(calls) == 1
        blocked = _client(lambda req: httpx.Response(202, text="<div class='anomaly-modal'></div>"))
        with pytest.raises(ddg.DDGError):
            await ddg.search("something new", blocked)
        assert ddg.blocked()
        with pytest.raises(ddg.DDGError):          # backing off: no request at all
            await ddg.search("another", _client(lambda req: pytest.fail("should not call")))
    asyncio.run(run())
    ddg.reset()


def test_lane_falls_back_to_wikipedia_when_ddg_refuses(monkeypatch):
    ddg.reset()
    monkeypatch.setattr(config, "WEB_BACKEND", "duckduckgo")
    monkeypatch.setattr(config, "ROUTER", False, raising=False)

    async def refuse(query, k=None):
        raise ddg.DDGError(429, "bot check")

    async def wiki(query, k=4, entity="", k_entity=2):
        return [lanes.Doc(id="w", title="Ankara", text="Ankara is the capital.", source="wikipedia")]

    monkeypatch.setattr(lanes, "duckduckgo", refuse)
    monkeypatch.setattr(lanes, "wikipedia", wiki)
    meter = {}
    docs = asyncio.run(lanes._live("capital of turkey", "general", "", "", True, None, meter))
    assert meter["backend"] == "wikipedia" and "DDGError" in meter["fallback"] and docs[0].source == "wikipedia"


def test_lane_uses_ddg_docs(monkeypatch):
    monkeypatch.setattr(config, "WEB_BACKEND", "duckduckgo")

    async def ok(query, k=None):
        return [lanes.Doc(id="u", title="Ankara", text="Ankara is the capital.", url="u", source="duckduckgo")]

    monkeypatch.setattr(lanes, "duckduckgo", ok)
    meter = {}
    res = asyncio.run(lanes._live("capital of turkey", "general", "", "", False, None, meter))
    assert meter["backend"] == "duckduckgo" and meter["credits"] == 0.0 and res.backend == "duckduckgo"
