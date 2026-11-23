"""DuckDuckGo web lane: free, keyless web search (WEB_BACKEND=duckduckgo, the default when no Tavily / Telem key is set).

One POST to DuckDuckGo's HTML endpoint (html.duckduckgo.com/html, the no-JavaScript results page) returns ~10 organic
results: title, URL and a ~200-character snippet. Ads are skipped. There is no API contract, so this module is careful:

- a per-process cache (WEB_DDG_CACHE_TTL_S, default 15 min) so the speculator's repeated partial queries cost nothing;
- at most WEB_DDG_CONCURRENCY requests in flight and WEB_DDG_MIN_GAP_S between request starts (speculation fires
  several queries per spoken question; DuckDuckGo throttles bursts);
- when DuckDuckGo answers with its bot check (HTTP 202 / 403 / 429, or the "anomaly" page) the lane backs off for
  WEB_DDG_BACKOFF_S and raises DDGError: the web lane then falls back to Wikipedia (server/lanes.py `_live`).
"""
from __future__ import annotations

import asyncio
import html
import os
import re
import time
from urllib.parse import parse_qs, unquote, urlparse

import httpx

ENDPOINT = os.environ.get("WEB_DDG_URL", "https://html.duckduckgo.com/html/")
REGION = os.environ.get("WEB_DDG_REGION", "us-en")
MAX_RESULTS = int(os.environ.get("WEB_DDG_MAX_RESULTS", "6"))
CACHE_TTL_S = float(os.environ.get("WEB_DDG_CACHE_TTL_S", "900"))
CONCURRENCY = int(os.environ.get("WEB_DDG_CONCURRENCY", "2"))
MIN_GAP_S = float(os.environ.get("WEB_DDG_MIN_GAP_S", "0.25"))
BACKOFF_S = float(os.environ.get("WEB_DDG_BACKOFF_S", "60"))
# A browser-like User-Agent: the HTML endpoint serves its bot check to unknown clients.
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/17.0 Safari/605.1.15",
      "Accept": "text/html", "Accept-Language": "en-US,en;q=0.8"}


class DDGError(Exception):
    def __init__(self, status: int, msg: str = ""):
        super().__init__(f"duckduckgo {status}: {msg}"[:200])
        self.status = status


_cache: dict[str, tuple[float, list[dict]]] = {}
_sem: asyncio.Semaphore | None = None
_sem_loop = None
_last_start = 0.0
_blocked_until = 0.0
_gap_lock: asyncio.Lock | None = None


def reset():
    global _cache, _sem, _sem_loop, _last_start, _blocked_until, _gap_lock
    _cache, _sem, _sem_loop, _last_start, _blocked_until, _gap_lock = {}, None, None, 0.0, 0.0, None


def blocked() -> bool:
    return time.monotonic() < _blocked_until


def _limits():
    global _sem, _sem_loop, _gap_lock
    loop = asyncio.get_running_loop()
    if _sem is None or _sem_loop is not loop:
        _sem, _gap_lock, _sem_loop = asyncio.Semaphore(CONCURRENCY), asyncio.Lock(), loop
    return _sem, _gap_lock


_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r'<div class="result results_links[^"]*"(.*?)<div class="clear"></div>', re.S)
_A = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.S)
_SNIP = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.S)


def _text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", s))).strip()


def _real_url(href: str) -> str:
    """Result links are direct, or (older pages) DuckDuckGo redirects: //duckduckgo.com/l/?uddg=<url>."""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    u = urlparse(href)
    if u.netloc.endswith("duckduckgo.com") and u.path.startswith("/l/"):
        q = parse_qs(u.query).get("uddg")
        if q:
            return unquote(q[0])
    return href


def parse(page: str, k: int = MAX_RESULTS) -> list[dict]:
    """[{title, url, snippet}] from a results page, ads skipped, duplicates (same URL) dropped."""
    out, seen = [], set()
    for m in _BLOCK.finditer(page):
        block = m.group(0)
        if "result--ad" in block[:200]:
            continue
        a = _A.search(block)
        if not a:
            continue
        url = _real_url(a.group(1))
        if not url.startswith("http") or url in seen or "duckduckgo.com/y.js" in url:
            continue
        sn = _SNIP.search(block)
        title, snippet = _text(a.group(2)), _text(sn.group(1)) if sn else ""
        if not title:
            continue
        seen.add(url)
        out.append({"title": title, "url": url, "snippet": snippet})
        if len(out) >= k:
            break
    return out


def _is_block_page(status: int, page: str) -> bool:
    return status in (202, 403, 429) or "anomaly-modal" in page or "anomaly.js" in page


async def search(query: str, client: httpx.AsyncClient, k: int = MAX_RESULTS, timeout: float = 4.0) -> list[dict]:
    """Results for `query` (cached). Raises DDGError when DuckDuckGo refuses or is backing off, httpx errors when the
    network fails."""
    global _last_start, _blocked_until
    key = " ".join(query.lower().split())
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
        return hit[1][:k]
    if blocked():
        raise DDGError(429, f"backing off for {(_blocked_until - time.monotonic()):.0f}s after a bot check")
    sem, gap = _limits()
    async with sem:
        async with gap:
            wait = _last_start + MIN_GAP_S - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            _last_start = time.monotonic()
        r = await client.post(ENDPOINT, data={"q": query, "kl": REGION}, headers=UA, timeout=timeout,
                              follow_redirects=True)
    page = r.text
    if _is_block_page(r.status_code, page):
        _blocked_until = time.monotonic() + BACKOFF_S
        raise DDGError(r.status_code if r.status_code != 200 else 429, "bot check")
    if r.status_code != 200:
        raise DDGError(r.status_code, page[:80])
    rows = parse(page, k=max(k, MAX_RESULTS))
    _cache[key] = (time.monotonic(), rows)
    if len(_cache) > 512:
        for old in sorted(_cache, key=lambda q: _cache[q][0])[:128]:
            _cache.pop(old, None)
    return rows[:k]
