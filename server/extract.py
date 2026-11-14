"""Tavily /extract: read the cited page when the snippets could not confirm the answer (verify escalation).

VERIFY_EXTRACT=1 (default off): after the answer is shown, when verify (server/verify.py) could not confirm the key
claim and the free re-check (server/recheck.py: Wikidata, local leads) came back "still unconfirmed", ONE /extract
call reads the top cited URL with the claim's sentence as the query (Tavily reranks the page's chunks by it), and
verify's matcher runs again on those chunks: "re-checked ✓ page" when the page states the claim, else "still
unconfirmed". It never runs before or during the answer, so time to answer is unchanged.

Cost (docs.tavily.com, 2026-11-14): extract_depth basic = 1 credit per 5 successful URLs; failed URLs are free.
Process credit cap (TAVILY_PROCESS_CREDITS) applies like /search. Timeout EXTRACT_TIMEOUT_S (Tavily's own `timeout`
plus a hard wall-clock cap); any failure = "still unconfirmed", never an error on the answer.
"""
from __future__ import annotations

import asyncio
import os
import re
import time

from . import config

VERIFY_EXTRACT = os.environ.get("VERIFY_EXTRACT", "0") == "1"
EXTRACT_TIMEOUT_S = float(os.environ.get("EXTRACT_TIMEOUT_S", "3.0"))
EXTRACT_CHUNKS = int(os.environ.get("EXTRACT_CHUNKS", "3"))          # 1-5 chunks of the page, ranked by the claim
EXTRACT_DEPTH = os.environ.get("EXTRACT_DEPTH", "basic")              # basic (1 credit / 5 urls) | advanced (2 / 5)
COST_PER_URL = {"basic": 0.2, "advanced": 0.4}


class ExtractError(Exception):
    pass


def body(urls: list[str], query: str = "", chunks: int | None = None, depth: str | None = None,
         timeout_s: float | None = None) -> dict:
    b = {"urls": list(urls)[:20], "extract_depth": depth or EXTRACT_DEPTH, "format": "text", "include_usage": True,
         "timeout": max(1.0, min(60.0, timeout_s or EXTRACT_TIMEOUT_S))}
    if query:
        b["query"] = query[:400]
        b["chunks_per_source"] = max(1, min(5, chunks or EXTRACT_CHUNKS))
    return b


async def extract(urls: list[str], query: str = "", chunks: int | None = None, depth: str | None = None,
                  timeout_s: float | None = None) -> dict:
    """POST /extract. Returns {"pages": [{"url", "text"}], "failed": [...], "credits", "ms", "response_time"}.
    Raises ExtractError on no key / over the process cap / HTTP error / timeout."""
    from . import lanes
    urls = [u for u in urls if u and u.startswith("http")]
    if not urls:
        raise ExtractError("no url")
    if not config.TAVILY_API_KEY:
        raise ExtractError("no key")
    b = body(urls, query, chunks, depth, timeout_s)
    est = COST_PER_URL.get(b["extract_depth"], 0.4) * len(b["urls"])
    if config.TAVILY_PROCESS_CREDITS > 0 and lanes._PROCESS["spent"] + est > config.TAVILY_PROCESS_CREDITS:
        raise ExtractError("process credit budget")
    t0 = time.perf_counter()
    try:
        r = await asyncio.wait_for(
            lanes.client().post(config.TAVILY_BASE_URL + "/extract", json=b, timeout=b["timeout"] + 0.5,
                                headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"}),
            timeout=b["timeout"] + 0.5)
    except asyncio.TimeoutError:
        raise ExtractError(f"timeout after {b['timeout']:g}s") from None
    except Exception as e:
        raise ExtractError(f"{type(e).__name__}: {e}"[:120]) from None
    ms = round((time.perf_counter() - t0) * 1000)
    if r.status_code != 200:
        raise ExtractError(f"HTTP {r.status_code}")
    data = r.json()
    pages = [{"url": x.get("url", ""), "text": (x.get("raw_content") or "").strip()} for x in data.get("results") or []]
    usage = (data.get("usage") or {}).get("credits")
    # include_usage can read 0 until 5 extractions accumulate: meter the documented rate when it does
    credits = float(usage) if usage else COST_PER_URL.get(b["extract_depth"], 0.4) * len(pages)
    lanes._PROCESS["spent"] += credits
    return {"pages": pages, "failed": data.get("failed_results") or [], "credits": credits, "usage": usage,
            "ms": ms, "response_time": data.get("response_time")}


_CITE_N = re.compile(r"\[(\d+)")


def cited_url(answer: str, docs: list, claim_text: str = "") -> str:
    """The URL to read: the source cited in the claim's sentence, else the first cited source, else source 1.
    Only web pages (http urls; cards / facts / local moments have none)."""
    def url(d):
        u = d.get("url") if isinstance(d, dict) else getattr(d, "url", "")
        return u if u and u.startswith("http") else ""

    sents = re.split(r"(?<=[.!?])\s+", answer or "")
    order = [s for s in sents if claim_text and claim_text in s] + [answer or ""]
    for s in order:
        for m in _CITE_N.finditer(s):
            k = int(m.group(1)) - 1
            if 0 <= k < len(docs) and url(docs[k]):
                return url(docs[k])
    return next((url(d) for d in docs if url(d)), "")


def claim_sentence(answer: str, claim_text: str) -> str:
    from . import verify
    s = next((s for s in re.split(r"(?<=[.!?])\s+", answer or "") if claim_text in s), claim_text)
    return re.sub(r"\s+", " ", verify._CITE.sub(" ", s)).strip()


async def recheck_page(claim: dict, answer: str, docs: list) -> dict:
    """The /extract escalation for one claim. {"status": rechecked|unconfirmed|skip, "url", "credits", "ms", ...}"""
    from . import verify
    u = cited_url(answer, docs, claim.get("text", ""))
    out = {"status": "skip", "url": u, "credits": 0.0}
    if not u:
        out["why"] = "no web source"
        return out
    try:
        from . import pagefetch          # VERIFY_FETCH=tavily (this module's /extract, default) | telem
        r = await pagefetch.fetch([u], query=claim_sentence(answer, claim.get("text", "")))
    except ExtractError as e:
        out.update(status="unconfirmed", error=str(e))
        return out
    out.update(credits=r["credits"], ms=r["ms"], response_time=r["response_time"])
    texts = [p["text"] for p in r["pages"] if p["text"]]
    if not texts:
        out.update(status="unconfirmed", why="page not readable")
        return out
    out["chars"] = sum(len(t) for t in texts)
    out["status"] = "rechecked" if any(verify._supports(claim, t) for t in texts) else "unconfirmed"
    return out
