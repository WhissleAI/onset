"""Read a cited page for the verify escalation, from a pluggable backend (VERIFY_FETCH=tavily|telem).

One contract for both, the one server/extract.extract (Tavily /extract, feat/tavily) returns:
  {"pages": [{"url", "text", ...}], "failed": [...], "credits" (Tavily-credit units; Telem $ converted), "ms", ...}
and one error type: FetchError, a subclass of extract.ExtractError when that module is present, so the escalation's
existing `except ExtractError` handles either backend. The escalation (extract.recheck_page) switches backends by
calling pagefetch.fetch(...) where it now calls extract(...).

  tavily  Tavily /extract with the claim sentence as the query (it ranks the page's chunks by it). Tavily credits.
  telem   Telem /v1/fetch: the whole page as text (no query ranking), the engine chain chosen by Telem (fails per
          URL, free when it fails). Telem $ (est.), counted against TELEM_PROCESS_USD.
"""
from __future__ import annotations

from . import telem

try:                                   # feat/tavily's /extract client (absent on branches without it)
    from .extract import ExtractError as _Base
except ImportError:                    # pragma: no cover - depends on merge order
    _Base = Exception


class FetchError(_Base):
    pass


def backend() -> str:
    return telem.VERIFY_FETCH if telem.VERIFY_FETCH in ("tavily", "telem") else "tavily"


async def fetch(urls: list[str], query: str = "", timeout_s: float | None = None) -> dict:
    b = backend()
    if b == "telem":
        try:
            r = await telem.fetch(urls, query=query, timeout_s=timeout_s)
        except telem.TelemError as e:
            raise FetchError(str(e)) from None
        r["backend"] = "telem"
        return r
    try:
        from . import extract
    except ImportError:
        raise FetchError("tavily extract not available") from None
    r = await extract.extract(urls, query=query, timeout_s=timeout_s)
    r["backend"] = "tavily"
    return r
