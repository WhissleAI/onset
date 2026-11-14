"""Telem (telem.ai): one router for web search and page reads across providers (brave, exa, tavily, perplexity, ...).

Used three ways, all OFF unless configured (no TELEM_API_KEY = nothing here runs):
  WEB_BACKEND=telem   the paid web lane goes to Telem instead of Tavily (server/lanes.py `_live`).
  WEB_FAILOVER=telem  (or =tavily with WEB_BACKEND=telem): a final question whose primary backend fails tries the
                      other one before Wikipedia and the recorded cache, so a provider outage never leaves the
                      user without sources.
  VERIFY_FETCH=telem  the verify escalation reads the cited page with Telem /v1/fetch (server/pagefetch.py).

Provider choice per question (TELEM_MODE):
  family  Onset's own router family (server/narrow.py: news / sports / finance / weather / reference / local ...)
          picks the engines (FAMILY_PROVIDERS). A hypothesis until eval/telem_ab.py measures it.
  auto    Telem's auto-routing ("accuracy"): Telem picks the engines per query (`routing` in the response).
  fixed   TELEM_PROVIDERS, always.

Verified against https://docs.telem.ai (llms-full.txt, read 2026-11-14):
  POST /v1/search {user_input:{query}, search:{providers:{include}, num_results 1-20 (per provider), tier
    minimalist|default|extended|max, auto_routing:"accuracy", include_domains (restrict only, 1-20), published_after
    (ISO date)}, session_id (UUID from an earlier response)} -> {status succeeded|partially_succeeded|failed,
    preprocessor_runs[{preprocessor_name, status, latency_ms, error, output_payload:{results[{url,title,rank,summary,
    publish_date?}], warnings}}], routing[{providers}] | null, session_id, interaction_id}. `rank` is per provider
    (never a cross-provider score); a failed provider run has output_payload null and is not billed.
  POST /v1/fetch {urls (<=5, no duplicates), options:{tier, providers (pinned chain), content_format text|markdown,
    inline_max_chars}} -> preprocessor_runs[{output_payload:{fetched_results[{url,status,content,provider,
    http_status?,warnings,error}]}}]. A URL fails per URL (free), never the call.
  Errors: 400 bad request, 401 key, 402 out of credit, 403, 422 schema, 429 (+Retry-After), 503 all_providers_failed,
    504 deadline_exceeded (free). Price: the provider's own list price per call, no markup (docs /pricing/); the
    response carries no $ figure at tier default, so every $ here is an estimate from that table ("est.").
  Observability: every call has an interaction_id and a session_id; sending the session_id back groups a
    conversation's calls as one session in the console (app.telem.ai). No per-trace URL scheme is documented, so
    the UI shows the ids, not a link.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx

from . import config

TELEM_API_KEY = os.environ.get("TELEM_API_KEY", "")
TELEM_BASE_URL = os.environ.get("TELEM_BASE_URL", "https://router.telem.ai").rstrip("/")
TELEM_MODE = os.environ.get("TELEM_MODE", "family")                 # family | auto | fixed
TELEM_PROVIDERS = [p for p in os.environ.get("TELEM_PROVIDERS", "brave").split(",") if p]
TELEM_AUTO_INCLUDE = [p for p in os.environ.get("TELEM_AUTO_INCLUDE", "").split(",") if p]  # auto: choose from these
TELEM_NUM_RESULTS = int(os.environ.get("TELEM_NUM_RESULTS", "5"))   # per provider
TELEM_TIER = os.environ.get("TELEM_TIER", "default")                 # default = url, title, rank, summary
TELEM_TIMEOUT_S = float(os.environ.get("TELEM_TIMEOUT_S", "4.0"))
TELEM_FETCH_TIMEOUT_S = float(os.environ.get("TELEM_FETCH_TIMEOUT_S", "5.0"))
TELEM_FETCH_MAX_CHARS = int(os.environ.get("TELEM_FETCH_MAX_CHARS", "20000"))
TELEM_PROCESS_USD = float(os.environ.get("TELEM_PROCESS_USD", "3.0"))   # hard $ cap for this process (0 = none)
TELEM_SESSIONS = os.environ.get("TELEM_SESSIONS", "1") == "1"       # group a conversation's calls in the console
WEB_FAILOVER = os.environ.get("WEB_FAILOVER", "")                   # "" | telem | tavily (second paid backend)
VERIFY_FETCH = os.environ.get("VERIFY_FETCH", "tavily")             # tavily | telem (server/pagefetch.py)

# Search $ per provider call (docs.telem.ai/pricing, 2026-11-14). exa bills "what exa reports", which tier default
# doesn't return: priced at exa's list $0.005 and flagged est.
SEARCH_USD = {"brave": 0.005, "ceramic": 0.00005, "exa": 0.005, "linkup": 0.005, "parallel": 0.001,
              "perplexity": 0.005, "seltz": 0.005, "serpapi": 0.006, "tavily": 0.005, "tinyfish": 0.0, "you": 0.005}
SEARCH_USD_LISTED = {k for k in SEARCH_USD if k != "exa"}
SEARCH_USD_UNKNOWN = 0.006                                           # a provider not in the table: the dearest listed
# Fetch $ per page read. jina bills $0.045 / M tokens (est. chars / 4); firecrawl 1 credit a page at $0.000798.
FETCH_USD = {"brightdata": 0.0015, "exa": 0.001, "firecrawl": 0.000798, "linkup": 0.005, "parallel": 0.001,
             "tavily": 0.001, "tinyfish": 0.0, "you": 0.001}
JINA_USD_PER_TOKEN = 0.045 / 1e6

# Onset family (server/narrow.py) -> engines. live: recency-strong engines with dates (brave page_age + news
# vertical; tinyfish dates news rows and is free); reference / entity: exa (embedding search, matches on meaning);
# places: brave (locations vertical). tinyfish rides along for free as a second opinion (cross-engine agreement).
LIVE, REFERENCE, PLACES = ["brave", "tinyfish"], ["exa", "tinyfish"], ["brave", "tinyfish"]
FAMILY_PROVIDERS = {"news": LIVE, "sports": LIVE, "finance": LIVE, "weather": LIVE,
                    "reference": REFERENCE, "definition": REFERENCE, "health": REFERENCE, "tech": REFERENCE,
                    "recipe": REFERENCE, "home": REFERENCE, "convert": REFERENCE,
                    "local": PLACES, "travel": PLACES, "reviews": PLACES, "": ["brave", "tinyfish"]}
try:
    FAMILY_PROVIDERS.update(json.loads(os.environ.get("TELEM_FAMILY_PROVIDERS", "{}")))
except ValueError:
    pass

_NEWS = {"discourse.intent.news_query"}
_FIN = {"discourse.intent.qa_stock", "discourse.intent.qa_currency"}
_WX = {"discourse.intent.weather_query"}

SPENT = {"usd": 0.0, "calls": 0, "fetch_urls": 0, "errors": 0}
_BLOCK = {"until": 0.0, "why": ""}


class TelemError(Exception):
    def __init__(self, status: int, msg: str = "", retry_after: float | None = None, code: str = ""):
        super().__init__(f"telem {status}{(' ' + code) if code else ''}: {msg}"[:200])
        self.status, self.retry_after, self.code = status, retry_after, code


def enabled() -> bool:
    return bool(TELEM_API_KEY)


def reset():
    """Tests: forget spend and the breaker."""
    SPENT.update(usd=0.0, calls=0, fetch_urls=0, errors=0)
    _BLOCK.update(until=0.0, why="")


def blocked() -> str | None:
    if time.monotonic() < _BLOCK["until"]:
        return _BLOCK["why"]
    return None


def _block(seconds: float, why: str):
    _BLOCK["until"] = max(_BLOCK["until"], time.monotonic() + seconds)
    _BLOCK["why"] = why


def credit_eq(usd: float) -> float:
    """$ in Tavily-credit units, so the session Ledger, the speculator's credit totals and server/meter.py (credits x
    METER_TAVILY_CREDIT_USD) price a Telem call at its own $ with no change to the meter."""
    return usd / config.METER_TAVILY_CREDIT_USD if config.METER_TAVILY_CREDIT_USD > 0 else 0.0


# ------------------------------------------------------------------------------------------------------- search
def family_of(intent: str = "", topic: str = "", hints: dict | None = None) -> str:
    h = hints or {}
    if h.get("family") is not None:
        return h["family"]
    if intent in _NEWS or topic == "news" or h.get("topic") == "news":
        return "news"
    if intent in _FIN or h.get("topic") == "finance":
        return "finance"
    if intent in _WX:
        return "weather"
    return ""


def plan(intent: str = "", text: str = "", final: bool = False, hints: dict | None = None, topic: str = "",
         mode: str | None = None) -> dict:
    """Telem parameters for one question (mode / providers / freshness / domains)."""
    from . import lanes
    hints = hints or {}
    mode = mode or TELEM_MODE
    fam = family_of(intent, topic, hints)
    p = {"mode": mode, "family": fam, "num_results": TELEM_NUM_RESULTS, "tier": TELEM_TIER}
    if mode == "auto":
        p["providers"] = list(TELEM_AUTO_INCLUDE)
    elif mode == "fixed":
        p["providers"] = list(TELEM_PROVIDERS)
    else:
        p["providers"] = list(FAMILY_PROVIDERS.get(fam) or FAMILY_PROVIDERS[""])
    # freshness: the same words Tavily's time_range uses, as a publish-date floor (results with no date are kept)
    tr = hints.get("time_range")
    if not tr and fam == "news":
        low = (text or "").lower()
        tr = "day" if lanes._DAY.search(low) else ("week" if lanes._WEEK.search(low) else None)
    if tr in ("day", "week", "month", "year"):
        days = {"day": 1, "week": 7, "month": 31, "year": 366}[tr]
        p["published_after"] = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
        p["time_range"] = tr
    # Telem's include_domains only restricts (no "prefer"): send narrowing only when Onset restricts
    if hints.get("include_domains") and hints.get("domains_mode") == "restrict":
        p["include_domains"] = list(hints["include_domains"])[:20]
    return p


def search_body(query: str, p: dict, session_id: str | None = None) -> dict:
    s: dict = {"num_results": max(1, min(20, int(p.get("num_results") or 5))), "tier": p.get("tier") or "default"}
    if p.get("mode") == "auto":
        s["auto_routing"] = "accuracy"
    if p.get("providers"):
        s["providers"] = {"include": list(dict.fromkeys(p["providers"]))}
    if p.get("published_after"):
        s["published_after"] = p["published_after"]
    if p.get("include_domains"):
        s["include_domains"] = list(p["include_domains"])[:20]
    body = {"user_input": {"query": query[:400]}, "search": s}
    if session_id:
        body["session_id"] = session_id
    return body


def est_usd(p: dict) -> float:
    """Pre-call estimate (budget check): the named providers' prices; auto = up to 3 engines at the dearest price."""
    provs = p.get("providers") or []
    if p.get("mode") == "auto":
        provs = provs or [None, None, None]
        return sum(SEARCH_USD.get(x, SEARCH_USD_UNKNOWN) if x else 0.005 for x in provs[:3])
    return sum(SEARCH_USD.get(x, SEARCH_USD_UNKNOWN) for x in provs)


async def _post(path: str, body: dict, timeout_s: float) -> dict:
    from . import lanes
    try:
        r = await asyncio.wait_for(
            lanes.client().post(TELEM_BASE_URL + path, json=body, timeout=timeout_s,
                                headers={"Authorization": f"Bearer {TELEM_API_KEY}"}),
            timeout=timeout_s)
    except asyncio.TimeoutError:
        raise TelemError(0, f"timeout after {timeout_s:g}s") from None
    if r.status_code not in (200, 201):          # 201: a monitor was created
        ra = r.headers.get("retry-after")
        code, msg = "", ""
        try:
            d = r.json()
            err, det = d.get("error"), d.get("detail")
            if isinstance(err, dict):
                code, msg = err.get("code") or "", err.get("message") or ""
            elif isinstance(det, dict):
                code, msg = det.get("code") or det.get("error") or "", det.get("reason") or det.get("message") or ""
            elif det is not None:
                msg = str(det)
        except Exception:
            msg = r.text[:120]
        raise TelemError(r.status_code, msg, float(ra) if ra and ra.replace(".", "", 1).isdigit() else None, code)
    return r.json()


def _breaker(e: TelemError):
    if e.status == 401:
        _block(600, "401 invalid key")
    elif e.status == 402:
        _block(900, "402 out of credit")
    elif e.status == 429:
        _block(e.retry_after or 10, "429 rate limited")


def parse_search(data: dict, keep: int | None = None):
    """(WebResult of Doc, info). Rows from every provider run are merged by URL; each Doc keeps the engine that ranked
    it best, every engine that found it (`agree`), its rank there and that engine's latency. Doc.score = reciprocal
    rank summed over engines / engines that answered (0..1; a cross-engine signal, NOT Tavily's relevance score)."""
    from . import lanes
    runs, rows = [], {}
    ok = 0
    for run in data.get("preprocessor_runs") or []:
        name = run.get("preprocessor_name") or "?"
        payload = run.get("output_payload") or None
        ms = run.get("latency_ms") if run.get("latency_ms") is not None else (payload or {}).get("latency_ms")
        failed = payload is None or run.get("status") == "failed" or bool(run.get("error"))
        res = [] if payload is None else (payload.get("results") or [])
        usd = 0.0 if (payload is None or run.get("status") == "failed") else SEARCH_USD.get(name, SEARCH_USD_UNKNOWN)
        err = run.get("error")
        runs.append({"provider": name, "ok": not failed, "ms": ms, "n": len(res), "usd": usd,
                     **({"error": (err.get("code") or err.get("message") or str(err))[:80] if isinstance(err, dict)
                         else str(err)[:80]} if err else {}),
                     **({"warnings": [w.get("code") for w in payload.get("warnings") or [] if isinstance(w, dict)]}
                        if payload and payload.get("warnings") else {})})
        if failed and not res:
            continue
        ok += 1
        for i, x in enumerate(res):
            url = x.get("url") or ""
            rank = int(x.get("rank") or (i + 1))
            key = url or f"{name}:{i}"
            e = rows.get(key)
            if e is None:
                e = rows[key] = {"x": x, "provider": name, "rank": rank, "ms": ms, "by": {}}
            elif rank < e["rank"]:
                e.update(x={**x, "summary": x.get("summary") or e["x"].get("summary")}, provider=name, rank=rank, ms=ms)
            e["by"].setdefault(name, rank)
    out = lanes.WebResult()
    for key, e in rows.items():
        x = e["x"]
        title = x.get("title") or ""
        pub = x.get("publish_date")
        if pub:
            title = f"{title} ({str(pub)[:16]})"
        text = (x.get("summary") or x.get("excerpt") or "").strip()
        rrf = sum(1.0 / r for r in e["by"].values()) / max(1, ok)
        meta = {"provider": e["provider"], "rank": e["rank"], "provider_ms": e["ms"], "agree": len(e["by"]),
                "providers": sorted(e["by"]), "telem_rrf": round(rrf, 4)}
        if pub:
            meta["published_date"] = pub
        out.append(lanes.Doc(id=lanes._doc_id("telem", x.get("url") or "", x.get("title") or "", text), title=title,
                             text=text, url=x.get("url") or "", source="telem", score=round(rrf, 4), meta=meta))
    out.sort(key=lambda d: (-d.score, d.meta["rank"]))
    if keep:
        del out[keep:]
    usd = sum(r["usd"] for r in runs)
    info = {"runs": runs, "usd": round(usd, 6),
            "usd_est": any(r["ok"] and r["provider"] not in SEARCH_USD_LISTED for r in runs),
            "routing": [p for r in (data.get("routing") or []) for p in (r.get("providers") or [])] or None,
            "status": data.get("status"), "session_id": data.get("session_id"),
            "interaction_id": data.get("interaction_id")}
    return out, info


def _cache_key(query: str, body: dict) -> tuple:
    from . import lanes
    s = body["search"]
    return ("telem", lanes.norm_query(query), s.get("auto_routing"), tuple((s.get("providers") or {}).get("include") or ()),
            s["num_results"], s["tier"], s.get("published_after"), tuple(s.get("include_domains") or ()))


async def search(query: str, p: dict | None = None, ledger=None, meter: dict | None = None, final: bool = False):
    """One Telem search for the web lane: the session Ledger's budget, cache and in-flight dedup (in credit-equivalent
    units), the process $ cap, and the breaker. Raises lanes.WebUnavailable / TelemError like the Tavily lane."""
    from . import lanes
    p = p or plan(final=final)
    ledger = ledger if ledger is not None else lanes.Ledger()
    meter = meter if meter is not None else {}
    meter.update(backend="telem", credits=0.0, cached=False)
    tm = meter.setdefault("telem", {})
    tm.update(mode=p.get("mode"), family=p.get("family"), asked=list(p.get("providers") or []))
    if not TELEM_API_KEY:
        raise lanes.WebUnavailable("telem: no TELEM_API_KEY")
    sess = getattr(ledger, "telem_session", None) if TELEM_SESSIONS else None
    body = search_body(query, p, sess)
    key = _cache_key(query, body)
    hit = ledger.cache.get(key)
    if hit and hit[0] > time.monotonic():
        ledger.cache_hits += 1
        meter["cached"] = True
        return _copy(hit[1], True, tm)
    task = ledger.inflight.get(key)
    if task is not None:
        ledger.cache_hits += 1
        meter["cached"] = True
        return _copy(await asyncio.shield(task), True, tm)
    why = blocked()
    if why:
        ledger.denied += 1
        raise lanes.WebUnavailable(f"telem blocked: {why}")
    est = est_usd(p)
    if TELEM_PROCESS_USD > 0 and SPENT["usd"] + est > TELEM_PROCESS_USD:
        ledger.denied += 1
        raise lanes.WebUnavailable(f"telem process $ cap {TELEM_PROCESS_USD:g}")
    cost = credit_eq(est)
    why = ledger.allow(cost, final)
    if why:
        ledger.denied += 1
        raise lanes.WebUnavailable(why)
    ledger.charge(cost)
    ledger.calls += 1
    meter["credits"] = cost
    task = asyncio.create_task(_run(ledger, key, body, cost, meter, tm, p, final))
    ledger.inflight[key] = task
    return _copy(await asyncio.shield(task), False, tm)


def _copy(res, cached: bool, tm: dict):
    from . import lanes
    out = lanes.WebResult(lanes.Doc(**{**d.__dict__, "meta": dict(d.meta)}) for d in res)
    out.backend, out.cached = "telem", cached
    out.credits = 0.0 if cached else res.credits
    out.request_id = res.request_id
    info = getattr(res, "telem", None) or {}
    out.telem = info
    if not cached:
        tm.update({k: info.get(k) for k in ("runs", "usd", "usd_est", "routing", "session_id", "interaction_id")})
    else:
        tm.update(cached=True, interaction_id=info.get("interaction_id"))
    return out


async def _run(ledger, key, body, cost, meter, tm, p, final):
    from . import lanes
    t0 = time.perf_counter()
    try:
        data = await _post("/v1/search", body, TELEM_TIMEOUT_S)
        res, info = parse_search(data, keep=max(5, int(p.get("num_results") or 5)))
        info["ms"] = round((time.perf_counter() - t0) * 1000)
        actual = credit_eq(info["usd"])
        ledger.charge(actual - cost)
        SPENT["usd"] += info["usd"]
        SPENT["calls"] += 1
        res.credits = actual
        res.request_id = info.get("interaction_id")
        res.telem = info
        meter["credits"] = actual
        if TELEM_SESSIONS and info.get("session_id") and not getattr(ledger, "telem_session", None):
            ledger.telem_session = info["session_id"]
        fresh = p.get("family") in ("news", "sports", "finance", "weather") or p.get("published_after")
        ttl = config.TAVILY_FRESH_TTL_S if fresh else config.TAVILY_CACHE_TTL_S
        ledger.cache[key] = (time.monotonic() + ttl, res)
        return res
    except BaseException as e:
        ledger.charge(-cost)                   # failed calls are free (docs /pricing)
        meter["credits"] = 0.0
        if isinstance(e, Exception):
            ledger.errors += 1
            SPENT["errors"] += 1
        if isinstance(e, TelemError):
            if e.status == 400 and "session" in (e.code + str(e)).lower():
                ledger.telem_session = None    # an unknown session: start a new one next call
            _breaker(e)
        raise
    finally:
        ledger.inflight.pop(key, None)


# -------------------------------------------------------------------------------------------------------- fetch
def fetch_usd(provider: str, content: str) -> float:
    if provider == "jina":
        return len(content or "") / 4 * JINA_USD_PER_TOKEN
    return FETCH_USD.get(provider, 0.005)


async def fetch(urls: list[str], query: str = "", timeout_s: float | None = None, providers: list[str] | None = None,
                session_id: str | None = None) -> dict:
    """POST /v1/fetch for up to 5 URLs. Same contract as server/extract.extract (Tavily /extract), so the verify
    escalation can use either (server/pagefetch.py): {"pages": [{"url", "text", "provider", "ms"}], "failed": [...],
    "credits" (credit-equivalent), "usd", "usd_est", "ms", "interaction_id"}. `query` is unused by Telem (no chunk
    ranking); the caller's matcher runs on the page text. Raises TelemError (no key / cap / HTTP / timeout)."""
    urls = list(dict.fromkeys(u for u in urls if u and u.startswith("http")))[:5]
    if not urls:
        raise TelemError(400, "no url")
    if not TELEM_API_KEY:
        raise TelemError(401, "no TELEM_API_KEY")
    why = blocked()
    if why:
        raise TelemError(0, f"blocked: {why}")
    est = 0.005 * len(urls)
    if TELEM_PROCESS_USD > 0 and SPENT["usd"] + est > TELEM_PROCESS_USD:
        raise TelemError(402, f"process $ cap {TELEM_PROCESS_USD:g}")
    opts = {"tier": "minimalist", "content_format": "text", "inline_max_chars": TELEM_FETCH_MAX_CHARS}
    if providers:
        opts["providers"] = list(providers)
    body = {"urls": urls, "options": opts}
    if session_id:
        body["session_id"] = session_id
    t0 = time.perf_counter()
    try:
        data = await _post("/v1/fetch", body, timeout_s or TELEM_FETCH_TIMEOUT_S)
    except TelemError as e:
        _breaker(e)
        SPENT["errors"] += 1
        raise
    ms = round((time.perf_counter() - t0) * 1000)
    pages, failed, usd = [], [], 0.0
    for run in data.get("preprocessor_runs") or []:
        for x in (run.get("output_payload") or {}).get("fetched_results") or []:
            content = (x.get("content") or "").strip()
            soft = x.get("http_status") and int(x.get("http_status")) >= 400
            if x.get("status") == "failed" or x.get("error") or not content or soft:
                err = x.get("error")
                failed.append({"url": x.get("url", ""), "error": (err.get("code") if isinstance(err, dict) else
                               str(err or (f"http {x.get('http_status')}" if soft else "empty")))[:80]})
                if not (x.get("status") == "failed" or x.get("error")):
                    usd += fetch_usd(x.get("provider") or "", content)    # an error page that "succeeded" is billed
                continue
            usd += fetch_usd(x.get("provider") or "", content)
            pages.append({"url": x.get("url") or x.get("canonical_url") or "", "text": content,
                          "provider": x.get("provider") or "", "ms": x.get("latency_ms") or run.get("latency_ms"),
                          "truncated": bool(x.get("content_truncated"))})
    usd = round(usd, 6)
    SPENT["usd"] += usd
    SPENT["fetch_urls"] += len(urls)
    return {"pages": pages, "failed": failed, "credits": credit_eq(usd), "usd": usd, "usd_est": True,
            "ms": ms, "interaction_id": data.get("interaction_id"), "session_id": data.get("session_id")}


def pages_as_docs(r: dict, query: str = "", n_sent: int = 3) -> list:
    """Fetched pages as lane Docs (source "telem-fetch"): the sentences that best match `query`, so a page read can
    go into Keystone (keystone/hook.ingest, after the speculator embeds it) like any other web result."""
    from . import lanes
    from .keystone import cells
    q_terms = cells.toks(query) if query else []
    out = []
    for pg in r.get("pages") or []:
        text = cells.key_sentences(pg["text"], q_terms, n=n_sent) if q_terms else pg["text"][:600]
        out.append(lanes.Doc(id=pg["url"] or lanes._doc_id("telem-fetch", "", "", text), title=pg["url"], text=text,
                             url=pg["url"], source="telem-fetch",
                             meta={"provider": pg.get("provider"), "provider_ms": pg.get("ms"), "fetched": True}))
    return out


# --------------------------------------------------------------------------------------------------- health
async def probe(timeout: float = 4.0):
    """GET /v1/preprocessors (no auth, no cost): the router is reachable and lists its engines."""
    from . import lanes
    return await asyncio.wait_for(lanes.client().get(TELEM_BASE_URL + "/v1/preprocessors"), timeout)


def health() -> dict:
    return {"enabled": enabled(), "mode": TELEM_MODE, "failover": WEB_FAILOVER or None, "verify_fetch": VERIFY_FETCH,
            "spent_usd_est": round(SPENT["usd"], 4), "calls": SPENT["calls"], "errors": SPENT["errors"],
            "cap_usd": TELEM_PROCESS_USD, "blocked": blocked()}


_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def trace_ref(meter: dict) -> dict | None:
    """The ids that find this question's calls in the Telem console (no URL scheme is documented)."""
    tm = (meter or {}).get("telem") or {}
    s, i = tm.get("session_id"), tm.get("interaction_id")
    if not (s and _UUID.match(str(s))):
        return None
    return {"session_id": s, **({"interaction_id": i} if i else {})}


def question_summary(meters: list) -> dict:
    """The per-question metric's Telem fields ({} when no Telem call ran): est. $, engines that answered with their
    latency, auto-routing's choice, and the console ids (session + each call's interaction id)."""
    tms = [m.get("telem") for m in meters if m and m.get("telem")]
    if not tms:
        return {}
    calls = [t for t in tms if t.get("runs") is not None and not t.get("cached")]
    runs = [r for t in calls for r in t.get("runs") or []]
    out = {"telem_usd": round(sum(t.get("usd") or 0.0 for t in calls), 6), "telem_usd_est": True,
           "telem_calls": len(calls), "telem_cached": sum(1 for t in tms if t.get("cached")),
           "telem_engines": [{k: r.get(k) for k in ("provider", "ok", "ms", "n")} for r in runs],
           "telem_failed_engines": sum(1 for r in runs if not r.get("ok"))}
    routing = [p for t in calls for p in (t.get("routing") or [])]
    if routing:
        out["telem_routing"] = list(dict.fromkeys(routing))
    sess = next((t.get("session_id") for t in calls if t.get("session_id")), None)
    ids = [t.get("interaction_id") for t in tms if t.get("interaction_id")]
    if sess or ids:
        out["telem_trace"] = {"session_id": sess, "interaction_ids": list(dict.fromkeys(ids))}
    return out
