"""Live-data verticals: free, no-key APIs (or pure local code) that answer a whole class of question as a card.

Every vertical returns a Card: structured data, a one-line spoken answer, its source and an "as of" time. A card that
fits the question answers it without the answer model (answer.py card path); its doc text is also what the model sees
if it is ranked into an ordinary answer.

Shared here: one pooled HTTP client with tight per-request timeouts, small TTL caches, and a per-host breaker so a
dead API costs one timeout, not one per question.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx

from .. import config


@dataclass
class Card:
    vertical: str                  # weather | fx | time | units | sun
    title: str
    spoken: str                    # one line, said as the answer
    detail: str = ""               # extra lines for the model / the card body
    source: str = ""               # "Open-Meteo", "ECB via Frankfurter", "zoneinfo (local)" ...
    url: str = ""
    as_of: str = ""                # ISO time the data is valid for / was fetched
    args: dict = field(default_factory=dict)    # what was asked (place, day, amount, units): fits() checks it
    data: dict = field(default_factory=dict)    # structured payload for the UI card
    live: bool = True              # fetched now (False: computed locally, e.g. units)

    def doc(self):
        from ..lanes import Doc
        key = "|".join(f"{k}={v}" for k, v in sorted(self.args.items()))
        text = self.spoken + (("\n" + self.detail) if self.detail else "") + \
            f"\n(Source: {self.source}{', as of ' + self.as_of if self.as_of else ''})"
        return Doc(id=f"card:{self.vertical}:{key}", title=self.title, text=text, url=self.url,
                   source=self.source, score=1.0,
                   meta={"tier": "card", "card": {"vertical": self.vertical, "title": self.title,
                                                  "spoken": self.spoken, "detail": self.detail,
                                                  "source": self.source, "url": self.url, "as_of": self.as_of,
                                                  "args": self.args, "data": self.data, "live": self.live}})


class TTLCache:
    def __init__(self, ttl_s: float, max_items: int = 512):
        self.ttl, self.max = ttl_s, max_items
        self.d: dict = {}

    def get(self, k):
        v = self.d.get(k)
        if v is None or v[0] < time.monotonic():
            return None
        return v[1]

    def put(self, k, v, ttl: float | None = None):
        if len(self.d) >= self.max:
            for kk in sorted(self.d, key=lambda x: self.d[x][0])[: self.max // 4]:
                self.d.pop(kk, None)
        self.d[k] = (time.monotonic() + (self.ttl if ttl is None else ttl), v)

    def clear(self):
        self.d.clear()


_client: httpx.AsyncClient | None = None
_transport: httpx.AsyncBaseTransport | None = None      # tests: httpx.MockTransport
_down: dict[str, float] = {}                              # host -> monotonic time until which it is skipped
STATS: dict[str, dict] = {}                               # host -> {calls, fails, ms(last)}: /api/health, eval


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        kw = {"transport": _transport} if _transport is not None else {}
        _client = httpx.AsyncClient(timeout=httpx.Timeout(config.ROUTER_TIMEOUT_S, connect=config.ROUTER_TIMEOUT_S),
                                    headers={"User-Agent": config.ROUTER_UA},
                                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10,
                                                        keepalive_expiry=120), **kw)
    return _client


def use_transport(transport: httpx.AsyncBaseTransport | None):
    """Tests: route every vertical / knowledge call through `transport` (None restores the network)."""
    global _client, _transport
    _client, _transport = None, transport
    _down.clear()


def host_down(host: str) -> bool:
    return time.monotonic() < _down.get(host, 0.0)


async def get_json(url: str, params: dict | None = None, timeout: float | None = None, host: str = "",
                   headers: dict | None = None, breaker_s: float = 30.0):
    """GET -> JSON with a hard total deadline. BREAKER_FAILS consecutive network failures (timeouts, connection
    errors, 5xx) open the host's breaker for `breaker_s`: callers then go straight to their fallback instead of paying
    a timeout per question. A 4xx is an answer about this request (e.g. no such page), not about the host. Raises on
    any failure."""
    host = host or httpx.URL(url).host
    if host_down(host):
        raise ConnectionError(f"{host}: breaker open")
    t = time.perf_counter()
    st = STATS.setdefault(host, {"calls": 0, "fails": 0, "ms": None, "streak": 0})
    st["calls"] += 1
    to = config.ROUTER_TIMEOUT_S if timeout is None else timeout
    try:
        # per-request httpx timeout too: the client's default (ROUTER_TIMEOUT_S per phase) would otherwise cut a
        # slower host (SPARQL, MET Norway) off before its own deadline
        r = await asyncio.wait_for(client().get(url, params=params, headers=headers,
                                                timeout=httpx.Timeout(to, connect=min(to, 1.0))), to)
        if r.status_code >= 500:
            r.raise_for_status()
    except asyncio.CancelledError:
        raise
    except BaseException as e:
        st["fails"] += 1
        st["streak"] = st.get("streak", 0) + 1
        if st["streak"] >= BREAKER_FAILS:
            _down[host] = time.monotonic() + breaker_s
        if isinstance(e, asyncio.TimeoutError):
            raise TimeoutError(f"{host}: no response in {to:g}s") from None
        raise
    st["ms"] = round((time.perf_counter() - t) * 1000)
    st["streak"] = 0
    _down.pop(host, None)
    r.raise_for_status()                      # 4xx: raised, but the host is healthy
    return r.json()


BREAKER_FAILS = 2


async def hedged(primary, fallback, delay_s: float = 0.25):
    """First successful result of primary() and fallback(); fallback() starts after `delay_s` (or at once if primary
    fails first). A slow or dead primary API then costs at most `delay_s`, not its full timeout."""
    t1 = asyncio.ensure_future(primary())
    tasks = {t1}
    try:
        done, _ = await asyncio.wait({t1}, timeout=delay_s)
        if t1 in done and not t1.exception() and t1.result() is not None:
            return t1.result()
        t2 = asyncio.ensure_future(fallback())
        tasks.add(t2)
        pending = set(t for t in tasks if not t.done()) | ({t1} if t1.done() else set())
        err = None
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None and t.result() is not None:
                    return t.result()
                err = t.exception() or err
        if err:
            raise err
        return None
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
            elif not t.cancelled():
                t.exception()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


async def warm():
    """Open keep-alive connections to the APIs the cards use (first real call then skips TLS setup)."""
    from . import fx, weather
    await asyncio.gather(weather.warm(), fx.warm(), return_exceptions=True)
