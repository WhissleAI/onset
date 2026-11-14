"""'Keep me posted' for live questions asked again and again (a team's score, a stock), backed by a Telem monitor.

TELEM_MONITORS=1 (default off; needs TELEM_API_KEY). Design:
  1. offer(): a live-family question (sports / finance / news / weather, server/narrow.py) asked TELEM_MONITOR_REPEAT
     times (normalized query) within TELEM_MONITOR_WINDOW_S gets a "keep me posted" offer in the UI. Nothing is
     created until the user accepts: a monitor runs (and bills) on a schedule.
  2. create(): POST /v1/monitors {query, options:{frequency}, ttl}. Telem's monitor agent checks every run's results
     against the query and keeps a change list (new entity, changed value, new verdict, alert), each with the quote
     and the page it was read from. Cost (docs /pricing): $0.003-$0.015 per run, plus the agent's own searches; an
     hourly monitor is ~$2-11 / 30 days, so the default is every 6 h for 7 days and the UI says so.
  3. poll(): GET /v1/monitors/{id}/agent/changes?cursor=N (oldest first; keep next_cursor, no change is read twice).
     Each change with a value or a page becomes a Doc (source "telem-monitor", meta live + as_of), which the
     speculator embeds and keystone/hook.ingest stores like any live result: Keystone's live TTL (learned per
     relation in keystone/relkey.py) decides how long it may answer for free.
  4. end(): DELETE /v1/monitors/{id} when the user says stop (or the ttl ends it).
State (monitor ids + cursors) lives in the data dir (telem_monitors.json), never in the bundle.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from . import config, telem

TELEM_MONITORS = os.environ.get("TELEM_MONITORS", "0") == "1"
TELEM_MONITOR_FREQ = os.environ.get("TELEM_MONITOR_FREQ", "6h")     # 1h | 6h | 12h | 1d | 7d
TELEM_MONITOR_TTL = os.environ.get("TELEM_MONITOR_TTL", "7d")       # at most 30d
TELEM_MONITOR_REPEAT = int(os.environ.get("TELEM_MONITOR_REPEAT", "2"))
TELEM_MONITOR_WINDOW_S = float(os.environ.get("TELEM_MONITOR_WINDOW_S", str(7 * 86400)))
TELEM_MONITOR_MAX = int(os.environ.get("TELEM_MONITOR_MAX", "3"))    # active monitors at once
LIVE_FAMILIES = {"sports", "finance", "news", "weather"}
FREQS = {"1h", "6h", "12h", "1d", "7d"}
RUN_USD = (0.003, 0.015)                                             # per run, docs /pricing
RUNS_PER_DAY = {"1h": 24, "6h": 4, "12h": 2, "1d": 1, "7d": 1 / 7}


def _path() -> Path:
    return Path(os.environ.get("TELEM_MONITOR_PATH") or (config.DATA_DIR / "telem_monitors.json"))


class Store:
    def __init__(self, path: Path | None = None):
        self.path = path or _path()
        self.data = {"monitors": {}, "asked": {}}
        try:
            self.data.update(json.loads(self.path.read_text()))
        except (OSError, ValueError):
            pass

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1))
        tmp.replace(self.path)

    @property
    def monitors(self) -> dict:
        return self.data["monitors"]


_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def set_store(s: Store | None):
    """Tests."""
    global _store
    _store = s


def cost_note(freq: str = TELEM_MONITOR_FREQ, ttl: str = TELEM_MONITOR_TTL) -> str:
    """Plain words for the offer: what the monitor may cost over its life (est., from the docs' price range)."""
    days = float(ttl[:-1]) * (1 / 24 if ttl.endswith("h") else 1) if ttl[:-1].replace(".", "", 1).isdigit() else 7
    runs = RUNS_PER_DAY.get(freq, 4) * days
    lo, hi = runs * RUN_USD[0], runs * RUN_USD[1]
    return f"checks every {freq} for {ttl}: about ${lo:.2f}-${hi:.2f} est."


def offer(query: str, family: str, now: float | None = None) -> dict | None:
    """Count a live-family question; the offer when it was asked often enough (None otherwise)."""
    if not (TELEM_MONITORS and telem.enabled()) or family not in LIVE_FAMILIES:
        return None
    from .lanes import norm_query
    now = time.time() if now is None else now
    k = norm_query(query)
    s = store()
    if any(m.get("key") == k for m in s.monitors.values()):
        return None
    hist = [t for t in s.data["asked"].get(k, []) if now - t <= TELEM_MONITOR_WINDOW_S] + [now]
    s.data["asked"][k] = hist[-10:]
    if len(s.data["asked"]) > 500:
        s.data["asked"] = dict(list(s.data["asked"].items())[-300:])
    if len(hist) < TELEM_MONITOR_REPEAT or len(s.monitors) >= TELEM_MONITOR_MAX:
        return None
    return {"query": query, "family": family, "frequency": TELEM_MONITOR_FREQ, "ttl": TELEM_MONITOR_TTL,
            "note": cost_note()}


async def create(query: str, family: str = "", frequency: str | None = None, ttl: str | None = None) -> dict:
    if not (TELEM_MONITORS and telem.enabled()):
        raise telem.TelemError(0, "monitors off")
    freq = frequency or TELEM_MONITOR_FREQ
    if freq not in FREQS:
        raise telem.TelemError(422, f"frequency {freq}")
    body = {"query": query[:400], "options": {"frequency": freq}, "ttl": ttl or TELEM_MONITOR_TTL}
    data = await telem._post("/v1/monitors", body, 10.0)
    from .lanes import norm_query
    m = {"id": data["id"], "query": query, "key": norm_query(query), "family": family, "frequency": freq,
         "cursor": 0, "created": time.time(), "expires_at": data.get("expires_at"),
         "session_id": data.get("session_id"), "agent": bool(data.get("agent_state"))}
    s = store()
    s.monitors[m["id"]] = m
    s.save()
    return m


def change_doc(mid: str, query: str, c: dict):
    """One agent change as a lane Doc (None for run / plan bookkeeping, or a change with nothing to show)."""
    from .lanes import Doc
    if c.get("kind") in ("run", "plan_changed"):
        return None
    after = c.get("after")
    subj = (c.get("subject") or {}).get("name") or query
    url, text = "", ""
    if isinstance(after, dict):
        url = after.get("url") or ""
        val = after.get("value", after.get("status"))
        quote = after.get("quote") or ""
        text = " ".join(x for x in (f"{subj}: {val}" if val is not None else "", quote) if x)
    elif after is not None:
        text = f"{subj}: {after}"
    if not text:
        return None
    return Doc(id=url or f"telem-monitor:{mid}:{c.get('seq')}", title=f"{subj} (update {str(c.get('at') or '')[:16]})",
               text=text[:600], url=url, source="telem-monitor",
               meta={"live": True, "as_of": c.get("at"), "monitor": mid, "kind": c.get("kind"), "seq": c.get("seq")})


async def poll(mid: str) -> list:
    """New changes since the stored cursor, as Docs; the cursor moves only after a successful read."""
    from . import lanes
    s = store()
    m = s.monitors.get(mid)
    if m is None:
        return []
    r = await asyncio.wait_for(lanes.client().get(
        f"{telem.TELEM_BASE_URL}/v1/monitors/{mid}/agent/changes", params={"cursor": int(m.get("cursor") or 0)},
        headers={"Authorization": f"Bearer {telem.TELEM_API_KEY}"}), 10.0)
    if r.status_code == 404:
        s.monitors.pop(mid, None)
        s.save()
        return []
    if r.status_code != 200:
        raise telem.TelemError(r.status_code, r.text[:120])
    data = r.json()
    docs = [d for d in (change_doc(mid, m["query"], c) for c in data.get("changes") or []) if d is not None]
    m["cursor"] = int(data.get("next_cursor") or m.get("cursor") or 0)
    m["polled"] = time.time()
    s.save()
    return docs


async def poll_all() -> dict:
    out = {}
    for mid in list(store().monitors):
        try:
            out[mid] = await poll(mid)
        except Exception as e:  # noqa: BLE001 - one monitor's failure never stops the others
            out[mid] = {"error": f"{type(e).__name__}: {e}"[:120]}
    return out


async def end(mid: str) -> bool:
    from . import lanes
    s = store()
    try:
        r = await asyncio.wait_for(lanes.client().delete(
            f"{telem.TELEM_BASE_URL}/v1/monitors/{mid}", headers={"Authorization": f"Bearer {telem.TELEM_API_KEY}"}),
            10.0)
        ok = r.status_code in (200, 204, 404)
    except Exception:  # noqa: BLE001
        ok = False
    if ok:
        s.monitors.pop(mid, None)
        s.save()
    return ok
