"""Learned volatility: a TTL per relation from data, replacing the fixed 90 d / 30 d / 10 min (relkey.TTL_S).

Two sources:
  wikidata  for every relation with a Wikidata property (knowledge.PROPS via relkey.FROM_PROP): a sample of up to 3,000
            statements and their temporal qualifiers. start (P580) / end (P582): a censored-exponential change rate
            lambda = ended statements / statement-years observed. Point in time (P585, e.g. population): lambda =
            1 / the mean interval between an item's successive values. Statements without any temporal qualifier
            on a property that never carries them (born, founded) give lambda ~ 0: stable.
  observed  for the live relations (weather, prices, scores, news, schedules, hours): the same questions re-asked
            through today's path at t0, t0 + 15 min and t0 + 60 min (free lanes + Wikipedia, no Tavily); the share
            whose answer changed gives a per-minute hazard.
TTL = the time by which the answer has changed with probability 5 %: -ln(0.95) / lambda, clamped to [5 min, 365 d].
Writes server/keystone/ttl_model.json (relation -> seconds, plus class fallbacks) and runs/index/ttl_fit.json.

    .venv/bin/python eval/keystone/fit_ttl.py [--live] [--no-wikidata]
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

os.environ["WEB_BACKEND"] = "wikipedia"
os.environ["TAVILY_API_KEY"] = ""
os.environ["INDEX"] = "0"
os.environ["WEB_CACHE_RECORD"] = "0"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import httpx  # noqa: E402

from server import knowledge  # noqa: E402
from server.keystone.relkey import FROM_PROP, RELATIONS, TTL_S  # noqa: E402

OUT = ROOT / "runs/index"
MODEL = ROOT / "server/keystone/ttl_model.json"
P_CHANGE = 0.05
LO, HI = 300.0, 365 * 86400.0
from server import config  # noqa: E402
UA = {"User-Agent": config.ROUTER_UA, "Accept": "application/sparql-results+json"}
SPARQL = "https://query.wikidata.org/sparql"
NOW = dt.datetime(2026, 11, 14, tzinfo=dt.timezone.utc)


def _t(s):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


async def q(client, query):
    for attempt in range(4):
        try:
            r = await client.get(SPARQL, params={"query": query, "format": "json"}, headers=UA, timeout=45)
            if r.status_code == 429:
                await asyncio.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:  # noqa: BLE001
            print("  sparql retry", type(e).__name__, str(e)[:80], flush=True)
            await asyncio.sleep(5 * (attempt + 1))
    return []


async def wikidata_rates():
    out = {}
    async with httpx.AsyncClient() as client:
        for prop, rel in FROM_PROP.items():
            pid = knowledge.PROPS[prop][0][0]
            se = await q(client, f"""SELECT ?s ?start ?end WHERE {{ ?i p:{pid} ?s . ?s pq:P580 ?start .
                                     OPTIONAL {{ ?s pq:P582 ?end }} }} LIMIT 3000""")
            pit = await q(client, f"""SELECT ?i ?t WHERE {{ ?i p:{pid} ?s . ?s pq:P585 ?t . }} LIMIT 3000""")
            smp = await q(client, f"""SELECT ?s ?st WHERE {{ ?i p:{pid} ?s . OPTIONAL {{ ?s pq:P580 ?st }} }} LIMIT 3000""")
            n_all = len(smp)
            n_with = sum(1 for b in smp if "st" in b)
            ended, years = 0, 0.0
            for b in se:
                s = _t(b["start"]["value"])
                e = _t(b["end"]["value"]) if "end" in b else None
                if s is None or s.year < 1800:
                    continue
                end = e if e is not None and e > s else NOW
                years += max(0.0, (min(end, NOW) - s).days / 365.25)
                ended += e is not None and e > s and e <= NOW
            by = {}
            for b in pit:
                t = _t(b["t"]["value"])
                if t is not None and 1800 < t.year <= NOW.year:
                    by.setdefault(b["i"]["value"], set()).add(t)
            gaps = []
            for ts in by.values():
                ts = sorted(ts)
                gaps += [(b - a).days / 365.25 for a, b in zip(ts, ts[1:]) if (b - a).days > 0]
            lam_se = ended / years if years > 0 else 0.0
            lam_pit = 1.0 / (sum(gaps) / len(gaps)) if len(gaps) >= 20 else 0.0
            # qualifier coverage: a property whose statements carry start/end at all (one holder at a time) vs. one
            # that never changes (born, founded): weight the rate by the share of statements that are time-bounded
            share_se = n_with / n_all if n_all else 0.0
            lam = max(lam_se * (1.0 if share_se > 0.02 else share_se / 0.02), lam_pit)
            ttl = HI if lam <= 0 else -math.log(1 - P_CHANGE) / lam * 365.25 * 86400
            out[rel] = {"pid": pid, "statements_with_start": len(se), "ended": ended, "years": round(years, 1),
                        "pit_items": len(by), "pit_gaps": len(gaps), "sample_statements": n_all,
                        "share_with_start": round(share_se, 4),
                        "lambda_per_year": round(lam, 5), "ttl_s": round(min(HI, max(LO, ttl))),
                        "ttl_days": round(min(HI, max(LO, ttl)) / 86400, 2),
                        "fixed_days": round(TTL_S[RELATIONS[rel][1]] / 86400, 2)}
            print(rel, out[rel], flush=True)
            await asyncio.sleep(1.0)
    return out


async def observed_live(n=80, waits=(15, 45)):
    """Re-ask live questions through today's path (no Tavily) at t0, +15 min, +60 min; share whose top answer changed."""
    sys.path.insert(0, str(ROOT / "eval/keystone"))
    import record_fixture as RF
    from server import router
    fx = json.loads((OUT / "kdir_fixture.json").read_text())
    from server import searchcache
    from replay_frames import frame_for
    picks = []
    for t, r in fx.items():
        if r.get("error"):
            continue
        v = (r.get("plan") or {}).get("vertical", "")
        live = r["tier"] == "card" or (r.get("plan") or {}).get("why", "").startswith("fresh")
        if live:
            picks.append((t, v or "web-fresh"))
    by = {}
    for t, v in picks:
        by.setdefault(v, []).append(t)
    sel = []
    for v, ts in sorted(by.items()):
        sel += [(t, v) for t in ts[: max(4, n // max(1, len(by)))]]
    await router.warm()
    sem = asyncio.Semaphore(3)

    async def snap():
        rows = await asyncio.gather(*[RF.one(t, sem) for t, _ in sel])
        return [(r["docs"][0]["text"][:300] if r["docs"] else "") for r in rows]
    s0 = await snap()
    res = {"n": len(sel), "verticals": {}}
    elapsed = 0
    snaps = [s0]
    for w in waits:
        await asyncio.sleep(w * 60)
        elapsed += w
        snaps.append(await snap())
    for v in sorted({v for _, v in sel}):
        idx = [i for i, (_, vv) in enumerate(sel) if vv == v]
        d = {}
        el = 0
        for k, w in enumerate(waits):
            el += w
            ch = sum(1 for i in idx if snaps[k + 1][i] != s0[i])
            d[f"changed_{el}m"] = round(ch / len(idx), 3)
        # hazard from the longest interval: P(change by t) = 1 - exp(-lam t)
        p = max(1e-3, min(0.999, d[f"changed_{el}m"]))
        lam = -math.log(1 - p) / (el * 60)
        d["n"] = len(idx)
        d["ttl_s"] = round(min(HI, max(LO, -math.log(1 - P_CHANGE) / lam)))
        res["verticals"][v] = d
        print(v, d, flush=True)
    return res


VERT_REL = {"weather": "weather_in", "fx": "price_of", "web-fresh": "news_about", "time": None, "sun": None,
            "units": None}


def main(a):
    rep = json.loads((OUT / "ttl_fit.json").read_text()) if (OUT / "ttl_fit.json").exists() else {}
    if not a.no_wikidata:
        rep["wikidata"] = asyncio.run(wikidata_rates())
    if a.live:
        rep["live"] = asyncio.run(observed_live())
    ttl = {}
    for rel, d in (rep.get("wikidata") or {}).items():
        ttl[rel] = d["ttl_s"]
    live = (rep.get("live") or {}).get("verticals") or {}
    for v, d in live.items():
        rel = VERT_REL.get(v)
        if rel:
            ttl[rel] = d["ttl_s"]
    if live:
        ttl["class:live"] = min(d["ttl_s"] for d in live.values())
    stable = [v for k, v in ttl.items() if RELATIONS.get(k, ("", ""))[1] == "stable"]
    slow = [v for k, v in ttl.items() if RELATIONS.get(k, ("", ""))[1] == "slow"]
    if stable:
        ttl["class:stable"] = sorted(stable)[len(stable) // 2]
    if slow:
        ttl["class:slow"] = sorted(slow)[len(slow) // 2]
    MODEL.write_text(json.dumps({"ttl_s": ttl, "p_change": P_CHANGE, "source": "eval/keystone/fit_ttl.py"}, indent=1))
    (OUT / "ttl_fit.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(ttl, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-wikidata", action="store_true")
    main(ap.parse_args())
