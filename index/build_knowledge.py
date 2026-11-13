"""Build the local Wikipedia lead index for the router's facts tier (server/knowledge.py).

    .venv/bin/python index/build_knowledge.py                 # vital articles (level 4) + top-viewed, ~12k leads
    .venv/bin/python index/build_knowledge.py --limit 2000    # quick build
    .venv/bin/python index/build_knowledge.py --skip-fetch    # re-embed the leads already fetched

Articles: Wikipedia's Vital Articles level 4 (~10k: the stable core of people, places, science, history, arts) plus the
1,000 most-viewed articles of each of the last N months (popular). Text: each article's lead section (plain text,
first 1,200 chars) from the MediaWiki API, 20 titles per request, a few requests in flight, descriptive User-Agent.
Vectors: EmbeddingGemma 2 through server/embed.py (encode_docs with titles, first 600 chars: the same call the server
uses for web results), float16.

Output (git-ignored): index/data/knowledge/{leads.jsonl, vectors.npy, stats.json}. Size ~40 MB for 12k leads.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import config  # noqa: E402

OUT = config.KNOWLEDGE_DIR
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": config.ROUTER_UA}
SKIP_PREFIX = ("Main_Page", "Special:", "Wikipedia:", "File:", "Portal:", "Help:", "Category:", "Talk:", "Template:",
               "User:", "Draft:", "-")


async def _get(c: httpx.AsyncClient, url: str, params: dict | None = None, tries: int = 4) -> dict:
    for k in range(tries):
        try:
            r = await c.get(url, params=params)
            if r.status_code == 429:
                await asyncio.sleep(2 + 2 * k)
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            if k == tries - 1:
                raise
            await asyncio.sleep(1 + k)
    return {}


async def vital_titles(c) -> list[str]:
    pages, cont = [], {}
    while True:
        d = await _get(c, API, {"action": "query", "list": "allpages", "apnamespace": 4, "format": "json",
                                "apprefix": "Vital articles/Level 4/", "apfilterredir": "nonredirects",
                                "aplimit": "max", **cont})
        pages += [p["title"] for p in d["query"]["allpages"]
                  if not any(x in p["title"] for x in ("Article alerts", "Removed", "Archive"))]
        if "continue" not in d:
            break
        cont = d["continue"]
    titles: list[str] = []
    for pg in pages:
        cont = {}
        while True:
            d = await _get(c, API, {"action": "query", "prop": "links", "titles": pg, "plnamespace": 0,
                                    "pllimit": "max", "format": "json", "formatversion": 2, **cont})
            for p in d["query"]["pages"]:
                titles += [x["title"] for x in p.get("links", [])]
            if "continue" not in d:
                break
            cont = d["continue"]
        print(f"  {pg}: {len(titles)} titles so far", flush=True)
    return titles


async def popular_titles(c, months: int) -> list[str]:
    out = []
    m = dt.date.today().replace(day=1)
    for _ in range(months):
        m = (m - dt.timedelta(days=1)).replace(day=1)        # previous full month
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/"
               f"{m.year}/{m.month:02d}/all-days")
        try:
            d = await _get(c, url)
        except Exception as e:
            print(f"  pageviews {m:%Y-%m}: {e}", flush=True)
            continue
        arts = d["items"][0]["articles"]
        out += [a["article"].replace("_", " ") for a in arts if not a["article"].startswith(SKIP_PREFIX)]
        print(f"  pageviews {m:%Y-%m}: {len(arts)}", flush=True)
    return out


async def fetch_leads(c, titles: list[str], conc: int = 4) -> list[dict]:
    batches = [titles[i:i + 20] for i in range(0, len(titles), 20)]
    out: dict[str, dict] = {}
    sem = asyncio.Semaphore(conc)
    done = [0]

    async def one(b):
        async with sem:
            d = await _get(c, API, {"action": "query", "prop": "extracts|info", "exintro": 1, "explaintext": 1,
                                    "exlimit": 20, "inprop": "url", "titles": "|".join(b), "redirects": 1,
                                    "format": "json", "formatversion": 2})
        for p in (d.get("query") or {}).get("pages", []):
            text = (p.get("extract") or "").strip()
            if len(text) < 80 or p.get("missing"):
                continue
            out[p["title"]] = {"title": p["title"], "text": " ".join(text.split())[:1200], "url": p.get("fullurl", "")}
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"  leads: {done[0]}/{len(batches)} requests, {len(out)} articles", flush=True)

    await asyncio.gather(*[one(b) for b in batches])
    return list(out.values())


def embed_all(docs: list[dict], bs: int = 64) -> np.ndarray:
    from server import embed
    vs = []
    t = time.perf_counter()
    for i in range(0, len(docs), bs):
        b = docs[i:i + bs]
        vs.append(embed.encode_docs([d["text"][:600] for d in b], [d["title"] for d in b]))
        if (i // bs) % 20 == 0:
            print(f"  embedded {i + len(b)}/{len(docs)} ({time.perf_counter() - t:.0f}s)", flush=True)
    return np.concatenate(vs).astype(np.float16)


async def main_async(a):
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"built": time.strftime("%Y-%m-%d"), "t_start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    leads_path = OUT / "leads.jsonl"
    if a.skip_fetch and leads_path.exists():
        docs = [json.loads(x) for x in leads_path.read_text().splitlines() if x.strip()]
    else:
        t = time.perf_counter()
        async with httpx.AsyncClient(headers=UA, timeout=20, http2=False) as c:
            vt = await vital_titles(c)
            pt = await popular_titles(c, a.months)
            titles = list(dict.fromkeys(vt + pt))
            if a.limit:
                titles = titles[: a.limit]
            print(f"{len(vt)} vital + {len(pt)} popular -> {len(titles)} titles", flush=True)
            docs = await fetch_leads(c, titles)
        stats.update(vital=len(vt), popular=len(pt), titles=len(titles), fetch_s=round(time.perf_counter() - t, 1))
        leads_path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs))
    print(f"{len(docs)} leads; embedding...", flush=True)
    t = time.perf_counter()
    vec = embed_all(docs)
    np.save(OUT / "vectors.npy", vec)
    stats.update(leads=len(docs), embed_s=round(time.perf_counter() - t, 1), dim=int(vec.shape[1]),
                 bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--skip-fetch", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
