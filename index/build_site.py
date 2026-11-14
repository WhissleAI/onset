"""Read one website with Tavily (/map, then /extract) into the site brain (server/sitebrain.py). Offline, once.

  .venv/bin/python index/build_site.py                       # dry run: what it would send and the credit estimate
  .venv/bin/python index/build_site.py --yes                 # spend (est. 5 + 6 = 11 credits with the defaults)
  .venv/bin/python index/build_site.py --rebuild             # re-chunk / re-embed the saved pages, no network

Defaults read docs.tavily.com's documentation pages. Writes <DATA_DIR>/sites/<site>/ (HEADSTART_DATA_DIR = the Mac
app's data dir; a repo run: runs/sites/){pages.json, site.json,
passages.jsonl, vectors.npz} (git-ignored). Cost (docs.tavily.com/documentation/api-credits, 2026-11-14): /map 1 credit
per 10 pages (2 with instructions, not used here); /extract basic 1 credit per 5 successful URLs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import config, sitebrain  # noqa: E402


async def post(path: str, body: dict, timeout: float) -> dict:
    from server import lanes
    r = await lanes.client().post(config.TAVILY_BASE_URL + path, json=body, timeout=timeout,
                                  headers={"Authorization": f"Bearer {config.TAVILY_API_KEY}"})
    if r.status_code != 200:
        raise SystemExit(f"{path}: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def encode(passages: list[dict], batch: int) -> "np.ndarray":
    """EmbeddingGemma 2 document vectors with the text-only CPU copy (server/embed.qmodel: ~1-2 GB RSS, not the
    multimodal MPS model, which reached 11 GB on 2.5k passages), small batches; same weights and space as encode_docs."""
    import numpy as np
    from server import embed
    m = embed.qmodel()
    out = []
    for i in range(0, len(passages), batch):
        b = passages[i:i + batch]
        x = m.encode([f"title: {p['title'] or 'none'} | text: {p['text']}" for p in b], batch_size=batch)
        out.append(embed._norm(x).astype(np.float16))
        if i // batch % 25 == 0:
            print(f"  embedded {i + len(b)}/{len(passages)}", flush=True)
    return np.concatenate(out) if out else np.zeros((0, 768), np.float16)


def pick(urls: list[str], want: list[str], skip: list[str], n: int) -> list[str]:
    """Documentation pages first (the `want` path fragments, in order), no changelog / blog / integrations noise."""
    urls = [u.split("#")[0].rstrip("/") for u in urls]
    urls = list(dict.fromkeys(u for u in urls if not any(s in u for s in skip)))
    ranked = sorted(urls, key=lambda u: next((i for i, w in enumerate(want) if w in u), len(want)))
    return ranked[:n]


async def main(a) -> int:
    out = (Path(a.out_root).expanduser() if a.out_root else config.DATA_DIR / "sites") / a.site
    n_map, n_ext = a.map_limit, a.pages
    est = math.ceil(n_map / 10) + math.ceil(n_ext / 5)
    print(f"site {a.site}: /map {a.url} (limit {n_map}, depth {a.depth}) + /extract {n_ext} pages "
          f"(basic) -> est. {est} credits; out {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out.name}")
    pages_path = out / "pages.json"
    if a.rebuild:
        pages = json.loads(pages_path.read_text())
    else:
        if not (a.yes and config.TAVILY_API_KEY):
            print("dry run (pass --yes with TAVILY_API_KEY set)")
            return 0
        t = time.perf_counter()
        m = await post("/map", {"url": a.url, "max_depth": a.depth, "max_breadth": 100, "limit": n_map,
                                "allow_external": False, "include_usage": True,
                                **({"select_paths": a.select} if a.select else {})}, 150)
        map_s = time.perf_counter() - t
        urls = m.get("results") or []
        print(f"/map: {len(urls)} urls in {map_s:.1f} s, credits {(m.get('usage') or {}).get('credits')}")
        chosen = pick(urls, a.want, a.skip, n_ext)
        pages, ext_credits, ext_s = [], 0.0, 0.0
        for i in range(0, len(chosen), 20):
            t = time.perf_counter()
            e = await post("/extract", {"urls": chosen[i:i + 20], "extract_depth": "basic", "format": "markdown",
                                        "include_usage": True, "timeout": 30}, 40)
            ext_s += time.perf_counter() - t
            ext_credits += float((e.get("usage") or {}).get("credits") or 0)
            for x in e.get("results") or []:
                pages.append({"url": x["url"], "text": x.get("raw_content") or ""})
            for x in e.get("failed_results") or []:
                print("  failed:", x.get("url"), x.get("error"))
        print(f"/extract: {len(pages)} pages in {ext_s:.1f} s, credits {ext_credits:g}")
        out.mkdir(parents=True, exist_ok=True)
        pages_path.write_text(json.dumps({"site": a.site, "read": time.strftime("%Y-%m-%d"), "map_urls": len(urls),
                                          "map_credits": (m.get("usage") or {}).get("credits"),
                                          "extract_credits": ext_credits, "map_s": round(map_s, 2),
                                          "extract_s": round(ext_s, 2), "pages": pages}, indent=1))
        pages = json.loads(pages_path.read_text())
    passages, seen = [], set()
    for p in pages["pages"]:
        title = next((ln.lstrip("# ").strip() for ln in p["text"].splitlines() if ln.startswith("# ")), "") or \
            p["url"].rsplit("/", 1)[-1].replace("-", " ")
        for i, c in enumerate(sitebrain.chunk(sitebrain.clean(p["text"]))):
            if c in seen:                     # navigation / banners repeated on every page
                continue
            seen.add(c)
            passages.append({"url": p["url"].removesuffix(".md"), "title": f"{title} · {a.site}", "i": i, "text": c})
    import numpy as np
    vecs = encode(passages, a.batch)
    np.savez(out / "vectors.npz", vecs=np.asarray(vecs, np.float16))
    (out / "passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages))
    (out / "site.json").write_text(json.dumps({"name": a.site, "terms": a.terms, "as_of": pages["read"],
                                               "pages": len(pages["pages"]), "passages": len(passages)}, indent=1))
    print(f"wrote {len(pages['pages'])} pages -> {len(passages)} passages ({vecs.shape[1]}-d)")
    return 0


def cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="docs.tavily.com")
    ap.add_argument("--url", default="https://docs.tavily.com")
    ap.add_argument("--terms", nargs="*", default=["tavily"], help="a question must name one to be served from here")
    ap.add_argument("--select", nargs="*", default=["/documentation/.*", "/sdk/python/.*"])
    ap.add_argument("--want", nargs="*", default=["/api-reference/endpoint/", "api-credits", "rate-limits",
                                                  "best-practices", "/about", "/quickstart", "/sdk/python/"])
    ap.add_argument("--skip", nargs="*", default=["changelog", "/integrations/", "/partnerships", "/enterprise"])
    ap.add_argument("--map-limit", type=int, default=50)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--pages", type=int, default=30)
    ap.add_argument("--out-root", default="", help="default: <DATA_DIR>/sites (Mac app: Application Support/Onset)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    sys.exit(asyncio.run(main(ap.parse_args())))


if __name__ == "__main__":
    cli()
