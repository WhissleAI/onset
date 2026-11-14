"""Site brain: one website read once with Tavily (/map + /extract, offline) and kept in Onset's local tier.

Spoken questions about that site ("how many credits is an advanced search on Tavily?") are answered from the stored
pages at 0 paid calls. It sits in the Keystone local tier (server/keystone/hook.wrap_web): after the Keystone lookup
misses and before the paid web lane, for speculative and final calls alike. Labelled like every local answer:
source "Onset index · <site>", snapshot "from your Onset index · <site> · as of <date it was read>".

Gate (cheap first): the question must name the site (one of its `terms`, e.g. "tavily": a word check, no embedding);
then one query embedding (shared with the speculator's when it is in flight), cosine over the site's passages
(EmbeddingGemma 2, same space as the web pool); served when the best passage >= SITE_MIN_SIM. Anything else (no
site loaded, no term, low similarity, timeout, error) = None: the paid lane runs as before.

Data: <DATA_DIR>/sites/<site>/ (the Mac app's data dir; repo runs: runs/) or <INDEX_DIR>/sites/<site>/, each
{site.json, passages.jsonl, vectors.npz} (git-ignored), written by index/build_site.py (--out). SITE_BRAIN=0 turns it off.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from . import config

SITE_BRAIN = os.environ.get("SITE_BRAIN", "1") == "1"
SITE_MIN_SIM = float(os.environ.get("SITE_MIN_SIM", "0.5"))
SITE_K = int(os.environ.get("SITE_K", "4"))
SITE_TIMEOUT_MS = float(os.environ.get("SITE_TIMEOUT_MS", "800"))   # query embedding (only for scoped questions)
SITE_TERM_SIM = float(os.environ.get("SITE_TERM_SIM", "0.66"))
SITE_DIR = None          # tests: override the sites directory


@dataclass
class Site:
    name: str
    terms: list[str]
    as_of: str
    passages: list[dict] = field(default_factory=list)
    vecs: np.ndarray | None = None

    def scoped(self, text: str) -> bool:
        """The question names the site: a term, or a near-miss of one as the ear wrote it ("tavali" for "tavily":
        same first 3 letters, string similarity >= SITE_TERM_SIM, words of 5+ letters)."""
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        for t in self.terms:
            if t in words:
                return True
            if len(t) >= 5 and any(len(w) >= 5 and w[:3] == t[:3] and SequenceMatcher(None, w, t).ratio() >= SITE_TERM_SIM
                                   for w in words):
                return True
        return False


_sites: list[Site] | None = None
STATS = {"lookups": 0, "scoped": 0, "hits": 0, "errors": 0}


def sites_dirs() -> list:
    """The app's data dir first (Mac app: ~/Library/Application Support/Onset/sites, writable), then the index dir."""
    if SITE_DIR is not None:
        return [SITE_DIR]
    return list(dict.fromkeys([config.DATA_DIR / "sites", config.INDEX_DIR / "sites"]))


def load(force: bool = False) -> list[Site]:
    """Every built site under sites_dir() (once per process)."""
    global _sites
    if _sites is not None and not force:
        return _sites
    out, seen = [], set()
    for d in sites_dirs():
        try:
            subs = sorted(p for p in d.iterdir() if p.is_dir() and p.name not in seen)
        except OSError:
            continue
        for sd in subs:
            try:
                meta = json.loads((sd / "site.json").read_text())
                ps = [json.loads(x) for x in (sd / "passages.jsonl").read_text().splitlines() if x.strip()]
                v = np.load(sd / "vectors.npz")["vecs"].astype(np.float32)
                if len(ps) == len(v) and len(ps):
                    out.append(Site(meta["name"], [t.lower() for t in meta.get("terms") or []], meta.get("as_of", ""),
                                    ps, v))
                    seen.add(sd.name)
            except (OSError, ValueError, KeyError):
                continue
    _sites = out
    return out


def set_sites(sites: list[Site] | None):
    """Tests."""
    global _sites
    _sites = sites


def _text(f, query: str) -> str:
    return " ".join(x for x in (getattr(f, "resolved_text", "") or "", getattr(f, "inline", "") or "",
                                " ".join(getattr(f, "words", []) or []), query) if x)


def search(site: Site, qvec: np.ndarray, k: int = SITE_K) -> list[tuple[dict, float]]:
    sims = site.vecs @ np.asarray(qvec, np.float32)[: site.vecs.shape[1]]
    top = np.argsort(-sims)[:k]
    return [(site.passages[i], float(sims[i])) for i in top]


def docs_of(site: Site, hits: list[tuple[dict, float]]) -> list:
    from .lanes import Doc
    label = f"from your Onset index · {site.name} · as of {site.as_of}"
    return [Doc(id=f"site:{p['url']}#{p.get('i', 0)}", title=p.get("title") or site.name, text=p["text"], url=p["url"],
                source=f"Onset index · {site.name}", score=s,
                meta={"tier": "index", "snapshot": label, "as_of": site.as_of, "index_path": "site",
                      "index_scope": "site", "site": site.name}) for p, s in hits]


async def lookup(f, query: str, meter: dict | None = None) -> list | None:
    """Docs from a loaded site, or None (see module doc). Never raises."""
    meter = meter if meter is not None else {}
    if not SITE_BRAIN:
        return None
    try:
        sites = load()
        if not sites:
            return None
        STATS["lookups"] += 1
        text = _text(f, query)
        mine = [s for s in sites if s.scoped(text)]
        if not mine:
            return None
        STATS["scoped"] += 1
        t0 = time.perf_counter()
        from . import embed
        from .keystone import hook
        qtext = getattr(f, "resolved_text", "") or getattr(f, "inline", "") or query
        shared = hook._QV.get(qtext)
        budget = SITE_TIMEOUT_MS / 1000
        qv = None
        if shared is not None and not shared.done():
            done, _ = await asyncio.wait({shared}, timeout=budget)
            if shared in done and not shared.cancelled() and shared.exception() is None:
                qv = shared.result()[0]
        elif shared is not None and not shared.cancelled() and shared.exception() is None:
            qv = shared.result()[0]
        if qv is None:
            qv = (await asyncio.wait_for(embed.arun(embed.encode_queries, [qtext]), budget))[0]
        best_site, best = None, []
        for s in mine:
            h = search(s, qv)
            if h and (not best or h[0][1] > best[0][1]):
                best_site, best = s, h
        meter["site_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        meter["site_sim"] = round(best[0][1], 3) if best else None
        if not best or best[0][1] < SITE_MIN_SIM:
            return None
        STATS["hits"] += 1
        return docs_of(best_site, [(p, s) for p, s in best if s >= SITE_MIN_SIM - 0.1])
    except Exception as e:  # noqa: BLE001  (timeout included: the paid lane runs as before)
        STATS["errors"] += 1
        meter["site_error"] = f"{type(e).__name__}: {e}"[:120]
        return None


async def local(f, query: str, meter: dict):
    """keystone.hook.wrap_web's hook: a LocalResult (served like a Keystone hit, path "site") or None."""
    docs = await lookup(f, query, meter)
    if not docs:
        return None
    from .keystone.hook import LocalResult
    meter["site"] = docs[0].meta.get("site")
    try:
        as_of = time.mktime(time.strptime(docs[0].meta.get("as_of", ""), "%Y-%m-%d"))
    except ValueError:
        as_of = time.time()
    return LocalResult(docs, "site", float(docs[0].score), as_of, "Onset index")


def clean(md: str) -> str:
    """Crawled markdown -> prose for passages (build time): no code blocks, images, quote banners, link targets,
    heading anchors or navigation rows; tables kept as text."""
    t = re.sub(r"```.*?```", " ", md or "", flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                       # images
    t = re.sub(r"\[\u200b?\]\(#[^)]*\)\s*", "", t)                    # heading anchors "[​](#x)"
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)                     # links -> their text
    keep = []
    for ln in t.splitlines():
        s = ln.strip()
        if s.startswith(">") or not s:
            keep.append("" if not s else None)
            continue
        if len(re.sub(r"[^A-Za-z]", "", s)) < 3:
            continue
        keep.append(ln)
    return "\n".join(x for x in keep if x is not None)


def chunk(text: str, max_chars: int = 700) -> list[str]:
    """Page text -> passages of <= max_chars at paragraph / sentence boundaries (build time)."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\n(?=#)", text or "")]
    out, cur = [], ""
    for p in paras:
        if not p:
            continue
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if len(cur) + len(p) + 1 > max_chars and cur:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    return [c for c in out if len(c) >= 40]
