"""Tavily /extract as the verify escalation (server/extract.py, VERIFY_EXTRACT), measured on recorded A/B rows.

  .venv/bin/python eval/extract_eval.py runs/tavily/ab_*.json                 # dry run: eligible rows + est. credits
  .venv/bin/python eval/extract_eval.py runs/tavily/ab_*.json --probe 10 --yes   # latency / credits on 10 cited urls
  .venv/bin/python eval/extract_eval.py runs/tavily/ab_*.json --confirm --yes    # confirm rate on unconfirmed claims

Rows need `source_docs` (eval/run_ab.py writes them). --confirm takes every row whose verify badge left the key claim
unconfirmed (verify status disagree / unverified / partial with a key claim, loose names as VERIFY_EXTRACT uses them,
and no free re-check that confirmed it), reads the cited url with the claim sentence as the query (exactly
extract.recheck_page), and reports: confirmed / eligible, credits, latency p50 / p90, and whether the confirmed answers
were scored correct. Spends credits only with --yes (basic: 1 credit per 5 successful urls).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from server import extract, recheck  # noqa: E402


def rows(paths):
    for p in paths:
        for r in json.loads(Path(p).read_text()).get("rows") or []:
            r["_file"] = Path(p).stem
            yield r


def eligible(r) -> dict | None:
    v = r.get("verify") or {}
    if not r.get("source_docs") or v.get("status") not in ("disagree", "unverified", "partial"):
        return None
    if (r.get("recheck") or {}).get("status") in ("rechecked", "corrected"):
        return None
    c = recheck.key_claim(v, loose=True)
    if not c or not extract.cited_url(r.get("answer") or "", r["source_docs"], c.get("text", "")):
        return None
    return recheck._full(c)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


async def main(a):
    rs = list(rows(a.files))
    el = [(r, c) for r in rs if (c := eligible(r))]
    print(f"{len(rs)} rows, {len(el)} with an unconfirmed key claim and a cited web url "
          f"(est. {0.2 * len(el):.1f} credits)")
    if not a.yes:
        print("dry run (pass --yes)")
        return
    if a.probe:
        urls = list(dict.fromkeys(extract.cited_url(r.get("answer") or "", r["source_docs"])
                                  for r in rs if r.get("source_docs")))
        urls = [u for u in urls if u][: a.probe]
        qs = {extract.cited_url(r.get("answer") or "", r["source_docs"]): r["text"] for r in rs if r.get("source_docs")}
        res = []
        for u in urls:
            try:
                x = await extract.extract([u], query=qs.get(u, ""))
                res.append({"url_ok": bool(x["pages"]), "ms": x["ms"], "server_s": x["response_time"],
                            "credits": x["usage"], "chars": sum(len(p["text"]) for p in x["pages"])})
            except extract.ExtractError as e:
                res.append({"url_ok": False, "error": str(e)})
        ms = [x["ms"] for x in res if "ms" in x]
        print(json.dumps({"probe_urls": len(urls), "ok": sum(x["url_ok"] for x in res), "p50_ms": pct(ms, .5),
                          "p90_ms": pct(ms, .9), "max_ms": max(ms) if ms else None,
                          "usage_credits_reported": [x.get("credits") for x in res],
                          "mean_chars": round(statistics.mean([x.get("chars", 0) for x in res]), 0)}, indent=1))
    if a.confirm:
        out = []
        for r, c in el:
            pg = await extract.recheck_page(c, r.get("answer") or "", r["source_docs"])
            out.append({"id": r["id"], "mode": r["mode"], "set": r["_file"], "claim": c["text"], "kind": c["kind"],
                        "status": pg["status"], "ms": pg.get("ms"), "credits": pg.get("credits"),
                        "error": pg.get("error"), "correct": r.get("correct")})
            print(f"  {r['id']:6s} {r['mode']:9s} {pg['status']:11s} {pg.get('ms')!s:>5} ms  {c['kind']:6s} "
                  f"{c['text'][:40]!r} correct={r.get('correct')}")
        ok = [x for x in out if x["status"] == "rechecked"]
        ms = [x["ms"] for x in out if x.get("ms")]
        summ = {"eligible": len(out), "confirmed": len(ok), "confirm_rate": round(len(ok) / len(out), 3) if out else None,
                "errors": sum(1 for x in out if x.get("error")), "p50_ms": pct(ms, .5), "p90_ms": pct(ms, .9),
                "credits_metered": round(sum(x.get("credits") or 0 for x in out), 2),
                "confirmed_and_scored_wrong": sum(1 for x in ok if x["correct"] is False)}
        print(json.dumps(summ, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps({"summary": summ, "rows": out}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--out", default="")
    asyncio.run(main(ap.parse_args()))
