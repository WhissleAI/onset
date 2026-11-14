"""Phase 2 offline measure (text path, untimed, no lock): verify badges and the verify-triggered re-search.

For each question (its scripted text): Arc Tagger frame -> Wikipedia search (query + entity, as the live web lane on
WEB_BACKEND=wikipedia) -> the real answer model -> verify.check -> recheck.run (free lanes only unless --paid).
Reports per set: badge distribution, false alarms ("disagree" on an answer the keyword scorer marks right), how many
answers the re-check confirmed / corrected / left, accuracy before and after the patch, added paid calls, re-check ms.

  WEB_BACKEND=wikipedia .venv/bin/python eval/recheck_eval.py --sets multipart,questions,showcase --out runs/loop/recheck_eval.json
"""
import argparse, asyncio, collections, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
os.environ.setdefault("VERIFY_RESEARCH", "1")
from run_ab import score  # noqa: E402
from replay_frames import frame_for  # noqa: E402
from server import answer, config, lanes, recheck, verify  # noqa: E402

SETS = {"multipart": ROOT / "eval/multipart/questions.jsonl", "questions": ROOT / "eval/questions.jsonl",
        "showcase": ROOT / "eval/showcase/questions.jsonl"}
ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="multipart,questions,showcase")
ap.add_argument("--paid", action="store_true")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--out", default=str(ROOT / "runs/loop/recheck_eval.json"))
args = ap.parse_args()


async def one(q, sem):
    async with sem:
        f = frame_for(q["text"])
        try:
            docs = (await lanes.wikipedia(f.query() or q["text"], k=args.k, entity=f.entity_query()))[:args.k + 2]
        except Exception as e:
            return {"id": q["id"], "error": repr(e)[:120]}
        if not docs:
            return {"id": q["id"], "error": "no docs"}
        st = answer.Stream(f.resolved_text, docs, tag="final")
        async for _ in st.follow():
            pass
        text = st.text()
        v = verify.check(text, docs, f.resolved_text)
        row = {"id": q["id"], "text": q["text"], "answer": text, "verify": v["status"], "label": v["label"],
               "correct": score(text, q.get("expect") or []) if q.get("expect") else None}
        if recheck.needs(v):
            async def web(qq):
                return await lanes.wikipedia(qq, k=3)
            r = await recheck.run(text, docs, f.resolved_text, v, web=web, paid_ok=args.paid)
            row["recheck"] = {k: r.get(k) for k in ("status", "label", "claim", "lanes", "paid", "ms", "value")}
            if r.get("patched"):
                row["patched"] = r["patched"]
                row["correct_patched"] = score(r["patched"], q.get("expect") or []) if q.get("expect") else None
        return row


def summarize(rows):
    rs = [r for r in rows if "verify" in r]
    c = collections.Counter(r["verify"] for r in rs)
    sc = [r for r in rs if r["correct"] is not None]
    rc = [r for r in rs if r.get("recheck")]
    rcs = collections.Counter(r["recheck"]["status"] for r in rc)
    after = [r.get("correct_patched") if r.get("correct_patched") is not None else r["correct"] for r in sc]
    return {"n": len(rs), "badges": dict(c),
            "disagree_on_correct (false alarm)": sum(1 for r in sc if r["verify"] == "disagree" and r["correct"]),
            "disagree_on_wrong": sum(1 for r in sc if r["verify"] == "disagree" and not r["correct"]),
            "unverified_or_partial_on_correct": sum(1 for r in sc if r["verify"] in ("unverified", "partial") and r["correct"]),
            "rechecked": len(rc), "recheck_outcomes": dict(rcs),
            "corrected_right": sum(1 for r in sc if r.get("correct_patched") and not r["correct"]),
            "corrected_wrongly": sum(1 for r in sc if r.get("correct_patched") is False and r["correct"]),
            "accuracy_before": round(sum(1 for r in sc if r["correct"]) / len(sc), 3) if sc else None,
            "accuracy_after": round(sum(1 for x in after if x) / len(sc), 3) if sc else None,
            "paid_per_q": round(sum((r.get("recheck") or {}).get("paid") or 0 for r in rs) / max(1, len(rs)), 3),
            "recheck_ms_p50": round(statistics.median([r["recheck"]["ms"] for r in rc])) if rc else None}


async def main():
    await answer.warm()
    from server import embed
    await embed.arun(embed.warmup)                 # the leads lane embeds the question (warm in the live server)
    out = {"config": {"WEB_BACKEND": config.WEB_BACKEND, "paid": args.paid}, "sets": {}}
    sem = asyncio.Semaphore(3)
    for name in args.sets.split(","):
        qs = [json.loads(x) for x in SETS[name].read_text().splitlines() if x.strip()]
        qs = [q for q in qs if q.get("kind") != "current" and not q.get("local")]
        rows = await asyncio.gather(*[one(q, sem) for q in qs])
        out["sets"][name] = {"summary": summarize(rows), "rows": rows}
        print(name, json.dumps(out["sets"][name]["summary"]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


asyncio.run(main())
