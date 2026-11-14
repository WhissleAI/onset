"""Per-part accuracy of a run_ab.py run over eval/multipart/questions.jsonl (and planner on vs off side by side).

  .venv/bin/python eval/multipart/score.py runs/ab_mp_planon.json [runs/ab_mp_planoff.json ...]

A part is answered when every one of its expect items matches the answer (eval/run_ab.score: case / accent folded,
number words, "a|b" alternatives). Overall = all parts + the overall keywords (e.g. which one is taller).
Also: ready p50 / p90 (ms), paid web calls per question, free calls, verify badge counts, plan path share.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from run_ab import pct, score  # noqa: E402

QS = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/multipart/questions.jsonl").read_text().splitlines()
      if l.strip()}


def summarize(path: str) -> dict:
    res = json.loads(Path(path).read_text())
    out = {}
    for mode in ("headstart", "baseline", "plain"):
        rows = [r for r in res["rows"] if r["mode"] == mode and r["id"] in QS]
        if not rows:
            continue
        parts_ok = parts_n = all_ok = 0
        by_kind: dict = {}
        for r in rows:
            q = QS[r["id"]]
            a = r.get("answer") or ""
            pk = [bool(score(a, p["expect"])) for p in q["parts"]]
            ov = bool(score(a, q["overall"])) if q["overall"] else True
            parts_ok += sum(pk)
            parts_n += len(pk)
            ok = all(pk) and ov
            all_ok += ok
            k = by_kind.setdefault(q["kind"], [0, 0, 0, 0])
            k[0] += sum(pk)
            k[1] += len(pk)
            k[2] += ok
            k[3] += 1
        ready = [r["ready_ms"] for r in rows if r.get("ready_ms") is not None]
        out[mode] = {
            "n": len(rows), "part_acc": round(parts_ok / parts_n, 3), "parts": f"{parts_ok}/{parts_n}",
            "all_parts_acc": round(all_ok / len(rows), 3), "p50_ready_ms": pct(ready, 50),
            "p90_ready_ms": pct(ready, 90), "mean_ready_ms": round(statistics.mean(ready)) if ready else None,
            "paid_per_q": round(statistics.mean([(r.get("web_paid") if r.get("web_paid") is not None
                                                   else r.get("paid_calls") or 0) for r in rows]), 2),
            "free_per_q": round(statistics.mean([r.get("free_calls") or 0 for r in rows]), 2),
            "plan_path": sum(1 for r in rows if r.get("path") == "plan"),
            "verify": dict(Counter((r.get("verify") or {}).get("status", "-") for r in rows)),
            "by_kind": {k: {"part_acc": round(v[0] / v[1], 3), "all_parts": f"{v[2]}/{v[3]}"} for k, v in by_kind.items()},
        }
    return out


def main():
    for p in sys.argv[1:]:
        print(p)
        for mode, s in summarize(p).items():
            print(f"  {mode:9s} {json.dumps(s)}")


if __name__ == "__main__":
    main()
