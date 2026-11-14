"""Text-level check of the planner (server/planner.py): which questions does it split, and into what?

  .venv/bin/python eval/multipart/plan_textcheck.py                 # regression sets: every word prefix of the 58 set,
                                                                     # showcase, local questions: must split NONE
  .venv/bin/python eval/multipart/plan_textcheck.py --corpus        # + the director's ~15k spoken-search texts by style
                                                                     # (multi_part / compare = recall; the rest = false splits)
  .venv/bin/python eval/multipart/plan_textcheck.py --show compare  # print the splits of one style

Uses the real Arc Tagger (tag_text) for spans, frame.build, then planner.decompose; no network, no audio.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server import frame, planner  # noqa: E402
from server.tagger import Tagger  # noqa: E402

CORPUS = Path("/Users/karan/Desktop/work/whissle/headstart/runs/director_queries")


def frames(tagger: Tagger, text: str, prefixes: bool):
    words = text.split()
    ks = range(1, len(words) + 1) if prefixes else [len(words)]
    for k in ks:
        ws = words[:k]
        r = tagger.tag_text(" ".join(ws))
        spans = [s for s in r.get("spans", []) if s.get("end", 0) <= len(ws)]
        yield k, frame.build(ws, spans, r.get("current"))


def rows(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--show", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tg = Tagger(domain="assistant")
    res: dict = {"regression": {}, "corpus": {}}
    t0 = time.perf_counter()
    bad = []
    n_frames = 0
    for name, p in [("eval58", ROOT / "eval/questions.jsonl"), ("local", ROOT / "index/questions_local.jsonl"),
                    ("showcase", ROOT / "eval/showcase/questions.jsonl")]:
        nq = split = 0
        for r in rows(p):
            nq += 1
            for k, f in frames(tg, r["text"], prefixes=True):
                n_frames += 1
                mp = planner.decompose(f)
                if mp is not None:
                    split += 1
                    bad.append((name, r["id"], " ".join(r["text"].split()[:k]), mp.why, [x.text for x in mp.parts]))
        res["regression"][name] = {"questions": nq, "split_frames": split}
    print(f"regression sets: {n_frames} prefix frames, {len(bad)} split")
    for b in bad:
        print("  SPLIT", b)
    mp_rows = rows(ROOT / "eval/multipart/questions.jsonl") if (ROOT / "eval/multipart/questions.jsonl").exists() else []
    hit = 0
    tot = 0
    for r in mp_rows:
        tot += 1
        f = list(frames(tg, r["text"], prefixes=False))[0][1]
        mp = planner.decompose(f)
        ok = mp is not None and len(mp.parts) >= 2
        hit += ok
        print(f"  {'ok ' if ok else 'MISS'} {r['id']} {r['text']!r} -> {[x.text for x in mp.parts] if mp else None}")
    if tot:
        res["multipart_set"] = {"n": tot, "split": hit}
        print(f"multipart set: {hit}/{tot} split into >= 2 parts")
    if a.corpus:
        by = collections.defaultdict(lambda: [0, 0])
        shown = 0
        for p in sorted(CORPUS.glob("part_*.jsonl")):
            for r in rows(p):
                st = r.get("style") or "?"
                f = list(frames(tg, r["text"], prefixes=False))[0][1]
                mp = planner.decompose(f)
                ok = mp is not None and len(mp.parts) >= 2
                by[st][0] += 1
                by[st][1] += ok
                if ok and a.show in (st, "all") and shown < 400:
                    shown += 1
                    print(f"  [{st}] {r['text']!r} -> {[x.text for x in mp.parts]}")
        print("corpus splits by style (n, split, %):")
        for st, (n, s) in sorted(by.items(), key=lambda x: -x[1][1] / max(1, x[1][0])):
            print(f"  {st:12s} {n:5d} {s:4d} {100 * s / max(1, n):5.1f}%")
            res["corpus"][st] = {"n": n, "split": s}
    print(f"{time.perf_counter() - t0:.1f}s")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
