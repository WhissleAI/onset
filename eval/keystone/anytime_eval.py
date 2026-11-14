"""Anytime retrieval over a streaming question, against a WARM Keystone store (the replayed workload store,
runs/index/warm_store, written by replay.py --save-store).

For each spoken eval question (eval/questions.jsonl + showcase; their word timings come from the audio duration),
words arrive one by one; at every word:
  key        the typed key of the prefix (no embedding): the word at which the relation is known, the word at which
             the full key resolves (and stays), and whether the store has an edge for it (a local hit ready)
  sparse     incremental BM25 over the cells: only the new word's posting list is touched (term-at-a-time
             accumulators); CPU us per update
  dense      the prefix's query embedding + the two-stage dense search; the word from which the local top-1 / top-5
             equal the final question's (convergence), CPU ms per update
Reported per question and summarised: words-to-convergence, share of questions whose local result is ready before
the last word, and the lead in ms (words left x the speaker's mean word duration).

    .venv/bin/python eval/keystone/anytime_eval.py [--store runs/index/warm_store]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval/keystone"))

from replay import prefix_frames  # noqa: E402
from server import config, embed  # noqa: E402
from server.keystone import hook  # noqa: E402
from server.keystone.cells import toks  # noqa: E402
from server.keystone.relkey import key_of_frame  # noqa: E402


class SparseAcc:
    """Term-at-a-time BM25 accumulators: adding a word touches only its posting list."""

    def __init__(self, cs):
        self.cs, self.acc, self.seen = cs, {}, set()
        n = max(1, len(cs.cells))
        self.n, self.avg = n, cs.len_sum / n

    def add(self, word: str):
        for w in toks(word):
            if w in self.seen:
                continue
            self.seen.add(w)
            post = self.cs.post.get(w, ())
            idf = math.log(1 + (self.n - len(post) + 0.5) / (len(post) + 0.5))
            for i in post:
                tf = self.cs.tf[i]
                f = tf.get(w, 0)
                L = sum(tf.values())
                self.acc[i] = self.acc.get(i, 0.0) + idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * L / self.avg))

    def top(self, k=5):
        return [i for i, _ in sorted(self.acc.items(), key=lambda x: -x[1])[:k]]


def main(a):
    config.INDEX, config.INDEX_PERSIST, config.KEYSTONE_DIR = True, True, Path(a.store)
    ks = hook.store()
    print("store", ks.stats(), flush=True)
    qs = []
    for p in ("eval/questions.jsonl", "eval/showcase/questions.jsonl"):
        qs += [json.loads(x) for x in (ROOT / p).read_text().splitlines() if x.strip()]
    rows = []
    embed.encode_queries(["warm up"])
    for q in qs:
        text = q.get("arc_text") or q["text"]
        pf = prefix_frames(text)
        n = len(pf)
        wdur = (q.get("dur_s") or n * 0.4) / max(1, n) * 1000
        keys = [key_of_frame(f) for f in pf]
        final_key = keys[-1]
        sp = SparseAcc(ks.cells)
        sparse_us, dense_ms, tops = [], [], []
        for i, f in enumerate(pf):
            t = time.perf_counter()
            sp.add(f.words[-1])
            sparse_us.append((time.perf_counter() - t) * 1e6)
            t = time.perf_counter()
            qv = embed.encode_queries([" ".join(f.words)])[0]
            res = ks.cells.search(qv, k=5, q_terms=toks(" ".join(f.words)), allow=lambda c: c.kind == "passage")
            dense_ms.append((time.perf_counter() - t) * 1000)
            tops.append([c.id for c, _, _ in res])
        fin = tops[-1]

        def conv(pred):
            at = None
            for i in range(n - 1, -1, -1):
                if pred(i):
                    at = i + 1
                else:
                    break
            return at
        r = {"id": q["id"], "n": n, "word_ms": round(wdur),
             "top1_at": conv(lambda i: tops[i][:1] == fin[:1]) if fin else None,
             "top5_at": conv(lambda i: set(tops[i]) == set(fin)) if fin else None,
             "rel_at": conv(lambda i: keys[i] is not None and final_key is not None
                            and keys[i].relation == final_key.relation) if final_key else None,
             "key_at": conv(lambda i: keys[i] is not None and final_key is not None and keys[i].text() ==
                            final_key.text()) if final_key and final_key.complete else None,
             "edge": bool(final_key and final_key.complete and ks.g.lookup(final_key.subject, final_key.relation)[1]),
             "sparse_us_p50": round(statistics.median(sparse_us)), "dense_ms_p50": round(statistics.median(dense_ms), 1)}
        rows.append(r)
    def share_before_end(k):
        xs = [r for r in rows if r[k] is not None]
        return round(sum(1 for r in xs if r[k] < r["n"]) / len(xs), 3) if xs else None

    def lead_ms(k):
        xs = [(r["n"] - r[k]) * r["word_ms"] for r in rows if r[k] is not None]
        return round(statistics.median(xs)) if xs else None
    summ = {"questions": len(rows),
            "top1_converged_before_end": share_before_end("top1_at"), "top1_lead_ms_p50": lead_ms("top1_at"),
            "top5_converged_before_end": share_before_end("top5_at"), "top5_lead_ms_p50": lead_ms("top5_at"),
            "relation_known_words_before_end_p50": statistics.median([r["n"] - r["rel_at"] for r in rows
                                                                      if r["rel_at"]]) if any(r["rel_at"] for r in rows)
            else None,
            "key_resolved_before_end": share_before_end("key_at"), "key_lead_ms_p50": lead_ms("key_at"),
            "keyed_questions": sum(1 for r in rows if r["key_at"] is not None),
            "edge_hit_ready": sum(1 for r in rows if r["edge"]),
            "edge_hit_ready_before_end": sum(1 for r in rows if r["edge"] and r["key_at"] and r["key_at"] < r["n"]),
            "sparse_update_us_p50": statistics.median([r["sparse_us_p50"] for r in rows]),
            "dense_update_ms_p50": statistics.median([r["dense_ms_p50"] for r in rows])}
    out = {"summary": summ, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(ROOT / "runs/index/warm_store"))
    ap.add_argument("--out", default=str(ROOT / "runs/index/anytime.json"))
    main(ap.parse_args())
