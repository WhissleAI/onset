"""Which first-pass structure for Keystone's cells: latency vs size, and recall vs truncation / quantization.

Corpus: the 11,085 real EmbeddingGemma 2 Wikipedia lead vectors (index/data/knowledge), padded to 100k with
synthetic vectors drawn around them (each = a real vector + noise, renormalised: realistic clustering, not uniform
noise). Queries: real query vectors of the eval / workload questions (runs/index/fixture_vecs.npz).

  latency   brute-force numpy matmul (fp32 768, fp16 768, int8 256, binary 256 Hamming) vs hnswlib HNSW
            (M=16, ef_construction=100, ef=64), at 1k / 10k / 100k cells, one query at a time (p50 / p90 us)
  recall    recall@10 of the two-stage search (first pass at dim d / quantization -> shortlist M=64 -> exact
            re-score at 768) against exact fp32 768 search, on the 11k real vectors
  size      bytes per cell for each representation

    .venv/bin/python eval/keystone/ann_bench.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from server.keystone.ahash import hamming  # noqa: E402
from server.keystone.cells import first_pass, quantize_i8  # noqa: E402

OUT = ROOT / "runs/index"


def corpus(n, base, rng):
    if n <= len(base):
        return base[:n].copy()
    extra = base[rng.integers(0, len(base), n - len(base))] + rng.normal(0, 0.035, (n - len(base), base.shape[1]))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    return np.concatenate([base, extra.astype(np.float32)])


def timeit(fn, qs, reps=1):
    ts = []
    for q in qs:
        t = time.perf_counter()
        fn(q)
        ts.append((time.perf_counter() - t) * 1e6)
    return {"p50_us": round(statistics.median(ts)), "p90_us": round(sorted(ts)[int(0.9 * (len(ts) - 1))])}


def main():
    rng = np.random.default_rng(0)
    base = np.load(ROOT / "index/data/knowledge/vectors.npy").astype(np.float32)[:, :768]
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    z = np.load(OUT / "fixture_vecs.npz")
    Qs = z["q_vecs"].astype(np.float32)[:300]
    res = {"latency": {}, "recall": {}, "bytes_per_cell": {}}
    import hnswlib
    for n in (1000, 10000, 100000):
        X = corpus(n, base, rng)
        X16 = X.astype(np.float16)
        fp = first_pass(X, 256)
        X8, s8 = quantize_i8(fp)
        B = np.packbits(fp > 0, axis=1)
        r = {}
        r["fp32_768"] = timeit(lambda q: np.argpartition(-(X @ q), 10)[:10], Qs[:200])
        r["fp16_768"] = timeit(lambda q: np.argpartition(-(X16 @ q.astype(np.float16)), 10)[:10], Qs[:200])

        def i8(q):
            qp = first_pass(q, 256)
            s = (X8 @ qp) * s8          # int8 matmul promotes to fp32 once; still 3x less memory traffic
            sh = np.argpartition(-s, 64)[:64]
            return sh[np.argsort(-(X16[sh].astype(np.float32) @ q))[:10]]
        r["int8_256+rescore"] = timeit(i8, Qs[:200])
        for d in (256, 512):
            Fd = first_pass(X, d)

            def f32(q, Fd=Fd, d=d):
                s = Fd @ first_pass(q, d)
                sh = np.argpartition(-s, 64)[:64]
                return sh[np.argsort(-(X16[sh].astype(np.float32) @ q))[:10]]
            r[f"fp32_{d}+rescore"] = timeit(f32, Qs[:200])

        def bits(q):
            h = hamming(B, np.packbits(first_pass(q, 256) > 0))
            sh = np.argpartition(h, 64)[:64]
            return sh[np.argsort(-(X16[sh].astype(np.float32) @ q))[:10]]
        r["bits_256+rescore"] = timeit(bits, Qs[:200])
        t = time.perf_counter()
        ix = hnswlib.Index(space="ip", dim=768)
        ix.init_index(max_elements=n, ef_construction=100, M=16)
        ix.add_items(X)
        ix.set_ef(64)
        build = time.perf_counter() - t
        r["hnsw_768"] = {**timeit(lambda q: ix.knn_query(q, k=10), Qs[:100]), "build_s": round(build, 2)}
        exact = [set(np.argsort(-(X @ q))[:10]) for q in Qs[:100]]
        got = [set(ix.knn_query(q, k=10)[0][0]) for q in Qs[:100]]
        r["hnsw_768"]["recall@10"] = round(float(np.mean([len(a & b) / 10 for a, b in zip(exact, got)])), 3)
        res["latency"][str(n)] = r
        print(n, json.dumps(r), flush=True)
    # recall vs truncation / quantization (real vectors only)
    X = base
    X16 = X.astype(np.float16)
    exact = [np.argsort(-(X @ q))[:10] for q in Qs]
    for d in (768, 512, 256, 128, 64):
        fp = first_pass(X, d)
        X8, s8 = quantize_i8(fp)
        B = np.packbits(fp > 0, axis=1)
        for quant in ("fp32", "int8", "bits"):
            rec1, recs = [], []
            for q, ex in zip(Qs, exact):
                qp = first_pass(q, d)
                if quant == "fp32":
                    s = fp @ qp
                elif quant == "int8":
                    s = (X8 @ qp) * s8
                else:
                    s = -hamming(B, np.packbits(qp > 0)).astype(np.float32)
                top_first = np.argsort(-s)[:10]
                rec1.append(len(set(top_first) & set(ex)) / 10)
                sh = np.argpartition(-s, 64)[:64]
                top = sh[np.argsort(-(X16[sh].astype(np.float32) @ q))[:10]]
                recs.append(len(set(top) & set(ex)) / 10)
            res["recall"][f"{d}_{quant}"] = {"first_pass_only": round(float(np.mean(rec1)), 3),
                                             "with_rescore_M64": round(float(np.mean(recs)), 3)}
        print(d, {k: v for k, v in res["recall"].items() if k.startswith(f"{d}_")}, flush=True)
    res["bytes_per_cell"] = {"fp32_768": 3072, "fp16_768": 1536, "int8_256": 256 + 4, "bits_256": 32,
                             "int8_128": 128 + 4, "bits_128": 16,
                             "keystone_default": "int8_256 (260) + bits_256 (32) + fp16_768 rescoring copy (1536)"}
    (OUT / "ann_bench.json").write_text(json.dumps(res, indent=1))
    print("->", OUT / "ann_bench.json")


if __name__ == "__main__":
    main()
