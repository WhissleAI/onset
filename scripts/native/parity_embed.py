"""EmbeddingGemma 2: ONNX Runtime vs the torch path the server runs today (queries CPU f32, docs + audio on MPS bf16).

    ONNX_RUNTIME=0 python scripts/native/parity_embed.py run --out runs/native/embed_torch.npz
    ONNX_RUNTIME=1 python scripts/native/parity_embed.py run --out runs/native/embed_onnx.npz
    python scripts/native/parity_embed.py compare runs/native/embed_torch.npz runs/native/embed_onnx.npz

One backend per process (memory). Inputs, all in the repo / index data:
  queries  eval/questions.jsonl + eval/showcase/questions.jsonl + index/questions_local.jsonl texts
  docs     every 37th Wikipedia lead of index/data/knowledge (title + first 600 chars: the server's encode_docs call)
           and the 222 local moments (index/data/docs.jsonl)
  audio    eval/audio/local_*.wav (the audio-lane questions)
compare: cosine between backends, plus
  rerank top-5 agreement: per question, the 20 leads nearest under the SHIPPED index vectors are re-ranked by each
    backend's own query x doc vectors (what speculator/rank does with fresh web docs); top-5 set overlap.
  index top-5 agreement: each backend's query against the shipped 11k lead vectors (vectors.npy, built with torch).
  latency p50: one query, a batch of 8 docs, one 3 s audio clip.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def texts():
    qs = []
    for f in ("eval/questions.jsonl", "eval/showcase/questions.jsonl", "index/questions_local.jsonl"):
        p = ROOT / f
        if p.exists():
            qs += [json.loads(line)["text"] for line in p.read_text().splitlines() if line.strip()]
    return qs


def run(out: str):
    import soundfile as sf
    from server import config, embed, modelpaths
    idx = modelpaths.resolve("index-data", "INDEX_DIR", ROOT / "index" / "data")
    leads = [json.loads(x) for x in (idx / "knowledge" / "leads.jsonl").read_text().splitlines() if x.strip()]
    sel = list(range(0, len(leads), 37))
    moments = [json.loads(x) for x in (idx / "docs.jsonl").read_text().splitlines() if x.strip()]
    qs = texts()
    t0 = time.perf_counter()
    embed.warmup()
    warm = time.perf_counter() - t0
    Q = embed.encode_queries(qs)
    D = np.concatenate([embed.encode_docs([leads[i]["text"][:600] for i in sel[j:j + 16]],
                                          [leads[i]["title"] for i in sel[j:j + 16]]) for j in range(0, len(sel), 16)])
    M = embed.encode_docs([m["text"][:600] for m in moments], [m["title"] for m in moments])
    wavs = sorted((modelpaths.resolve("demo-audio", "AUDIO_DIR", ROOT / "eval" / "audio")).glob("local_*.wav"))
    A = np.stack([embed.encode_audio([sf.read(str(w), dtype="float32")[0]])[0] for w in wavs]) if wavs else np.zeros((0, 768))
    lat = {"query": [], "docs8": [], "audio3s": []}
    rng = np.random.default_rng(0)
    for i in range(30):
        t = time.perf_counter(); embed.encode_queries([qs[i % len(qs)]]); lat["query"].append(time.perf_counter() - t)
    for i in range(10):
        b = [leads[sel[(i * 8 + k) % len(sel)]] for k in range(8)]
        t = time.perf_counter(); embed.encode_docs([d["text"][:600] for d in b], [d["title"] for d in b])
        lat["docs8"].append(time.perf_counter() - t)
    for i in range(8):
        w = rng.standard_normal(48000).astype(np.float32) * 0.02
        t = time.perf_counter(); embed.encode_audio([w]); lat["audio3s"].append(time.perf_counter() - t)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # macOS: bytes
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, Q=Q, D=D, M=M, A=A, sel=np.array(sel), warm=warm, rss_mb=rss,
             **{f"lat_{k}": np.array(v) for k, v in lat.items()}, onnx=config.ONNX_RUNTIME)
    print(f"{'onnx' if config.ONNX_RUNTIME else 'torch'}: {len(qs)} queries, {len(sel)} leads, {len(moments)} moments, "
          f"{len(wavs)} audio, warm {warm:.1f}s, peak RSS {rss:.0f} MB")


def compare(a: str, b: str):
    from server import modelpaths
    A, B = np.load(a), np.load(b)
    cos = lambda x, y: (x * y).sum(-1)                 # both normalised
    rep = {}
    for k in ("Q", "D", "M", "A"):
        if len(A[k]):
            c = cos(A[k], B[k])
            rep[f"cos_{k}"] = {"mean": round(float(c.mean()), 5), "min": round(float(c.min()), 5), "n": int(len(c))}
    idx = modelpaths.resolve("index-data", "INDEX_DIR", ROOT / "index" / "data")
    V = np.load(idx / "knowledge" / "vectors.npy", mmap_mode="r")
    V = np.asarray(V, np.float32)
    agree5, agree1, rr5 = [], [], []
    for qa, qb in zip(A["Q"], B["Q"]):
        ta, tb = np.argsort(-(V @ qa))[:5], np.argsort(-(V @ qb))[:5]
        agree5.append(len(set(ta) & set(tb)) / 5)
        agree1.append(float(ta[0] == tb[0]))
    rep["index_top5_agreement"] = round(float(np.mean(agree5)), 4)
    rep["index_top1_agreement"] = round(float(np.mean(agree1)), 4)
    # rerank: candidates = the 20 shipped-index neighbours of the torch query; each backend embeds them itself
    # (the sampled leads only: every 37th lead has fresh vectors from both backends)
    sel = A["sel"]
    Da, Db = A["D"], B["D"]
    for qa, qb in zip(A["Q"], B["Q"]):
        cand = np.argsort(-(Da @ qa))[:20]
        ra = cand[np.argsort(-(Da[cand] @ qa))][:5]
        rb = cand[np.argsort(-(Db[cand] @ qb))][:5]
        rr5.append(len(set(ra) & set(rb)) / 5)
    rep["rerank_top5_agreement"] = round(float(np.mean(rr5)), 4)
    rep["rerank_top5_exact_order"] = None
    # vs the shipped index vectors themselves (torch-built): how close each backend's doc vectors are to them
    Vs = V[sel]
    rep["cos_D_vs_shipped_index"] = {"torch": round(float(cos(A["D"], Vs).mean()), 5),
                                     "onnx": round(float(cos(B["D"], Vs).mean()), 5)}
    for k in ("query", "docs8", "audio3s"):
        rep[f"p50_ms_{k}"] = {"torch": round(float(np.median(A[f"lat_{k}"])) * 1000, 1),
                              "onnx": round(float(np.median(B[f"lat_{k}"])) * 1000, 1)}
    rep["warm_s"] = {"torch": round(float(A["warm"]), 1), "onnx": round(float(B["warm"]), 1)}
    rep["peak_rss_mb"] = {"torch": round(float(A["rss_mb"])), "onnx": round(float(B["rss_mb"]))}
    rep.pop("rerank_top5_exact_order")
    print(json.dumps(rep, indent=1))
    return rep


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.out)
    else:
        rep = compare(a.a, a.b)
        if a.json:
            Path(a.json).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
