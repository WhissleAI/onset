"""Arc v1 Mini streaming compute: ONNX (server/arc_onnx.py) vs torch (arc_audio), on the eval audio in 100 ms chunks.

    python scripts/native/parity_arc.py --arc <arc-v1 dir> --onnx <export dir> [--n 12] [--json out.json]

Both sides get the same chunks the live ear gets (100 ms, then flush). Compared per file:
  enc   the 50 Hz encoder frames (MSTStreamEncoder vs MSTStream): max |diff|, min cosine
  trunk the 10 Hz trunk outputs (Trunk.step with carried state vs TrunkStep), delay 0.3 s, domain "assistant":
        max |diff|, min cosine, and the argmax agreement of the frame-tag logits computed from both (torch head on
        each side's trunk output: does the ONNX trunk change any frame's top tag?)
  ms    compute per 100 ms of audio (p50), each side.
Developer machine only (needs torch for the reference side).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import soundfile as sf
    import torch
    sys.path.insert(0, a.arc)
    from arc_audio import Arc
    from arc_audio import mst as MST
    from arc_audio.api import Pooler as TPooler
    from server import arc_onnx as AO

    arc = Arc.from_pretrained(a.arc, device="cpu", threads=2, load_decoder=False)
    m = AO.ArcOnnx(a.onnx, threads=2)
    dom_i = arc.domains.index("assistant")
    d = arc.delay_frames(0.3)
    di = arc.net.trunk.delays.index(d)
    wavs = sorted((ROOT / "eval" / "audio").glob("q*.wav"))[: a.n]
    rep = {"files": [], "dom": "assistant", "delay_frames": d}
    t_ms, o_ms = [], []
    for w in wavs:
        x, sr = sf.read(str(w), dtype="float32")
        x = x.mean(1) if x.ndim > 1 else x
        assert sr == 16000
        x = np.concatenate([x, np.zeros(12000, np.float32)])         # the 1.2 s of silence the replay appends
        te, tp, tst, trows, tys = MST.MSTStreamEncoder(arc.enc), TPooler(), None, 0, []
        oe, op, ots, orows, oys = AO.MSTStream(m), AO.Pooler(), AO.TrunkStep(m, dom_i, di), 0, []
        enc_t, enc_o = [], []
        domt, dlt = torch.tensor([dom_i]), torch.tensor([d])
        chunks = [x[i:i + 1600] for i in range(0, len(x), 1600)]
        for k, c in enumerate(chunks + [None]):
            t0 = time.perf_counter()
            with torch.no_grad():
                f = te.feed(c) if c is not None else te.flush()
                xs = tp.push(f.float().numpy()) + (tp.flush() if c is None else [])
                rows = []
                for r in xs:
                    if trows >= d:
                        rows.append(r)
                    trows += 1
                if rows:
                    h, tst = arc.net.trunk.step(torch.as_tensor(np.stack(rows))[None], domt, dlt, tst)
                    tys.append(h[0].numpy())
            t1 = time.perf_counter()
            g = oe.feed(c) if c is not None else oe.flush()
            ys = op.push(g) + (op.flush() if c is None else [])
            rows2 = []
            for r in ys:
                if orows >= d:
                    rows2.append(r)
                orows += 1
            if rows2:
                oys.append(ots.step(rows2))
            t2 = time.perf_counter()
            if c is not None:
                t_ms.append((t1 - t0) * 1000)
                o_ms.append((t2 - t1) * 1000)
            enc_t.append(f.float().numpy())
            enc_o.append(g)
        E1, E2 = np.concatenate(enc_t), np.concatenate(enc_o)
        Y1, Y2 = np.concatenate(tys), np.concatenate(oys)
        cos = lambda p, q: (p * q).sum(-1) / np.maximum(np.linalg.norm(p, axis=-1) * np.linalg.norm(q, axis=-1), 1e-9)
        with torch.no_grad():
            K = arc.net.head.keys()
            z1 = arc.net.head.logits_q("frame", torch.as_tensor(Y1)[None], domt, K)[0].numpy()
            z2 = arc.net.head.logits_q("frame", torch.as_tensor(Y2)[None], domt, K)[0].numpy()
        p1, p2 = 1 / (1 + np.exp(-z1)), 1 / (1 + np.exp(-z2))
        on1, on2 = p1 >= 0.5, p2 >= 0.5
        rep["files"].append({"file": w.name, "enc_frames": [len(E1), len(E2)], "enc_maxdiff": float(np.abs(E1 - E2).max()),
                             "enc_mincos": float(cos(E1, E2).min()), "trunk_rows": [len(Y1), len(Y2)],
                             "trunk_maxdiff": float(np.abs(Y1 - Y2).max()), "trunk_mincos": float(cos(Y1, Y2).min()),
                             "frame_tag_maxdiff_p": float(np.abs(p1 - p2).max()),
                             "frame_tag_flips": int((on1 != on2).sum()), "frame_tag_cells": int(on1.size)})
        print(rep["files"][-1], flush=True)
    rep["p50_ms_per_100ms"] = {"torch": round(float(np.median(t_ms)), 2), "onnx": round(float(np.median(o_ms)), 2)}
    rep["p90_ms_per_100ms"] = {"torch": round(float(np.percentile(t_ms, 90)), 2),
                               "onnx": round(float(np.percentile(o_ms, 90)), 2)}
    rep["worst"] = {k: (max if "diff" in k or "flips" in k else min)(f[k] for f in rep["files"])
                    for k in ("enc_maxdiff", "enc_mincos", "trunk_maxdiff", "trunk_mincos", "frame_tag_maxdiff_p",
                              "frame_tag_flips")}
    print(json.dumps({k: rep[k] for k in ("p50_ms_per_100ms", "p90_ms_per_100ms", "worst")}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
