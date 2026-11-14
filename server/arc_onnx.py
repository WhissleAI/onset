"""Arc v1 Mini's streaming compute on ONNX Runtime: the Moonshine Streaming encoder and the trunk recurrence.

What is here (exported by scripts/native/export_arc.py, parity in scripts/native/parity_arc.py):
  MSTStream   arc_audio.mst.MSTStreamEncoder line for line (same buffers, windows, look-ahead and trimming), with
              each step's maths in ONNX (frame embedder, the 6 sliding-window layers, final norm) or numpy (the two
              stride-2 convs). feed(pcm) / flush() -> new encoder frames [n,320] (50 Hz).
  Pooler      arc_audio.api.Pooler: encoder frames -> 10 Hz mean+max rows [640] (pure numpy already).
  TrunkStep   Trunk.step for one 10 Hz row, the 5 layers' recurrent state carried as explicit arrays (h, bx).

What is NOT here yet (MACAPP.md "Native runtime", open): Arc's heads + Engine (tags, events, command_complete, names,
numbers, diarization, word channel) and the builtin transcript (Moonshine decoder beam search). Those still run in
torch through the arc_audio package, so the Arc ear itself still needs torch.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FRAME, HOP, STRIDE, RF, SR = 80, 320, 320, 3920, 16000


def _sess(path: Path, threads: int):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class ArcOnnx:
    """The loaded graphs, shared by every stream (sessions are thread-safe)."""

    def __init__(self, path: str | Path, threads: int | None = None):
        d = Path(path)
        th = int(threads or os.environ.get("ARC_THREADS", "2"))
        self.embed = _sess(d / "mst_embed.onnx", th)
        meta = np.load(d / "mst_meta.npz")
        self.win = [tuple(int(v) for v in w) for w in meta["windows"]]
        self.hidden = int(meta["hidden"])
        self.layers = [_sess(d / f"mst_layer{i}.onnx", th) for i in range(len(self.win))]
        self.norm = _sess(d / "mst_norm.onnx", th)
        self.conv = []
        for n in ("conv1", "conv2"):
            z = np.load(d / f"mst_{n}.npz")
            self.conv.append((z["w"].astype(np.float32), z["b"].astype(np.float32), int(z["stride"])))
        self.trunk = _sess(d / "trunk_step.onnx", th)
        sh = {i.name: i.shape for i in self.trunk.get_inputs()}["h_in"]
        self.state_shape = tuple(int(v) for v in sh[:1]) + (1,) + tuple(int(v) for v in sh[2:])


def _conv1d_s2(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int) -> np.ndarray:
    """F.conv1d(x.T[None], w, b, stride)[0].T for x [T,Cin], w [Cout,Cin,k] (no padding: callers pad)."""
    k = w.shape[2]
    n = (x.shape[0] - k) // stride + 1
    if n <= 0:
        return np.zeros((0, w.shape[0]), np.float32)
    cols = np.stack([x[i * stride:i * stride + k] for i in range(n)])          # [n,k,Cin]
    return np.einsum("nkc,ock->no", cols, w, optimize=True) + b


class MSTStream:
    """arc_audio.mst.MSTStreamEncoder on ONNX Runtime (see that class for the bookkeeping this mirrors)."""

    def __init__(self, m: ArcOnnx):
        self.m = m
        self.win = m.win
        self.L = len(m.layers)
        self.pcm = np.zeros(0, np.float32)
        self.e, self.e0, self.n_e = None, 0, 0
        self.c1, self.c10, self.n_c1 = None, 0, 0
        self.H = [None] * (self.L + 1)
        self.H0 = [0] * (self.L + 1)
        self.n_H = [0] * (self.L + 1)
        self.n_out = 0

    def _embed_frames(self, x):
        return self.m.embed.run(None, {"frames": np.ascontiguousarray(x.reshape(-1, FRAME), np.float32)})[0]

    @staticmethod
    def _conv(conv, buf, b0, n_have, n_done):
        J = (n_have + 1) // 2
        if J <= n_done or buf is None:
            return None, n_done
        lo, hi = 2 * n_done - 4, 2 * (J - 1) + 1
        x = buf[max(lo, 0) - b0:hi - b0]
        if lo < 0:
            x = np.concatenate([np.zeros((-lo, x.shape[1]), x.dtype), x], 0)
        w, b, s = conv
        return _conv1d_s2(x, w, b, s).astype(np.float32), J

    def _layer(self, l, q0, q1, k_hi):
        left, right = self.win[l]
        k_lo = max(0, q0 - left + 1)
        base = self.H0[l]
        X = self.H[l][k_lo - base:k_hi - base]
        qi = np.arange(q0, q1)[:, None]
        ki = np.arange(k_lo, k_hi)[None, :]
        d = qi - ki
        ok = ((d >= 0) & (d < left)) | ((d < 0) & (-d < right))
        mask = np.where(ok, 0.0, np.finfo(np.float32).min).astype(np.float32)
        return self.m.layers[l].run(None, {"X": np.ascontiguousarray(X, np.float32),
                                           "q0": np.asarray(q0 - k_lo, np.int64), "mask": mask})[0]

    @staticmethod
    def _cat(buf, new):
        return new if buf is None else np.concatenate([buf, new], 0)

    def _trim(self):
        keep = max(w[0] for w in self.win) + 4
        for l in range(self.L + 1):
            if self.H[l] is None:
                continue
            need_from = self.n_H[l + 1] - keep if l < self.L else self.n_H[l]
            cut = max(0, need_from - self.H0[l])
            if cut > 0:
                self.H[l] = self.H[l][cut:]
                self.H0[l] += cut
        if self.e is not None:
            cut = max(0, 2 * self.n_c1 - 8 - self.e0)
            if cut > 0:
                self.e = self.e[cut:]
                self.e0 += cut
        if self.c1 is not None:
            cut = max(0, 2 * self.n_H[0] - 8 - self.c10)
            if cut > 0:
                self.c1 = self.c1[cut:]
                self.c10 += cut

    def _advance(self, final):
        y, J = self._conv(self.m.conv[0], self.e, self.e0, self.n_e, self.n_c1)
        if y is not None:
            self.c1 = self._cat(self.c1, y / (1.0 + np.exp(-y)))           # silu
            self.n_c1 = J
        y, J = self._conv(self.m.conv[1], self.c1, self.c10, self.n_c1, self.n_H[0])
        if y is not None:
            self.H[0] = self._cat(self.H[0], y)
            self.n_H[0] = J
        for l in range(self.L):
            right = self.win[l][1]
            hi = self.n_H[l] if final else self.n_H[l] - max(0, right - 1)
            if hi > self.n_H[l + 1]:
                k_hi = self.n_H[l] if final else min(self.n_H[l], hi + max(0, right - 1))
                self.H[l + 1] = self._cat(self.H[l + 1], self._layer(l, self.n_H[l + 1], hi, k_hi))
                self.n_H[l + 1] = hi
        out = None
        if self.n_H[self.L] > self.n_out:
            new = self.H[self.L][self.n_out - self.H0[self.L]:]
            out = self.m.norm.run(None, {"h": new[None].astype(np.float32)})[0][0]
            self.n_out = self.n_H[self.L]
        self._trim()
        return out

    def feed(self, pcm):
        self.pcm = np.concatenate([self.pcm, np.asarray(pcm, np.float32).reshape(-1)])
        n = len(self.pcm) // FRAME
        if n:
            self.e = self._cat(self.e, self._embed_frames(self.pcm[:n * FRAME]))
            self.n_e += n
            self.pcm = self.pcm[n * FRAME:]
        out = self._advance(final=False)
        return out if out is not None else np.zeros((0, self.m.hidden), np.float32)

    def flush(self):
        self.pcm = np.zeros(0, np.float32)
        out = self._advance(final=True)
        return out if out is not None else np.zeros((0, self.m.hidden), np.float32)


def _frame_of_enc(g):
    return int(np.ceil((g * STRIDE + RF) / SR * 10 - 1e-09)) - 1


class Pooler:
    """arc_audio.api.Pooler (unchanged maths): 50 Hz encoder frames -> 10 Hz [mean, max] rows, f16-rounded."""

    def __init__(self):
        self.g, self.cur_f, self.buf, self.prev, self.next_f = 0, None, [], None, 0

    def _emit_upto(self, f_stop, out):
        while self.next_f < f_stop:
            if self.cur_f == self.next_f and self.buf:
                E = np.stack(self.buf)
                x = np.concatenate([E.mean(0), E.max(0)]).astype(np.float16).astype(np.float32)
                self.buf = []
            else:
                x = self.prev if self.prev is not None else np.zeros(640, np.float32)
            self.prev = x
            out.append(x)
            self.next_f += 1

    def push(self, enc):
        out = []
        for e in np.asarray(enc, np.float32):
            f = _frame_of_enc(self.g)
            self._emit_upto(f, out)
            self.cur_f = f
            self.buf.append(e)
            if _frame_of_enc(self.g + 1) != f:
                self._emit_upto(f + 1, out)
            self.g += 1
        return out

    def flush(self):
        out = []
        if self.buf:
            self._emit_upto(self.cur_f + 1, out)
        return out


class TrunkStep:
    """Trunk.step with explicit state; zeros == state None."""

    def __init__(self, m: ArcOnnx, dom: int, delay_i: int):
        self.m = m
        self.dom = np.asarray([dom], np.int64)
        self.dl = np.asarray([delay_i], np.int64)
        self.h = np.zeros(m.state_shape, np.float32)
        self.bx = np.zeros(m.state_shape, np.float32)

    def step(self, rows: list[np.ndarray]) -> np.ndarray:
        ys = []
        for x in rows:
            y, self.h, self.bx = self.m.trunk.run(None, {"x": np.asarray(x, np.float32)[None], "dom": self.dom,
                                                         "delay_i": self.dl, "h_in": self.h, "bx_in": self.bx})
            ys.append(y[0])
        return np.stack(ys) if ys else np.zeros((0, 512), np.float32)
