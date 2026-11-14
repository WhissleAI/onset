"""Export Arc v1 Mini's streaming compute to ONNX with the streaming state as explicit inputs/outputs.

    python scripts/native/export_arc.py --arc <arc-v1 dir> --out <dir> [--part trunk|mst|all]

  trunk_step.onnx   one 10 Hz frame through Trunk.step (5 SeqLayers + MLPs + FiLM + domain adapter):
                    x [B,640] f32, dom [B] i64, delay_i [B] i64 (index into trunk.delays), h_in [L,B,H,N,P],
                    bx_in [L,B,H,N,P]  ->  y [B,512], h_out, bx_out.  Zero state == Trunk.step's state=None (the
                    first step's alpha*h + beta*prev_bx term is zero either way), so callers start from zeros.
  mst_embed.onnx    Moonshine Streaming tiny's frame embedder: pcm frames [n,80] -> [n,320] (cmvn, comp, linear, silu)
  mst_conv{1,2}.npz the two stride-2 convs (applied in numpy on the streaming buffers, see arc_onnx.MSTStream)
  mst_layer{l}.onnx one encoder layer on a key window: X [K,320], q0 [] i64, mask [Q,K] f32 -> [Q,320]
  mst_norm.onnx     the final norm ([1,n,320] -> [1,n,320])

Parity: scripts/native/parity_arc.py. Developer machine only (torch + transformers + the arc_audio package).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _rms_exportable():
    """nn.RMSNorm as plain ops (aten::rms_norm has no ONNX export below opset 23). Same math: eps None = finfo eps."""
    import torch

    def fwd(self, x):
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return y * self.weight if self.weight is not None else y

    torch.nn.RMSNorm.forward = fwd


def load(arc_dir: str):
    sys.path.insert(0, arc_dir)
    from arc_audio import Arc
    return Arc.from_pretrained(arc_dir, device="cpu", threads=4, load_decoder=False)


def export_trunk(a, out: Path):
    import torch
    from arc_audio.model import layer_step
    tr = a.net.trunk
    L = len(tr.layers)
    lay0 = tr.layers[0]

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tr = tr

        def forward(self, x, dom, delay_i, h_in, bx_in):
            t = self.tr
            h = t.inp(t.in_norm(x)) + t.dom_emb(dom) + t.delay_emb(delay_i)
            e = t.dom_emb(dom) + t.delay_film(delay_i)
            D = h.shape[-1]
            hs, bxs = [], []
            for i, (lay, mlp) in enumerate(zip(t.layers, t.mlps)):
                gb = t.film[i](e)
                h = h * (1 + gb[..., :D]) + gb[..., D:]
                # layer_step with an explicit (h, -, -, bx) state: it reads only h0 and prev_bx
                y, st = layer_step(lay, h, (h_in[i], None, None, bx_in[i]))
                h = h + y
                h = h + mlp(h)
                hs.append(st[0])
                bxs.append(st[3])
            y = t.out_norm(t._adapt(h[:, None], dom))[:, 0]
            return y, torch.stack(hs), torch.stack(bxs)

    w = Step().eval()
    B = 1
    shp = (L, B, lay0.H, lay0.N, lay0.P)
    args = (torch.randn(B, tr.cfg["d_in"]), torch.tensor([12]), torch.tensor([3]), torch.zeros(shp), torch.zeros(shp))
    with torch.no_grad():
        torch.onnx.export(w, args, str(out / "trunk_step.onnx"), input_names=["x", "dom", "delay_i", "h_in", "bx_in"],
                          output_names=["y", "h_out", "bx_out"],
                          dynamic_axes={"x": {0: "b"}, "dom": {0: "b"}, "delay_i": {0: "b"}, "h_in": {1: "b"},
                                        "bx_in": {1: "b"}, "y": {0: "b"}, "h_out": {1: "b"}, "bx_out": {1: "b"}},
                          opset_version=17, dynamo=False)
    print("trunk_step exported", shp)


def export_mst(a, out: Path):
    import torch
    import torch.nn.functional as F
    enc = a.enc
    em = enc.embedder

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.em = em

        def forward(self, frames):
            z = torch.exp(self.em.comp.log_k) * self.em.cmvn(frames[None])
            h = torch.sign(z) * torch.log(z.abs() + torch.sqrt(z * z + 1))   # = comp (asinh): no ONNX symbolic
            return F.silu(self.em.linear(h))[0]

    with torch.no_grad():
        torch.onnx.export(Embed().eval(), (torch.randn(7, 80),), str(out / "mst_embed.onnx"), input_names=["frames"],
                          output_names=["e"], dynamic_axes={"frames": {0: "n"}, "e": {0: "n"}}, opset_version=17,
                          dynamo=False)
    for name in ("conv1", "conv2"):
        c = getattr(em, name)
        np.savez(out / f"mst_{name}.npz", w=c.weight.detach().numpy(), b=c.bias.detach().numpy(),
                 stride=c.stride[0])

    class Layer(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, X, q0, mask):
            # = arc_audio.mst.MSTStreamEncoder._layer after the key window X and the additive mask are cut
            layer = self.layer
            at = layer.self_attn
            hd = at.head_dim
            X = X[None]
            xn = layer.input_layernorm(X)
            Q = mask.shape[0]
            xq = xn[:, q0:q0 + Q]
            Qh = at.q_proj(xq).view(1, Q, -1, hd).transpose(1, 2)
            K = at.k_proj(xn).view(1, X.shape[1], -1, hd).transpose(1, 2)
            V = at.v_proj(xn).view(1, X.shape[1], -1, hd).transpose(1, 2)
            w = torch.softmax(Qh @ K.transpose(2, 3) * at.scaling + mask, dim=-1)
            o = (w @ V).transpose(1, 2).reshape(1, Q, -1)
            h = X[:, q0:q0 + Q] + at.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            return h[0]

    for li, layer in enumerate(enc.layers):
        assert layer.self_attn.num_key_value_groups == 1
        with torch.no_grad():
            torch.onnx.export(Layer(layer).eval(), (torch.randn(20, 320), torch.tensor(3), torch.zeros(5, 20)),
                              str(out / f"mst_layer{li}.onnx"), input_names=["X", "q0", "mask"], output_names=["h"],
                              dynamic_axes={"X": {0: "k"}, "mask": {0: "q", 1: "k"}, "h": {0: "q"}},
                              opset_version=17, dynamo=False)
    with torch.no_grad():
        torch.onnx.export(enc.final_norm.eval(), (torch.randn(1, 5, 320),), str(out / "mst_norm.onnx"),
                          input_names=["h"], output_names=["y"], dynamic_axes={"h": {1: "n"}, "y": {1: "n"}},
                          opset_version=17, dynamo=False)
    np.savez(out / "mst_meta.npz", windows=np.array([list(w) for w in enc.config.sliding_windows]),
             hidden=enc.config.hidden_size)
    print("mst exported:", len(enc.layers), "layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=["trunk", "mst", "all"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arc = load(a.arc)
    _rms_exportable()
    if a.part in ("trunk", "all"):
        export_trunk(arc, out)
    if a.part in ("mst", "all"):
        export_mst(arc, out)


if __name__ == "__main__":
    main()
