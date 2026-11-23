"""Build the on-device answer model (models/manifest.json "answer-lm"): Qwen2.5-0.5B-Instruct as one ~330 MB ONNX graph.

    python scripts/native/quantize_answer_lm.py --src <onnx-community/Qwen2.5-0.5B-Instruct snapshot> --out <dir>
                                                [--accuracy-level 4]

Source: onnx-community/Qwen2.5-0.5B-Instruct @ 22942cb7d7ba4cc81bb4673549ca4d4614469b5e, onnx/model_q4.onnx (786 MB):
every transformer MatMul is already 4-bit (MatMulNBits, block 32), but the tied embedding / LM head is a float32
151936 x 896 table (545 MB) that the graph also transposes into an f32 MatMul for the logits. This script:

  1. quantises that table once to 4 bits (symmetric, block 32 along the hidden size, f32 scale per block, zero point 8):
     the MatMulNBits weight layout [vocab, 28 blocks, 16 bytes] IS a row-major embedding table, so ONE initializer
     (68 MB + 17 MB of scales) serves both
       - the LM head: MatMulNBits(hidden, table) -> logits, and
       - the token embedding: Gather rows of the table, unpack the nibbles with plain ONNX ops, scale;
  2. computes logits for the LAST position only (a Slice before the LM head): prefill no longer materialises
     prompt_len x 151936 floats (~330 MB for a 550-token prompt), which is most of the time to first token;
  3. optionally sets MatMulNBits accuracy_level (4 = int8 dot products on Apple CPUs).

Everything else (24 layers, KV-cache inputs past_key_values.N.key/value, outputs present.N.*) is unchanged. Plain
onnxruntime runs it (com.microsoft MatMulNBits 4-bit: onnxruntime >= 1.17); server/answer_lm.py drives the decode loop.
The tokenizer files are copied from the same snapshot.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EMB = "model.embed_tokens.weight"
BLOCK = 32
TOKENIZER_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")


def quantize_rows(w: np.ndarray, block: int = BLOCK):
    """[V, H] f32 -> (packed uint8 [V, H/block, block/2], scales f32 [V * H/block]); w ~= (q - 8) * scale."""
    v, h = w.shape
    nb = h // block
    b = w.reshape(v, nb, block).astype(np.float32)
    scale = np.abs(b).max(axis=2) / 7.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(b / scale[..., None]), -8, 7).astype(np.int16) + 8
    q = q.astype(np.uint8)
    packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
    return packed, scale.reshape(-1).astype(np.float32)


def dequantize_rows(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    v, nb, half = packed.shape
    lo, hi = packed & 0x0F, packed >> 4
    q = np.stack([lo, hi], axis=-1).reshape(v, nb, half * 2).astype(np.float32) - 8.0
    return (q * scales.reshape(v, nb, 1)).reshape(v, nb * half * 2)


def build(src: Path, out: Path, accuracy_level: int | None):
    m = onnx.load(str(src / "onnx" / "model_q4.onnx"))
    g = m.graph
    init = {t.name: t for t in g.initializer}
    w = numpy_helper.to_array(init[EMB])
    V, H = w.shape
    nb = H // BLOCK
    packed, scales = quantize_rows(w)
    err = np.abs(dequantize_rows(packed[:2000], scales[: 2000 * nb]) - w[:2000]).mean()
    print(f"  embedding {V}x{H}: 4-bit, mean abs error {err:.5f} (mean |w| {np.abs(w[:2000]).mean():.5f})")

    g.initializer.remove(init[EMB])
    g.initializer.extend([numpy_helper.from_array(packed, "embed_q4"), numpy_helper.from_array(scales, "embed_scales"),
                          numpy_helper.from_array(np.array([V, nb], np.int64), "embed_scales_shape"),
                          numpy_helper.from_array(np.array([16], np.int32), "c16_i32"),
                          numpy_helper.from_array(np.array([8.0], np.float32), "c8_f32"),
                          numpy_helper.from_array(np.array([-1], np.int64), "ax_last"),
                          numpy_helper.from_array(np.array([-1], np.int64), "slice_start"),
                          numpy_helper.from_array(np.array([np.iinfo(np.int64).max], np.int64), "slice_end"),
                          numpy_helper.from_array(np.array([1], np.int64), "ax_1"),
                          numpy_helper.from_array(np.array([nb, BLOCK], np.int64), "nb_block"),
                          numpy_helper.from_array(np.array([H], np.int64), "hidden")])

    nodes = list(g.node)
    gather = next(n for n in nodes if n.op_type == "Gather" and n.input[0] == EMB)
    transpose = next(n for n in nodes if n.op_type == "Transpose" and n.input[0] == EMB)
    lm_head = next(n for n in nodes if n.op_type == "MatMul" and n.input[1] == transpose.output[0])
    ids, emb_out = gather.input[1], gather.output[0]
    hidden, logits = lm_head.input[0], lm_head.output[0]

    P = "/answer_lm/"
    emb_nodes = [
        helper.make_node("Gather", ["embed_q4", ids], [P + "rows"], axis=0),                  # [b, s, nb, 16] uint8
        helper.make_node("Cast", [P + "rows"], [P + "rows_i"], to=TensorProto.INT32),
        helper.make_node("Mod", [P + "rows_i", "c16_i32"], [P + "lo"]),
        helper.make_node("Div", [P + "rows_i", "c16_i32"], [P + "hi"]),
        helper.make_node("Unsqueeze", [P + "lo", "ax_last"], [P + "lo1"]),
        helper.make_node("Unsqueeze", [P + "hi", "ax_last"], [P + "hi1"]),
        helper.make_node("Concat", [P + "lo1", P + "hi1"], [P + "nib"], axis=-1),            # [b, s, nb, 16, 2]
        helper.make_node("Shape", [ids], [P + "ids_shape"]),
        helper.make_node("Concat", [P + "ids_shape", "nb_block"], [P + "q_shape"], axis=0),
        helper.make_node("Reshape", [P + "nib", P + "q_shape"], [P + "q"]),                  # [b, s, nb, 32]
        helper.make_node("Cast", [P + "q"], [P + "qf"], to=TensorProto.FLOAT),
        helper.make_node("Sub", [P + "qf", "c8_f32"], [P + "qc"]),
        helper.make_node("Reshape", ["embed_scales", "embed_scales_shape"], [P + "scales2d"]),
        helper.make_node("Gather", [P + "scales2d", ids], [P + "s"], axis=0),                 # [b, s, nb]
        helper.make_node("Unsqueeze", [P + "s", "ax_last"], [P + "s1"]),
        helper.make_node("Mul", [P + "qc", P + "s1"], [P + "deq"]),
        helper.make_node("Concat", [P + "ids_shape", "hidden"], [P + "e_shape"], axis=0),
        helper.make_node("Reshape", [P + "deq", P + "e_shape"], [emb_out]),
    ]
    head_nodes = [
        helper.make_node("Slice", [hidden, "slice_start", "slice_end", "ax_1"], [P + "last"]),
        helper.make_node("MatMulNBits", [P + "last", "embed_q4", "embed_scales"], [logits], domain="com.microsoft",
                         K=H, N=V, bits=4, block_size=BLOCK, name="/answer_lm/lm_head_Q4"),
    ]
    out_nodes = []
    for n in nodes:
        if n is gather:
            out_nodes.extend(emb_nodes)
        elif n is lm_head:
            out_nodes.extend(head_nodes)
        elif n is transpose:
            continue
        else:
            out_nodes.append(n)
    del g.node[:]
    g.node.extend(out_nodes)
    for o in g.output:                       # logits: [batch, 1, vocab] now
        if o.name == logits:
            o.type.tensor_type.shape.dim[1].Clear()
            o.type.tensor_type.shape.dim[1].dim_value = 1
    if accuracy_level is not None:
        for n in g.node:
            if n.op_type == "MatMulNBits":
                n.attribute.append(helper.make_attribute("accuracy_level", accuracy_level))
    out.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(out / "model.onnx"))
    for f in TOKENIZER_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    print(f"  wrote {out / 'model.onnx'} ({(out / 'model.onnx').stat().st_size / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="snapshot of onnx-community/Qwen2.5-0.5B-Instruct (onnx/model_q4.onnx)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--accuracy-level", type=int, default=None)
    a = ap.parse_args()
    build(Path(a.src), Path(a.out), a.accuracy_level)


if __name__ == "__main__":
    main()
