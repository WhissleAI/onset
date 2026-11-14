"""Smaller weights for the EmbeddingGemma 2 ONNX graphs.

    python scripts/native/quantize_embed.py --dir <export dir> [--mode nbits8|nbits4|dynamic] [--block 32] [--keep-fp32]

text_fp32.onnx -> text.onnx, audio_fp32.onnx -> audio.onnx.
  nbits8 (default)  weight-only int8 (MatMulNBits, symmetric, one scale per --block weights along K), computed with
                    --accuracy-level 4. Needs a recent onnxruntime (8-bit MatMulNBits; tested 1.32.0).
  nbits4            the same at 4 bits (half the size again; measured too lossy for this model, see MACAPP.md)
  dynamic           QDQ-free dynamic int8 (MatMulInteger: activations quantised per call). Measured: query cosine 0.990
                    and doc cosine 0.967 against torch, rerank top-5 agreement 0.70: NOT shipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--mode", default="nbits8", choices=["nbits8", "nbits4", "dynamic"])
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--accuracy-level", type=int, default=4,
                    help="MatMulNBits compute: 4 = int8 dot products on block-quantised activations (3-6x faster on "
                         "Apple CPUs than 0 = f32 compute; measured cost: query cosine 0.99988 vs 0.99995)")
    ap.add_argument("--keep-fp32", action="store_true")
    ap.add_argument("--parts", default="text,audio")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    d = Path(a.dir)
    for part in a.parts.split(","):
        src, dst = d / f"{part}_fp32.onnx", d / f"{part}{a.suffix}.onnx"
        if a.mode == "dynamic":
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True,
                             op_types_to_quantize=["MatMul"], extra_options={"MatMulConstBOnly": True})
        else:
            import onnx
            from onnxruntime.quantization.matmul_nbits_quantizer import (DefaultWeightOnlyQuantConfig,
                                                                         MatMulNBitsQuantizer)
            bits = 8 if a.mode == "nbits8" else 4
            cfg = DefaultWeightOnlyQuantConfig(block_size=a.block, is_symmetric=True, bits=bits,
                                               accuracy_level=a.accuracy_level)
            q = MatMulNBitsQuantizer(onnx.load(str(src)), bits=bits, block_size=a.block, is_symmetric=True,
                                     accuracy_level=a.accuracy_level, algo_config=cfg)
            q.process()
            q.model.save_model_to_file(str(dst), use_external_data_format=False)
        print(part, a.mode, f"{src.stat().st_size / 1e6:.0f} MB -> {dst.stat().st_size / 1e6:.0f} MB")
        if not a.keep_fp32:
            src.unlink()


if __name__ == "__main__":
    main()
