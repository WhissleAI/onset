"""Export EmbeddingGemma 2 (text + audio towers; the vision tower is not used by Onset) to ONNX for server/embed_onnx.py.

    python scripts/native/export_embed.py --src <embeddinggemma-2 dir> --out <dir> [--part text|audio|tables|all]

Writes into --out:
  text_fp32.onnx      inputs_embeds [B,T,512] f32 + attention_mask [B,T] i64 -> pooled [B,768] (masked mean of the
                      projected token states, not normalised: the same numbers sentence-transformers' Pooling gives)
  audio_fp32.onnx     input_features [B,F,128] f32 + input_features_mask [B,F] bool -> soft [B,T',512], mask [B,T']
  embed_tokens.npz    the token table as int8 rows + one f32 scale per row (gather in numpy), and the embed scale
  audio_frontend.npz  Gemma4AudioFeatureExtractor's window + HTK mel filter bank (numpy front end, no transformers)
  tokenizer.json, config_sentence_transformers.json (prompts), LICENSE notes are copied by quantize_embed.py

Run it on a developer machine with torch + transformers (never in the app). One part per process keeps RSS ~2-3 GB.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load(src: str, audio: bool):
    import torch
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(src)
    cfg.vision_config = None
    if not audio:
        cfg.audio_config = None
    m = AutoModel.from_pretrained(src, config=cfg, dtype=torch.float32, attn_implementation="eager").eval()
    return m


def export_text(src: str, out: Path):
    import torch

    m = load(src, audio=False)
    lm = m.language_model

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = lm

        def forward(self, inputs_embeds, attention_mask):
            h = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state
            w = attention_mask.to(h.dtype).unsqueeze(-1)
            return (h * w).sum(1) / w.sum(1).clamp(min=1.0)

    w = Text().eval()
    x = torch.randn(2, 19, lm.config.hidden_size)
    am = torch.ones(2, 19, dtype=torch.long)
    am[1, 15:] = 0
    with torch.no_grad():
        torch.onnx.export(w, (x, am), str(out / "text_fp32.onnx"), input_names=["inputs_embeds", "attention_mask"],
                          output_names=["pooled"], dynamic_axes={"inputs_embeds": {0: "b", 1: "t"},
                                                                 "attention_mask": {0: "b", 1: "t"}, "pooled": {0: "b"}},
                          opset_version=17, dynamo=False)
    # the token table: int8 per row (gathered in numpy; the graph takes embeddings so audio soft tokens can go in too)
    W = lm.embed_tokens.weight.detach().float().numpy()
    s = np.maximum(np.abs(W).max(1), 1e-12) / 127.0
    q = np.clip(np.rint(W / s[:, None]), -127, 127).astype(np.int8)
    np.savez(out / "embed_tokens.npz", q=q, scale=s.astype(np.float32),
             embed_scale=np.float32(lm.config.hidden_size ** 0.5))
    print("text exported")


def export_audio(src: str, out: Path):
    import torch

    m = load(src, audio=True)
    from transformers.models.gemma4 import modeling_gemma4 as G
    ac = m.config.audio_config
    left, right = ac.attention_context_left - 1, ac.attention_context_right

    def mask_no_vmap(config=None, inputs_embeds=None, attention_mask=None, and_mask_function=None, **_):
        # = masking_utils.create_bidirectional_mask(..., sliding_window_mask_function((left, right))) as the sdpa path
        # returns it: a BOOL mask (True = attend), which is what Gemma4AudioAttention's masked_fill expects. (In eager
        # mode HF hands it an additive float mask, which that masked_fill inverts: eager output is wrong, sdpa = the
        # production path.) Plain ops: the vmap inside the original does not trace to ONNX.
        T = inputs_embeds.shape[1]
        i = torch.arange(T)
        d = i[:, None] - i[None, :]
        ok = ((d >= 0) & (d < left)) | ((d < 0) & (-d < right))
        return ok[None, None] & attention_mask[:, None, None, :].bool()

    G.create_bidirectional_mask = mask_no_vmap

    def block_context(self, hidden_states):
        # = the original pad + unfold(1, context_size, chunk_size) + movedim, as an index gather (Unfold has no
        # ONNX export with a dynamic length)
        hs = torch.nn.functional.pad(
            hidden_states, (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1))
        nb = (hs.shape[1] - self.context_size) // self.chunk_size + 1
        idx = torch.arange(nb)[:, None] * self.chunk_size + torch.arange(self.context_size)[None, :]
        return hs[:, idx.reshape(-1)].reshape(hs.shape[0], nb, self.context_size, hs.shape[2], hs.shape[3])

    G.Gemma4AudioAttention._extract_block_context = block_context

    class Audio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = m.audio_tower
            self.proj = m.embed_audio

        def forward(self, input_features, input_features_mask):
            o = self.tower(input_features, input_features_mask, return_dict=True)
            return self.proj(inputs_embeds=o.last_hidden_state), o.attention_mask

    w = Audio().eval()
    f = torch.randn(1, 199, 128)
    fm = torch.ones(1, 199, dtype=torch.bool)
    with torch.no_grad():
        torch.onnx.export(w, (f, fm), str(out / "audio_fp32.onnx"), input_names=["input_features", "input_features_mask"],
                          output_names=["soft", "mask"], dynamic_axes={"input_features": {0: "b", 1: "f"},
                                                                        "input_features_mask": {0: "b", 1: "f"},
                                                                        "soft": {0: "b", 1: "t"}, "mask": {0: "b", 1: "t"}},
                          opset_version=17, dynamo=False)
    print("audio exported")


def export_tables(src: str, out: Path):
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    pc = json.loads((Path(src) / "processor_config.json").read_text())
    fe = Gemma4AudioFeatureExtractor(**{k: v for k, v in pc["feature_extractor"].items()
                                        if k not in ("feature_extractor_type",)})
    np.savez(out / "audio_frontend.npz", window=fe.window.astype(np.float32), mel=fe.mel_filters.astype(np.float64),
             frame_length=fe.frame_length, hop_length=fe.hop_length, fft_length=fe.fft_length,
             mel_floor=np.float64(fe.mel_floor), ms_per_token=pc.get("audio_ms_per_token", 40),
             max_samples=480000, pad_multiple=128)
    for f in ("tokenizer.json", "config_sentence_transformers.json"):
        shutil.copy(Path(src) / f, out / f)
    print("tables exported")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=["text", "audio", "tables", "all"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.part in ("tables", "all"):
        export_tables(a.src, out)
    if a.part in ("text", "all"):
        export_text(a.src, out)
    if a.part in ("audio", "all"):
        export_audio(a.src, out)


if __name__ == "__main__":
    main()
