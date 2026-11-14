"""EmbeddingGemma 2 on ONNX Runtime (ONNX_RUNTIME=1): no torch, transformers or sentence-transformers.

Same three calls and the same vectors (up to quantisation) as server/embed.py's sentence-transformers path:
  encode_queries(texts)        "task: search result | query: " + text
  encode_docs(texts, titles)   "title: <t> | text: <x>" (or "title: none | text: " without titles)
  encode_audio(wavs)           <bos><|audio>{soft tokens}<audio|><eos>, the soft tokens from the audio tower

Files (MODELS_DIR/embeddinggemma-2-onnx, built by scripts/native/export_embed.py + quantize_embed.py):
  text.onnx, audio.onnx       the text encoder (embeddings in -> masked-mean pooled 768) and the audio tower
  embed_tokens.npz            the token table, int8 per row (gathered here in numpy)
  audio_frontend.npz          Gemma4AudioFeatureExtractor's window + mel filter bank (log-mel in numpy)
  tokenizer.json              read with `tokenizers` (Rust, no torch)
  config_sentence_transformers.json   the prompts
"""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path

import numpy as np

AUDIO_TOKEN, BOA, EOA = 258881, 256000, 258883
BOS, EOS, PAD = 2, 1, 0
MAX_TOKENS = 2048              # sentence-transformers leaves it unbounded; docs are cut to 600 chars upstream


class OnnxEmbedder:
    def __init__(self, path: str | Path, threads: int | None = None, audio: bool = True):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.dir = Path(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads or os.environ.get("ONNX_THREADS", "4"))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def sess(name):
            for n in (f"{name}.onnx", f"{name}_int8.onnx", f"{name}_fp32.onnx"):
                if (self.dir / n).exists():
                    return ort.InferenceSession(str(self.dir / n), so, providers=["CPUExecutionProvider"])
            raise FileNotFoundError(self.dir / f"{name}.onnx")

        self.text = sess("text")
        self.audio = sess("audio") if audio else None
        self.tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        z = np.load(self.dir / "embed_tokens.npz")
        self.E_q, self.E_s, self.e_scale = z["q"], z["scale"], float(z["embed_scale"])
        self.prompts = json.loads((self.dir / "config_sentence_transformers.json").read_text())["prompts"]
        f = np.load(self.dir / "audio_frontend.npz")
        self.win, self.mel = f["window"], f["mel"]
        self.flen, self.hop, self.nfft = int(f["frame_length"]), int(f["hop_length"]), int(f["fft_length"])
        self.mel_floor, self.ms_tok = float(f["mel_floor"]), int(f["ms_per_token"])
        self.max_samples, self.pad_mult = int(f["max_samples"]), int(f["pad_multiple"])
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ text
    def ids(self, text: str) -> list[int]:
        ids = self.tok.encode(text).ids          # the tokenizer's post-processor adds <bos> ... <eos>
        return ids[:MAX_TOKENS]

    def embed_ids(self, ids: np.ndarray) -> np.ndarray:
        return self.E_q[ids].astype(np.float32) * (self.E_s[ids] * self.e_scale)[..., None]

    def _run(self, seqs: list[np.ndarray]) -> np.ndarray:
        """seqs: per item [T,512] embeddings -> pooled [N,768] (padded batch, one graph call)."""
        T = max(len(s) for s in seqs)
        x = np.zeros((len(seqs), T, seqs[0].shape[1]), np.float32)
        am = np.zeros((len(seqs), T), np.int64)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = s
            am[i, :len(s)] = 1
        return self.text.run(None, {"inputs_embeds": x, "attention_mask": am})[0]

    def encode_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            b = texts[i:i + batch_size]
            seqs = [self.embed_ids(np.asarray(self.ids(t), np.int64)) for t in b]
            # sort-free: batches are small; pad to the longest in the batch
            out.append(self._run(seqs))
        return np.concatenate(out, 0) if out else np.zeros((0, 768), np.float32)

    def encode(self, texts: list[str], prompt_name: str | None = None, batch_size: int = 16) -> np.ndarray:
        p = self.prompts.get(prompt_name, "") if prompt_name else ""
        return self.encode_texts([p + t for t in texts], batch_size)

    # ------------------------------------------------------------------ audio
    def log_mel(self, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gemma4AudioFeatureExtractor for one mono 16 kHz clip (no dither, preemphasis 0, scale 1)."""
        w = np.asarray(wav, np.float32).reshape(-1)[: self.max_samples]
        n = len(w)
        padded = int(math.ceil(n / self.pad_mult) * self.pad_mult)
        mask = np.zeros(padded, np.int32)
        mask[:n] = 1
        w = np.pad(w, (0, padded - n))
        left = self.flen // 2
        w = np.pad(w, (left, 0))
        mask = np.pad(mask, (left, 0))
        size = self.flen + 1
        nfr = (len(w) - size) // self.hop + 1
        if nfr <= 0:
            return np.zeros((0, self.mel.shape[1]), np.float32), np.zeros(0, bool)
        fr = np.lib.stride_tricks.as_strided(w, shape=(nfr, size), strides=(w.strides[0] * self.hop, w.strides[0]))
        fr = fr[:, :-1] * self.win
        mag = np.abs(np.fft.rfft(fr, n=self.nfft, axis=-1))
        logm = np.log(mag @ self.mel + self.mel_floor).astype(np.float32)
        m = mask[np.arange(nfr) * self.hop + size - 1].astype(bool)
        return logm * m[:, None], m

    def encode_audio(self, wavs: list[np.ndarray]) -> np.ndarray:
        out = []
        for w in wavs:                  # one clip per call: the server sends one at a time
            f, m = self.log_mel(w)
            soft, sm = self.audio.run(None, {"input_features": f[None], "input_features_mask": m[None]})
            soft = soft[0][sm[0].astype(bool)]
            pre = self.embed_ids(np.asarray([BOS, BOA], np.int64))
            post = self.embed_ids(np.asarray([EOA, EOS], np.int64))
            out.append(self._run([np.concatenate([pre, soft.astype(np.float32), post], 0)])[0])
        return np.stack(out, 0) if out else np.zeros((0, 768), np.float32)
