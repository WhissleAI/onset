"""Native runtime (ONNX_RUNTIME=1): the switch, the numpy front ends, the models manifest entry, the native requirements.

Model-free tests always run. The ones that need the ONNX export (MODELS_DIR/embeddinggemma-2-onnx or EMBED_ONNX_DIR) or
torch/transformers (reference side) skip when those are missing; parity at scale is scripts/native/parity_*.py."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from server import arc_onnx, config, embed

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "models" / "manifest.json").read_text())


# ------------------------------------------------------------------------------------------------ switch
def test_flag_defaults_off():
    src = (ROOT / "server" / "config.py").read_text()
    assert 'ONNX_RUNTIME = os.environ.get("ONNX_RUNTIME", "0") == "1"' in src


class _Fake:
    def __init__(self):
        self.calls = []

    def encode(self, texts, prompt_name=None, batch_size=16):
        self.calls.append(("text", list(texts), prompt_name))
        return np.ones((len(texts), 768), np.float32)

    def encode_audio(self, wavs):
        self.calls.append(("audio", len(wavs), None))
        return np.ones((len(wavs), 768), np.float32) * 2


def test_switch_routes_every_call_to_onnx(monkeypatch):
    fake = _Fake()
    monkeypatch.setattr(config, "ONNX_RUNTIME", True)
    monkeypatch.setattr(embed, "_onnx", fake)
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setattr(embed, "_qmodel", None)
    q = embed.encode_queries(["who founded nvidia"])
    d = embed.encode_docs(["text"], ["Title"])
    d2 = embed.encode_docs(["text"])
    a = embed.encode_audio([np.zeros(16000, np.float32)])
    assert fake.calls[0] == ("text", ["who founded nvidia"], "SearchQuery")
    assert fake.calls[1] == ("text", ["title: Title | text: text"], None)
    assert fake.calls[2] == ("text", ["text"], "Document")
    assert fake.calls[3] == ("audio", 1, None)
    for v in (q, d, d2, a):                                          # normalised like the torch path
        assert np.allclose(np.linalg.norm(v, axis=-1), 1.0, atol=1e-5)
    assert embed._model is None and embed._qmodel is None            # torch never loaded


# ------------------------------------------------------------------------------------------------ numpy pieces
def test_conv1d_stride2_matches_direct_loop():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((23, 5)).astype(np.float32)
    w = rng.standard_normal((7, 5, 3)).astype(np.float32)
    b = rng.standard_normal(7).astype(np.float32)
    y = arc_onnx._conv1d_s2(x, w, b, 2)
    ref = np.stack([np.array([(x[i * 2:i * 2 + 3].T * w[o]).sum() + b[o] for o in range(7)]) for i in range(11)])
    assert y.shape == (11, 7) and np.allclose(y, ref, atol=1e-5)


def test_pooler_matches_arc_audio_when_available():
    arc_dir = config.ARC_DIR
    if not (Path(arc_dir) / "arc_audio" / "api.py").exists() or importlib.util.find_spec("torch") is None:
        pytest.skip("arc-v1 / torch not here")
    import sys
    sys.path.insert(0, str(arc_dir))
    from arc_audio.api import Pooler as Ref
    rng = np.random.default_rng(1)
    a, b = arc_onnx.Pooler(), Ref()
    out_a, out_b = [], []
    for n in (3, 7, 0, 12, 5):
        e = rng.standard_normal((n, 320)).astype(np.float32)
        out_a += a.push(e)
        out_b += b.push(e)
    out_a += a.flush()
    out_b += b.flush()
    assert len(out_a) == len(out_b) and all(np.array_equal(x, y) for x, y in zip(out_a, out_b))


def _eg_dir():
    p = Path(config.EMBED_ONNX_DIR)
    return p if (p / "text.onnx").exists() and (p / "audio_frontend.npz").exists() else None


def test_log_mel_matches_transformers_feature_extractor():
    d = _eg_dir()
    if d is None or importlib.util.find_spec("transformers") is None:
        pytest.skip("ONNX export / transformers not here")
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    from server.embed_onnx import OnnxEmbedder
    e = OnnxEmbedder.__new__(OnnxEmbedder)              # front end only: no ONNX sessions
    f = np.load(d / "audio_frontend.npz")
    e.win, e.mel = f["window"], f["mel"]
    e.flen, e.hop, e.nfft = int(f["frame_length"]), int(f["hop_length"]), int(f["fft_length"])
    e.mel_floor, e.max_samples, e.pad_mult = float(f["mel_floor"]), int(f["max_samples"]), int(f["pad_multiple"])
    fe = Gemma4AudioFeatureExtractor()
    rng = np.random.default_rng(2)
    for n in (16000, 24077, 48000 + 5, 8000):
        w = (rng.standard_normal(n) * 0.05).astype(np.float32)
        ref = fe([w], return_tensors="np")
        mel, m = e.log_mel(w)
        assert np.array_equal(m, ref["input_features_mask"][0])
        assert np.abs(mel - ref["input_features"][0]).max() < 1e-4


def test_onnx_embedder_is_batch_invariant_and_unit_norm():
    d = _eg_dir()
    if d is None:
        pytest.skip("ONNX export not here")
    from server.embed_onnx import OnnxEmbedder
    try:
        e = OnnxEmbedder(d, threads=2, audio=False)
    except Exception as ex:  # noqa: BLE001 (onnxruntime < 1.2x: 4-bit MatMulNBits only; requirements-native pins 1.32)
        if "4b" in str(ex) or "MatMulNBits" in str(ex):
            pytest.skip(f"onnxruntime too old for 8-bit MatMulNBits: {ex}")
        raise
    one = e.encode(["what is the capital of australia"], "SearchQuery")
    two = e.encode(["what is the capital of australia", "a much longer question about the fall of the roman empire"],
                   "SearchQuery")
    n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    # padding is masked out of attention and the mean; what is left is the 8-bit activation blocks (accuracy_level 4)
    assert float(n(one)[0] @ n(two)[0]) > 0.9999
    assert e.ids("task: search result | query: hi")[0] == 2 and e.ids("x")[-1] == 1     # <bos> ... <eos>


# ------------------------------------------------------------------------------------------------ manifest / reqs
def _entry(name):
    return next((m for m in MANIFEST["models"] if m.get("dir") == name), None)


def test_manifest_lists_the_onnx_embedder_with_checksums():
    m = _entry("embeddinggemma-2-onnx")
    assert m is not None and m["required"] is False                  # only with ONNX_RUNTIME=1
    paths = {f["path"] for f in m["files"]}
    assert {"text.onnx", "audio.onnx", "embed_tokens.npz", "audio_frontend.npz", "tokenizer.json",
            "config_sentence_transformers.json", "LICENSE", "NOTICE"} <= paths
    for f in m["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0
    assert m["size"] == sum(f["size"] for f in m["files"])
    assert "Apache" in m["license"]
    # the NOTICE / LICENSE shipped with the model are the repo's (the export copies them)
    for f in ("LICENSE", "NOTICE"):
        assert (ROOT / "models" / "embeddinggemma-2-onnx" / f).exists()


def test_native_requirements_have_no_torch_stack():
    req = (ROOT / "macapp" / "requirements-native.txt").read_text()
    names = {re.split(r"[=<> ]", l.strip())[0].lower() for l in req.splitlines() if l.strip() and not l.startswith("#")}
    assert "onnxruntime" in names and "tokenizers" in names
    assert not names & {"torch", "transformers", "sentence-transformers", "torchvision", "accelerate", "scipy",
                        "safetensors", "pillow"}
