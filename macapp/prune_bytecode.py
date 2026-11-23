"""Build step (scripts/build_macapp.sh): keep precompiled bytecode only for the modules the server imports.

`uv pip install --compile-bytecode` compiles every module (~230 MB of __pycache__, mostly transformers' 500 model
families and sympy). This imports what the server loads at startup (server.app + the model classes its warm-up
instantiates), then deletes every .pyc whose source module was not loaded. Anything imported later still works: it is
compiled in memory (the bundle is read-only; PYTHONDONTWRITEBYTECODE=1).

    <bundled python> -I macapp/prune_bytecode.py <app root> <site-packages>
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

app_root, site = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(app_root))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MODELS_DIR", str(app_root / "_no_models_"))

import server.app  # noqa: E402,F401  (imports the whole server: lanes, session, ear, tagger, embed, ...)

# loaded lazily at warm-up: sentence-transformers + the transformers model classes behind EmbeddingGemma 2 and Arc,
# onnxruntime + tokenizers behind the answer model (server/answer_lm.py)
for name in ["sentence_transformers", "transformers.models.auto.modeling_auto", "transformers.models.auto.processing_auto",
             "transformers.models.auto.tokenization_auto", "transformers.models.auto.feature_extraction_auto",
             "transformers.models.auto.image_processing_auto", "transformers.models.auto.video_processing_auto",
             "torch.nn", "safetensors.torch", "soundfile", "anthropic", "httpx", "websockets", "uvicorn",
             "uvicorn.protocols.websockets.websockets_impl", "uvicorn.protocols.http.h11_impl", "rapidfuzz", "jellyfish",
             "server.answer_lm", "onnxruntime", "tokenizers"]:
    try:
        importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        print(f"  (prune: could not import {name}: {type(e).__name__})")
models = site / "transformers" / "models"
for fam in sorted(p.name for p in models.iterdir() if p.is_dir()):
    if fam.startswith(("embedding_gemma", "gemma", "moonshine", "siglip", "auto")):
        for py in (models / fam).glob("*.py"):
            try:
                importlib.import_module(f"transformers.models.{fam}.{py.stem}")
            except Exception:  # noqa: BLE001
                pass

keep = set()
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and f.endswith(".py"):
        keep.add(Path(f).resolve())

removed = kept = freed = 0
for pyc in site.rglob("*.pyc"):
    if pyc.parent.name != "__pycache__":
        continue
    src = pyc.parent.parent / (pyc.name.split(".")[0] + ".py")
    if src.resolve() in keep:
        kept += 1
        continue
    freed += pyc.stat().st_size
    pyc.unlink()
    removed += 1
for d in list(site.rglob("__pycache__")):
    try:
        d.rmdir()
    except OSError:
        pass
print(f"  bytecode: kept {kept} modules, removed {removed} ({freed / 1e6:.0f} MB)")
