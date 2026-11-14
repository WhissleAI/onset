"""Where the models and the app's writable data live. No machine-specific paths: everything resolves from

  1. an explicit env var for that one model (ARC_DIR, EMBED_MODEL, ARCTAG_MODEL_DIR, INDEX_DIR, ...), else
  2. MODELS_DIR/<name> (MODELS_DIR env, default ~/Library/Application Support/<APP_NAME>/models), else
  3. a legacy location inside this repo (models/arctag, index/data, eval/audio) if it exists, else
  4. MODELS_DIR/<name> anyway (missing: scripts/fetch_models.py fills it; the error says so).

APP_NAME comes from env APP_NAME, else macapp/app.conf (the one build setting the macOS app uses), else "Onset".
The names under MODELS_DIR are the "dir" fields of models/manifest.json (kept in sync by tests/test_modelpaths.py).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_APP_NAME = "Onset"


def app_name(env=None, root: Path = ROOT) -> str:
    env = os.environ if env is None else env
    if env.get("APP_NAME"):
        return env["APP_NAME"]
    conf = root / "macapp" / "app.conf"
    try:
        for line in conf.read_text().splitlines():
            m = re.match(r'\s*APP_NAME\s*=\s*"?([^"#]+?)"?\s*(#.*)?$', line)
            if m:
                return m.group(1).strip()
    except OSError:
        pass
    return DEFAULT_APP_NAME


def support_dir(env=None, root: Path = ROOT) -> Path:
    """~/Library/Application Support/<APP_NAME> (HOME from the environment, never a hard-coded user)."""
    env = os.environ if env is None else env
    home = Path(env.get("HOME") or Path.home())
    return home / "Library" / "Application Support" / app_name(env, root)


def models_dir(env=None, root: Path = ROOT) -> Path:
    env = os.environ if env is None else env
    if env.get("MODELS_DIR"):
        return Path(env["MODELS_DIR"]).expanduser()
    return support_dir(env, root) / "models"


def data_dir(env=None, root: Path = ROOT) -> Path:
    """Writable state (memory, caches). The app sets HEADSTART_DATA_DIR (Application Support); a repo run keeps runs/."""
    env = os.environ if env is None else env
    if env.get("HEADSTART_DATA_DIR"):
        return Path(env["HEADSTART_DATA_DIR"]).expanduser()
    return root / "runs"


def resolve(name: str, env_var: str | None = None, legacy: Path | None = None, env=None, root: Path = ROOT) -> Path:
    """Path of model `name` (a manifest "dir"), by the order in the module docstring."""
    env = os.environ if env is None else env
    if env_var and env.get(env_var):
        return Path(env[env_var]).expanduser()
    p = models_dir(env, root) / name
    if p.exists():
        return p
    if legacy is not None and legacy.exists():
        return legacy
    return p


def embed_model(env=None, root: Path = ROOT) -> str:
    """EmbeddingGemma 2: a local folder (MODELS_DIR/embeddinggemma-2) when present, else the HF id (HF cache / download).
    EMBED_MODEL env always wins (a local path or another HF id)."""
    env = os.environ if env is None else env
    if env.get("EMBED_MODEL"):
        return env["EMBED_MODEL"]
    p = models_dir(env, root) / "embeddinggemma-2"
    if (p / "config.json").exists():
        return str(p)
    return "google/embeddinggemma-2"


def tagger_lib(env=None, root: Path = ROOT) -> Path:
    """libarctag.dylib is built from vendor/arctag/src (scripts/fetch_models.py --build-tagger, or the app build)."""
    env = os.environ if env is None else env
    if env.get("ARCTAG_LIB"):
        return Path(env["ARCTAG_LIB"]).expanduser()
    repo = root / "vendor" / "arctag" / "libarctag.dylib"
    if repo.exists():
        return repo
    return models_dir(env, root) / "arctag" / "libarctag.dylib"
