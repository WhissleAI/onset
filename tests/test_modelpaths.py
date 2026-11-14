"""Model path resolution (server/modelpaths.py) and the model manifest (models/manifest.json, scripts/fetch_models.py).
No network, no real models: everything runs against tmp folders."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from server import modelpaths as mp

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "models" / "manifest.json").read_text())


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ app name / dirs
def test_app_name_from_conf_env_and_default(tmp_path):
    (tmp_path / "macapp").mkdir()
    (tmp_path / "macapp" / "app.conf").write_text('# x\nAPP_NAME="My App"   # the name\nBUNDLE_ID="a.b"\n')
    assert mp.app_name({}, tmp_path) == "My App"
    assert mp.app_name({"APP_NAME": "Other"}, tmp_path) == "Other"
    assert mp.app_name({}, tmp_path / "nowhere") == mp.DEFAULT_APP_NAME


def test_repo_app_conf_is_the_single_name_setting():
    conf = (ROOT / "macapp" / "app.conf").read_text()
    name = re.search(r'^APP_NAME="([^"]+)"', conf, re.M).group(1)
    assert mp.app_name({}, ROOT) == name
    assert re.search(r'^BUNDLE_ID="[a-z0-9.\-]+"', conf, re.M)


def test_models_dir_is_application_support_under_home(tmp_path):
    env = {"HOME": str(tmp_path), "APP_NAME": "Onset"}
    assert mp.models_dir(env) == tmp_path / "Library" / "Application Support" / "Onset" / "models"
    assert mp.models_dir({**env, "MODELS_DIR": str(tmp_path / "m")}) == tmp_path / "m"


def test_data_dir_app_vs_repo(tmp_path):
    assert mp.data_dir({}, tmp_path) == tmp_path / "runs"
    assert mp.data_dir({"HEADSTART_DATA_DIR": str(tmp_path / "d")}) == tmp_path / "d"


# ------------------------------------------------------------------------------------------------ resolution order
def test_resolve_order_env_then_models_dir_then_legacy(tmp_path):
    md, legacy = tmp_path / "models", tmp_path / "repo" / "index" / "data"
    env = {"MODELS_DIR": str(md)}
    # nothing anywhere: MODELS_DIR/<name> (so the error names the place fetch_models.py fills)
    assert mp.resolve("index-data", "INDEX_DIR", legacy, env) == md / "index-data"
    legacy.mkdir(parents=True)
    assert mp.resolve("index-data", "INDEX_DIR", legacy, env) == legacy
    (md / "index-data").mkdir(parents=True)
    assert mp.resolve("index-data", "INDEX_DIR", legacy, env) == md / "index-data"
    assert mp.resolve("index-data", "INDEX_DIR", legacy, {**env, "INDEX_DIR": "/x/y"}) == Path("/x/y")


def test_embed_model_local_folder_or_hf_id(tmp_path):
    env = {"MODELS_DIR": str(tmp_path)}
    assert mp.embed_model(env) == "google/embeddinggemma-2"
    (tmp_path / "embeddinggemma-2").mkdir()
    (tmp_path / "embeddinggemma-2" / "config.json").write_text("{}")
    assert mp.embed_model(env) == str(tmp_path / "embeddinggemma-2")
    assert mp.embed_model({**env, "EMBED_MODEL": "org/other"}) == "org/other"


def test_tagger_lib_env_repo_then_models_dir(tmp_path):
    root = tmp_path / "repo"
    env = {"MODELS_DIR": str(tmp_path / "m")}
    assert mp.tagger_lib(env, root) == tmp_path / "m" / "arctag" / "libarctag.dylib"
    lib = root / "vendor" / "arctag" / "libarctag.dylib"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    assert mp.tagger_lib(env, root) == lib
    assert mp.tagger_lib({**env, "ARCTAG_LIB": "/l.dylib"}, root) == Path("/l.dylib")


def test_no_machine_specific_paths_in_server():
    """Config and the server never point at one developer's home folder."""
    for p in (ROOT / "server").rglob("*.py"):
        text = p.read_text()
        assert "/Users/" not in text, p
        assert "dv3work" not in text, p


def test_config_dirs_come_from_modelpaths():
    from server import config
    assert config.ARC_DIR.name == "arc-v1" or "ARC_DIR" in __import__("os").environ
    assert config.MEM_DIR.parent == config.DATA_DIR or "MEM_DIR" in __import__("os").environ


# ------------------------------------------------------------------------------------------------ manifest
def test_manifest_shape_and_dirs_match_config():
    dirs = {m["dir"] for m in MANIFEST["models"] if m["dir"]}
    assert {"arc-v1", "arctag", "embeddinggemma-2", "index-data", "demo-audio"} <= dirs
    src = (ROOT / "server" / "config.py").read_text() + (ROOT / "server" / "modelpaths.py").read_text() + \
        (ROOT / "server" / "tagger.py").read_text()
    for d in dirs:
        assert f'"{d}"' in src, f"manifest dir {d} is not resolved anywhere in the server"
    for m in MANIFEST["models"]:
        assert m["name"] and m["license"] and m["source"]["type"] in ("hf", "url", "local", "build", "repo")
        assert m["size"] == sum(f["size"] for f in m["files"])
        for f in m["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] >= 0 and not f["path"].startswith("/")
        if m["source"]["type"] == "hf":
            assert re.fullmatch(r"[0-9a-f]{40}", m["source"]["revision"]), "HF sources are pinned to a commit"
        if m["source"]["type"] == "local":
            assert m["source"].get("note"), "a model with no public source says so"


def test_manifest_repo_files_match_git_checkout():
    """The director ships in the repo: its manifest checksums must match the files."""
    for m in MANIFEST["models"]:
        if m["source"]["type"] == "repo":
            for f in m["files"]:
                p = ROOT / m["source"]["path"] / f["path"]
                assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"], p


def test_manifest_has_no_secrets_or_home_paths():
    text = (ROOT / "models" / "manifest.json").read_text()
    assert "/Users/" not in text and "sk-ant" not in text and "tvly-" not in text


# ------------------------------------------------------------------------------------------------ fetch script
@pytest.fixture()
def fake_manifest(tmp_path):
    payload = b"hello model"
    man = {"version": 1, "models": [
        {"name": "Tiny", "dir": "tiny", "required": True, "license": "x", "source": {"type": "local", "note": "n"},
         "files": [{"path": "w.bin", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                    "also": [{"size": 3, "sha256": hashlib.sha256(b"old").hexdigest()}]}]}]}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(man))
    return p, payload


def test_fetch_status_link_import_verify(tmp_path, fake_manifest, capsys):
    fm = _fetch_module()
    man, payload = fake_manifest
    dest = tmp_path / "models"
    args = ["--manifest", str(man), "--dest", str(dest)]
    assert fm.main(args + ["status"]) == 2                       # missing
    assert fm.main(args + ["fetch"]) == 2                        # local-only: needs import, nothing downloaded
    assert "needs_import" in capsys.readouterr().out
    src = tmp_path / "copy"
    src.mkdir()
    (src / "w.bin").write_bytes(b"wrong size!!")
    assert fm.main(args + ["link", f"tiny={src}"]) == 2          # size mismatch: refused
    (src / "w.bin").write_bytes(b"wrong bytes")                  # same size: link checks sizes, --verify hashes
    assert fm.main(args + ["link", "--verify", f"tiny={src}"]) == 2
    (src / "w.bin").write_bytes(payload)
    assert fm.main(args + ["link", f"tiny={src}"]) == 0
    assert (dest / "tiny").is_symlink() and fm.main(args + ["status"]) == 0
    assert fm.main(args + ["verify"]) == 0
    (dest / "tiny").unlink()
    assert fm.main(args + ["import", f"tiny={src}"]) == 0        # a real copy, sha256-checked
    assert not (dest / "tiny").is_symlink() and (dest / "tiny" / "w.bin").read_bytes() == payload
    (dest / "tiny" / "w.bin").write_bytes(b"old")                # an accepted older variant ("also")
    assert fm.main(args + ["verify"]) == 0
    (dest / "tiny" / "w.bin").write_bytes(b"bad")                # same size, wrong hash
    assert fm.main(args + ["verify"]) == 2


def test_fetch_json_progress_is_one_object_per_line(tmp_path, fake_manifest, capsys):
    fm = _fetch_module()
    man, _ = fake_manifest
    fm.main(["--manifest", str(man), "--dest", str(tmp_path / "m"), "--json-progress", "fetch"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert lines and all(json.loads(ln)["event"] for ln in lines)
    fm.JSON_OUT = False


def test_hf_urls_are_pinned_to_the_revision():
    fm = _fetch_module()
    for m in MANIFEST["models"]:
        if m["source"]["type"] == "hf":
            u = fm.url_for(m, m["files"][0])
            assert u.startswith(f"https://huggingface.co/{m['source']['repo']}/resolve/{m['source']['revision']}/")
