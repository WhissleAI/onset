"""The public Mac release (MACAPP.md "Release"): the bundled-models profile (scripts/fetch_models.py bundle), the
Replay list filtered to what is installed, and the release guards in scripts/build_macapp.sh. No network, no models."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILE = json.loads((ROOT / "models" / "release_profile.json").read_text())


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _file(root: Path, rel: str, data: bytes) -> dict:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return {"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_profile_names_only_manifest_models_and_says_why():
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    dirs = {m["dir"] for m in man["models"] if m.get("dir")}
    assert set(PROFILE["models"]) <= dirs
    for rule in PROFILE["models"].values():
        assert rule.get("note"), "every release rule says why"


def test_profile_never_ships_the_local_moments_recordings_or_system_voices():
    fm = _fetch_module()
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    plan = {m["dir"]: files for m, files, _ in fm.bundle_plan(man, PROFILE)}
    idx = [f["path"] for f in plan["index-data"]]
    assert idx and all(p.startswith("knowledge/") for p in idx)
    say = set()
    for rel in PROFILE["models"]["demo-audio"]["questions"]:
        for line in (ROOT / rel).read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if str(r.get("voice", "")).startswith("say:"):
                say.add(r["file"])
    audio = {f["path"] for f in plan.get("demo-audio", [])}
    assert audio and not (audio & say)
    assert not any(a.startswith("local_") for a in audio)


def test_bundle_copies_the_subset_and_writes_a_checkable_manifest(tmp_path):
    fm = _fetch_module()
    src, out = tmp_path / "src", tmp_path / "out"
    keep = _file(src / "big", "w.bin", b"weights")
    drop = _file(src / "big", ".gitattributes", b"x")
    k = _file(src / "idx", "knowledge/leads.jsonl", b"{}\n")
    priv = _file(src / "idx", "audio/rec.wav", b"RIFF")
    opt = _file(src / "opt", "a.bin", b"o")
    man = {"version": 1, "models": [
        {"name": "Big", "dir": "big", "required": True, "source": {"type": "hf"}, "files": [keep, drop]},
        {"name": "Index", "dir": "idx", "required": True, "source": {"type": "local"}, "files": [k, priv]},
        {"name": "Optional", "dir": "opt", "required": False, "source": {"type": "local"}, "files": [opt]},
        {"name": "New", "dir": "new", "required": True, "source": {"type": "hf"},
         "files": [_file(src / "new", "m.onnx", b"onnx")]},
    ]}
    (tmp_path / "man.json").write_text(json.dumps(man))
    prof = {"models": {"big": {"exclude": [".gitattributes"], "note": "n"},
                       "idx": {"include": ["knowledge/"], "note": "n"}}}
    (tmp_path / "prof.json").write_text(json.dumps(prof))
    rc = fm.main(["--manifest", str(tmp_path / "man.json"), "bundle", "--out", str(out), "--src", str(src),
                  "--profile", str(tmp_path / "prof.json")])
    assert rc == 0
    assert (out / "big" / "w.bin").read_bytes() == b"weights"
    assert not (out / "big" / ".gitattributes").exists()
    assert (out / "idx" / "knowledge" / "leads.jsonl").exists() and not (out / "idx" / "audio").exists()
    assert not (out / "opt").exists()                   # optional + not in the profile: left out
    assert (out / "new" / "m.onnx").exists()            # required + not in the profile (a new manifest): bundled whole
    bm = json.loads((out / "manifest.json").read_text())
    assert bm["bundled"] is True
    assert all(m["source"]["type"] == "bundled" for m in bm["models"])
    # the app's model check against the read-only bundle: everything present
    assert fm.main(["--manifest", str(out / "manifest.json"), "--dest", str(out), "status"]) == 0


def test_bundle_refuses_a_bad_copy(tmp_path):
    fm = _fetch_module()
    src = tmp_path / "src"
    f = _file(src / "m", "w.bin", b"good")
    (src / "m" / "w.bin").write_bytes(b"evil")       # same size, different bytes
    man = {"models": [{"name": "M", "dir": "m", "required": True, "source": {"type": "hf"}, "files": [f]}]}
    (tmp_path / "man.json").write_text(json.dumps(man))
    rc = fm.main(["--manifest", str(tmp_path / "man.json"), "bundle", "--out", str(tmp_path / "out"),
                  "--src", str(src), "--profile", str(ROOT / "models" / "release_profile.json")])
    assert rc == 2


def test_replay_list_only_offers_what_is_installed(tmp_path):
    from server.app import replayable
    audio, idx = tmp_path / "audio", tmp_path / "idx"
    audio.mkdir()
    idx.mkdir()
    (audio / "q01.wav").write_bytes(b"x")
    (audio / "local_01.wav").write_bytes(b"x")
    assert replayable({"file": "q01.wav"}, audio, idx)
    assert not replayable({"file": "q02.wav"}, audio, idx)                       # audio not bundled
    assert not replayable({"file": "local_01.wav", "kind": "local"}, audio, idx)  # no local-moments index
    (idx / "docs.jsonl").write_text("{}\n")
    assert replayable({"file": "local_01.wav", "kind": "local"}, audio, idx)


def test_release_build_refuses_dev_env_and_unsigned():
    sh = ROOT / "scripts" / "build_macapp.sh"
    r = subprocess.run(["bash", str(sh), "--release", "--dev-env"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "ONSET_SIGN_ID": "Developer ID Application: X (Y)", "NOTARY_PROFILE": "p"})
    assert r.returncode == 2 and ".env" in r.stdout
    r = subprocess.run(["bash", str(sh), "--release"], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 2 and "Developer ID" in r.stdout


def test_swift_keys_and_backend_rules():
    src = "\n".join(p.read_text() for p in (ROOT / "macapp" / "Sources").glob("*.swift"))
    for k in ("DEEPGRAM_API_KEY", "TAVILY_API_KEY", "TELEM_API_KEY", "ANTHROPIC_API_KEY"):
        assert f'"{k}"' in src
    assert re.search(r'tavily \? "tavily" : \(telem \? "telem" : "duckduckgo"\)', src)
    assert "type" in src.lower() and "speak" in src.lower()       # onboarding: you can type as well as speak
    # no key material in the app sources
    assert not re.search(r"sk-ant-[A-Za-z0-9]|tvly-[A-Za-z0-9]{8}", src)
