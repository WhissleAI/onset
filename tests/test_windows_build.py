"""Windows build kit: the shell's pure parts, the committed DLLs, and the POSIX traps that break Windows."""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _shell(tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_HOME", str(tmp_path))
    sys.path.insert(0, str(ROOT / "winapp"))
    try:
        import onset_win
        return importlib.reload(onset_win)
    finally:
        sys.path.remove(str(ROOT / "winapp"))


def test_shell_env_defaults_free_and_on_device(tmp_path, monkeypatch):
    for k in ("TAVILY_API_KEY", "TELEM_API_KEY", "DEEPGRAM_API_KEY", "WEB_BACKEND"):
        monkeypatch.delenv(k, raising=False)
    w = _shell(tmp_path, monkeypatch)
    monkeypatch.setattr(w, "get_key", lambda k: None)
    env = w.server_env({})
    assert env["WEB_BACKEND"] == "duckduckgo" and env["EAR"] == "arc" and env["ONNX_RUNTIME"] == "1"
    assert env["MODELS_DIR"] == str(tmp_path / "models")
    monkeypatch.setattr(w, "get_key", lambda k: "x" if k == "DEEPGRAM_API_KEY" else None)
    assert w.server_env({"ear": "auto"})["EAR"] == "flux"
    assert w.server_env({"trainCollect": False})["TRAIN_COLLECT"] == "0"


def test_shell_versions_and_settings(tmp_path, monkeypatch):
    w = _shell(tmp_path, monkeypatch)
    assert w._ver("0.2.10") > w._ver("0.2.9") > w._ver("0.2")
    w.save_settings({"ear": "auto", "port": 123})
    assert w.load_settings()["port"] == 123
    assert w.APP_NAME == "Onset" and w.VERSION.count(".") == 2


def test_open_found_file_never_runs_code(tmp_path, monkeypatch):
    w = _shell(tmp_path, monkeypatch)
    for ext in (".exe", ".bat", ".ps1", ".lnk", ".url", ".msi", ".js", ".vbs", ".hta"):
        assert ext in w.RISKY


def test_dlls_committed_and_build_kit_present():
    for f in ("vendor/arc/arc.dll", "vendor/arctag/arctag.dll", "winapp/build_windows.ps1", "winapp/onset.iss",
              "winapp/requirements-win.txt", "winapp/Onset.ico", "docs/BUILD_WINDOWS.md", "scripts/smoke_typed.py"):
        assert (ROOT / f).exists(), f
    assert (ROOT / "vendor/arc/arc.dll").read_bytes()[:2] == b"MZ"
    for req in ("winapp/requirements-win.txt", "macapp/requirements-native.txt"):
        pkgs = "".join(l.split("#")[0] for l in (ROOT / req).read_text().splitlines())
        assert "torch" not in pkgs, req


def test_library_names_per_platform(monkeypatch):
    from server import modelpaths
    monkeypatch.setattr(modelpaths.sys, "platform", "win32")
    assert modelpaths.arc_lib({}).name == "arc.dll" and modelpaths.tagger_lib({}).name == "arctag.dll"
    monkeypatch.setattr(modelpaths.sys, "platform", "darwin")
    assert modelpaths.arc_lib({}).name == "libarc.dylib"


def test_no_parent_check_that_kills_on_windows():
    src = (ROOT / "macapp" / "server_main.py").read_text()
    assert "_alive_windows" in src and 'sys.platform == "win32"' in src
    # POSIX-only modules are imported behind a guard
    for p in (ROOT / "server").glob("*.py"):
        for line in p.read_text().splitlines():
            if line.startswith(("import resource", "import fcntl", "import termios", "import pwd")):
                raise AssertionError(f"{p.name}: unguarded {line}")
