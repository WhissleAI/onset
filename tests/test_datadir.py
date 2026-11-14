"""Runtime writes go to the app's data dir (HEADSTART_DATA_DIR), never into the (read-only, signed) app bundle."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_web_cache_seed_is_read_and_writes_go_elsewhere(tmp_path):
    from server import lanes
    seed = tmp_path / "bundle" / "web_cache.json"
    seed.parent.mkdir()
    seed.write_text(json.dumps({"version": 1, "entries": {"capital france": {"query": "capital france", "docs": []}}}))
    seed.chmod(0o444)
    out = tmp_path / "data" / "web_cache.json"
    c = lanes.WebCache(seed, out)
    assert len(c) == 1                                   # the seed is read
    c.put("who founded nvidia", [], "wikipedia")
    c.save()
    assert out.exists() and json.loads(seed.read_text())["entries"].keys() == {"capital france"}
    c2 = lanes.WebCache(seed, out)                       # the data-dir copy wins once it exists
    assert len(c2) == 2


def test_every_writable_path_is_in_the_data_dir(tmp_path):
    """With HEADSTART_DATA_DIR set (the macOS app), every path the server writes resolves under it."""
    code = ("import json; from server import config as c; print(json.dumps({k: str(getattr(c, k)) for k in "
            "['CACHE_PATH', 'MEM_DIR', 'KEYSTONE_DIR', 'KEYSTONE_SHARED', 'WEB_CACHE_WRITE_PATH']}))")
    env = {k: v for k, v in os.environ.items() if not k.endswith("_PATH") and k not in ("MEM_DIR", "KEYSTONE_DIR",
                                                                                       "KEYSTONE_SHARED")}
    env["HEADSTART_DATA_DIR"] = str(tmp_path)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    paths = json.loads(r.stdout.strip().splitlines()[-1])
    for k, v in paths.items():
        assert Path(v).is_relative_to(tmp_path), (k, v)
