"""First-run model downloads from public archives (fetch_models.py source type "archive", publish_models.py pack)."""
import hashlib
import http.server
import io
import json
import sys
import tarfile
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_models as F  # noqa: E402


def _tar(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def server():
    blobs = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            b = blobs.get(self.path.split("?")[0])
            if b is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def log_message(self, *a):
            pass

    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield blobs, f"http://127.0.0.1:{s.server_address[1]}"
    s.shutdown()


def _manifest(tmp, url, archive, files, tamper_sha=False):
    m = {"name": "Toy", "dir": "toy", "required": True,
         "source": {"type": "archive", "url": url, "size": len(archive),
                    "sha256": hashlib.sha256(archive).hexdigest()},
         "files": [{"path": p, "size": len(d), "sha256": ("0" * 64 if tamper_sha else hashlib.sha256(d).hexdigest())}
                   for p, d in files.items()]}
    path = tmp / "manifest.json"
    path.write_text(json.dumps({"version": 1, "models": [m]}))
    return path


FILES = {"weights.bin": b"w" * 5000, "sub/tokenizer.json": b'{"a": 1}'}


def test_archive_fetch_unpacks_and_verifies(tmp_path, server):
    blobs, base = server
    arc = _tar(FILES)
    blobs["/toy.tar"] = arc
    man = _manifest(tmp_path, base + "/toy.tar?dl=1", arc, FILES)
    dest = tmp_path / "models"
    assert F.main(["--manifest", str(man), "--dest", str(dest), "fetch"]) == 0
    assert (dest / "toy" / "weights.bin").read_bytes() == FILES["weights.bin"]
    assert (dest / "toy" / "sub" / "tokenizer.json").read_bytes() == FILES["sub/tokenizer.json"]
    assert not (dest / ".downloads" / "toy.tar").exists()          # the archive is removed once unpacked
    assert F.main(["--manifest", str(man), "--dest", str(dest), "status"]) == 0
    assert F.main(["--manifest", str(man), "--dest", str(dest), "verify"]) == 0


def test_archive_with_wrong_bytes_is_rejected(tmp_path, server):
    blobs, base = server
    arc = _tar(FILES)
    blobs["/toy.tar"] = arc[:-512] + b"x" * 512                     # served bytes differ from the pinned sha256
    man = _manifest(tmp_path, base + "/toy.tar", arc, FILES)
    dest = tmp_path / "models"
    assert F.main(["--manifest", str(man), "--dest", str(dest), "fetch"]) == 2
    assert not (dest / "toy" / "weights.bin").exists()


def test_file_that_fails_its_hash_after_unpacking_is_not_installed(tmp_path, server):
    blobs, base = server
    arc = _tar(FILES)
    blobs["/toy.tar"] = arc
    man = _manifest(tmp_path, base + "/toy.tar", arc, FILES, tamper_sha=True)
    dest = tmp_path / "models"
    assert F.main(["--manifest", str(man), "--dest", str(dest), "fetch"]) == 2
    assert not (dest / "toy" / "weights.bin").exists()


def test_archive_models_are_downloadable_in_status(tmp_path, server, capsys):
    blobs, base = server
    arc = _tar(FILES)
    man = _manifest(tmp_path, base + "/toy.tar", arc, FILES)
    F.main(["--manifest", str(man), "--dest", str(tmp_path / "m"), "status", "--json"])
    row = json.loads(capsys.readouterr().out)["models"][0]
    assert row["downloadable"] is True and row["source"] == "archive" and not row["ok"]


def test_release_manifest_only_lists_what_the_release_profile_allows():
    rel = ROOT / "models" / "manifest.release.json"
    if not rel.exists():
        pytest.skip("no release manifest")
    man = json.loads(rel.read_text())
    for m in man["models"]:
        assert m["source"]["type"] == "archive" and m["source"]["url"].startswith("https://")
        assert m["source"]["url"].endswith("dl=1")
    idx = next(m for m in man["models"] if m["dir"] == "index-data")
    assert all(f["path"].startswith("knowledge/") for f in idx["files"])     # never the private local-moments index
    arc = next(m for m in man["models"] if m["dir"] == "arc-v1")
    assert not any("decoder_v1e" in f["path"] for f in arc["files"])
