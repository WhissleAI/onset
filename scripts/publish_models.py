#!/usr/bin/env python3
"""Publish the release models to Dropbox and write models/manifest.release.json (maintainers only).

    DROPBOX_TOKEN=... scripts/publish_models.py [--src DIR] [--tag v1] [--keep DIR] [--dry-run]

For every model the release profile allows (models/release_profile.json, the same plan `fetch_models.py bundle` uses):
  1. pack exactly those files into one uncompressed tar (paths relative to the model's folder; .ogg / safetensors / onnx
     do not compress), recording each file's real size and sha256;
  2. upload it to Dropbox:/onset-models/<tag>/<dir>.tar (upload sessions, 64 MB chunks), and make a public, view-only
     shared link (dl=1: a direct download);
  3. write models/manifest.release.json: the release manifest the app ships. Every model in it has
     source {"type": "archive", "url", "size", "sha256"}, so `fetch_models.py fetch` downloads it on first run
     (resumable, the archive sha256-checked, then every file sha256-checked after unpacking).

The token needs files.content.write and sharing.write. It is read from the environment only and never written anywhere.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import fetch_models as F  # noqa: E402
from server import modelpaths  # noqa: E402

OUT_MANIFEST = ROOT / "models" / "manifest.release.json"
CHUNK = 64 << 20


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Dropbox:
    def __init__(self, token: str):
        self.h = {"Authorization": f"Bearer {token}"}

    def _call(self, url: str, body: bytes, headers: dict, tries: int = 5) -> dict:
        for i in range(tries):
            req = urllib.request.Request(url, data=body, headers={**self.h, **headers}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    t = r.read()
                    return json.loads(t) if t else {}
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")
                if e.code == 409:
                    return {"_conflict": json.loads(msg) if msg.startswith("{") else msg}
                if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                    time.sleep(int(e.headers.get("Retry-After") or 2 ** (i + 1)))
                    continue
                raise SystemExit(f"Dropbox {url.rsplit('/', 2)[-2:]} -> HTTP {e.code}: {msg[:300]}")
            except urllib.error.URLError:
                if i < tries - 1:
                    time.sleep(2 ** (i + 1))
                    continue
                raise
        raise SystemExit("unreachable")

    def rpc(self, endpoint: str, args: dict) -> dict:
        return self._call(f"https://api.dropboxapi.com/2/{endpoint}", json.dumps(args).encode(),
                          {"Content-Type": "application/json"})

    def content(self, endpoint: str, args: dict, data: bytes) -> dict:
        return self._call(f"https://content.dropboxapi.com/2/{endpoint}", data,
                          {"Content-Type": "application/octet-stream", "Dropbox-API-Arg": json.dumps(args)})

    def upload(self, src: Path, path: str) -> None:
        size, sent = src.stat().st_size, 0
        with open(src, "rb") as f:
            sid = self.content("files/upload_session/start", {"close": False}, f.read(CHUNK))["session_id"]
            sent = f.tell()
            while True:
                b = f.read(CHUNK)
                cursor = {"session_id": sid, "offset": sent}
                if len(b) < CHUNK and sent + len(b) == size:
                    self.content("files/upload_session/finish",
                                 {"cursor": cursor, "commit": {"path": path, "mode": "overwrite", "mute": True}}, b)
                    break
                self.content("files/upload_session/append_v2", {"cursor": cursor, "close": False}, b)
                sent += len(b)
                print(f"    {sent / 1e6:.0f}/{size / 1e6:.0f} MB", flush=True)

    def public_link(self, path: str) -> str:
        r = self.rpc("sharing/create_shared_link_with_settings",
                     {"path": path, "settings": {"requested_visibility": "public", "audience": "public", "access": "viewer"}})
        if "_conflict" in r:
            links = self.rpc("sharing/list_shared_links", {"path": path, "direct_only": True}).get("links", [])
            if not links:
                raise SystemExit(f"could not make a link for {path}: {r['_conflict']}")
            url = links[0]["url"]
        else:
            url = r["url"]
        return url.replace("&dl=0", "&dl=1").replace("?dl=0", "?dl=1")


def pack(m: dict, files: list[dict], src: Path, out: Path) -> list[dict]:
    """Tar the listed files (sorted, fixed metadata: the same files give the same archive). Returns the files with their
    actual size + sha256 (a local copy may be an accepted older variant: the archive records what it really holds)."""
    real = []
    with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as t:
        for f in sorted(files, key=lambda f: f["path"]):
            p = (src / m["dir"] / f["path"]).resolve()
            ti = t.gettarinfo(str(p), arcname=f["path"])
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mtime = 0
            ti.mode = 0o644
            with open(p, "rb") as fh:
                t.addfile(ti, fh)
            real.append({"path": f["path"], "size": p.stat().st_size, "sha256": sha256(p)})
    return real


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", help="models dir to pack from (default MODELS_DIR)")
    ap.add_argument("--tag", default="v1", help="Dropbox folder: /onset-models/<tag>/")
    ap.add_argument("--profile", default=str(F.PROFILE))
    ap.add_argument("--keep", help="keep the archives in this folder (default: a temp folder, deleted)")
    ap.add_argument("--dry-run", action="store_true", help="pack + hash only: no upload, no manifest")
    a = ap.parse_args(argv)
    token = os.environ.get("DROPBOX_TOKEN", "")
    if not token and not a.dry_run:
        raise SystemExit("set DROPBOX_TOKEN (scopes: files.content.write, sharing.write)")
    man = F.load_manifest()
    src = Path(a.src).expanduser() if a.src else modelpaths.models_dir()
    plan = F.bundle_plan(man, json.loads(Path(a.profile).read_text()))
    dbx = Dropbox(token) if not a.dry_run else None
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="onset-models-"))
    work.mkdir(parents=True, exist_ok=True)
    out_models = []
    for m, files, note in plan:
        bad = [f["path"] for f in files if not F.file_ok(src / m["dir"] / f["path"], f, deep=True)]
        if bad:
            raise SystemExit(f"{m['dir']}: {src / m['dir']} lacks / differs in {', '.join(bad[:4])}")
        tar = work / f"{m['dir']}.tar"
        print(f"== {m['dir']}: packing {len(files)} files", flush=True)
        real = pack(m, files, src, tar)
        size, digest = tar.stat().st_size, sha256(tar)
        print(f"   {size / 1e6:.1f} MB  sha256 {digest[:16]}", flush=True)
        url = ""
        if dbx:
            path = f"/onset-models/{a.tag}/{m['dir']}.tar"
            print(f"   uploading to Dropbox:{path}", flush=True)
            dbx.upload(tar, path)
            meta = dbx.rpc("files/get_metadata", {"path": path})
            if meta.get("size") != size:
                raise SystemExit(f"{path}: Dropbox has {meta.get('size')} bytes, expected {size}")
            url = dbx.public_link(path)
            print(f"   {url}", flush=True)
        if not a.keep:
            tar.unlink()
        out_models.append({**{k: v for k, v in m.items() if k not in ("files", "source", "dev_paths")},
                           "source": {"type": "archive", "format": "tar", "url": url, "size": size, "sha256": digest,
                                      "host": "Dropbox", "origin": m["source"].get("repo") or m["source"]["type"]},
                           **({"release_note": note} if note else {}), "files": real})
    if a.dry_run:
        print(json.dumps([{"dir": m["dir"], "size": m["source"]["size"], "files": len(m["files"])} for m in out_models]))
        return 0
    OUT_MANIFEST.write_text(json.dumps(
        {"version": man.get("version", 1), "release": a.tag,
         "about": "What the public app downloads on first run (scripts/publish_models.py writes this; "
                  "scripts/fetch_models.py fetch reads it). One public Dropbox archive per model, sha256-pinned; after "
                  "unpacking every file is sha256-checked too. Only what models/release_profile.json allows is here.",
         "models": out_models}, indent=1) + "\n")
    print(f"wrote {OUT_MANIFEST.relative_to(ROOT)}  ({sum(m['source']['size'] for m in out_models) / 1e6:.0f} MB in "
          f"{len(out_models)} archives)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
