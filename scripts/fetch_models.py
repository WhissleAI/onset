#!/usr/bin/env python3
"""Fetch, link, verify the models in models/manifest.json into MODELS_DIR (default ~/Library/Application Support/<APP_NAME>/models).
Standard library only: runs with any Python 3.9+ (the repo venv, the app's bundled Python, /usr/bin/python3).

    scripts/fetch_models.py status [--json]            what is there (size check), what is missing
    scripts/fetch_models.py fetch [NAME ...] [--json-progress]
                                                        download every missing / bad file of downloadable models, sha256-checked;
                                                        local-only models are reported as "needs import"
    scripts/fetch_models.py verify [NAME ...]          sha256 every file (slow for the big ones)
    scripts/fetch_models.py link NAME=PATH [...]       dev: symlink MODELS_DIR/<dir> to an existing copy (no disk used);
                                                        PATH must hold the manifest's files (sizes checked, --verify hashes)
    scripts/fetch_models.py link --dev                 dev: link whatever this machine already has (the HF cache for HF
                                                        models, this repo's legacy folders models/arctag, index/data, eval/audio)
    scripts/fetch_models.py import NAME=PATH [...]     copy an existing copy in (a model with no public source), sha256-checked
    scripts/fetch_models.py build-tagger [--out FILE]  compile vendor/arctag/src -> libarctag.dylib (cc + Accelerate)
    scripts/fetch_models.py build-arc [--out FILE]     compile vendor/arc/src -> libarc.dylib (Arc Mini in C: the ear)
    scripts/fetch_models.py hash DIR                   print a manifest "files" list for DIR (maintainers)
    scripts/fetch_models.py bundle --out DIR [--src SRC] [--profile FILE]
                                                        release builds: copy the models the release profile allows
                                                        (models/release_profile.json) from SRC (default MODELS_DIR) into
                                                        DIR (the app's Contents/Resources/models), sha256-checked, and
                                                        write DIR/manifest.json listing exactly what was bundled
                                                        ("bundled": true). `status --manifest DIR/manifest.json --dest DIR`
                                                        then checks the read-only bundle like any models folder.

--dest DIR overrides MODELS_DIR. --json-progress prints one JSON object per line (the macOS app reads them).
Exit codes: 0 ok, 1 error, 2 something still missing (needs import or a failed download).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "manifest.json"
sys.path.insert(0, str(ROOT))
from server import modelpaths  # noqa: E402  (stdlib-only module)

CHUNK = 1 << 20
UA = "model-fetch/1.0 (python urllib)"
JSON_OUT = False


def emit(event: str, **kw):
    if JSON_OUT:
        print(json.dumps({"event": event, **kw}), flush=True)
    else:
        msg = kw.get("message") or " ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[{event}] {msg}", flush=True)


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text())


def models(man: dict, names=None) -> list[dict]:
    out = [m for m in man["models"] if m.get("dir")]
    if names:
        unknown = set(names) - {m["dir"] for m in out} - {m["name"] for m in out}
        if unknown:
            raise SystemExit(f"unknown model(s): {', '.join(sorted(unknown))}")
        out = [m for m in out if m["dir"] in names or m["name"] in names]
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def accepted(f: dict) -> list[dict]:
    """The pinned file, plus any "also" variants accepted for a local copy (e.g. an older tested release)."""
    return [{"size": f["size"], "sha256": f["sha256"]}, *f.get("also", [])]


def file_ok(p: Path, f: dict, deep: bool) -> bool:
    if not p.is_file():
        return False
    ok = [v for v in accepted(f) if v["size"] == p.stat().st_size]
    if not ok:
        return False
    return (not deep) or sha256(p) in {v["sha256"] for v in ok}


def state(m: dict, dest: Path, deep: bool = False) -> dict:
    d = dest / m["dir"]
    bad = [f["path"] for f in m["files"] if not file_ok(d / f["path"], f, deep)]
    total = sum(f["size"] for f in m["files"])
    return {"model": m["dir"], "name": m["name"], "required": m.get("required", True), "path": str(d),
            "linked": d.is_symlink(), "ok": not bad, "missing": bad, "bytes": total,
            "source": m["source"]["type"], "downloadable": m["source"]["type"] in DOWNLOADABLE}


def cmd_status(man, dest, a) -> int:
    rows = [state(m, dest) for m in models(man, a.names)]
    if a.json:
        print(json.dumps({"models_dir": str(dest), "models": rows}))
    else:
        print(f"models dir: {dest}")
        for r in rows:
            st = "OK" if r["ok"] else ("MISSING" if r["required"] else "missing (optional)")
            how = "" if r["ok"] else (" -> fetch" if r["downloadable"] else " -> needs import (no public source)")
            print(f"  [{st}] {r['model']:<18} {r['bytes'] / 1e6:8.1f} MB  {'(link) ' if r['linked'] else ''}"
                  f"{len(r['missing'])} file(s) missing{how}" if not r["ok"] else
                  f"  [{st}] {r['model']:<18} {r['bytes'] / 1e6:8.1f} MB  {'(link)' if r['linked'] else ''}")
    return 0 if all(r["ok"] or not r["required"] for r in rows) else 2


DOWNLOADABLE = ("hf", "url", "archive")


def url_for(m: dict, f: dict) -> str:
    s = m["source"]
    remote = f.get("remote", f["path"])
    if s["type"] == "hf":
        return f"https://huggingface.co/{s['repo']}/resolve/{s['revision']}/{remote}"
    return s["base_url"].rstrip("/") + "/" + remote


def download(url: str, out: Path, f: dict, progress) -> None:
    """Resumable download to out.part, sha256-checked, then renamed into place."""
    part = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > f["size"]:
        part.unlink()
        have = 0
    hdr = {"User-Agent": UA, **({"Range": f"bytes={have}-"} if have else {})}
    tok = os.environ.get("HF_TOKEN")
    if tok and url.startswith("https://huggingface.co/"):          # gated repos (WhissleAI/arc-v1.1)
        hdr["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:          # server ignored the range: start over
            have = 0
        if have:
            progress(have)                     # resumed: the bytes already on disk count toward the total
        with open(part, "ab" if have else "wb") as w:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                w.write(b)
                have += len(b)
                progress(len(b))
    got = sha256(part)
    if got != f["sha256"]:
        part.unlink()
        raise IOError(f"checksum mismatch for {out.name}: got {got[:12]}, want {f['sha256'][:12]}")
    os.replace(part, out)


def unpack(m: dict, archive: Path, d: Path) -> None:
    """Unpack a model archive into d: only the manifest's paths are written (anything else in the archive is ignored, so
    no member can land outside d), each through a .part file, and each sha256-checked before it is moved into place."""
    want = {f["path"]: f for f in m["files"]}
    seen = set()
    with tarfile.open(archive, "r:") as t:
        for ti in t:
            f = want.get(ti.name)
            if f is None or not ti.isfile():
                continue
            out = d / f["path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            h = hashlib.sha256()
            src = t.extractfile(ti)
            with open(part, "wb") as w:
                for b in iter(lambda: src.read(CHUNK), b""):
                    h.update(b)
                    w.write(b)
            if h.hexdigest() != f["sha256"] or part.stat().st_size != f["size"]:
                part.unlink()
                raise IOError(f"{m['dir']}/{f['path']}: checksum mismatch after unpacking")
            os.replace(part, out)
            seen.add(f["path"])
    missing = sorted(set(want) - seen)
    if missing:
        raise IOError(f"{m['dir']}: archive lacks {', '.join(missing[:4])}")


def cmd_fetch(man, dest, a) -> int:
    todo, rc = [], 0
    for m in models(man, a.names):
        st = state(m, dest)
        if st["ok"]:
            emit("ok", model=m["dir"], message=f"{m['dir']}: present")
            continue
        if m["source"]["type"] not in DOWNLOADABLE:
            emit("needs_import", model=m["dir"], name=m["name"], required=m.get("required", True),
                 message=f"{m['dir']}: no public source; import a copy: fetch_models.py import {m['dir']}=PATH")
            rc = 2 if m.get("required", True) else rc
            continue
        d = dest / m["dir"]
        if d.is_symlink():
            emit("error", model=m["dir"], message=f"{d} is a link to an incomplete copy; remove the link to download")
            rc = 2
            continue
        if m["source"]["type"] == "archive":     # one archive per model: fetched whole, then unpacked
            todo.append((m, {"path": Path(m["source"]["url"].split("?")[0]).name or m["dir"] + ".tar",
                             "size": m["source"]["size"], "sha256": m["source"]["sha256"], "archive": True}))
        else:
            todo += [(m, f) for f in m["files"] if not file_ok(d / f["path"], f, deep=False)]
    total = sum(f["size"] for _, f in todo)
    done = 0
    last = [0.0]

    def progress(n, m, f):
        nonlocal done
        done += n
        now = time.monotonic()
        if now - last[0] > 0.25 or done == total:
            last[0] = now
            emit("progress", model=m["dir"], file=f["path"], done=done, total=total,
                 message=f"{done / 1e6:.0f}/{total / 1e6:.0f} MB  {m['dir']}/{f['path']}")

    for m, f in todo:
        out = dest / m["dir"] / f["path"]
        if f.get("archive"):
            out = dest / ".downloads" / f"{m['dir']}.tar"
        for attempt in range(3):
            try:
                if f.get("archive"):
                    if not out.is_file() or out.stat().st_size != f["size"]:
                        download(m["source"]["url"], out, f, lambda n, m=m, f=f: progress(n, m, f))
                    else:
                        progress(f["size"], m, f)
                    emit("unpack", model=m["dir"], message=f"{m['dir']}: unpacking and checking {len(m['files'])} files")
                    unpack(m, out, dest / m["dir"])
                    out.unlink()
                else:
                    download(url_for(m, f), out, f, lambda n, m=m, f=f: progress(n, m, f))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    emit("error", model=m["dir"], file=f["path"], message=f"{m['dir']}/{f['path']}: {e}")
                    rc = 2
                else:
                    time.sleep(2 * (attempt + 1))
    emit("done", ok=rc == 0, message="all models present" if rc == 0 else "some models still missing")
    return rc


def cmd_verify(man, dest, a) -> int:
    rc = 0
    for m in models(man, a.names):
        st = state(m, dest, deep=True)
        emit("verified" if st["ok"] else "bad", model=m["dir"],
             message=f"{m['dir']}: " + ("sha256 OK" if st["ok"] else f"bad/missing: {', '.join(st['missing'][:5])}"))
        if not st["ok"] and (m.get("required", True) or (dest / m["dir"]).exists()):
            rc = 2
    return rc


def _check_copy(m: dict, src: Path, deep: bool) -> list[str]:
    return [f["path"] for f in m["files"] if not file_ok(src / f["path"], f, deep)]


def hf_cache_snapshot(m: dict) -> Path | None:
    s = m["source"]
    if s["type"] != "hf":
        return None
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub")
    p = hub / ("models--" + s["repo"].replace("/", "--")) / "snapshots" / s["revision"]
    return p if p.exists() else None


def cmd_link(man, dest, a) -> int:
    by_dir = {m["dir"]: m for m in models(man)}
    pairs = []
    for spec in a.specs:
        name, _, path = spec.partition("=")
        if name not in by_dir or not path:
            raise SystemExit(f"bad spec {spec!r}: want NAME=PATH, NAME one of {', '.join(by_dir)}")
        pairs.append((by_dir[name], Path(path).expanduser().resolve()))
    if a.dev:
        for m in by_dir.values():
            if any(m is p[0] for p in pairs):
                continue
            cand = [hf_cache_snapshot(m)] + [ROOT / x for x in m.get("dev_paths", [])]
            for c in cand:
                if c is not None and c.exists() and not _check_copy(m, c, deep=False):
                    pairs.append((m, c.resolve()))
                    break
    rc = 0
    dest.mkdir(parents=True, exist_ok=True)
    for m, src in pairs:
        d = dest / m["dir"]
        bad = _check_copy(m, src, deep=a.verify)
        if bad:
            emit("error", model=m["dir"], message=f"{m['dir']}: {src} lacks / differs in {', '.join(bad[:4])}")
            rc = 2
            continue
        if a.copy:
            if d.is_symlink():
                d.unlink()
            for f in m["files"]:
                out = d / f["path"]
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / f["path"], out)
            emit("imported", model=m["dir"], message=f"{m['dir']}: copied from {src}")
            continue
        if d.is_symlink() or not d.exists():
            if d.is_symlink():
                d.unlink()
            d.symlink_to(src, target_is_directory=True)
            emit("linked", model=m["dir"], message=f"{m['dir']} -> {src}")
        elif d.resolve() != src:
            emit("error", model=m["dir"], message=f"{d} is a real folder; move it away to link")
            rc = 2
    return rc


def cmd_build_tagger(a) -> int:
    v = ROOT / "vendor" / "arctag"
    out = Path(a.out) if a.out else v / "libarctag.dylib"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["cc", "-O3", "-shared", "-fPIC", f"-I{v / 'include'}", f"-I{v / 'src'}", f"-I{v / 'third_party/cjson'}",
           *sorted(str(p) for p in (v / "src").glob("*.c")), str(v / "third_party/cjson/cJSON.c"),
           "-framework", "Accelerate", "-install_name", "@rpath/libarctag.dylib", "-o", str(out)]
    if a.arch:
        cmd[1:1] = ["-arch", a.arch]
    r = subprocess.run(cmd)
    if r.returncode == 0:
        emit("built", message=f"{out}")
    return r.returncode


def cmd_build_arc(a) -> int:
    """libarc: C11, libc + libm + threads (NEON kernels on arm64), no frameworks."""
    v = ROOT / "vendor" / "arc"
    out = Path(a.out) if a.out else v / "libarc.dylib"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["cc", "-O2", "-std=c11", "-shared", "-fPIC", "-pthread", f"-I{v / 'include'}", f"-I{v / 'src'}",
           *sorted(str(p) for p in (v / "src").glob("*.c")), "-lm",
           "-install_name", "@rpath/libarc.dylib", "-o", str(out)]
    if a.arch:
        cmd[1:1] = ["-arch", a.arch]
    r = subprocess.run(cmd)
    if r.returncode == 0:
        emit("built", message=f"{out}")
    return r.returncode


def cmd_hash(a) -> int:
    d = Path(a.dir)
    skip = {".cache", "__pycache__", ".git"}
    files = []
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if p.is_file() and not (set(rel.parts) & skip) and p.name != ".DS_Store":
            files.append({"path": str(rel), "size": p.stat().st_size, "sha256": sha256(p)})
    print(json.dumps(files, indent=1))
    return 0


PROFILE = ROOT / "models" / "release_profile.json"


def _voiced(rows_files: list[str], prefixes: list[str]) -> set[str]:
    """Audio files whose question rows were spoken by an allowed voice (e.g. "kokoro:": Apache-2.0 synthetic speech)."""
    out = set()
    for rel in rows_files:
        p = ROOT / rel
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("voice", "")).startswith(tuple(prefixes)):
                out.update([r["file"]] if r.get("file") else [])
                out.update(r.get("files") or [])
    return out


def bundle_plan(man: dict, profile: dict) -> list[tuple[dict, list[dict], str]]:
    """(model, files to bundle, note) per bundled model. A model the profile does not name is bundled whole when
    required and left out when optional, so a new manifest (e.g. the ONNX one) needs no profile change."""
    plan = []
    rules = profile.get("models", {})
    for m in models(man):
        rule = rules.get(m["dir"])
        if rule is None:
            rule = {"files": "all"} if m.get("required", True) else {"skip": "optional, not in the release profile"}
        if rule.get("skip"):
            continue
        files = list(m["files"])
        if rule.get("include"):
            files = [f for f in files if any(f["path"] == i or f["path"].startswith(i) for i in rule["include"])]
        if rule.get("exclude"):
            files = [f for f in files if not any(f["path"] == x or f["path"].startswith(x) for x in rule["exclude"])]
        if rule.get("voices"):
            keep = _voiced(rule.get("questions", []), rule["voices"])
            files = [f for f in files if f["path"] in keep]
        if files:
            plan.append((m, files, rule.get("note", "")))
    return plan


def _clone(src: Path, out: Path) -> None:
    """APFS clone (no extra disk) when possible, else a plain copy."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", str(src), str(out)], capture_output=True).returncode == 0:
        return
    shutil.copy2(src, out)


def cmd_bundle(man, a) -> int:
    profile = json.loads(Path(a.profile).read_text())
    src = Path(a.src).expanduser() if a.src else modelpaths.models_dir()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bundled, rc, total = [], 0, 0
    for m, files, note in bundle_plan(man, profile):
        bad = [f["path"] for f in files if not file_ok(src / m["dir"] / f["path"], f, deep=not a.no_verify)]
        if bad:
            emit("error", model=m["dir"], message=f"{m['dir']}: {src / m['dir']} lacks / differs in {', '.join(bad[:4])}")
            rc = 2
            continue
        for f in files:
            _clone((src / m["dir"] / f["path"]).resolve(), out / m["dir"] / f["path"])
        size = sum(f["size"] for f in files)
        total += size
        bundled.append({**{k: v for k, v in m.items() if k not in ("files", "source", "dev_paths")},
                        "source": {"type": "bundled", "from": m["source"]["type"]},
                        **({"bundle_note": note} if note else {}), "files": files})
        emit("bundled", model=m["dir"], message=f"{m['dir']}: {len(files)}/{len(m['files'])} files, {size / 1e6:.0f} MB")
    (out / "manifest.json").write_text(json.dumps(
        {"version": man.get("version", 1), "bundled": True,
         "about": "Models bundled read-only inside the app (scripts/fetch_models.py bundle). Only these files ship.",
         "models": bundled}, indent=1) + "\n")
    emit("done", ok=rc == 0, message=f"bundled {total / 1e6:.0f} MB into {out}")
    return rc


def main(argv=None) -> int:
    global JSON_OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", help="models dir (default: MODELS_DIR or ~/Library/Application Support/<APP_NAME>/models)")
    ap.add_argument("--manifest", default=os.environ.get("MODELS_MANIFEST") or str(MANIFEST))
    ap.add_argument("--json-progress", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status"); p.add_argument("names", nargs="*"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("fetch"); p.add_argument("names", nargs="*")
    p = sub.add_parser("verify"); p.add_argument("names", nargs="*")
    p = sub.add_parser("link"); p.add_argument("specs", nargs="*"); p.add_argument("--dev", action="store_true")
    p.add_argument("--verify", action="store_true")
    p = sub.add_parser("import"); p.add_argument("specs", nargs="+")
    p = sub.add_parser("build-tagger"); p.add_argument("--out"); p.add_argument("--arch")
    p = sub.add_parser("build-arc"); p.add_argument("--out"); p.add_argument("--arch")
    p = sub.add_parser("hash"); p.add_argument("dir")
    p = sub.add_parser("bundle"); p.add_argument("--out", required=True); p.add_argument("--src")
    p.add_argument("--profile", default=str(PROFILE)); p.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    JSON_OUT = a.json_progress
    if a.cmd == "build-tagger":
        return cmd_build_tagger(a)
    if a.cmd == "build-arc":
        return cmd_build_arc(a)
    if a.cmd == "hash":
        return cmd_hash(a)
    man = load_manifest(Path(a.manifest))
    if a.cmd == "bundle":
        return cmd_bundle(man, a)
    dest = Path(a.dest).expanduser() if a.dest else modelpaths.models_dir()
    if a.cmd == "import":
        a.dev, a.verify, a.copy = False, True, True
        return cmd_link(man, dest, a)
    if a.cmd == "link":
        a.copy = False
    return {"status": cmd_status, "fetch": cmd_fetch, "verify": cmd_verify, "link": cmd_link}[a.cmd](man, dest, a)


if __name__ == "__main__":
    sys.exit(main())
