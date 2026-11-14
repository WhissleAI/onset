"""Scrub recorded demo runs before they are published (the repo and the Pages site are public).

- drops ids that tie a run to a browser or a machine (client ids, run ids, request ids)
- rewrites absolute file paths to their base name
- refuses (raises) if any value from .env (API keys) or a home-dir path survives
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DROP_KEYS = {"client", "client_id", "rid", "api_key", "authorization", "request_id", "x-request-id"}
PATH_RE = re.compile(r"(?:/Users|/home|/private|/var/folders|/tmp|/Volumes|[A-Za-z]:\\\\Users)[^\s\"'<>]*")


def _secrets() -> list[str]:
    out = []
    for p in (ROOT / ".env", ROOT.parent / "headstart" / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    v = line.split("=", 1)[1].strip().strip("'\"")
                    if len(v) >= 12:
                        out.append(v)
    for k, v in os.environ.items():
        if any(s in k for s in ("KEY", "TOKEN", "SECRET")) and len(v) >= 12:
            out.append(v)
    return out


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items() if str(k).lower() not in DROP_KEYS}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, str):
        return PATH_RE.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], x)
    return x


def check(blob: str):
    for s in _secrets():
        if s in blob:
            raise ValueError("a secret from .env is in the recorded run: not writing it")
    if re.search(r"/Users/|/home/[a-z]", blob):
        raise ValueError("a home-dir path survived sanitizing")


def sanitize(rec: dict) -> dict:
    rec = _clean(rec)
    check(json.dumps(rec))
    return rec


if __name__ == "__main__":       # check every published run: python scripts/pages/sanitize.py
    import sys
    bad = 0
    for p in sorted((ROOT / "site" / "demo-data" / "runs").glob("*.json")):
        try:
            check(p.read_text())
        except ValueError as e:
            bad = 1
            print(p.name, e)
    sys.exit(bad)
