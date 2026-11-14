"""Results view data (web/scorecard.js): server/scorecard.json (tracked, numbers + where each was measured) plus what
this server runs now, and the provenance of a local eval A/B file (runs/ab_latest.json) so it is never shown as current
when it was run on other settings."""
from __future__ import annotations

import json
from pathlib import Path

from . import config

PATH = Path(__file__).with_name("scorecard.json")


def load() -> dict:
    return json.loads(PATH.read_text())


def payload() -> dict:
    from . import cost
    d = load()
    d["now"] = {"cost_mode": cost.profile().name, "web_backend": config.WEB_BACKEND,
                "answer_model": config.ANSWER_MODEL, "keystone": bool(config.INDEX)}
    return d


def ab_latest(p: Path) -> dict:
    """{} when there is no local A/B. Otherwise the summary / rows / paired blocks (no config: machine paths) and
    "provenance": created, web backend, budget ("cost_mode", None = recorded before the Parity budget existed),
    current = run on this server's web backend and budget."""
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(d, dict):
        return {}
    from . import cost
    rows = d.get("rows") or []
    cfg = d.get("config") or {}
    modes = {r.get("cost_mode") for r in rows if isinstance(r, dict) and r.get("mode") == "headstart"} - {None}
    cm = modes.pop() if len(modes) == 1 else ("mixed" if modes else None)
    backend = cfg.get("web_backend")
    if backend is None:
        bs = {r.get("web_backend") for r in rows if isinstance(r, dict)} - {None}
        backend = bs.pop() if len(bs) == 1 else None
    prov = {"created": d.get("created"), "web_backend": backend, "cost_mode": cm,
            "answer_model": cfg.get("answer_model"),
            "current": bool(cm == cost.profile().name and backend == config.WEB_BACKEND)}
    out = {k: d[k] for k in ("created", "summary", "rows", "paired") if k in d}
    out["rows"] = [{k: v for k, v in r.items() if k not in ("trace",)} for r in rows if isinstance(r, dict)]
    out["provenance"] = prov
    return out
