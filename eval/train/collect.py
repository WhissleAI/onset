"""Harvest real per-question rows into runs/train/real/rows.jsonl (local only, never committed).

Sources
  app       <Onset data dir>/train/rows.jsonl: what server/trainlog.py writes when TRAIN_COLLECT=1 (already redacted)
  app_log   <Onset data dir>/../logs/server.log*: the request log's "done" lines (ready ms, path, lane, endpoint by,
            first 80 characters of the question) from before the collector existed
  eval      every agent's eval replays: <checkout>/runs/**/*.json files with per-question rows (run_ab.py shape:
            ready / sources / endpoint lag, path, route, paid / free, verify / re-check, conf_feats, correct)
Every text field goes through server.trainlog.redact (PII_YEARS handling, 5+ digit numbers, emails, handles, the
tagger's redacted words); audio is never read or stored; client ids are salted hashes. Rows are de-duplicated by a
content hash, so re-running only adds what is new.

  .venv/bin/python eval/train/collect.py [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.sys_path()
from server import trainlog  # noqa: E402

OUT = C.TRAIN / "real" / "rows.jsonl"
SEEN = C.TRAIN / "real" / "seen.json"
EVAL_ROW_KEYS = set(trainlog.KEEP) | {"id", "kind", "voice", "correct", "correct_patched", "q_wer", "arc_wer",
                                      "answers", "phantom_answers", "verify", "recheck", "ans_in", "ans_out",
                                      "llm_requests", "order", "error", "trace"}
_DONE = re.compile(r"\[req (\d\d:\d\d:\d\d)\.\d+\] run (\d+) (\w+) q(\d+) done ready=(-?\d+|None)ms path=(\S+) "
                   r"lane=(\S+)(?: fallback=(\S+))?(?: ERROR=(.*?))? \| (.*)$")
_ENDP = re.compile(r"\[req [\d:.]+\] run (\d+) (\w+) q(\d+) endpoint by=(\S+) speech_end=(\S+)ms")


def row_hash(r: dict) -> str:
    keep = {k: v for k, v in r.items() if k not in ("harvested", "h", "file")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:20]


def scrub(r: dict) -> dict:
    """Redact every text field of a row (idempotent on already-redacted rows) and drop anything audio-like."""
    out = {}
    for k, v in r.items():
        if k in ("audio", "wav", "pcm", "samples", "file") or k.endswith("_path"):
            continue
        if k in ("client", "client_id", "cid") and v:
            out["cid"] = v if (isinstance(v, str) and re.fullmatch(r"[0-9a-f]{16}", v)) else trainlog.hash_id(v)
            continue
        if k in trainlog.TEXT or k in ("text", "answer"):
            if k == "answer":
                out["answer_words"] = len(str(v or "").split())
                continue
            out[k] = trainlog.redact(str(v or "")[:300])
            continue
        if k == "trace" and isinstance(v, list):
            out[k] = [{kk: (trainlog.redact(str(vv)) if kk == "query" else vv) for kk, vv in t.items()}
                      for t in v[:40] if isinstance(t, dict)]
            continue
        if k == "spec_queries" and isinstance(v, list):
            out[k] = [trainlog.redact(str(x)) for x in v[:10]]
            continue
        if k == "sources":
            continue                                   # titles of what was read: not needed for training
        out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ sources
def from_app(data_dir: Path) -> list[dict]:
    rows = []
    for p in sorted((data_dir / "train").glob("rows*.jsonl")):
        for r in C.read_jsonl(p):
            rows.append({**scrub(r), "src": "app"})
    return rows


def parse_log(lines: list[str], day: str = "") -> list[dict]:
    rows, ends = [], {}
    for ln in lines:
        m = _ENDP.search(ln)
        if m:
            ends[(m.group(1), m.group(2), m.group(3))] = {"endpoint_by": m.group(4), "speech_end_ms": m.group(5)}
            continue
        m = _DONE.search(ln)
        if not m:
            continue
        t, run, mode, q, ready, path, lane, fb, err, words = m.groups()
        words = words.strip()
        if len(words) >= 2 and words[0] in "'\"" and words[-1] == words[0]:
            words = words[1:-1]
        r = {"src": "app_log", "time": f"{day} {t}".strip(), "run": int(run), "mode": mode, "q": int(q),
             "ready_ms": None if ready == "None" else int(ready), "path": path, "lane": None if lane == "-" else lane,
             "answer_fallback": fb, "failed": bool(err), "question": trainlog.redact(words)}
        r.update(ends.get((run, mode, q), {}))
        rows.append(r)
    return rows


def from_logs(data_dir: Path) -> list[dict]:
    import datetime as dt
    rows = []
    for p in sorted(glob.glob(str(data_dir.parent / "logs" / "server.log*"))):
        try:
            day = dt.date.fromtimestamp(Path(p).stat().st_mtime).isoformat()
            rows += parse_log(Path(p).read_text(errors="replace").splitlines(), day)
        except OSError:
            pass
    return rows


def eval_files(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        rr = root / "runs"
        if not rr.is_dir():
            continue
        for p in rr.rglob("*.json"):
            if "train" in p.parts[len(rr.parts):len(rr.parts) + 1] or "models" in p.parts:
                continue
            try:
                if p.stat().st_size > 30_000_000:
                    continue
            except OSError:
                continue
            out.append(p)
    return out


def from_eval_file(p: Path, root: Path) -> list[dict]:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(d, dict) or not isinstance(d.get("rows"), list):
        return []
    rows = [r for r in d["rows"] if isinstance(r, dict) and "ready_ms" in r and "mode" in r]
    if not rows:
        return []
    cfg = d.get("config") or {}
    meta = {"src": "eval", "file": f"{root.name}/{p.relative_to(root)}", "created": d.get("created"),
            "cfg": {k: cfg.get(k) for k in ("web_backend", "ear", "flux_soft", "cost_mode", "answer_model", "index",
                                             "kdir", "director") if k in cfg}}
    return [{**scrub({k: v for k, v in r.items() if k in EVAL_ROW_KEYS or k in trainlog.TEXT or k == "text"}), **meta}
            for r in rows]


def checkouts() -> list[Path]:
    return sorted({Path(p) for p in glob.glob(str(C.MAIN.parent / "headstart*")) if Path(p, "server").is_dir()})


def harvest(app_data: Path, roots: list[Path], seen: set) -> tuple[list[dict], dict]:
    new, stats = [], {}
    batches = [("app", from_app(app_data)), ("app_log", from_logs(app_data))]
    ev = []
    for root in roots:
        for p in eval_files([root]):
            ev += from_eval_file(p, root)
    batches.append(("eval", ev))
    for src, rows in batches:
        k = 0
        for r in rows:
            h = row_hash(r)
            if h in seen:
                continue
            seen.add(h)
            new.append({**r, "h": h, "harvested": C.now_iso()})
            k += 1
        stats[src] = {"seen": len(rows), "new": k}
    return new, stats


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    seen = set(C.read_json(SEEN, []))
    new, stats = harvest(C.APP_DATA, checkouts(), seen)
    if not a.dry_run:
        C.append_jsonl(OUT, new)
        C.write_json(SEEN, sorted(seen))
    total = len(C.read_jsonl(OUT)) if OUT.exists() else 0
    rep = {"new": len(new), "total": total, "sources": stats}
    print(json.dumps(rep))
    return rep


if __name__ == "__main__":
    main()
