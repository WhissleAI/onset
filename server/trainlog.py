"""Training rows from real use (TRAIN_COLLECT=1): one redacted line per answered question, kept on this Mac.

What a row holds: the timings (ready / sources / endpoint), the route and tiers, paid / free calls, the verify and
re-check outcome, the calibration features and confidence, the endpoint trigger, the ear, and the question / query
with personal data masked (numbers of 5+ digits, emails, handles, spoken or written birth years, the tagger's own
redacted words). Never: audio, keys, the raw client id (a salted hash), the answer text (only its length).
User corrections ("no, I meant ...") arrive as the next row's `corrected` / `heard_fix` fields.

Where: <data dir>/train/rows.jsonl (Application Support in the Mac app; runs/ in a repo run). eval/train/collect.py
reads it into the training set; nothing here ever leaves the machine or goes into git. Any failure is swallowed:
collection must never slow or break a question.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

from . import config

_LOCK = threading.Lock()
_RUNS: dict[int, dict] = {}            # run id -> {"cid": hashed client id, "kind": mic / replay}
MAX_BYTES = 20_000_000                 # rows.jsonl rolls to rows.1.jsonl past this

KEEP = ("mode", "ready_ms", "sources_ms", "done_ms", "endpoint_lag_ms", "ready_vad_ms", "path", "lane", "route",
        "route_plan", "route_conf", "route_why", "routes", "tiers", "answer_tier", "answer_src", "answer_by",
        "answer_fallback", "endpoint_by", "endpoint_trigger", "endpoint_via", "ear", "ear_fallback", "web_backend",
        "web_calls", "web_paid", "paid_calls", "free_calls", "search_cached", "web_cached", "credits", "credits_spec",
        "credits_final", "usd", "usd_web", "usd_llm", "spec_fires", "spec_used", "wasted_fires", "first_fire_lead_ms",
        "used_fire_lead_ms", "drafts", "draft_reused", "cost_mode", "intent", "hedged", "stale", "n_finals",
        "conf", "conf_band", "conf_pre", "conf_feats", "ks_hit", "keystone", "kdir", "failed", "skipped",
        "superseded", "corrected", "followup", "heard_fix", "answer_patched", "cover", "plan_hedged", "q")
TEXT = ("question", "query", "arc_heard", "question_heard")


def salt() -> str:
    """A per-install random salt (data dir), so a client-id hash can't be matched across machines."""
    p = config.DATA_DIR / "train" / ".salt"
    try:
        return p.read_text().strip()
    except OSError:
        import secrets
        p.parent.mkdir(parents=True, exist_ok=True)
        s = secrets.token_hex(16)
        p.write_text(s)
        return s


def hash_id(cid, s: str | None = None) -> str | None:
    if not cid:
        return None
    return hashlib.sha256(f"{s if s is not None else salt()}:{cid}".encode()).hexdigest()[:16]


_EMAIL = re.compile(r"\S+@\S+")
_HANDLE = re.compile(r"(^|\s)@\w+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def redact(text: str, hide_idx=()) -> str:
    """Mask personal data in a transcript: the tagger's redacted token indices (hide_idx), digit runs of 5+ digits
    (phones, cards, ids), emails / handles / urls, and birth years near a birth cue (PII_YEARS, spoken or written)."""
    if not text:
        return text or ""
    from . import frame
    text = _URL.sub(" <url> ", _EMAIL.sub(" <email> ", str(text)))
    text = _HANDLE.sub(r"\1<handle>", text)
    words = text.split()
    toks = [frame.norm(w) for w in words]
    hide = set(i for i in (hide_idx or ()) if isinstance(i, int))
    # a number said digit by digit ("four one five five five ...") or written: hide runs carrying 5+ digits
    run: list[int] = []
    digits = 0
    for i, t in enumerate(toks + [""]):
        d = sum(c.isdigit() for c in t) if t else 0
        spoken = t in _DIGIT_WORDS
        if d or spoken:
            run.append(i)
            digits += d or 1
            continue
        if digits >= 5:
            hide.update(run)
        run, digits = [], 0
    if config.PII_YEARS:
        hide |= frame.birth_year_tokens(toks)
    return " ".join("<pii>" if i in hide else w for i, w in enumerate(words) if w)


_DIGIT_WORDS = {"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}


def enabled() -> bool:
    return bool(getattr(config, "TRAIN_COLLECT", False))


def bind(run_id: int, client_id, kind: str = ""):
    if enabled():
        _RUNS[run_id] = {"cid": hash_id(client_id) if client_id else None, "kind": kind}
        if len(_RUNS) > 512:
            for k in list(_RUNS)[:256]:
                _RUNS.pop(k, None)


def row_of(e: dict, run_id: int, run: dict | None = None) -> dict:
    hide = e.get("redact") or ((e.get("frame") or {}).get("redact") if isinstance(e.get("frame"), dict) else None)
    r = {k: e[k] for k in KEEP if k in e}
    for k in TEXT:
        if e.get(k):
            r[k] = redact(str(e[k])[:300], hide if k == "question" else ())
    if isinstance(e.get("verify"), dict):
        r["verify"] = {k: e["verify"].get(k) for k in ("status", "n_claims", "found", "ms")}
    if isinstance(e.get("recheck"), dict):
        r["recheck"] = {k: e["recheck"].get(k) for k in ("status", "changed", "ms")}
    if e.get("answer"):
        r["answer_words"] = len(str(e["answer"]).split())
    r.update({"src": "app", "ts": round(time.time(), 1), "run": run_id, **(run or {})})
    return r


def record(e: dict, mode: str, run_id: int):
    """Called for every event a run emits; keeps only answered-question metrics. Never raises."""
    if not enabled() or e.get("t") != "metric":
        return
    try:
        r = row_of({**e, "mode": e.get("mode") or mode}, run_id, _RUNS.get(run_id))
        p = Path(config.DATA_DIR) / "train" / "rows.jsonl"
        line = json.dumps(r, default=str) + "\n"
        with _LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and p.stat().st_size > MAX_BYTES:
                p.replace(p.with_name("rows.1.jsonl"))
            with p.open("a") as f:
                f.write(line)
    except Exception:
        pass
