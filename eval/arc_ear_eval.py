"""Arc ear A/B: the old Arc v1 Mini (arc_audio 1.0.1 + torch, models/arc-v1) vs Arc Mini in C (libarc, models/arc-mini),
streamed exactly as server/ear.py streams them (100 ms chunks, builtin transcript, partials) over the spoken eval sets.

Per ear, in its own process (so cold start and RSS are that ear's alone):
  wer             word error rate of the finals against the scripted text (lowercase, punctuation stripped; "Mt."
                  and digits as written: both models spell numbers the same way, as digits)
  complete_ms     audio time when the ear first SHOWED the whole final text (partial or final) - speech end
  final_ms        audio time of the utterance's final - speech end
  behind_words    mean words behind while speaking: words the speaker had finished (this ear's own final word times)
                  minus words shown, every 100 ms chunk from the first word to the speech end
  first_word_ms   audio time the first word was shown - its end
  chunk_ms        wall time of feed() per 100 ms chunk (p50 / p90 / max) on this ear's worker thread
  load_s          cold start: import + model load + first stream; rss_mb: peak resident set at the end

Speech end is model-independent: the clean file's energy (last 10 ms frame within 35 dB of the peak), or
meta.json's speech_end_s for eval/audio_noisy.

  .venv/bin/python eval/arc_ear_eval.py --ear mini   --out runs/arc_ear/mini.json
  .venv/bin/python eval/arc_ear_eval.py --ear v1     --out runs/arc_ear/v1.json
  .venv/bin/python eval/arc_ear_eval.py --summary runs/arc_ear/v1.json runs/arc_ear/mini.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SETS = {  # rows file -> (text field, file field)
    "questions": "eval/questions.jsonl",
    "current": "eval/current/questions.jsonl",
    "multipart": "eval/multipart/questions.jsonl",
    "showcase": "eval/showcase/questions.jsonl",
    "sitebrain": "eval/sitebrain/questions.jsonl",
    "local": "index/questions_local.jsonl",
    "conversations": "eval/conversations.jsonl",
}


def refs(sets: list[str], root: Path) -> list[tuple[str, str, str]]:
    out = []
    for s in sets:
        p = root / SETS[s]
        if not p.exists():
            p = ROOT / SETS[s]
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("files"):
                out += [(s, f, t) for f, t in zip(r["files"], r["turns"])]
            elif r.get("file"):
                out.append((s, r["file"], r["text"]))
    return out


_P = re.compile(r"[^\w'\s]")


def norm(t: str) -> list[str]:
    t = re.sub(r"(?<=\w)\.(?=\w)", "", t.lower()).replace("%", " percent").replace("-", " ")   # "U.S." = "us"
    return [w.strip("'") for w in _P.sub(" ", t).split() if w.strip("'")]


def edits(a: list[str], b: list[str]) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def speech_end(x: np.ndarray) -> float:
    n = len(x) // 160
    db = 10 * np.log10(np.mean(x[: n * 160].reshape(n, 160) ** 2, axis=1) + 1e-12)
    on = np.where(db > db.max() - 35)[0]
    return float((on[-1] + 1) * 0.01) if len(on) else 0.0


# ---------------------------------------------------------------------------------------------------- ears
def make_v1(threads: int):
    arc_dir = os.environ.get("ARC_V1_DIR") or str(Path(os.environ.get("MODELS_DIR", "")) / "arc-v1")
    from server import modelpaths
    if not Path(arc_dir, "model.safetensors").exists():
        arc_dir = str(modelpaths.resolve("arc-v1", "ARC_V1_DIR"))
    sys.path.insert(0, arc_dir)
    from arc_audio import Arc
    a = Arc.from_pretrained(arc_dir, device="cpu", threads=threads)
    # background=False: partials and finals inside feed(), as libarc does them, so the audio-time lags below are the
    # model's own (the server's background worker adds wall time on top whenever it falls behind) and chunk_ms
    # carries their compute
    return lambda: a.stream(domain="assistant", transcript="builtin", partials=True, background=False), arc_dir


def make_mini(threads: int, partial_ms: int, encode: bool = True):
    from server import arc_c, modelpaths
    d = os.environ.get("ARC_MINI_DIR") or str(modelpaths.resolve("arc-mini", "ARC_MINI_DIR"))
    return (lambda: arc_c.open_stream(d, threads, "assistant", partial_ms=partial_ms, partial_encode=encode)), d


def run_file(new_stream, x: np.ndarray, tail_s: float) -> dict:
    s = new_stream()
    x = np.concatenate([x, np.zeros(int(tail_s * 16000), np.float32)])
    finals, shown_log, chunk_ms = [], [], []
    for i in range(0, len(x), 1600):
        t0 = time.perf_counter()
        out = s.feed(x[i:i + 1600])
        chunk_ms.append((time.perf_counter() - t0) * 1000)
        at = min(len(x), i + 1600) / 16000
        for u in out.get("transcript") or []:
            finals.append((at, u))
        p = out.get("transcript_partial")
        pw = [w["text"] for w in (p.get("words") or [])] if p and not p.get("final") else []
        shown = [w for _, u in finals for w in norm(u.get("text", ""))] + norm(" ".join(pw))
        shown_log.append((at, shown))
    t0 = time.perf_counter()
    out = s.flush()
    flush_ms = (time.perf_counter() - t0) * 1000
    for u in out.get("transcript") or []:
        finals.append((len(x) / 16000, u))
    if hasattr(s, "close"):
        s.close()
    return {"finals": finals, "shown": shown_log, "chunk_ms": chunk_ms, "flush_ms": flush_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ear", choices=["v1", "mini"])
    ap.add_argument("--sets", default="questions,current,multipart,showcase,sitebrain,local,conversations")
    ap.add_argument("--audio", default=os.environ.get("AUDIO_DIR") or "")
    ap.add_argument("--noisy", default="", help="e.g. snr10: eval/audio_noisy/<x> (only files there)")
    ap.add_argument("--data-root", default=os.environ.get("EVAL_ROOT") or str(ROOT))
    ap.add_argument("--threads", type=int, default=0, help="0: the server's default for that ear")
    ap.add_argument("--partial-ms", type=int, default=400)
    ap.add_argument("--streamed-partials", action="store_true",
                    help="mini: partials from the streamed encoder states (upstream libarc) instead of a re-encode")
    ap.add_argument("--tail-s", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--summary", nargs="*")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    import soundfile as sf
    from server import config, modelpaths
    root = Path(a.data_root)
    audio = Path(a.audio) if a.audio else modelpaths.resolve("demo-audio", "AUDIO_DIR", legacy=root / "eval/audio")
    meta = {}
    if a.noisy:
        audio = root / "eval/audio_noisy" / a.noisy
        meta = json.loads((audio / "meta.json").read_text())
    rows = [r for r in refs(a.sets.split(","), root) if (audio / r[1]).exists()]
    if a.limit:
        rows = rows[: a.limit]
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    t0 = time.perf_counter()
    if a.ear == "v1":
        threads = a.threads or config.ARC_THREADS
        new_stream, mdir = make_v1(threads)
    else:
        threads = a.threads or config.ARC_C_THREADS
        new_stream, mdir = make_mini(threads, a.partial_ms, not a.streamed_partials)
    s = new_stream()                                   # first stream + a short warm feed, as ear.warmup()
    for _ in range(3):
        s.feed(np.zeros(1600, np.float32))
    s.flush()
    if hasattr(s, "close"):
        s.close()
    load_s = time.perf_counter() - t0
    res, allchunks = [], []
    for k, (set_, f, text) in enumerate(rows):
        x, sr = sf.read(str(audio / f), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == 16000, f
        se = meta[f]["speech_end_s"] if f in meta else speech_end(x)
        r = run_file(new_stream, x, a.tail_s)
        hyp = " ".join(u.get("text", "") for _, u in r["finals"])
        ref_w, hyp_w = norm(text), norm(hyp)
        fw = [w for _, u in r["finals"] for w in (u.get("words") or [])]
        full = [w for _, u in r["finals"] for w in norm(u.get("text", ""))]
        complete = next((t for t, sh in r["shown"] if full and sh == full), None)
        final_at = r["finals"][-1][0] if r["finals"] else None
        ends = [w["end"] for w in fw if w.get("end") is not None]
        behind = []
        for t, sh in r["shown"]:
            if ends and ends[0] <= t <= se + 1e-6:
                behind.append(max(0, sum(e <= t for e in ends) - len(sh)))
        first = next((t for t, sh in r["shown"] if sh), None)
        res.append({"set": set_, "file": f, "ref": text, "hyp": hyp, "n_ref": len(ref_w), "err": edits(ref_w, hyp_w),
                    "speech_end": round(se, 2), "n_utts": len(r["finals"]),
                    "complete_ms": None if complete is None else round((complete - se) * 1000),
                    "final_ms": None if final_at is None else round((final_at - se) * 1000),
                    "first_word_ms": None if first is None or not ends else round((first - ends[0]) * 1000),
                    "behind_words": round(float(np.mean(behind)), 2) if behind else 0.0,
                    "chunk_ms_p50": round(float(np.median(r["chunk_ms"])), 2), "flush_ms": round(r["flush_ms"], 1)})
        allchunks += r["chunk_ms"]
        print(f"{k + 1:3d}/{len(rows)} {f:16s} err {res[-1]['err']}/{res[-1]['n_ref']}  {hyp[:70]}", file=sys.stderr)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20      # macOS: bytes
    size = sum(p.stat().st_size for p in Path(mdir).rglob("*") if p.is_file())
    out = {"ear": a.ear + ("-streamed" if a.ear == "mini" and a.streamed_partials else ""), "model_dir": mdir, "threads": threads, "partial_ms": a.partial_ms, "noisy": a.noisy or None,
           "load_s": round(load_s, 2), "rss_mb": round(rss), "rss_before_load_mb": round(rss0),
           "model_mb": round(size / 1e6, 1), "chunk_ms": _pct(allchunks), "rows": res, **_agg(res)}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


def _pct(xs):
    xs = np.asarray([x for x in xs if x is not None], float)
    if not len(xs):
        return None
    return {"p50": round(float(np.median(xs)), 1), "p90": round(float(np.percentile(xs, 90)), 1),
            "max": round(float(xs.max()), 1), "mean": round(float(xs.mean()), 2), "n": len(xs)}


def _agg(res):
    n = sum(r["n_ref"] for r in res)
    e = sum(r["err"] for r in res)
    return {"n_files": len(res), "wer": round(e / max(1, n), 4), "exact": round(sum(r["err"] == 0 for r in res) / max(1, len(res)), 3),
            "complete_ms": _pct([r["complete_ms"] for r in res]), "final_ms": _pct([r["final_ms"] for r in res]),
            "first_word_ms": _pct([r["first_word_ms"] for r in res]),
            "behind_words": round(float(np.mean([r["behind_words"] for r in res])), 2) if res else None,
            "by_set": {s: round(sum(r["err"] for r in res if r["set"] == s) / max(1, sum(r["n_ref"] for r in res if r["set"] == s)), 4)
                       for s in sorted({r["set"] for r in res})}}


def summary(paths):
    rs = [json.loads(Path(p).read_text()) for p in paths]
    keys = ["ear", "noisy", "threads", "n_files", "wer", "exact", "by_set", "complete_ms", "final_ms", "first_word_ms",
            "behind_words", "chunk_ms", "load_s", "rss_mb", "model_mb"]
    for k in keys:
        print(f"{k:14s}" + "".join(f" | {json.dumps(r.get(k))[:60]:60s}" for r in rs))
    if len(rs) == 2:                                                   # paired per-file errors
        a = {r["file"]: r for r in rs[0]["rows"]}
        b = {r["file"]: r for r in rs[1]["rows"]}
        both = sorted(set(a) & set(b))
        better = sum(b[f]["err"] < a[f]["err"] for f in both)
        worse = sum(b[f]["err"] > a[f]["err"] for f in both)
        print(f"paired: {len(both)} files, second better on {better}, worse on {worse}")
        for f in both:
            if a[f]["err"] != b[f]["err"]:
                print(f"  {f:14s} {a[f]['err']}->{b[f]['err']}  ref: {a[f]['ref']}\n{'':18s}1: {a[f]['hyp']}\n{'':18s}2: {b[f]['hyp']}")


if __name__ == "__main__":
    main()
