"""Site-brain set: 10 spoken questions about the Tavily API, answered from docs.tavily.com read once into the local
tier (index/build_site.py, server/sitebrain.py). Gold = docs.tavily.com as read on 2026-11-14.

  .venv/bin/python eval/sitebrain/make_audio.py            # synthesize missing wavs (eval/audio/tav_NN.wav)
  .venv/bin/python eval/run_ab.py --questions eval/sitebrain/questions.jsonl --kinds all --no-latest ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
import make_audio as base  # noqa: E402

OUT = HERE / "questions.jsonl"
AUDIO = EVAL / "audio"
ASKED = "2026-11-14"

# (id, text, kokoro voice@speed, kind, expect, notes)
Q = [
    ("tav01", "how many credits does an advanced search cost on Tavily", "am_michael@1.0", "site", ["2|two"],
     "api-credits"),
    ("tav02", "how many URLs can I send to Tavily extract in one request", "af_heart@1.0", "site", ["20|twenty"],
     "extract: urls 1-20"),
    ("tav03", "um, how much does Tavily charge to extract pages at basic depth", "am_fenrir@1.0", "site",
     ["5|five"], "1 credit per 5 successful URLs"),
    ("tav04", "how many free credits does Tavily give you every month", "af_bella@1.0", "site", ["1000|thousand"],
     "Researcher plan"),
    ("tav05", "what's the most results a single Tavily search can return", "af_aoede@1.0", "site",
     ["20|twenty"], "max_results max 20"),
    ("tav06", "which topics can I choose for a Tavily search", "am_michael@1.0", "site", ["news", "finance"],
     "topic enum"),
    ("tav07", "what's the rate limit on the Tavily crawl endpoint", "af_nova@1.0", "site", ["100"],
     "100 requests per minute"),
    ("tav08", "how much does the Tavily map endpoint cost", "am_fenrir@1.0", "site", ["10|ten"],
     "1 credit per 10 pages"),
    ("tav09", "hey, what does one Tavily credit cost on pay as you go", "af_heart@1.0", "site", ["0.008"],
     "$0.008 per credit"),
    ("tav10", "what's the longest timeout I can set for a Tavily crawl", "af_bella@1.0", "site", ["150"],
     "timeout 10-150 s"),
]


def synth(rows, force):
    todo = [r for r in rows if r[0] in force or not (AUDIO / f"{r[0]}.wav").exists()]
    if not todo:
        return
    tmp = Path(tempfile.mkdtemp(prefix="hs_cur_"))
    jobs = []
    for qid, text, voice, *_ in todo:
        name, speed = voice.rsplit("@", 1)
        jobs.append({"text": text, "voice": name, "speed": float(speed), "out": str(tmp / f"{qid}.wav")})
    p = subprocess.run([base.KOKORO_PY, str(EVAL / "tts_kokoro.py")], input=json.dumps(jobs), text=True,
                       capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit("kokoro failed")
    for qid, *_ in todo:
        x, sr = sf.read(str(tmp / f"{qid}.wav"), dtype="float32")
        base.finish(x, sr, AUDIO / f"{qid}.wav")
        print("kokoro", qid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", default="")
    a = ap.parse_args()
    force = {r[0] for r in Q} if a.force == "all" else set(filter(None, a.force.split(",")))
    AUDIO.mkdir(parents=True, exist_ok=True)
    synth(Q, force)
    with OUT.open("w") as f:
        for qid, text, voice, kind, expect, notes in Q:
            name, speed = voice.rsplit("@", 1)
            row = {"id": qid, "file": f"{qid}.wav", "text": text, "voice": f"kokoro:{name}", "rate": float(speed),
                   "kind": kind, "expect": expect, "notes": notes, "asked": ASKED,
                   "dur_s": round(sf.info(str(AUDIO / f"{qid}.wav")).duration, 2)}
            f.write(json.dumps(row) + "\n")
    print("wrote", OUT, len(Q), "rows")


if __name__ == "__main__":
    main()
