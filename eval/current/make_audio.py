"""Current-events set (asked on 2026-11-14): questions whose answer only fresh web search has, for the real-Tavily A/B.

  .venv/bin/python eval/current/make_audio.py              # synthesize missing wavs, write questions.jsonl
  .venv/bin/python eval/current/make_audio.py --force all  # re-voice everything

Writes eval/current/questions.jsonl (same row format as eval/questions.jsonl) and eval/audio/cur_NN.wav (16 kHz mono,
Kokoro voices, eval/make_audio.finish). Gold ("expect") was checked by hand against several news sources on
2026-11-14 (event dates in "notes"); [] = changes by the hour (weather, prices, news) -> judged by hand.
Run the A/B:  .venv/bin/python eval/run_ab.py --questions eval/current/questions.jsonl --kinds all --no-latest ...
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
    ("cur01", "who won the World Series this year", "am_michael@1.0", "current", ["brewers"],
     "2026-10-28, Brewers beat Rays in 5"),
    ("cur02", "who won the Nobel Peace Prize this year", "af_heart@1.0", "current", ["pillay"],
     "2026-10-09, Navi Pillay"),
    ("cur03", "which party won the House in the midterm elections", "am_fenrir@1.0", "current", ["democrat"],
     "2026-11-03 midterms"),
    ("cur04", "um, who won the World Cup final this summer", "af_bella@1.0", "current", ["spain"],
     "2026-07-19, Spain 1-0 Argentina"),
    ("cur05", "who got the Nobel Prize in literature this year", "af_aoede@1.0", "current", ["carson"],
     "2026-10-08, Anne Carson"),
    ("cur06", "who won the men's US Open tennis final in September", "am_michael@1.0", "current", ["zverev"],
     "2026-09-13, Zverev beat Shelton"),
    ("cur07", "who won the Tour de France this year", "af_nova@1.0", "current", ["pogacar"],
     "2026-07-26, Pogacar 5th title"),
    ("cur08", "hey, who won the men's race at the New York City Marathon", "am_fenrir@1.0", "current", ["almgren"],
     "2026-11-01, Andreas Almgren"),
    ("cur09", "who won the Ballon d'Or this year", "af_heart@1.0", "current", ["kane"],
     "2026-10-26, Harry Kane"),
    ("cur10", "what's the price of gold per ounce today", "af_bella@1.0", "current", [], "commodity price"),
    ("cur11", "what's the latest news about the James Webb telescope", "am_michael@1.0", "current", [], "news"),
    ("cur12", "is it going to rain in Austin this weekend", "af_nova@1.0", "current", [], "weather"),
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
