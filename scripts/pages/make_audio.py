"""Voice the demo-only turns of scripts/pages/scenarios.json ("tts") with Kokoro (Apache-2.0 voices).

  .venv/bin/python scripts/pages/make_audio.py --out <dir>     # 16 kHz mono PCM16 wavs, same padding as eval/

The wavs are inputs to scripts/pages/record.py; only compressed copies are published (site/demo-data/audio).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import make_audio as base  # noqa: E402  (finish, KOKORO_PY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((Path(__file__).with_name("scenarios.json")).read_text())["tts"]
    todo = [r for r in rows if a.force or not (out / f"{r[0]}.wav").exists()]
    if not todo:
        print("nothing to do")
        return
    tmp = Path(tempfile.mkdtemp(prefix="pages_tts_"))
    jobs = [{"text": t, "voice": v, "speed": 1.0, "out": str(tmp / f"{i}.wav")} for i, t, v in todo]
    subprocess.run([base.KOKORO_PY, str(ROOT / "eval" / "tts_kokoro.py")], input=json.dumps(jobs), text=True,
                   check=True)
    for i, *_ in todo:
        x, sr = sf.read(str(tmp / f"{i}.wav"), dtype="float32")
        base.finish(x, sr, out / f"{i}.wav")
        print("wrote", out / f"{i}.wav")


if __name__ == "__main__":
    main()
