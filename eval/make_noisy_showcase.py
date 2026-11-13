"""Noisy-venue showcase audio for the director vs fused A/B: every showcase question mixed with babble of the OTHER
showcase questions (3-8 talkers) at 10 dB SNR over active speech, a synthetic room (RT60 0.4 s, near mic), level
-20 dBFS, noise running 1 s past the speech; fixed seed per question so every policy hears identical audio. Uses the
director recipe's acoustic chain (PromptingNemo-private recipes/headstart_director/augment.py; --augment points at it).

  .venv/bin/python eval/make_noisy_showcase.py [--augment <recipe dir>]   -> eval/audio_noisy/sc_babble10_room/
  .venv/bin/python eval/run_ab.py --questions eval/showcase/questions.jsonl --audio-dir eval/audio_noisy/sc_babble10_room
"""
import argparse, json, sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", default="/Users/karan/Desktop/work/whissle/pnp-headstart-director/recipes/headstart_director")
    a = ap.parse_args()
    sys.path.insert(0, a.augment)
    import augment
    import make_variants as mv
    qs = [json.loads(l) for l in open(ROOT / "eval" / "showcase" / "questions.jsonl")]
    clean = {q["file"]: sf.read(str(ROOT / "eval" / "audio" / q["file"]), dtype="float32")[0] for q in qs}
    out = ROOT / "eval" / "audio_noisy" / "sc_babble10_room"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"_": {"kind": "noise+room", "recipe": __doc__.splitlines()[0]}}
    for i, q in enumerate(qs):
        x = clean[q["file"]]
        pool = [w for f, w in clean.items() if f != q["file"]]
        c = {"level_dbfs": -20.0, "noise": "babble", "snr": 10.0, "room": {"rt60": 0.4, "far": False}}
        sf.write(str(out / q["file"]), augment.apply(x, c, pool, 1000 + i, (0, 0)), 16000, subtype="PCM_16")
        s0, s1 = mv.speech_span(x)
        meta[q["file"]] = {"speech_end_s": round(s1 / 16000, 3), "speech_start_s": round(s0 / 16000, 3), "cond": c}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(len(qs), "files ->", out)


if __name__ == "__main__":
    main()
