"""Noisy showcase for the Arc-value A/B: the scored showcase questions (expect non-empty) mixed exactly like
eval/make_variants.py noise (babble of 6 OTHER eval questions + 25 % pink, SNR over active speech, 2 s noise tail),
at 20 and 10 dB -> eval/audio_arcvalue_noisy/snr20|snr10/ (git-ignored) + meta.json (true speech end per file) and
eval/arcvalue/showcase_scored.jsonl.

  .venv/bin/python eval/arcvalue/make_noisy.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import make_variants as mv  # noqa: E402

SEED = 11


def main():
    qs = [json.loads(l) for l in (ROOT / "eval/showcase/questions.jsonl").read_text().splitlines() if l.strip()]
    qs = [q for q in qs if q.get("expect")]
    (ROOT / "eval/arcvalue/showcase_scored.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    pool = [(q["id"], mv.load(mv.CLEAN / q["file"])) for q in mv.questions() if q["id"].startswith("q")]
    for snr in (20, 10):
        rng = np.random.default_rng(SEED)
        d = ROOT / "eval" / "audio_arcvalue_noisy" / f"snr{snr}"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"_": {"kind": "noise", "snr_db": snr, "noise": "babble x6 (other questions) + pink 25% power",
                      "noise_tail_s": mv.NOISE_TAIL_S, "seed": SEED, "set": "showcase scored"}}
        for q in qs:
            x = mv.load(mv.CLEAN / q["file"])
            a, b = mv.speech_span(x)
            n = len(x) + int(mv.NOISE_TAIL_S * mv.SR)
            nz = np.sqrt(1 - mv.PINK_SHARE) * mv.babble(n, q["id"], pool, rng) + np.sqrt(mv.PINK_SHARE) * mv.pink(n, rng)
            nz *= np.sqrt(mv.speech_power(x) / 10 ** (snr / 10)) / (np.sqrt(np.mean(nz ** 2)) + 1e-12)
            y = np.concatenate([x, np.zeros(n - len(x), np.float32)]) + nz
            peak = float(np.max(np.abs(y)))
            if peak > 0.97:
                y *= 0.97 / peak
            sf.write(str(d / q["file"]), y.astype(np.float32), mv.SR, subtype="PCM_16")
            meta[q["file"]] = {"speech_end_s": round(b / mv.SR, 3), "speech_start_s": round(a / mv.SR, 3)}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"wrote {len(qs)} files to {d}")


if __name__ == "__main__":
    main()
