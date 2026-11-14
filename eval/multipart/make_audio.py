"""Multi-part spoken questions (planner eval): comparisons, conjunctions, multi-hop. Keyword answers per part.

  .venv/bin/python eval/multipart/make_audio.py                 # synthesize missing wavs, write questions.jsonl
  .venv/bin/python eval/multipart/make_audio.py --force mp03     # re-voice some ('all' for everything)

Writes eval/multipart/questions.jsonl (eval/questions.jsonl row format + "parts": [{"q", "expect"}]) and
eval/audio_multipart/mp_NN.wav (16 kHz mono PCM16 via eval/make_audio.finish; git-ignored). Voices: Kokoro (studio venv).
Answers are stable facts (Wikipedia / Wikidata) so the Wikipedia lane can score them; mp15 (flights + weather) and
mp20 (two weather cards) are live data: their weather parts are scored on the card's wording ("46°F" -> "f" / "degrees").

A/B:  .venv/bin/python eval/run_ab.py --questions eval/multipart/questions.jsonl --audio-dir eval/audio_multipart \
          --out runs/ab_mp_planon.json
Score per part:  .venv/bin/python eval/multipart/score.py runs/ab_mp_planon.json [runs/ab_mp_planoff.json]
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
AUDIO = EVAL / "audio_multipart"

EVEREST = "8848|8849|8,848|8,849|29,03|29032|29031|29,029|29029"
K2 = "8611|8,611|28,251|28251"
EIFFEL_H = "330|324|300|1,083|1083|1,063|1063"

# (id, text, voice@speed, kind, overall expect, [(part question, expect)], notes)
Q = [
    ("mp01", "which is taller, Mount Everest or K2", "af_heart@1.0", "compare", ["everest"],
     [("how tall is Mount Everest", [EVEREST]), ("how tall is K2", [K2])], "comparison: height"),
    ("mp02", "who was born first, Mozart or Beethoven", "am_michael@1.0", "compare", ["mozart"],
     [("when was Mozart born", ["1756"]), ("when was Beethoven born", ["1770"])], "comparison: birth"),
    ("mp03", "which is older, Harvard or Yale", "af_nova@1.0", "compare", ["harvard"],
     [("when was Harvard founded", ["1636"]), ("when was Yale founded", ["1701"])], "comparison: founding"),
    ("mp04", "which came out first, the iPhone or the iPod", "am_fenrir@1.0", "compare", ["ipod"],
     [("when was the iPhone released", ["2007"]), ("when was the iPod released", ["2001"])], "comparison: release"),
    ("mp05", "is the Eiffel Tower taller than the Statue of Liberty", "af_bella@1.0", "compare", ["eiffel"],
     [("how tall is the Eiffel Tower", [EIFFEL_H]), ("how tall is the Statue of Liberty", ["93|305|46"])],
     "comparison: than"),
    ("mp06", "who is older, Barack Obama or Donald Trump", "af_aoede@1.0", "compare", ["trump"],
     [("when was Barack Obama born", ["1961"]), ("when was Donald Trump born", ["1946"])], "comparison: age"),
    ("mp07", "which country has more people, Germany or France", "am_michael@1.0", "compare", ["germany"],
     [("population of Germany", ["83|84|82"]), ("population of France", ["68|67|66"])], "comparison: population"),
    ("mp08", "which is bigger, Texas or Alaska", "af_heart@1.0", "compare", ["alaska"],
     [("how big is Texas", ["268|695|696"]), ("how big is Alaska", ["663|665|1,717|1717|1,723|1723|586"])],
     "comparison: area"),
    ("mp09", "compare the populations of Japan and Germany", "am_fenrir@1.0", "compare", [],
     [("population of Japan", ["123|124|125|122"]), ("population of Germany", ["83|84|82"])], "compare verb"),
    ("mp10", "compare Nvidia and AMD revenue", "af_nova@1.0", "compare", [],
     [("Nvidia revenue", ["billion"]), ("AMD revenue", ["billion"])], "compare verb, brief's example"),
    ("mp11", "compare the heights of the Burj Khalifa and the Shanghai Tower", "af_bella@1.0", "compare",
     ["burj"], [("how tall is the Burj Khalifa", ["828|829|2,717|2717|2,722|2722"]),
                ("how tall is the Shanghai Tower", ["632|2,073|2073"])], "compare verb: height"),
    ("mp12", "Python versus Java, which came out first", "am_michael@1.0", "compare", ["python"],
     [("when was Python released", ["1991"]), ("when was Java released", ["1995|1996"])], "options first"),
    ("mp13", "compare the capitals of Australia and Canada", "af_aoede@1.0", "compare", [],
     [("capital of Australia", ["canberra"]), ("capital of Canada", ["ottawa"])], "compare verb: capitals"),
    ("mp14", "what's the capital of Japan and the capital of Brazil", "am_fenrir@1.0", "conj", [],
     [("capital of Japan", ["tokyo"]), ("capital of Brazil", ["brasilia"])], "conjunction: parallel NP"),
    ("mp15", "flights to Denver and the weather there", "af_heart@1.0", "conj", [],
     [("flights to Denver", ["denver"]), ("weather in Denver", ["f|degrees"])], "conjunction: brief's example; live"),
    ("mp16", "who founded Nvidia and when was it founded", "am_michael@1.0", "conj", [],
     [("who founded Nvidia", ["huang"]), ("when was Nvidia founded", ["1993"])], "conjunction: pronoun"),
    ("mp17", "what is the population of France and Germany", "af_nova@1.0", "conj", [],
     [("population of France", ["68|67|66"]), ("population of Germany", ["83|84|82"])], "conjunction: two slots"),
    ("mp18", "who wrote Hamlet and when was it written", "af_bella@1.0", "conj", [],
     [("who wrote Hamlet", ["shakespeare"]), ("when was Hamlet written", ["1599|1600|1601|1602|1603|1609"])],
     "conjunction: pronoun"),
    ("mp19", "when was the Eiffel Tower built and how tall is it", "am_fenrir@1.0", "conj", [],
     [("when was the Eiffel Tower built", ["1887|1889"]), ("how tall is the Eiffel Tower", [EIFFEL_H])],
     "conjunction: pronoun"),
    ("mp20", "what's the weather in Paris and in London", "af_aoede@1.0", "conj", ["paris", "london"],
     [("weather in Paris", ["f|degrees"]), ("weather in London", ["f|degrees"])], "conjunction: two cards; live"),
    ("mp21", "how old was the founder of Amazon when it was founded", "am_michael@1.0", "hop", ["30"],
     [("who founded Amazon", ["bezos"]), ("when was Jeff Bezos born", ["1964"]), ("when was Amazon founded", ["1994"])],
     "multi-hop + arithmetic"),
    ("mp22", "where was the author of Pride and Prejudice born", "af_heart@1.0", "hop", [],
     [("who wrote Pride and Prejudice", ["austen"]), ("where was Jane Austen born", ["steventon|hampshire"])],
     "multi-hop"),
    ("mp23", "when was the director of Jaws born", "af_nova@1.0", "hop", [],
     [("who directed Jaws", ["spielberg"]), ("when was Steven Spielberg born", ["1946"])], "multi-hop"),
    ("mp24", "what is the population of the capital of Japan", "am_fenrir@1.0", "hop", [],
     [("capital of Japan", ["tokyo"]), ("population of Tokyo", ["13|14|37"])], "multi-hop"),
    ("mp25", "where was the CEO of Nvidia born", "af_bella@1.0", "hop", [],
     [("who is the CEO of Nvidia", ["huang"]), ("where was Jensen Huang born", ["tainan|taiwan"])], "multi-hop"),
    ("mp26", "how old was the founder of Microsoft when it was founded", "af_aoede@1.0", "hop", ["19|20|22"],
     [("who founded Microsoft", ["gates|allen"]), ("when was the founder born", ["1955|1953"]),
      ("when was Microsoft founded", ["1975"])], "multi-hop + arithmetic"),
]


def synth(force: set[str]):
    todo = [r for r in Q if r[0] in force or not (AUDIO / f"mp_{r[0][2:]}.wav").exists()]
    if not todo:
        return
    tmp = Path(tempfile.mkdtemp(prefix="hs_mp_"))
    jobs = []
    for qid, text, voice, *_ in todo:
        name, speed = voice.rsplit("@", 1)
        jobs.append({"text": text, "voice": name, "speed": float(speed), "out": str(tmp / f"{qid}.wav"), "qid": qid})
    p = subprocess.run([base.KOKORO_PY, str(EVAL / "tts_kokoro.py")], input=json.dumps(jobs), text=True,
                       capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-3000:])
        raise SystemExit("kokoro failed")
    for j in jobs:
        x, sr = sf.read(j["out"], dtype="float32")
        base.finish(x, sr, AUDIO / f"mp_{j['qid'][2:]}.wav")
        print("kokoro", j["qid"], j["voice"], j["speed"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", default="")
    ap.add_argument("--no-audio", action="store_true", help="only write questions.jsonl")
    a = ap.parse_args()
    force = {r[0] for r in Q} if a.force == "all" else set(filter(None, a.force.split(",")))
    AUDIO.mkdir(parents=True, exist_ok=True)
    if not a.no_audio:
        synth(force)
    with OUT.open("w") as f:
        for qid, text, voice, kind, expect, parts, notes in Q:
            name, speed = voice.rsplit("@", 1)
            fn = f"mp_{qid[2:]}.wav"
            wav = AUDIO / fn
            row = {"id": qid, "file": fn, "text": text, "voice": f"kokoro:{name}", "rate": float(speed), "kind": kind,
                   "expect": list(expect) + [e for _, ex in parts for e in ex], "overall": list(expect),
                   "parts": [{"q": q, "expect": ex} for q, ex in parts], "notes": notes,
                   "dur_s": round(sf.info(str(wav)).duration, 2) if wav.exists() else None}
            f.write(json.dumps(row) + "\n")
    print("wrote", OUT, len(Q), "rows")


if __name__ == "__main__":
    main()
