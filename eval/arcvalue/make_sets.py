"""Arc-value eval sets (docs/ARC_VALUE.md): new spoken corrections / restarts and live-data questions, voiced with
Kokoro or macOS say into eval/audio_arcvalue/ (git-ignored), plus the question files run_ab.py reads.

  .venv/bin/python eval/arcvalue/make_sets.py            # synthesize missing wavs, write the jsonl files

corrections.jsonl = the existing correction items (q26-q33, sc27, sc28, q50; their wavs symlinked from eval/audio)
                    + 22 new ones (cx01-cx22): self-corrections and restarts. "wrong" = keywords of the answer to the
                    corrected-away question (an answer with a wrong keyword and no expected one = harmful).
livedata.jsonl    = 16 live-data questions (weather / currency / clock / units); ground truth at answer time
                    (eval/arcvalue/score_live.py).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
import make_audio  # noqa: E402

OUT = ROOT / "eval" / "audio_arcvalue"
HERE = Path(__file__).resolve().parent

# (id, text, voice, expect, wrong, style)
CX = [
    ("cx01", "who directed Titanic, sorry, I mean Jurassic Park", "kokoro:am_michael@1.0", ["spielberg"], ["cameron"], "sorry I mean"),
    ("cx02", "what's the capital of Spain, no wait, Portugal", "say:Samantha@175", ["lisbon"], ["madrid"], "no wait"),
    ("cx03", "when was the Eiffel Tower built, actually make that the Empire State Building", "kokoro:af_heart@1.0", ["1931|1930"], ["1889|1887"], "actually make that"),
    ("cx04", "who wrote The Odyssey, scratch that, The Divine Comedy", "kokoro:am_fenrir@1.0", ["dante"], ["homer"], "scratch that"),
    ("cx05", "what's the largest ocean, I mean the smallest ocean", "say:Daniel@175", ["arctic"], ["pacific"], "I mean (adjective)"),
    ("cx06", "um, what's the, uh, the capital of the the Netherlands", "kokoro:af_nova@1.0", ["amsterdam"], [], "restart + fillers"),
    ("cx07", "who invented the, um, who invented the light bulb", "say:Reed (English (US))@180", ["edison|swan"], [], "restart"),
    ("cx08", "what currency does, uh, what currency does India use", "kokoro:af_bella@1.0", ["rupee"], [], "restart"),
    ("cx09", "where was Einstein born, sorry, Newton", "kokoro:am_michael@0.95", ["woolsthorpe|lincolnshire|grantham"], ["ulm"], "sorry"),
    ("cx10", "who founded Netflix, sorry, I mean Spotify", "say:Karen@175", ["ek|lorentzon"], ["hastings|randolph"], "sorry I mean"),
    ("cx11", "what's the boiling point of water, no wait, the freezing point", "kokoro:af_aoede@1.0", ["0|32|zero"], ["100|212"], "no wait (attribute)"),
    ("cx12", "which planet is closest to the sun, actually make that farthest", "kokoro:am_fenrir@1.0", ["neptune"], ["mercury"], "actually make that (adjective)"),
    ("cx13", "who painted Starry Night, sorry, The Scream", "say:Samantha@170", ["munch"], ["gogh"], "sorry"),
    ("cx14", "what's the national animal of Scotland, no wait, of Australia", "kokoro:af_heart@1.0", ["kangaroo|emu"], ["unicorn"], "no wait, partial restatement"),
    ("cx15", "how many legs does a spider have, I mean an insect", "say:Daniel@170", ["6|six"], ["8|eight"], "I mean"),
    ("cx16", "what language do they speak in Argentina, sorry, Brazil", "kokoro:am_michael@1.0", ["portuguese"], [], "sorry"),
    ("cx17", "who is the CEO of Microsoft, actually make that Apple", "kokoro:af_nova@1.0", ["cook"], ["nadella"], "actually make that"),
    ("cx18", "uh, what, what year did World War Two end", "say:Reed (English (US))@175", ["1945"], [], "restart"),
    ("cx19", "when did the Titanic sink, scratch that, when did the Hindenburg crash", "kokoro:af_bella@1.0", ["1937"], ["1912"], "scratch that, full restatement"),
    ("cx20", "what's the tallest mountain in Africa, I mean in South America", "kokoro:am_fenrir@0.95", ["aconcagua"], ["kilimanjaro"], "I mean, partial restatement"),
    ("cx21", "so who wrote, um, who wrote the Harry Potter books", "say:Karen@170", ["rowling"], [], "restart"),
    ("cx22", "what's the capital of Germany, I mean Austria", "kokoro:af_aoede@1.0", ["vienna"], ["berlin"], "I mean"),
]
# existing correction items (wavs in eval/audio): their "wrong" keywords, and expect for the two weather ones
OLD_WRONG = {"q26": ["shakespeare"], "q27": ["ottawa"], "q28": ["1998"], "q29": ["gates|allen"], "q30": ["salzburg"],
             "q31": ["seine"], "q32": ["yen"], "q33": ["napoleon|bonaparte"], "sc27": ["boston"],
             "sc28": ["3 teaspoons|three teaspoons"], "q50": ["london"]}
OLD_EXPECT = {"sc27": ["denver"], "q50": ["tokyo"]}

# live data: (id, text, voice, kind, args for the ground truth)
LD = [
    ("ld01", "what's the temperature in Denver right now", "kokoro:af_heart@1.0", "weather", {"city": "Denver"}),
    ("ld02", "how hot is it in Phoenix right now", "say:Samantha@175", "weather", {"city": "Phoenix"}),
    ("ld03", "what's the weather like in London right now", "kokoro:am_michael@1.0", "weather", {"city": "London"}),
    ("ld04", "how cold is it in Chicago today", "say:Daniel@175", "weather", {"city": "Chicago"}),
    ("ld05", "what's the temperature in Tokyo right now", "kokoro:af_nova@1.0", "weather", {"city": "Tokyo"}),
    ("ld06", "what's a hundred US dollars in euros", "kokoro:am_fenrir@1.0", "fx", {"amount": 100, "src": "USD", "dst": "EUR"}),
    ("ld07", "how much is fifty British pounds in US dollars", "say:Karen@175", "fx", {"amount": 50, "src": "GBP", "dst": "USD"}),
    ("ld08", "convert a thousand Japanese yen to dollars", "kokoro:af_bella@1.0", "fx", {"amount": 1000, "src": "JPY", "dst": "USD"}),
    ("ld09", "what's two hundred euros in Canadian dollars", "kokoro:af_aoede@1.0", "fx", {"amount": 200, "src": "EUR", "dst": "CAD"}),
    ("ld10", "what time is it in Tokyo right now", "say:Reed (English (US))@175", "time", {"tz": "Asia/Tokyo"}),
    ("ld11", "what time is it in London", "kokoro:am_michael@1.0", "time", {"tz": "Europe/London"}),
    ("ld12", "what's the time in Sydney right now", "kokoro:af_heart@1.0", "time", {"tz": "Australia/Sydney"}),
    ("ld13", "how many kilometers is twenty six miles", "say:Samantha@170", "units", {"value": 26 * 1.609344, "unit": "km"}),
    ("ld14", "what's seventy two fahrenheit in celsius", "kokoro:am_fenrir@1.0", "units", {"value": (72 - 32) * 5 / 9, "unit": "c"}),
    ("ld15", "how many ounces are in three pounds", "kokoro:af_nova@1.0", "units", {"value": 48, "unit": "oz"}),
    ("ld16", "how many cups are in a liter", "say:Daniel@170", "units", {"value": 1000 / 236.5882365, "unit": "cups", "alt": 4.0}),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_audio.AUDIO = OUT
    rows = [(i, t, v, "", [], "") for i, t, v, *_ in CX] + [(i, t, v, "", [], "") for i, t, v, *_ in LD]
    make_audio.synth(rows, set())
    old = {}
    for p in (ROOT / "eval" / "questions.jsonl", ROOT / "eval" / "showcase" / "questions.jsonl"):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["id"] in OLD_WRONG:
                old[r["id"]] = r
    out = []
    for qid in OLD_WRONG:
        r = dict(old[qid])
        src = ROOT / "eval" / "audio" / r["file"]
        dst = OUT / r["file"]
        if not dst.exists():
            os.symlink(os.path.relpath(src.resolve(), OUT), dst)
        r.update(kind="correction", wrong=OLD_WRONG[qid], expect=OLD_EXPECT.get(qid, r.get("expect") or []),
                 style="existing " + (r.get("notes") or ""))
        out.append(r)
    for qid, text, voice, expect, wrong, style in CX:
        out.append({"id": qid, "file": f"{qid}.wav", "text": text, "voice": voice, "kind": "correction",
                    "expect": expect, "wrong": wrong, "style": style})
    (HERE / "corrections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    (HERE / "livedata.jsonl").write_text("".join(json.dumps(
        {"id": qid, "file": f"{qid}.wav", "text": text, "voice": voice, "kind": "live_" + kind, "expect": [],
         "truth": args}) + "\n" for qid, text, voice, kind, args in LD))
    print(len(out), "corrections,", len(LD), "live-data questions")


if __name__ == "__main__":
    main()
