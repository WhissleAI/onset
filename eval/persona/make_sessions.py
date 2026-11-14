"""Scripted multi-turn sessions where personalization SHOULD change the result or the style -> eval/persona/audio/*.wav
(git-ignored) + eval/persona/sessions.jsonl. macOS `say` voices; rate (words / min) and [[slnc ms]] pauses make the
"how it was said" signals (quick vs hesitant). Re-run safe (existing wavs kept; --force rewrites).

Per turn:
  expect  keys that must all appear in the answer ("a|b" = either); [] = not scored for content
  sense   {"want": regex, "not": regex} on the answer: which meaning of an ambiguous name the user means (from history)
  style   what a good answer to THIS asker looks like: short | fuller | fix | more | units:imperial | "" (no change)
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf, librosa

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "audio"
P = "[[slnc 450]]"
CAR = r"car|xj|xk|f-type|e-type|vehicle|engine|model|coupe|sedan|automobile|mph|km/h"
CAT = r"\bcat\b|feline|predator|panthera|big cat|prey|animal"
S = [
    # ambiguous names resolved by what the user asked before (interest rerank + sense search)
    ("p01", "Samantha", [
        ("How much horsepower does a Porsche 911 have?", 185, ["horsepower|hp|kw"], None, ""),
        ("What is the top speed of a Ford Mustang?", 185, ["mph|km/h|kilomet|miles"], None, ""),
        ("How fast is a Jaguar?", 185, [], {"want": r"xj|xk|f-type|e-type|jaguar cars|sports car|car\b|vehicle", "not": r"\bcat\b|feline|panthera|big cat|predator", "src": r"^jaguar (cars|x|f-|e-|s-|i-|mark|land rover)"}, ""),
    ]),
    ("p02", "Daniel", [
        ("How fast can a cheetah run?", 185, ["km/h|mph|kilomet|miles"], None, ""),
        ("Where do lions live?", 185, ["africa"], None, ""),
        ("How fast is a Jaguar?", 185, [], {"want": CAT, "not": r"xj|xk|f-type|e-type|jaguar cars|horsepower|engine", "src": r"^jaguar( ·|$)"}, ""),
    ]),
    ("p03", "Karen", [
        ("How big is Jupiter?", 185, ["earth|km|kilomet|miles|diameter|radius|mass"], None, ""),
        ("How far is Venus from the Sun?", 185, ["million|km|miles|au"], None, ""),
        ("How hot is Mercury?", 185, [], {"want": r"planet|surface|sun|day|430|800|°", "not": r"element|liquid metal|thermometer|melting|boiling|toxic", "src": r"^mercury \(planet\)"}, ""),
    ]),
    ("p04", "Fred", [
        ("What is the melting point of gold?", 185, ["1,064|1064|1,063|1063|1948|1,948"], None, ""),
        ("Is lead toxic?", 185, ["yes|toxic|poison"], None, ""),
        ("What is Mercury used for?", 185, [], {"want": r"thermometer|barometer|lamp|amalgam|element|metal|batter|switch", "not": r"planet|orbit|the sun|spacecraft", "src": r"^mercury \(element\)|amalgam"}, ""),
    ]),
    ("p05", "Moira", [
        ("How long can an anaconda grow?", 185, ["m|feet|metre|meter|ft"], None, ""),
        ("Are boa constrictors venomous?", 185, ["no|not|non"], None, ""),
        ("How long can a python grow?", 185, [], {"want": r"snake|metre|meter|feet|ft|reticulated|\bm\b|long", "not": r"programming|guido|language|software|code", "src": r"python(idae)?\b(?!.*programming)"}, ""),
    ]),
    ("p06", "Tessa", [
        ("How long is the Nile?", 185, ["6,6|6650|6,650|4,1|4130|4,130|km|miles"], None, ""),
        ("Where is the Congo River?", 185, ["africa"], None, ""),
        ("How long is the Amazon?", 185, [], {"want": r"river|6,4|6,9|6,5|7,0|km|kilomet|miles", "not": r"company|bezos|retail|seattle|founded", "src": r"^amazon river"}, ""),
    ]),
    ("p13", "Rishi", [
        ("When was Microsoft founded?", 185, ["1975"], None, ""),
        ("Who is the CEO of Google?", 185, ["pichai"], None, ""),
        ("How big is Amazon?", 185, [], {"want": r"employ|revenue|billion|compan|market|retail", "not": r"rainforest|river|square kilomet|km2|km²|basin", "src": r"^amazon \(company\)|amazon\.com"}, ""),
    ]),
    ("p14", "Fred", [
        ("How much does a Tesla Model 3 cost?", 185, [], None, ""),
        ("What engine does a Chevrolet Corvette have?", 185, ["v8|engine|liter|litre"], None, ""),
        ("How fast is a Mustang?", 185, [], {"want": r"ford|car\b|engine|horsepower|hp\b|coupe|v8", "not": r"horse(?!power)|feral|herd|gallop", "src": r"^ford mustang"}, ""),
    ]),
    ("p15", "Tessa", [
        ("Where do mangoes grow?", 185, ["india|tropic|asia"], None, ""),
        ("How many calories are in a banana?", 185, ["calor|kcal"], None, ""),
        ("Where do apples come from?", 185, [], {"want": r"central asia|kazakhstan|tree|fruit|malus|orchard", "not": r"cupertino|iphone|jobs|wozniak|company", "src": r"^apple( ·|$)|malus"}, ""),
    ]),
    # length / tone / units matched to how the user asks (style hint)
    ("p07", "Daniel", [
        ("Boiling point of ethanol?", 270, ["78"], None, "short"),
        ("Speed of light in a vacuum?", 270, ["299|300,000|186"], None, "short"),
        ("Atomic number of tungsten?", 270, ["74"], None, "short"),
    ]),
    ("p08", "Samantha", [
        (f"Um, {P} so, {P} what is, uh, {P} photosynthesis?", 150, ["light|sun"], None, "fuller"),
        (f"And, um, {P} what does chlorophyll, {P} uh, do exactly?", 150, ["light|absorb"], None, "fuller"),
    ]),
    ("p09", "Karen", [
        ("Who wrote the novel Dracula?", 185, ["stoker"], None, ""),
        ("No, that's not what I meant. Who wrote Frankenstein?", 185, ["shelley"], None, "fix"),
    ]),
    ("p10", "Fred", [
        ("What is a black hole?", 185, ["gravity|light|spacetime|space"], None, ""),
        ("Tell me more about how they form.", 185, ["star|collapse"], None, "more"),
    ]),
    ("p11", "Tessa", [
        ("How tall is Mount Everest in feet?", 185, ["29,0|29,03|29,02|feet"], None, "units:imperial"),
        ("How tall is K2?", 185, ["28,2|28,25|feet|ft"], None, "units:imperial"),
    ]),
    ("p12", "Moira", [
        ("Just give me the short answer: who painted the Mona Lisa?", 185, ["leonardo"], None, "short"),
        ("Where is it displayed?", 185, ["louvre"], None, "short"),
        ("Who designed the Eiffel Tower?", 185, ["eiffel|koechlin|nouguier|sauvestre"], None, "short"),
    ]),
]


def say(text, voice, rate):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(p), text], check=True)
        w, sr = sf.read(str(p), dtype="float32")
    w = w.mean(1) if w.ndim > 1 else w
    return librosa.resample(w, orig_sr=sr, target_sr=16000)


if __name__ == "__main__":
    force = "--force" in sys.argv
    AUDIO.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid, voice, turns in S:
        files = []
        for i, (text, rate, *_rest) in enumerate(turns, 1):
            fn = f"{sid}_{i}.wav"
            if force or not (AUDIO / fn).exists():
                w = say(text, voice, rate)
                sf.write(AUDIO / fn, np.concatenate([np.zeros(4000, np.float32), w, np.zeros(8000, np.float32)]), 16000)
            files.append(fn)
        rows.append({"id": sid, "voice": voice, "files": files,
                     "turns": [{"text": t.replace(P, "").replace("  ", " "), "rate": r, "expect": e, "sense": s, "style": st}
                               for t, r, e, s, st in turns]})
    (HERE / "sessions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} sessions, {sum(len(r['files']) for r in rows)} turns -> {HERE / 'sessions.jsonl'}")
