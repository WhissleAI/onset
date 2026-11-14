"""Spoken multi-turn conversations (follow-ups use pronouns, references and ellipsis) -> eval/audio/conv_*.wav +
eval/conversations.jsonl. Existing wavs are kept (re-run safe); conv01-06 are the original two-turn set.
expect: per turn, a list of keys that must all appear in the answer ("a|b" = either); [] = not scored."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, soundfile as sf, librosa
ROOT = Path(__file__).resolve().parent
CONVS = [
    ("conv01", "Samantha", ["Who founded the company Nvidia?", "When was it founded?"], [["jensen", "huang"], ["1993"]]),
    ("conv02", "Daniel", ["Who painted the Mona Lisa?", "Which museum is it in?"], [["leonardo"], ["louvre"]]),
    ("conv03", "Samantha", ["Who wrote Pride and Prejudice?", "When was she born?"], [["austen"], ["1775"]]),
    ("conv04", "Daniel", ["What is the capital of Australia?", "What is the population there?"], [["canberra"], []]),
    ("conv05", "Samantha", ["Who developed the theory of general relativity?", "Where was he born?"], [["einstein"], ["ulm"]]),
    ("conv06", "Daniel", ["Which company makes the PlayStation?", "Where is its headquarters?"], [["sony"], ["tokyo|san mateo"]]),
    # three-turn: multi-turn references, ellipsis, kinds, user facts
    ("conv07", "Karen", ["Who founded Nvidia?", "And AMD?", "Going back to the first one, when was it founded?"],
     [["huang"], ["sanders"], ["1993"]]),
    ("conv08", "Fred", ["What is the capital of France?", "What is the population there?", "And Germany?"],
     [["paris"], ["million"], ["million"]]),
    ("conv09", "Moira", ["Who wrote Hamlet?", "When was he born?", "Where did he die?"],
     [["shakespeare"], ["1564"], ["stratford"]]),
    ("conv10", "Samantha", ["Which company makes the iPhone?", "Who is its CEO?", "Where is its headquarters?"],
     [["apple"], ["cook"], ["cupertino"]]),
    ("conv11", "Daniel", ["Who discovered penicillin?", "Where was he born?", "When did he die?"],
     [["fleming"], ["darvel|ayrshire|scotland"], ["1955"]]),
    ("conv12", "Tessa", ["What is the tallest mountain in the world?", "How tall is it?", "Which country is it in?"],
     [["everest"], ["8,8|8848|8849|29,0|29,03"], ["nepal|china"]]),
    ("conv13", "Rishi", ["Who painted The Starry Night?", "Where is it displayed?", "When did he die?"],
     [["gogh"], ["museum of modern art|moma|new york"], ["1890"]]),
    ("conv14", "Karen", ["What is the capital of Japan?", "What's the weather there tomorrow?", "And in Osaka?"],
     [["tokyo"], ["tokyo"], ["osaka"]]),
    ("conv15", "Fred", ["Who directed Jurassic Park?", "When was he born?", "What other movies did he make?"],
     [["spielberg"], ["1946"], ["jaws|schindler|e.t|indiana|saving private"]]),
    ("conv16", "Moira", ["Who invented the telephone?", "Where was he born?", "And when did he die?"],
     [["bell"], ["edinburgh"], ["1922"]]),
    ("conv17", "Samantha", ["Where is the Eiffel Tower?", "How tall is it?", "Who designed it?"],
     [["paris"], ["330|324|300|1,0"], ["eiffel|koechlin|nouguier|sauvestre"]]),
    ("conv18", "Daniel", ["Who founded Microsoft?", "And Apple?", "When was that company founded?"],
     [["gates|allen"], ["jobs|wozniak"], ["1976"]]),
    ("conv19", "Tessa", ["I live in Denver.", "What's the weather tomorrow?", "What time is sunset?"],
     [["denver"], ["denver"], ["denver"]]),
    ("conv20", "Rishi", ["What is the weather in Chicago today?", "What about tomorrow?", "How cold does it get there in winter?"],
     [["chicago"], ["chicago"], []]),
    ("conv21", "Karen", ["Who was Marie Curie?", "Who was her husband?", "Where was she born?"],
     [["physicist|chemist|radioactiv"], ["pierre"], ["warsaw"]]),
]


def say(text, voice):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(p), text], check=True)
        w, sr = sf.read(str(p), dtype="float32")
    w = w.mean(1) if w.ndim > 1 else w
    return librosa.resample(w, orig_sr=sr, target_sr=16000)


if __name__ == "__main__":
    force = "--force" in sys.argv
    rows = []
    for cid, voice, turns, expect in CONVS:
        files = []
        for i, t in enumerate(turns):
            f = f"{cid}_{i+1}.wav"
            if force or not (ROOT / "audio" / f).exists():
                w = say(t, voice)
                w = np.concatenate([np.zeros(6400, np.float32), w, np.zeros(4800, np.float32)])
                sf.write(ROOT / "audio" / f, w, 16000, subtype="PCM_16")
            files.append(f)
        rows.append({"id": cid, "voice": voice, "turns": turns, "files": files, "expect": expect, "kind": "conversation"})
    (ROOT / "conversations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(len(rows), "conversations,", sum(len(r["turns"]) - 1 for r in rows), "follow-up turns")
