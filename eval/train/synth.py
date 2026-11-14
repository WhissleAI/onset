"""Synthetic spoken-question sessions for the failure classes users hit (topic switch, elliptical follow-up,
self-correction, open-mic chatter, multi-part, live data), with gold labels.

  .venv/bin/python eval/train/synth.py --cycle N [--sessions 24] [--paraphrase-usd 2.0] [--no-audio] [--noise]

1. Templates: eval/train/seeds/sessions.json (committed, small). Slots are filled at random (seeded by the cycle).
2. Gold: Wikidata (RELATIONS: SPARQL on the slot's QID -> answer labels, cached in runs/train/synth/wikidata.json).
   A turn without an answerable relation carries route / addressed / must_query / never labels only.
3. Paraphrases: Claude Haiku (config.ANSWER_MODEL) rewrites first turns as a person would say them, keeping the
   entity words. Real token usage x config.METER_MODEL_PRICES = measured $; stops at --paraphrase-usd per cycle.
4. Voicing: macOS `say` (16 voices, rate 150-215 wpm) -> 16 kHz wav; with --noise ~40 % get babble + pink noise at
   8-25 dB SNR (eval/make_variants.py's mixer, eval/audio_noisy's recipe). Each turn also gets Flux-style word
   timings (start / end per word from syllable counts at the voice's rate, the shape of Flux TurnInfo words) for the
   text-level replays; the director's traces come from real Flux on the wav (eval/train/director_step.py).
Writes runs/train/synth/c<N>/sessions.jsonl + clips.jsonl (+ audio/, deleted once featurized). Never committed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

C.sys_path()

SEEDS = Path(__file__).resolve().parent / "seeds" / "sessions.json"
SYN = C.TRAIN / "synth"
WD_CACHE = SYN / "wikidata.json"
SPARQL = "https://query.wikidata.org/sparql"
# relation -> SPARQL body over wd:{q}; ?a = the answer item (labels in English)
RELATIONS = {
    "capital": "wd:{q} wdt:P36 ?a .",
    "head_of_state": "wd:{q} p:P35 ?s . ?s ps:P35 ?a . FILTER NOT EXISTS {{ ?s pq:P582 ?e }}",
    "prev_head_of_state": "wd:{q} p:P35 ?s . ?s ps:P35 ?a . ?s pq:P582 ?end .",
    "founder": "wd:{q} wdt:P112 ?a .",
    "ceo": "wd:{q} p:P169 ?s . ?s ps:P169 ?a . FILTER NOT EXISTS {{ ?s pq:P582 ?e }}",
    "director": "wd:{q} wdt:P57 ?a .",
    "screenwriter": "wd:{q} wdt:P58 ?a .",
    "author": "wd:{q} wdt:P50 ?a .",
}
ORDER = {"prev_head_of_state": "ORDER BY DESC(?end) LIMIT 1"}


# ------------------------------------------------------------------------------------------------ gold (Wikidata)
def wikidata(rel: str, qid: str, cache: dict, fetch=None) -> list[str]:
    key = f"{rel}:{qid}"
    if key in cache:
        return cache[key]
    q = ("SELECT DISTINCT ?aLabel WHERE { " + RELATIONS[rel].format(q=qid) +
         " SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\". } } " + ORDER.get(rel, "LIMIT 6"))
    try:
        if fetch is None:
            import httpx
            from server import config
            r = httpx.get(SPARQL, params={"query": q, "format": "json"}, timeout=20,
                          headers={"User-Agent": config.ROUTER_UA, "Accept": "application/sparql-results+json"})
            r.raise_for_status()
            rows = r.json()["results"]["bindings"]
        else:
            rows = fetch(q)
        out = [b["aLabel"]["value"] for b in rows if not re.fullmatch(r"Q\d+", b["aLabel"]["value"])]
    except Exception:
        return []                      # not cached: retried next cycle
    cache[key] = out
    if fetch is None:
        time.sleep(0.3)                # polite to the public endpoint
    return out


PEOPLE = {"head_of_state", "prev_head_of_state", "founder", "ceo", "director", "screenwriter", "author"}


def expect_of(labels: list[str], rel: str = "") -> list[str]:
    """Keyword gold in run_ab.score's format, alternatives joined by '|': a person's surname (last word), anything
    else its whole label ("mexico city", not "city")."""
    clean = [re.sub(r"[^\w\- ]", "", lab).lower().split() for lab in labels if lab.strip()]
    keys = sorted({(w[-1] if rel in PEOPLE else " ".join(w)) for w in clean if w})
    return ["|".join(keys)] if keys else []


# ------------------------------------------------------------------------------------------------ paraphrase (Haiku)
class Paraphraser:
    def __init__(self, budget_usd: float):
        from server import config
        self.model = config.ANSWER_MODEL
        self.price = config.METER_MODEL_PRICES.get(self.model, (1.0, 5.0))
        self.budget = budget_usd
        self.usd = 0.0
        self.calls = 0
        self.tok_in = self.tok_out = 0
        self._client = None

    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def __call__(self, text: str, keep: list[str], n: int = 2) -> list[str]:
        if self.usd >= self.budget:
            return []
        prompt = (f"Rewrite this voice-assistant question {n} ways, the way different people would say it out loud "
                  f"(casual, with a filler word or a restart is fine). Keep these words exactly: {', '.join(keep) or '-'}. "
                  f"One per line, no numbering, no quotes.\n\nQuestion: {text}")
        try:
            r = self.client().messages.create(model=self.model, max_tokens=120,
                                              messages=[{"role": "user", "content": prompt}])
        except Exception:
            self.budget = 0.0           # no key / offline: stop trying this cycle
            return []
        self.calls += 1
        self.tok_in += r.usage.input_tokens
        self.tok_out += r.usage.output_tokens
        self.usd = (self.tok_in * self.price[0] + self.tok_out * self.price[1]) / 1e6
        lines = [ln.strip(" -•\"'") for ln in "".join(b.text for b in r.content if hasattr(b, "text")).splitlines()]
        return [ln for ln in lines if ln and all(k.lower() in ln.lower() for k in keep)][:n]

    def report(self) -> dict:
        return {"model": self.model, "calls": self.calls, "in": self.tok_in, "out": self.tok_out,
                "usd": round(self.usd, 5), "budget_usd": self.budget}


# ------------------------------------------------------------------------------------------------ timings + audio
def _syll(w: str) -> int:
    return max(1, len(re.findall(r"[aeiouy]+", w.lower())))


def flux_like_words(text: str, wpm: int, t0: float = 0.4, rng: random.Random | None = None) -> list[dict]:
    """Per-word start / end the way Flux reports them (seconds from audio start), from syllables at the voice's rate:
    the text-level stand-in when a turn has no real Flux trace."""
    rng = rng or random.Random(0)
    words = [w for w in re.sub(r"[^\w' ]", " ", text).split() if w]
    if not words:
        return []
    tot = sum(_syll(w) for w in words)
    per_syl = (len(words) / (wpm / 60.0)) / tot
    t, out = t0, []
    for w in words:
        d = _syll(w) * per_syl * rng.uniform(0.85, 1.15)
        out.append({"word": w.lower(), "start": round(t, 3), "end": round(t + d, 3), "confidence": 0.9})
        t += d + rng.uniform(0.02, 0.09)
    return out


def voice_wav(text: str, voice: str, wpm: int, out: Path, noise: dict | None, pool: list) -> dict:
    import numpy as np
    import soundfile as sf
    from make_audio import finish
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "x.aiff"
        subprocess.run(["say", "-v", voice, "-r", str(wpm), "-o", str(aiff), text], check=True, timeout=60)
        x, sr = sf.read(str(aiff), dtype="float32")
    finish(x, sr, out)
    meta = {}
    if noise:
        import make_variants as MV
        y, _ = sf.read(str(out), dtype="float32")
        rng = np.random.default_rng(noise["seed"])
        n = len(y) + int(MV.NOISE_TAIL_S * 16000)
        base = MV.pink(n, rng) * MV.PINK_SHARE ** 0.5
        if len(pool) >= MV.N_BABBLE + 1:
            base = base + MV.babble(n, "", pool, rng) * (1 - MV.PINK_SHARE) ** 0.5
        ps = MV.speech_power(y)
        g = np.sqrt(ps / (np.mean(base ** 2) + 1e-12) / (10 ** (noise["snr"] / 10)))
        z = np.concatenate([y, np.zeros(n - len(y), np.float32)]) + g * base
        z = z / (np.max(np.abs(z)) + 1e-9) * 0.8
        sf.write(str(out), z.astype(np.float32), 16000, subtype="PCM_16")
        meta = {"snr": noise["snr"], "noise": "babble+pink"}
    return meta


# ------------------------------------------------------------------------------------------------ sessions
def _fill(s: str, slots: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: str(slots.get(m.group(1), m.group(0))), s)


def pick_slots(seed: dict, rng: random.Random) -> tuple[dict, dict]:
    """Slot words + their QIDs (two distinct values per kind: {film} and {film2})."""
    words, qids = {}, {}
    for kind, vals in seed["slots"].items():
        a, b = rng.sample(vals, 2)
        for name, v in ((kind, a), (kind + "2", b)):
            if isinstance(v, list):
                words[name], qids[name] = v[0], v[1]
            else:
                words[name] = v
    return words, qids


def build_sessions(seed: dict, n: int, rng: random.Random, cache: dict, para=None, fetch=None) -> list[dict]:
    out = []
    tmpl = seed["sessions"]
    for i in range(n):
        t = tmpl[i % len(tmpl)] if i < len(tmpl) else rng.choice(tmpl)
        words, qids = pick_slots(seed, rng)
        sess = {"sid": f"s{i:03d}", "cls": t["cls"], "turns": []}
        for k, tu in enumerate(t["turns"]):
            canon = _fill(tu["say"], words)
            g = {"route": tu.get("route"), "addressed": tu.get("addressed", True),
                 "must_query": [_fill(x, words) for x in tu.get("must_query", [])],
                 "never": [_fill(x, words) for x in tu.get("never", [])], "followup": bool(tu.get("followup"))}
            ans = []
            for rk, ok in (("rel", "of"), ("rel2", "of2")):
                if tu.get(rk) and qids.get(tu.get(ok)):
                    labs = wikidata(tu[rk], qids[tu[ok]], cache, fetch)
                    ans.append({"rel": tu[rk], "of": words[tu[ok]], "qid": qids[tu[ok]], "labels": labs})
            g["answers"] = ans
            g["expect"] = [e for a in ans for e in expect_of(a["labels"], a["rel"])]
            if any(a["rel"] and not a["labels"] for a in ans):
                g["gold_missing"] = True
            said = canon
            if para is not None and k == 0 and t["cls"] not in ("open_mic", "self_correction") and rng.random() < 0.5:
                alts = para(canon, g["must_query"])
                if alts:
                    said = rng.choice(alts)
            sess["turns"].append({"tid": f"{sess['sid']}-{k}", "text": said, "canon": canon, "gold": g})
        out.append(sess)
    return out


STYLE = {"topic_switch": "plain", "elliptical": "followup", "self_correction": "correction", "open_mic": "not_search",
         "multi_part": "multi_part", "live_data": "plain"}


def clips_of(sessions: list[dict], cycle: int) -> list[dict]:
    """One clip per turn in the director recipe's manifest shape (split train: the fixed held-out sets never grow)."""
    out = []
    for s in sessions:
        prev = None
        for t in s["turns"]:
            style = STYLE[s["cls"]]
            if s["cls"] == "open_mic" and t["gold"]["addressed"]:
                style = "plain"
            cid = f"tc{cycle}-{t['tid']}"
            out.append({"id": cid, "file": f"{cid}.wav", "text": t["text"], "kind": s["cls"], "style": style,
                        "followup_of": prev if t["gold"].get("followup") else None, "split": "train",
                        "voice": t.get("voice"), "rate": t.get("wpm"), "cond": t.get("cond") or {"clean": True}})
            prev = cid
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", type=int, required=True)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--paraphrase-usd", type=float, default=2.0)
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--noise", action="store_true")
    a = ap.parse_args(argv)
    rng = random.Random(1000 + a.cycle)
    seed = json.loads(SEEDS.read_text())
    cache = C.read_json(WD_CACHE, {})
    para = None if a.no_paraphrase else Paraphraser(a.paraphrase_usd)
    sessions = build_sessions(seed, a.sessions, rng, cache, para)
    C.write_json(WD_CACHE, cache)
    d = SYN / f"c{a.cycle}"
    (d / "audio").mkdir(parents=True, exist_ok=True)
    pool = []
    voiced = 0
    for s in sessions:
        voice = rng.choice(seed["voices"])          # one speaker per session
        for t in s["turns"]:
            wpm = rng.randint(150, 215)
            t.update(voice=f"say:{voice}", wpm=wpm, words=flux_like_words(t["text"], wpm, rng=rng))
            if a.no_audio:
                continue
            cid = f"tc{a.cycle}-{t['tid']}"
            noise = {"snr": rng.choice([8, 12, 16, 20, 25]), "seed": rng.randrange(10 ** 6)} \
                if a.noise and rng.random() < 0.4 else None
            try:
                t["cond"] = voice_wav(t["text"], voice, wpm, d / "audio" / f"{cid}.wav", noise, pool) or {"clean": True}
                voiced += 1
                if len(pool) < 12 and not noise:
                    import soundfile as sf
                    pool.append((cid, sf.read(str(d / "audio" / f"{cid}.wav"), dtype="float32")[0]))
            except (subprocess.SubprocessError, OSError) as e:
                t["voice_error"] = str(e)[:120]
    C.write_json(d / "sessions.json", sessions)
    clips = clips_of(sessions, a.cycle)
    (d / "clips.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips))
    by = {}
    for s in sessions:
        by[s["cls"]] = by.get(s["cls"], 0) + 1
    rep = {"cycle": a.cycle, "sessions": len(sessions), "turns": sum(len(s["turns"]) for s in sessions),
           "voiced": voiced, "by_class": by,
           "gold_answers": sum(1 for s in sessions for t in s["turns"] if t["gold"]["expect"]),
           "gold_missing": sum(1 for s in sessions for t in s["turns"] if t["gold"].get("gold_missing")),
           "paraphrase": para.report() if para else None}
    C.write_json(d / "report.json", rep)
    print(json.dumps(rep))
    return rep


if __name__ == "__main__":
    main()
