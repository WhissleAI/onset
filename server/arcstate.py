"""Arc state carried across push-to-talk presses (ARC_STATE=1).

Every press builds a new ear, a new ArcMeta stream and a new Session (server/app.py Run), and the tagger resets per
question, so Arc's acoustic read-out starts cold on every question. Text memory (server/memory.py) already carries
entities across turns; this module carries the ACOUSTIC side, one ArcState per conversation (one WebSocket
connection in the app), handed to every Run, its ear and its Sessions:

  voice       who is speaking. Arc Mini's diarization head has a voice embedding (DiarHeads.embed, the input of its
              "voice" TrackBank). Arc's streams can't carry hidden state across streams, so ArcMeta taps the per-frame
              embedding (diarization="voice"; the TrackBank itself is skipped) and we carry a centroid per voice here.
              The first voice that asks is the device user ("you", S1); a later press whose mean embedding is below
              VOICE_SAME cosine of every known voice is a new voice (flagged "another voice").
  baselines   per voice, running mean / spread of pace (syllables / s), loudness (dB), pitch (median f0, semitones) and
              hesitation (fillers + long pauses per word). relative() turns this question into "vs how this voice
              usually sounds" (persona style hints, director features).
  trajectory  the last few intents / dialogue acts and Arc's utterance emotion / arousal (when the stream gives them).
  keyterms    entity names from conversation memory, sent to Deepgram Flux as `keyterm` prompts on the next press
              (ear.flux_url), so names from earlier turns are heard right.

Reset: reset("all") ("Reset voice" in the memory panel, and "forget" clears it with memory); reset("topic") on a clear
topic switch (a new question that is not a follow-up and names a new subject) drops the trajectory and narrows the
keyterms to the new subject; idle for ARC_STATE_IDLE_S: the trajectory is dropped and the baselines' weight is cut,
the enrolled voice stays (it is this Mac's user). Persisted to <data dir>/arcstate/ only with ARC_STATE_PERSIST=1
(a voice embedding is biometric: default per-session, in memory only).

Everything is best effort: no method raises into the caller, and with ARC_STATE=0 nothing here runs."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from . import config

ENABLED = os.environ.get("ARC_STATE", "1") == "1"      # measured: PROGRESS.md "ArcState"
VOICE = os.environ.get("ARC_STATE_VOICE", "1") == "1"              # speaker enrollment (Arc voice embedding tap)
KEYTERMS = os.environ.get("ARC_STATE_KEYTERMS", "1") == "1"        # memory entities -> Flux keyterm prompts
MAX_KEYTERMS = int(os.environ.get("ARC_STATE_MAX_KEYTERMS", "12"))
IDLE_S = float(os.environ.get("ARC_STATE_IDLE_S", "600"))          # long idle: drop the trajectory, soften baselines
PERSIST = os.environ.get("ARC_STATE_PERSIST", "0") == "1"          # opt-in: keep across app restarts (data dir)
VOICE_SAME = float(os.environ.get("ARC_STATE_VOICE_SAME", "0.40"))  # centred cosine: same voice (eval/arcstate_eval.py)
MIN_FRAMES = int(os.environ.get("ARC_STATE_MIN_FRAMES", "6"))      # speech frames (100 ms) for a voice decision
MAX_VOICES = 4
TRAJ = 6                                                            # intents / acts / emotions kept

FILLERS = {"um", "uh", "er", "erm", "hmm", "uhm", "ah", "mm", "umm", "uhh"}
_VOWELS = re.compile(r"[aeiouy]+")


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", (w or "").lower())


def syllables(w: str) -> int:
    w = _norm(w)
    if not w:
        return 0
    if w.isdigit():
        return max(1, len(w))
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


# --------------------------------------------------------------------------------------------- measurements
def pitch_st(pcm: np.ndarray | None, sr: int = 16000) -> float | None:
    """Median f0 of the voiced 40 ms frames, in semitones re 100 Hz (autocorrelation; ~2 ms per second of audio)."""
    if pcm is None or len(pcm) < sr // 4:
        return None
    x = np.asarray(pcm, np.float32)
    n, hop = 640, 320
    k = (len(x) - n) // hop + 1
    if k < 3:
        return None
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    rms = np.sqrt((fr * fr).mean(1))
    keep = rms > max(0.01, 0.3 * float(np.percentile(rms, 90)))
    fr = fr[keep]
    if len(fr) < 3:
        return None
    spec = np.fft.rfft(fr * np.hanning(n), 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:, :n]
    lo, hi = sr // 400, sr // 60                          # 60..400 Hz
    seg = ac[:, lo:hi]
    lag = seg.argmax(1) + lo
    strength = seg.max(1) / np.maximum(ac[:, 0], 1e-9)
    f0 = sr / lag[strength > 0.3]
    if len(f0) < 3:
        return None
    return round(12 * math.log2(float(np.median(f0)) / 100.0), 2)


def _voiced_rms(pcm) -> float | None:
    """Median RMS of the question's voiced 100 ms frames (the session's own level samples start only once words have
    arrived, so on a short question they are mostly the quiet tail: measured -10 dB off on equal-level files)."""
    if pcm is None or len(pcm) < 3200:
        return None
    x = np.asarray(pcm, np.float32)
    fr = x[: len(x) // 1600 * 1600].reshape(-1, 1600)
    r = np.sqrt((fr * fr).mean(1))
    v = r[r > max(0.006, 0.25 * float(r.max()))]
    return float(np.median(v)) if len(v) >= 3 else None


def measure(words: list[str], times: list[tuple] | None, levels: list[float] | None, pcm=None) -> dict:
    """How one question was said: pace (syllables / s), level (dB of the voiced RMS), pitch (st), hes (per word)."""
    toks = [_norm(w) for w in words]
    toks = [t for t in toks if t]
    m = {"n": len(toks), "pace": None, "level": None, "pitch": None, "hes": None}
    tw = [(t, s, e) for t, (s, e) in zip(toks, times or []) if s is not None and e is not None] if times else []
    if len(tw) >= 3:
        dur = float(tw[-1][2]) - float(tw[0][1])
        if dur > 0.3:
            m["pace"] = round(sum(syllables(t) for t, _, _ in tw if t not in FILLERS) / dur, 2)
        pauses = sum(1 for a, b in zip(tw, tw[1:]) if float(b[1]) - float(a[2]) >= 0.45)
        m["hes"] = round((sum(1 for t in toks if t in FILLERS) + pauses) / max(1, len(toks)), 3)
    elif toks:
        m["hes"] = round(sum(1 for t in toks if t in FILLERS) / len(toks), 3)
    lv = _voiced_rms(pcm)
    if lv is None and levels:
        lv = float(np.median(levels))
    if lv is not None:
        m["level"] = round(20 * math.log10(max(lv, 1e-5)), 1)
    m["pitch"] = pitch_st(pcm)
    return m


class Baseline:
    """Running mean / spread of each measure (exponential, ~8 questions of memory)."""
    KEYS = ("pace", "level", "pitch", "hes")
    ALPHA = 0.25

    def __init__(self):
        self.mean: dict[str, float] = {}
        self.var: dict[str, float] = {}
        self.n: dict[str, float] = {}

    def add(self, m: dict):
        for k in self.KEYS:
            v = m.get(k)
            if v is None:
                continue
            v = float(v)
            if k not in self.mean:
                self.mean[k], self.var[k], self.n[k] = v, 0.0, 1.0
                continue
            a = max(self.ALPHA, 1.0 / (self.n[k] + 1))
            d = v - self.mean[k]
            self.mean[k] += a * d
            self.var[k] = (1 - a) * (self.var[k] + a * d * d)
            self.n[k] += 1

    def soften(self, n_max: float = 2.0):
        for k in self.n:
            self.n[k] = min(self.n[k], n_max)

    def ready(self, k: str, n_min: int = 2) -> bool:
        return self.n.get(k, 0) >= n_min

    def to_json(self) -> dict:
        return {"mean": self.mean, "var": self.var, "n": self.n}

    @classmethod
    def from_json(cls, d: dict | None) -> "Baseline":
        b = cls()
        if d:
            b.mean, b.var, b.n = dict(d.get("mean") or {}), dict(d.get("var") or {}), dict(d.get("n") or {})
        return b


def relative_of(m: dict, base: Baseline | None) -> dict:
    """This question vs the voice's usual: ratios for pace / hesitation (persona.relative's contract), dB / semitone
    differences for level / pitch. Only measures with >= 2 earlier questions."""
    out = {}
    if base is None:
        return out
    if m.get("pace") and base.ready("pace") and base.mean["pace"] > 0:
        out["pace_rel"] = round(m["pace"] / base.mean["pace"], 2)
    if m.get("hes") is not None and base.ready("hes"):
        out["hes_rel"] = round((m["hes"] + 0.05) / (base.mean["hes"] + 0.05), 2)
    if m.get("level") is not None and base.ready("level"):
        out["level_rel_db"] = round(m["level"] - base.mean["level"], 1)
    if m.get("pitch") is not None and base.ready("pitch"):
        out["pitch_rel_st"] = round(m["pitch"] - base.mean["pitch"], 2)
    return out


def describe(rel: dict) -> list[str]:
    """Plain words for the read-out: only clear differences."""
    out = []
    pr = rel.get("pace_rel")
    if pr is not None and pr <= 0.8:
        out.append("slower than you usually speak")
    elif pr is not None and pr >= 1.25:
        out.append("faster than you usually speak")
    hr = rel.get("hes_rel")
    if hr is not None and hr >= 1.8:
        out.append("more hesitant than usual")
    lv = rel.get("level_rel_db")
    if lv is not None and lv <= -6:
        out.append("quieter than usual")
    elif lv is not None and lv >= 6:
        out.append("louder than usual")
    return out


# --------------------------------------------------------------------------------------------- the state
class Voice:
    def __init__(self, label: str, centroid: np.ndarray | None = None):
        self.label = label
        self.c = centroid
        self.n = 0
        self.base = Baseline()
        self.last = 0.0

    def to_json(self) -> dict:
        return {"label": self.label, "c": None if self.c is None else [round(float(x), 5) for x in self.c],
                "n": self.n, "base": self.base.to_json(), "last": self.last}

    @classmethod
    def from_json(cls, d: dict) -> "Voice":
        v = cls(d.get("label") or "S1", None if d.get("c") is None else np.asarray(d["c"], np.float32))
        v.n, v.base, v.last = int(d.get("n") or 0), Baseline.from_json(d.get("base")), float(d.get("last") or 0)
        return v


def _unit(x: np.ndarray) -> np.ndarray:
    return x / max(1e-6, float(np.linalg.norm(x)))


def _load_mu() -> np.ndarray | None:
    try:
        d = json.loads((Path(__file__).with_name("arcstate_voice_mu.json")).read_text())
        return np.asarray(d["mu"], np.float32)
    except Exception:  # noqa: BLE001
        return None


VOICE_MU = _load_mu()      # mean voice-embedding frame: centring it out took the same/other-voice EER from 14 % to 10 %


def utt_vector(frames: list[np.ndarray], mu: np.ndarray | None = None) -> np.ndarray:
    """One press's voice: frames centred on the mean frame, each normalised, averaged, normalised."""
    x = np.stack(frames).astype(np.float32)
    mu = VOICE_MU if mu is None else mu
    if mu is not None and mu.shape[-1] == x.shape[-1]:
        x = x - mu
    x = x / np.maximum(1e-6, np.linalg.norm(x, axis=1, keepdims=True))
    return _unit(x.mean(0))


class ArcState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames: collections.deque = collections.deque(maxlen=1500)   # (audio time s, embedding) speech frames
        self.emo_now: dict | None = None                                    # latest Arc utterance emotion / arousal
        self.voices: list[Voice] = []
        self.intents: collections.deque = collections.deque(maxlen=TRAJ)
        self.emotions: collections.deque = collections.deque(maxlen=TRAJ)
        self.acts: collections.deque = collections.deque(maxlen=TRAJ)
        self.subjects: collections.deque = collections.deque(maxlen=TRAJ)  # (turn, subject) of answered questions
        self.topic_from = 0                    # keyterms come from memory turns >= this one (topic switch narrows)
        self.turns = 0
        self.last_active = time.time()
        self.seen: set = set()                 # questions already observed (both A/B sessions close the same one)
        self.last: dict | None = None          # read-out of the last observed question
        self.resets: collections.Counter = collections.Counter()
        self.timing: list[float] = []          # ms per observe(), for the cost line in PROGRESS / health

    # ------------------------------------------------------------------ Arc thread (ArcMeta)
    def add_frames(self, items: list[tuple[float, np.ndarray]]):
        if not items:
            return
        with self.lock:
            self.frames.extend(items)

    def set_emotion(self, emo: dict | None):
        if emo:
            self.emo_now = dict(emo)

    def new_stream(self):
        """A new press: the audio clock restarts at 0, so earlier frames' times mean nothing any more."""
        with self.lock:
            self.frames.clear()
        self.emo_now = None

    # ------------------------------------------------------------------ lifecycle
    def touch(self, now: float | None = None) -> str | None:
        """Call at the start of a press. Long idle: drop the trajectory and soften baselines (the voice stays)."""
        now = time.time() if now is None else now
        why = None
        if IDLE_S > 0 and now - self.last_active > IDLE_S and (self.turns or self.voices):
            self.intents.clear()
            self.emotions.clear()
            self.subjects.clear()
            self.topic_from = 10 ** 9            # keyterms: only what is said from now on (memory has its own TTL)
            for v in self.voices:
                v.base.soften()
            self.resets["idle"] += 1
            why = "idle"
        self.last_active = now
        return why

    def reset(self, scope: str = "all"):
        if scope == "topic":
            self.intents.clear()
            self.emotions.clear()
            self.acts.clear()
            self.topic_from = self.turns
            self.resets["topic"] += 1
            return
        with self.lock:
            self.frames.clear()
        self.voices, self.last, self.emo_now = [], None, None
        self.intents.clear()
        self.emotions.clear()
        self.subjects.clear()
        self.topic_from, self.turns = 0, 0
        self.seen = set()
        self.resets["all"] += 1

    # ------------------------------------------------------------------ voice
    def utt_embedding(self, t0: float | None, t1: float | None) -> tuple[np.ndarray | None, int]:
        with self.lock:
            fr = list(self.frames)
        if not fr:
            return None, 0
        if t0 is not None and t1 is not None:
            sel = [e for t, e in fr if t0 - 0.2 <= t <= t1 + 0.3]
        else:
            sel = [e for _, e in fr[-40:]]
        if len(sel) < MIN_FRAMES:
            return None, len(sel)
        return utt_vector(sel), len(sel)

    def assign(self, e: np.ndarray | None, enroll: bool = True) -> dict:
        """Which known voice this is. enroll: add it (first voice = you; else a new voice up to MAX_VOICES)."""
        if e is None:
            return {"who": None, "sim": None, "voice": None}
        best, sim = None, -1.0
        for v in self.voices:
            if v.c is None:
                continue
            s = float(_unit(v.c) @ e)
            if s > sim:
                best, sim = v, s
        if best is not None and sim >= VOICE_SAME:
            if enroll:
                best.c = _unit(best.c * min(best.n, 8) + e)          # running centroid (recent presses weigh more)
                best.n += 1
        elif enroll and len(self.voices) < MAX_VOICES:
            best = Voice(f"S{len(self.voices) + 1}", e)
            best.n = 1
            self.voices.append(best)
        elif best is None:
            return {"who": None, "sim": None, "voice": None}
        you = best is self.voices[0] if self.voices else False
        return {"who": "you" if you else "other", "sim": None if sim < -0.5 else round(sim, 3), "voice": best.label}

    def voice(self, label: str | None) -> Voice | None:
        for v in self.voices:
            if v.label == label:
                return v
        return None

    def you(self) -> Voice | None:
        return self.voices[0] if self.voices else None

    # ------------------------------------------------------------------ per question
    def observe(self, key, words: list[str], times: list[tuple] | None, levels: list[float] | None, pcm=None,
                intent: str = "", acts=(), answered: bool = True) -> dict | None:
        """A question closed (Session._close_question). Measures it against the speaker's baseline, then folds it in.
        key: identifies the question across the A/B sessions (both close the same words): observed once."""
        t = time.perf_counter()
        if key in self.seen:
            return self.last
        self.seen.add(key)
        if len(self.seen) > 200:
            self.seen = set(list(self.seen)[-100:])
        st = [x[0] for x in (times or []) if x and x[0] is not None]
        en = [x[1] for x in (times or []) if x and x[1] is not None]
        e, nfr = self.utt_embedding(st[0] if st else None, en[-1] if en else None) if VOICE else (None, 0)
        who = self.assign(e)
        m = measure(words, times, levels, pcm)
        # no voice decision (Arc stream not up, too short): treat as the enrolled user if there is one
        v = self.voice(who["voice"]) if who["voice"] else None
        if v is None:
            v = self.you()
            if v is None:
                v = Voice("S1")
                self.voices.append(v)
            who = {"who": "you", "sim": None, "voice": v.label, "assumed": True}
        rel = relative_of(m, v.base)
        v.base.add(m)
        v.last = time.time()
        if intent:
            self.intents.append(intent.split(".")[-1])
        if acts:
            self.acts.append(",".join(str(a).split(".")[-1] for a in acts))
        emo = self.emo_now
        if emo:
            self.emotions.append({k: round(float(x), 2) for k, x in emo.items()})
        self.turns += 1
        self.last_active = time.time()
        self.last = {"who": who.get("who"), "voice": who.get("voice"), "sim": who.get("sim"), "frames": nfr,
                     "voices": len([x for x in self.voices if x.c is not None]), "m": m, "rel": rel,
                     "notes": describe(rel) if who.get("who") == "you" else [],
                     "intents": list(self.intents), "acts": list(self.acts), "emotion": emo, "turn": self.turns}
        self.timing.append((time.perf_counter() - t) * 1000)
        self.timing = self.timing[-200:]
        return self.last

    def note_turn(self, f, followup: bool, subject: str, key=None) -> bool:
        """An answered question. A new subject that is not a follow-up = topic switch: partial reset. Returns it.
        key: the question (both A/B sessions answer it): noted once."""
        if key is not None:
            if key == getattr(self, "_turn_key", None):
                return False
            self._turn_key = key
        prev = self.subjects[-1][1] if self.subjects else None
        switched = bool(prev and subject and not followup and subject != prev)
        if switched:
            self.reset("topic")
        if subject:
            self.subjects.append((self.turns, subject))
        return switched

    # ------------------------------------------------------------------ getters
    def relative_now(self, words: list[str], times: list[tuple] | None) -> dict | None:
        """The question being heard vs the enrolled voice's baseline (persona: pace_rel / hes_rel)."""
        v = self.you()
        if v is None:
            return None
        if self.last and self.last.get("who") == "other":
            v = self.voice(self.last.get("voice")) or v
        rel = relative_of(measure(words, times, None), v.base)
        return rel or None

    def keyterms(self, mem) -> list[str]:
        """Entity names from conversation memory for Flux keyterm prompting: names of the current topic first, newest
        first; single common words are skipped (a keyterm biases the ear towards that word everywhere)."""
        if mem is None:
            return []
        ents = list((getattr(mem, "ents", None) or {}).values())
        out, seen = [], set()

        def last_turn(e):
            ts = e.get("turns") or [0]
            return max(ts)
        ents.sort(key=last_turn, reverse=True)
        for e in ents:
            if last_turn(e) < self.topic_from - 1 and len(out) >= 3:
                continue
            name = re.sub(r"['’]s$", "", (e.get("name") or e.get("key") or "").strip())
            if not name or len(name) < 3 or len(name) > 40:
                continue
            k = name.lower()
            if k in seen:
                continue
            if " " not in name and (name.islower() or k in _COMMON):
                continue
            seen.add(k)
            out.append(name)
            if len(out) >= MAX_KEYTERMS:
                break
        return out

    def features(self) -> dict:
        """Numbers for the director (director.ARCSTATE_F) and the health read-out."""
        last = self.last or {}
        rel = last.get("rel") or {}
        return {"as_ok": 1.0 if self.voices else 0.0, "as_other": 1.0 if last.get("who") == "other" else 0.0,
                "as_sim": float(last.get("sim") or 0.0), "as_nvoices": min(len(self.voices), 4) / 4,
                "as_pace_rel": float(np.clip((rel.get("pace_rel") or 1.0) - 1.0, -1, 1)),
                "as_hes_rel": float(np.clip(((rel.get("hes_rel") or 1.0) - 1.0) / 2, -1, 1)),
                "as_level_rel": float(np.clip((rel.get("level_rel_db") or 0.0) / 12, -1, 1)),
                "as_turns": min(self.turns, 10) / 10}

    def style_hints(self) -> dict:
        """For the persona agent: who is asking and how they sound vs their usual (plain values, no prompt text)."""
        last = self.last or {}
        return {"speaker": last.get("who"), "voice": last.get("voice"), "relative": dict(last.get("rel") or {}),
                "notes": list(last.get("notes") or []), "recent_intents": list(self.intents), "recent_acts": list(self.acts),
                "emotion": (list(self.emotions) or [None])[-1]}

    def summary(self) -> dict:
        """The UI read-out (memory panel)."""
        last = self.last or {}
        n = len([v for v in self.voices if v.c is not None])
        return {"on": True, "voices": n, "enrolled": n > 0, "turns": self.turns, "who": last.get("who"),
                "notes": list(last.get("notes") or []), "persist": PERSIST}

    # ------------------------------------------------------------------ persistence (opt-in)
    def to_json(self) -> dict:
        return {"v": 1, "voices": [v.to_json() for v in self.voices], "turns": self.turns,
                "last_active": self.last_active}

    @classmethod
    def from_json(cls, d: dict | None) -> "ArcState":
        s = cls()
        if not d or d.get("v") != 1:
            return s
        s.voices = [Voice.from_json(v) for v in d.get("voices") or []][:MAX_VOICES]
        s.last_active = float(d.get("last_active") or time.time())
        return s


_COMMON = set("""the and what when where who how why this that apple time day week year city country company""".split())


def state_dir(data_dir: Path | None = None) -> Path:
    return Path(data_dir or config.DATA_DIR) / "arcstate"


def _path(client: str, data_dir: Path | None = None) -> Path:
    return state_dir(data_dir) / (hashlib.sha256(client.encode()).hexdigest()[:24] + ".json")


def load(client: str | None, data_dir: Path | None = None) -> ArcState:
    if not (PERSIST and client):
        return ArcState()
    try:
        return ArcState.from_json(json.loads(_path(client, data_dir).read_text()))
    except Exception:  # noqa: BLE001
        return ArcState()


def save(client: str | None, st: ArcState, data_dir: Path | None = None):
    if not (PERSIST and client) or st is None:
        return
    try:
        p = _path(client, data_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st.to_json()))
        tmp.replace(p)
    except Exception:  # noqa: BLE001
        pass


def forget(client: str | None, data_dir: Path | None = None):
    if not client:
        return
    try:
        _path(client, data_dir).unlink()
    except (FileNotFoundError, OSError):
        pass


# --------------------------------------------------------------------------------------------- module getters
def of(sess) -> ArcState | None:
    st = getattr(sess, "arcstate", None)
    return st if (ENABLED and st is not None) else None


def relative(sess) -> dict | None:
    """persona.relative's input for the question being heard: {"pace_rel", "hes_rel", ...} or None."""
    st = of(sess)
    if st is None:
        return None
    try:
        return st.relative_now(list(getattr(sess, "words", []) or []), list(getattr(sess, "times", []) or []))
    except Exception:  # noqa: BLE001
        return None


def style_hints(sess) -> dict | None:
    st = of(sess)
    return st.style_hints() if st is not None else None


# --------------------------------------------------------------------------------------------- Arc stream tap
# ArcMeta: Arc Mini's engine computes the voice embedding only with a TrackBank (diarization "fused" / "voice"); we
# ask for "voice" and replace the bank step with a tap that keeps the embedding of every speech frame (the bank's own
# tracking is not run: its labels only feed the offline diarization output, which ArcMeta never reads).
EMO = {"e_neu": "affect.emotion.neutral", "e_hap": "affect.emotion.happy", "e_ang": "affect.emotion.angry",
       "e_sad": "affect.emotion.sad", "e_sur": "affect.emotion.surprised", "a_lo": "affect.arousal.low",
       "a_hi": "affect.arousal.high"}


class Tap:
    def __init__(self, stream, offset_s: float = 0.0):
        self.eng = getattr(stream, "eng", None)
        self.offset = float(offset_s)
        self.buf: list = []
        self.ok = False
        self.emo_cols = None
        if self.eng is not None and getattr(self.eng, "bank", None) is not None:
            def tap(f, e, slot_z, is_speech, _b=self.buf):
                if is_speech:
                    _b.append((f, np.asarray(e, np.float32)))
            self.eng._bank_step = tap
            self.ok = True

    def drain(self, offset: float | None = None) -> list[tuple[float, np.ndarray]]:
        if not self.buf:
            return []
        o = self.offset if offset is None else float(offset)
        out = [(f / 10.0 + o, e) for f, e in self.buf]
        self.buf.clear()
        return out

    def emotion(self) -> dict | None:
        eng = self.eng
        up = getattr(eng, "utt_p", None) if eng is not None else None
        if up is None:
            return None
        try:
            if self.emo_cols is None:
                cs = eng.R.cols.get("utt") or []
                self.emo_cols = {k: cs.index(eng.tab.idx[t]) for k, t in EMO.items()
                                 if eng.tab.idx.get(t) is not None and eng.tab.idx[t] in cs}
            return {k: float(up[j]) for k, j in self.emo_cols.items()} or None
        except Exception:  # noqa: BLE001
            return None
