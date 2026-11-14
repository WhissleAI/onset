"""Persona: who is asking and how to answer them (PERSONA=1; default off until measured, docs/PERSONALIZATION.md).

Two things, both small and both explained to the user in plain words ("Answered for you" chip):

1. How to answer THIS question (`style`), from how it was said plus the profile:
   - Arc / ear timing: speaking pace (syllables per second over the question's word times), pauses inside the question,
     spoken fillers and hedges ("um", "I think", "what's it called"), the ear's low-confidence words, Arc bio events
     (throat clear), Arc audio emotion when the director streams are on.
   - The words: an explicit ask for less / more ("just the number", "tell me more"), "the last answer was wrong"
     ("no, that's not what I meant"), units ("in feet"), a definition ask, a follow-up (memory resolved a reference).
   - The profile: learned answer length, expertise, units.
   The result is ONE short line for the answer model (`hint`, ~15-30 tokens) and a chip reason ("shorter: you asked
   quickly").

2. Who is asking (`Profile`), built over turns and persisted per browser like conversation memory:
   - interests: the titles of the PUBLIC sources the answers used, each with that source's embedding and a coarse
     category ("car", "planet"); recency-weighted. Nothing derived from the user's own words is stored as a vector.
   - preferred answer length, expertise (-1 novice .. 1 expert), units ("imperial" / "metric"), usual pace.
   Retrieval uses it twice: the final / draft rerank adds `PERSONA_BIAS` x cosine(doc, interest vector), and on the free
   Wikipedia lane an ambiguous subject gets one extra "<subject> <category>" search ("mercury planet") whose results
   are kept only when the subject is ambiguous (a disambiguation page or a "Subject (sense)" title among the results).

Never stored: audio, transcripts, demographics (Arc emits no age / gender buckets in this build; we would not keep them
anyway), personal data the tagger redacted (a turn with redacted spans adds no interest; PII_YEARS-hidden years never
reach a source title we keep because only source titles are kept). Export / clear: `Profile.export()`, `Store.forget`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

ENABLED = os.environ.get("PERSONA", "0") == "1"
BIAS = float(os.environ.get("PERSONA_BIAS", "0.25"))          # rerank: + BIAS x cos(doc, interest vector)
SENSE = os.environ.get("PERSONA_SENSE", "1") == "1"            # free extra "<subject> <category>" Wikipedia search
STYLE = os.environ.get("PERSONA_STYLE", "1") == "1"            # the per-question answer-style hint
TTL_S = float(os.environ.get("PERSONA_TTL_S", str(30 * 86400)))
MAX_ITEMS = 16
TURN_DECAY = 0.6            # interest weight per turn of age
DAY_HALF = 7.0              # ... and per day of age (half-life)
FAST_SPS = float(os.environ.get("PERSONA_FAST_SPS", "7.0"))   # syllables / s that counts as "asked quickly"
SLOW_SPS = float(os.environ.get("PERSONA_SLOW_SPS", "2.8"))   # ... as slow / halting

FILLERS = {"um", "uh", "er", "erm", "hmm", "uhm", "umm", "uhh", "mm", "hm"}
_HEDGE = re.compile(r"\b(i think|i guess|maybe|not sure|i'm not sure|kind of|sort of|what's it called|what is it called|"
                    r"i don't know|i forget|i forgot|whatever it is|or something|something like)\b")
_LESS = re.compile(r"\b(just (?:the|give me|tell me)|short answer|briefly|in short|quick(?:ly)? answer|in one word|"
                   r"bottom line|keep it short|one word|just the number|just the year)\b")
_MORE = re.compile(r"\b(tell me more|more detail|more details|in detail|explain|elaborate|go deeper|how does that work|"
                   r"why is that|walk me through|more about)\b")
_SIMPLE = re.compile(r"\b(in simple terms|simply|like i'm five|for a beginner|for beginners|layman)\b")
_EXACT = re.compile(r"\b(exact|exactly|precise|precisely|technically|specifically)\b")
_DEFN = re.compile(r"^(?:\W*(?:um|uh|so|and|ok|okay)\W+)*(?:what(?:'s| is| are) (?:a |an |the )?\w+(?: \w+)?\W*$|"
                   r"what does .+ mean|what do you mean by|define )")
_WRONG = re.compile(r"^(?:\W*(?:um|uh|oh|hmm)\W+)*(?:no\b|nope\b|wrong\b|that's wrong|that is wrong|that's not|that is not|"
                    r"that isn't|not that\b|not what i|i said\b|i asked\b|i meant\b|you're wrong|incorrect)")
_WRONG_ANY = re.compile(r"\b(that's not what i (?:asked|meant)|not what i (?:asked|meant)|that's wrong|that is wrong|"
                        r"you got (?:it|that) wrong|wrong answer)\b")
_IMPERIAL = re.compile(r"\b(miles?|mph|feet|foot|ft|inches|pounds|lbs?|fahrenheit|gallons?|ounces|yards?)\b")
_METRIC = re.compile(r"\b(kilomet(?:er|re)s?|km|kph|met(?:er|re)s?|centimet(?:er|re)s?|kilos?|kilograms?|celsius|"
                     r"litres?|liters?|grams?)\b")

# answer-model notes (appended to the prompt as "Note: ..."; ~20-30 tokens each)
SHORT_HINT = ("Style (overrides the length rule): ONE sentence, at most 15 words, just the fact; if no source states it, "
              "give it uncited without saying so.")
FULLER_HINT = ("Style (overrides the length rule): the user sounds unsure: give the answer, then one plain sentence "
               "explaining the key term; under 60 words.")
MORE_HINT = "Style (overrides the length rule): more detail, up to four sentences, under 80 words."
FIX_HINT = "The user says the last answer missed: start with a brief \"Sorry,\" then answer what they mean now."

# coarse interest categories: a source's embedding is matched to these labels (query side of EmbeddingGemma).
CATEGORIES = ["car", "animal", "planet", "programming language", "company", "river", "chemical element", "band",
              "film", "city", "country", "sports team", "food", "video game", "game console", "book", "song",
              "disease", "medicine", "bird", "snake", "plant", "mythology", "ship", "aircraft", "software",
              "musician", "television series", "language", "mountain", "island", "drink", "computer", "star",
              "politician", "scientist", "athlete", "painting", "religion", "weapon", "currency", "university"]
CAT_MIN = 0.30              # best category cosine, below: no category
CAT_MARGIN = 0.015          # best - second best

_cat_vecs: np.ndarray | None = None


def _encode_cats() -> np.ndarray:
    global _cat_vecs
    if _cat_vecs is None:
        from . import embed
        _cat_vecs = embed.encode_queries([f"a {c}" for c in CATEGORIES])
    return _cat_vecs


def category(vec: np.ndarray, cat_vecs: np.ndarray | None = None) -> str:
    """The coarse category of a source embedding ("" when no label clearly fits)."""
    try:
        C = _encode_cats() if cat_vecs is None else cat_vecs
        s = C @ vec
        o = np.argsort(-s)
        if s[o[0]] < CAT_MIN or (len(o) > 1 and s[o[0]] - s[o[1]] < CAT_MARGIN):
            return ""
        return CATEGORIES[int(o[0])]
    except Exception:
        return ""


def _norm(t: str) -> str:
    return re.sub(r"[^\w' ]+", " ", (t or "").lower()).strip()


# ------------------------------------------------------------------------------------------------- per question
def syllables(w: str) -> int:
    """Rough English syllable count (vowel groups, silent final e): pace in syllables / s is steadier than words / s
    on 3-6 word questions ("atomic number of tungsten" vs "how fast is a jaguar")."""
    digits = sum(ch.isdigit() for ch in w)
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w:
        return max(1, digits) if digits else 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def signals(words: list[str], ws: list[dict] | None = None, *, followup: bool = False, prev_answered: bool = False,
            arc: dict | None = None, arc_x2: dict | None = None, intent: str = "") -> dict:
    """How this question was said. words: the question's words; ws: the ear's words with start / end / conf."""
    toks = [_norm(w) for w in words]
    toks = [t for t in toks if t]
    text = " ".join(toks)
    n = len(toks)
    s = {"n": n, "pace": None, "pauses": 0, "fillers": sum(1 for t in toks if t in FILLERS),
         "hedges": len(_HEDGE.findall(text)), "low_conf": 0, "followup": bool(followup), "less": bool(_LESS.search(text)),
         "more": bool(_MORE.search(text)), "simple": bool(_SIMPLE.search(text)), "exact": bool(_EXACT.search(text)),
         "definition": bool(_DEFN.search(text)) or intent == "discourse.intent.qa_definition",
         "wrong": bool(prev_answered and (_WRONG.search(text) or _WRONG_ANY.search(text))),
         "units": "imperial" if _IMPERIAL.search(text) else ("metric" if _METRIC.search(text) else ""),
         "throat": False, "angry": None}
    tw = [w for w in (ws or []) if w.get("start") is not None and w.get("end") is not None and _norm(w.get("text", ""))]
    if len(tw) >= 3:
        dur = float(tw[-1]["end"]) - float(tw[0]["start"])
        content = [w for w in tw if _norm(w["text"]) not in FILLERS]
        if dur > 0.3:
            s["pace"] = round(sum(syllables(_norm(w["text"])) for w in content) / dur, 2)   # syllables / s
        s["pauses"] = sum(1 for a, b in zip(tw, tw[1:]) if float(b["start"]) - float(a["end"]) >= 0.45)
        s["low_conf"] = sum(1 for w in tw if w.get("conf") is not None and w["conf"] < 0.6)
    s["quick"] = bool(s["pace"] is not None and s["pace"] >= FAST_SPS and n >= 3)
    for b in (arc or {}).get("bio") or []:
        if (b.get("tag") or "").endswith("throat_clear"):
            s["throat"] = True
    utt = (arc_x2 or {}).get("utt") or {}
    if utt.get("e_ang") is not None:
        s["angry"] = round(float(utt["e_ang"]), 2)        # Arc audio emotion: only with DIRECTOR_STREAMS=1
    return s


def relative(sig: dict, rel: dict | None) -> dict:
    """Fold in the speaker's own norm from Arc state (feat/arcstate owns per-speaker baselines; persona builds none):
    rel = {"pace_rel": this pace / the speaker's usual, "hes_rel": hesitation vs usual}. Missing / None: absolute bars."""
    if not rel:
        return sig
    pr, hr = rel.get("pace_rel"), rel.get("hes_rel")
    if pr is not None and sig["n"] >= 3:
        sig["quick"] = bool(pr >= 1.25 or (sig["quick"] and pr >= 1.0))
        sig["pace_rel"] = round(float(pr), 2)
    if hr is not None and hr >= 1.5:
        sig["hedges"] += 1                   # noticeably more hesitant than this speaker usually is
    return sig


def style(sig: dict, prof: "Profile | None" = None) -> dict:
    """{"kind", "hint", "why"}: how to answer this question. kind "" = nothing to change (no hint, no chip)."""
    prof = prof or Profile()
    hes = sig["fillers"] + sig["hedges"] + min(sig["pauses"], 2) + (1 if sig["throat"] else 0)
    pace = sig.get("pace")
    quick = sig.get("quick", False)
    slow = pace is not None and pace <= SLOW_SPS
    units = sig["units"] or prof.units
    unit_note = ""
    if units and not sig["units"]:
        unit_note = (" Use miles, feet and °F." if units == "imperial" else " Use km, metres and °C.")
    out = {"kind": "", "hint": "", "why": "", "units": units if unit_note else ""}
    if sig["wrong"] or (sig.get("angry") or 0) >= 0.6 and sig.get("followup"):
        out.update(kind="fix", why="fixing it: you said the last answer missed",
                   hint=FIX_HINT)
    elif sig["more"]:
        out.update(kind="more", why="more detail: you asked for it",
                   hint=MORE_HINT)
    elif sig["less"]:
        out.update(kind="short", why="shorter: you asked for just the answer",
                   hint=SHORT_HINT)
    elif hes >= 2 or (hes >= 1 and (slow or sig["definition"])) or (slow and sig["definition"]) or sig["simple"]:
        out.update(kind="fuller", why="a little fuller: you sounded unsure" if not sig["simple"] else
                   "simpler: you asked for simple terms",
                   hint=FULLER_HINT)
    elif quick and sig["n"] <= 9 and not sig["definition"]:
        out.update(kind="short", why="shorter: you asked quickly",
                   hint=SHORT_HINT)
    elif prof.turns >= 1 and prof.length <= -0.45:
        out.update(kind="short", why="shorter: you usually want just the answer",
                   hint=SHORT_HINT)
    elif prof.turns >= 1 and prof.length >= 0.45:
        out.update(kind="more", why="more detail: you usually ask for it",
                   hint=MORE_HINT)
    elif prof.turns >= 3 and prof.expertise <= -0.5:
        out.update(kind="fuller", why="a little fuller: plain words, like before",
                   hint=FULLER_HINT)
    if unit_note:
        out["hint"] = (out["hint"] + unit_note).strip()
        out["why"] = out["why"] or ("in miles and feet: how you asked before" if units == "imperial"
                                    else "in km and metres: how you asked before")
        out["kind"] = out["kind"] or "units"
    return out


# ------------------------------------------------------------------------------------------------- profile
def _enc(v: np.ndarray) -> str:
    return base64.b64encode(np.asarray(v, np.float16).tobytes()).decode()


def _dec(s: str) -> np.ndarray:
    v = np.frombuffer(base64.b64decode(s), np.float16).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class Profile:
    def __init__(self):
        self.turns = 0
        self.pace: float | None = None       # unused: per-speaker baselines live in Arc state (feat/arcstate)
        self.length = 0.0                    # -1 short .. 1 long
        self.expertise = 0.0                 # -1 novice .. 1 expert
        self.units = ""
        self.items: list[dict] = []          # {"label", "cat", "vec": np, "turn", "at"}
        self.updated = 0.0

    def __bool__(self):
        return bool(self.turns or self.items)

    # ---- learning
    def observe(self, sig: dict, sources: list[tuple[str, np.ndarray]] | None = None, private: bool = False,
                cat_vecs: np.ndarray | None = None):
        """One answered turn. sources: (title, embedding) of the top sources the answer used (public pages). private:
        the turn had redacted personal data (nothing from it becomes an interest)."""
        self.turns += 1
        self.updated = time.time()
        d = 0.0
        if sig.get("less"):
            d -= 1.0
        if sig.get("more"):
            d += 1.0
        if sig.get("quick") and not d:
            d -= 0.5                         # asked quickly: leans short (a single quick turn alone doesn't flip it)
        if d:
            self.length = max(-1.0, min(1.0, 0.6 * self.length + 0.4 * d * 1.6))
        else:
            self.length *= 0.9
        e = 0.0
        if sig.get("simple"):
            e -= 1.0
        if sig.get("definition"):
            e -= 0.5
        if sig.get("hedges") or sig.get("fillers", 0) >= 2:
            e -= 0.3
        if sig.get("exact"):
            e += 0.5
        if sig.get("less") and not sig.get("definition"):
            e += 0.3
        self.expertise = max(-1.0, min(1.0, 0.7 * self.expertise + 0.3 * e * 2)) if e else self.expertise * 0.95
        if sig.get("units"):
            self.units = sig["units"]
        if private or not sources:
            return
        for title, vec in sources[:2]:
            title = (title or "").split(" · ")[0].strip()       # "Jaguar · local Wikipedia" -> "Jaguar"
            if not title or vec is None:
                continue
            v = np.asarray(vec, np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-9)
            self.items = [x for x in self.items if x["label"] != title]
            self.items.append({"label": title, "cat": category(v, cat_vecs), "vec": v, "turn": self.turns,
                               "at": time.time()})
        self.items = self.items[-MAX_ITEMS:]

    # ---- reading
    def _w(self, x: dict, now: float | None = None) -> float:
        now = now or time.time()
        age_t = max(0, self.turns - x["turn"])
        age_d = max(0.0, (now - x.get("at", now)) / 86400)
        return TURN_DECAY ** age_t * 0.5 ** (age_d / DAY_HALF)

    def vec(self) -> np.ndarray | None:
        if not self.items:
            return None
        ws = np.array([self._w(x) for x in self.items], np.float32)
        if float(ws.sum()) < 0.15:
            return None
        v = (np.stack([x["vec"] for x in self.items]) * ws[:, None]).sum(0)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def sense(self) -> str:
        """The category the recent interests share ("car"), or "" when there is no clear recent one."""
        tot: dict = {}
        for x in self.items:
            if x.get("cat"):
                tot[x["cat"]] = tot.get(x["cat"], 0.0) + self._w(x)
        if not tot:
            return ""
        cat, w = max(tot.items(), key=lambda kv: kv[1])
        all_w = sum(self._w(x) for x in self.items)
        return cat if (w >= 0.5 and w >= 0.6 * all_w) else ""

    def because(self, doc_vec: np.ndarray, cat: str = "") -> dict | None:
        """The remembered interest closest to this source (for the chip: "you asked about the Ford Mustang earlier")."""
        best, bs = None, -1.0
        for x in self.items:
            if cat and x.get("cat") != cat:
                continue
            s = float(x["vec"] @ doc_vec) * (0.5 + self._w(x))
            if s > bs:
                best, bs = x, s
        return best

    def topics(self, n: int = 8) -> list[dict]:
        xs = sorted(self.items, key=lambda x: -self._w(x))[:n]
        return [{"label": x["label"], "cat": x.get("cat") or "", "weight": round(self._w(x), 2)} for x in xs]

    def summary(self) -> dict:
        """What the profile panel shows (and what export writes): no vectors."""
        ln = "short" if self.length <= -0.45 else ("detailed" if self.length >= 0.45 else "normal")
        ex = "beginner" if self.expertise <= -0.5 else ("expert" if self.expertise >= 0.5 else "")
        return {"turns": self.turns, "length": ln, "expertise": ex, "units": self.units,
                "pace": self.pace, "topics": self.topics(), "sense": self.sense(),
                "updated": round(self.updated)}

    def export(self) -> dict:
        return {"what": "Onset profile: what it learned about how you ask (no audio, no transcripts)",
                **self.summary(), "length_score": round(self.length, 2), "expertise_score": round(self.expertise, 2)}

    def clear(self):
        self.__init__()

    def to_json(self) -> dict:
        return {"v": 1, "turns": self.turns, "pace": self.pace, "length": round(self.length, 3),
                "expertise": round(self.expertise, 3), "units": self.units, "updated": self.updated,
                "items": [{"label": x["label"], "cat": x.get("cat", ""), "turn": x["turn"], "at": x["at"],
                           "vec": _enc(x["vec"])} for x in self.items]}

    @classmethod
    def from_json(cls, d: dict | None, ttl_s: float | None = None) -> "Profile":
        p = cls()
        if not d or d.get("v") != 1:
            return p
        ttl = TTL_S if ttl_s is None else ttl_s
        if d.get("updated") and time.time() - float(d["updated"]) > ttl:
            return p
        try:
            p.turns = int(d.get("turns") or 0)
            p.pace = d.get("pace")
            p.length = float(d.get("length") or 0.0)
            p.expertise = float(d.get("expertise") or 0.0)
            p.units = d.get("units") or ""
            p.updated = float(d.get("updated") or 0.0)
            now = time.time()
            p.items = [{"label": x["label"], "cat": x.get("cat", ""), "turn": int(x["turn"]), "at": float(x["at"]),
                        "vec": _dec(x["vec"])} for x in d.get("items") or [] if now - float(x.get("at", now)) <= ttl]
        except Exception:
            return cls()
        return p


def interest_sources(f, answer_text: str, docs: list, vecs: dict, n: int = 2) -> list:
    """The answered turn's interests: (title, embedding) of up to n public articles the answer actually used: cited
    in the answer ([1]), or, when it cites nothing, a source whose title shares a word with the question. Never a
    Keystone cell (its title can be the user's own question), a local recording moment, a card or memory."""
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer_text or "")}
    try:                                     # the question's subject words (Arc entity phrases, else content words)
        qw = {w for w in _norm(" ".join(list(getattr(f, "phrases", ()) or ()) + [subject(f)])).split()
              if len(w) > 2 and w not in _STOPS and w not in _GENERIC}
    except Exception:
        qw = set()
    out = []
    for k, d in enumerate(docs, 1):
        if d is None or d.id not in vecs or d.t is not None or str(d.id).startswith("index:"):
            continue
        if (d.source or "") in ("local", "card", "memory") or (d.meta or {}).get("tier") == "index":
            continue
        title = (d.title or "").split(" · ")[0]
        tw = {w for w in _norm(title).split() if len(w) > 2}
        if (k in cited) if cited else bool(tw & qw):
            out.append((title, vecs[d.id]))
        if len(out) >= n:
            break
    return out


_GENERIC = {"answer", "short", "exactly", "give", "tell", "more", "just", "about", "the", "what", "does", "much"}


# ------------------------------------------------------------------------------------------------- retrieval
def bias_rank(pool_ids: list[str], vecs: dict, qvec: np.ndarray, pvec: np.ndarray | None, k: int = 5,
              beta: float | None = None, skip=(), only=None) -> list[tuple[str, float]]:
    """Top-k ids by q.d + beta * p.d (the plain ranking when pvec is None). Ids in `skip` get no bias; with `only`, only
    those ids do (the docs of the user's recent category)."""
    if not pool_ids:
        return []
    M = np.stack([vecs[i] for i in pool_ids])
    s = M @ qvec
    if pvec is not None:
        b = (BIAS if beta is None else beta) * (M @ pvec)
        if skip or only is not None:
            b = b * np.array([0.0 if (i in skip or (only is not None and i not in only)) else 1.0 for i in pool_ids],
                             np.float32)
        s = s + b
    o = np.argsort(-s)[:k]
    return [(pool_ids[i], float(s[i])) for i in o]


_STOPS = set("""a an the of is how what when where who fast big hot cold long old tall far much many high deep large heavy
small good best top speed grow grows live lives run runs used use made make born die died work works mean called weigh
weighs cost costs headquartered located founded invented discovered created built start started eat eats found last
get go come do does""".split())


def subject(f) -> str:
    """The question's subject for the sense search: Arc's entity phrase, else its last content word."""
    try:
        e = (f.entity_query() or "").strip()
        if not e and getattr(f, "phrases", None):
            e = f.phrases[-1]
        if not e:
            c = [w for w in (f.content or ()) if w not in _STOPS]
            e = c[-1] if c else ""
        return e.strip()
    except Exception:
        return ""


def ambiguous(docs, subj: str) -> bool:
    """Wikipedia says this subject has several senses: a disambiguation page, or a "Subject (sense)" title."""
    s = subj.lower().strip()
    if not s:
        return False
    for d in docs:
        t = (d.title or "").lower()
        head = (d.text or "")[:240].lower()
        if "(disambiguation)" in t:
            return True
        if t == s and ("may refer to" in head or "refers to:" in head or "may also refer to" in head):
            return True
        if t.startswith(s + " (") and t.endswith(")"):
            return True
    return False


# ------------------------------------------------------------------------------------------------- per session
class View:
    """What the speculator reads for one question: the interest vector, the sense, and the chip it may add."""

    def __init__(self, prof: Profile):
        self.prof = prof
        self.vec = prof.vec() if prof else None
        self.sense = prof.sense() if prof else ""
        self.changed: dict | None = None         # the rerank changed the top source: {"title", "because"}
        self._cats: dict = {}

    def cat(self, doc_id: str, vec: np.ndarray) -> str:
        if doc_id not in self._cats:
            self._cats[doc_id] = category(vec)
        return self._cats[doc_id]

    def other_sense(self, doc_id: str, vec) -> bool:
        """This source is of a named category other than the user's recent one (the case the bias may overturn)."""
        if not self.sense or vec is None:
            return False
        c = self.cat(doc_id, vec)
        return bool(c) and c != self.sense

    def eligible(self, ids, vecs: dict, skip=()) -> set:
        """Docs the interest bias may lift: of the user's recent category, not an earlier turn's own source."""
        if not self.sense:
            return set()
        return {i for i in ids if i not in skip and self.cat(i, vecs[i]) == self.sense}

    def sense_query(self, f) -> str:
        if not self.sense or not SENSE:
            return ""
        s = subject(f)
        if not s or self.sense in s.lower() or len(s) > 40:
            return ""
        return f"{s} {self.sense}"

    def note_rank(self, plain_top, biased_top, doc_vec: np.ndarray | None) -> dict | None:
        if plain_top is None or biased_top is None or doc_vec is None:
            return self.changed
        if plain_top.id == biased_top.id:
            self.changed = None
            return None
        if self.changed and self.changed["title"] == biased_top.title and self.changed["instead"] == plain_top.title:
            return self.changed
        b = self.prof.because(doc_vec, self.sense)
        self.changed = {"title": biased_top.title, "instead": plain_top.title,
                        "because": b["label"] if b else "", "cat": self.sense}
        return self.changed

    def hint(self) -> str:
        if not self.changed:
            return ""
        return f"Answer about {self.changed['title'].split(' · ')[0]} (the sense this user has been asking about)."

    def why(self) -> str:
        c = self.changed
        if not c:
            return ""
        t = c["title"].split(" · ")[0]
        if c.get("because"):
            return f"{t}: you asked about {c['because']} earlier"
        return f"{t}: closest to what you asked before"


# ------------------------------------------------------------------------------------------------- store
class Store:
    """<data dir>/persona/<sha256(client)[:24]>.json, the same browser id as conversation memory."""

    def __init__(self, root: Path | None = None):
        if root is None:
            from . import config
            root = Path(os.environ.get("PERSONA_DIR", config.DATA_DIR / "persona"))
        self.root = Path(root)

    @staticmethod
    def valid(client) -> bool:
        return isinstance(client, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{8,64}", client))

    def path(self, client: str) -> Path:
        return self.root / (hashlib.sha256(client.encode()).hexdigest()[:24] + ".json")

    def load(self, client: str | None) -> Profile:
        if not self.valid(client):
            return Profile()
        try:
            return Profile.from_json(json.loads(self.path(client).read_text()))
        except Exception:
            return Profile()

    def save(self, client: str | None, prof: Profile):
        if not self.valid(client) or not prof:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            p = self.path(client)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(prof.to_json()))
            tmp.replace(p)
        except Exception:
            pass

    def forget(self, client: str | None):
        if not self.valid(client):
            return
        try:
            self.path(client).unlink()
        except FileNotFoundError:
            pass


def warmup():
    """Embed the category labels once (so the first answered turn doesn't pay for it on the event loop)."""
    if ENABLED:
        try:
            _encode_cats()
        except Exception:
            pass


def hint_tokens(text: str) -> int:
    """Estimated tokens a hint adds to the prompt (the 'Note:' wrapper included). Calibrated on claude-haiku-5-5's
    count_tokens for the hints here: ~3 characters per token (exact: short 50, fuller 49, more 32, fix 34, units
    15, sense line 28; docs/PERSONALIZATION.md)."""
    return math.ceil((len(text) + 7) / 3) if text else 0
