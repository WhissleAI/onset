"""Streaming redaction with the Arc Tagger's own safety heads (TAGGER_SAFETY=1).

The tagger already streams profanity / toxicity spans (span heads safety.profanity.* / safety.toxicity.*) and per-
utterance safety flags (safety.crisis.self_harm, safety.toxicity.*, safety.threat.escalation). This module turns them
into three decisions, the same for spoken and typed input:

  mask_idx(words, spans)    word indices to mask: never in the web query, Keystone, memory, the persona profile, logs;
                            shown masked ("f***") in the transcript when the user keeps "Mask profanity" on
  turn_flag(...)            what the turn is, safety-wise: None | "insult" | "hate" | "threat" | "self_harm".
                            hate / threat: no web search for the turn; self_harm: a short supportive line with a crisis
                            resource instead of results
  AnswerMasker              masks the answer text as it streams (a profane source quote shows masked too)

Precision guard: a toxicity span inside a tagged entity mention is not masked ("Dick Van Dyke", "Scunthorpe", "Peru"
right after an insult); a profanity span inside one is masked only if the word itself is on the fallback list.
The old word list (WORDS) is the fallback when the tagger is not available (ARC_OFF / plain arm, TAGGER_SAFETY=0, a
handle that failed to load). eval/redaction/ measures both (eval/redaction_eval.py).
"""
from __future__ import annotations

import os
import re

ENABLED = os.environ.get("TAGGER_SAFETY", "1") == "1"
PROF_THR = float(os.environ.get("SAFETY_PROF_THR", "0.5"))     # the tagger's own span-head floor
TOX_THR = float(os.environ.get("SAFETY_TOX_THR", "0.6"))   # 0.5 masked "hole" in "black hole" (eval/redaction)
TOX_LO = float(os.environ.get("SAFETY_TOX_LO", "0.5"))     # other spans of a turn already over TOX_THR
# self-harm: the tagger's utterance flag is a low-threshold review trigger (configs/trust_safety_gaming.json uses 0.3);
# a few unambiguous first-person phrases back it up (measured: the flag alone misses "I want to kill myself")
SELF_HARM_THR = float(os.environ.get("SAFETY_SELF_HARM_THR", "0.3"))
# the old hard-coded list (server/frame.py PROFANITY): fallback only
WORDS = frozenset("fuck fucking fucked fucker shit shitty bullshit bitch bastard asshole dick damn crap piss cunt".split())
EMIT = "profanity,toxicity"
TAGGER_OPTIONS = {"redact": "profanity=mask,toxicity=mask", "thr:safety.crisis.self_harm": str(SELF_HARM_THR)}

_PUNCT = re.compile(r"[^\w\s'-]")
_SELF_HARM_CUES = re.compile(
    r"\b(kill(ing)? myself|end(ing)? my (own )?life|take my (own )?life|hurt(ing)? myself|harm(ing)? myself|"
    r"want(ed)? to die|wanna die|don'?t want to (live|be alive)|suicidal|commit(ting)? suicide|"
    r"painless way to die|better off dead|cut(ting)? myself)\b")
_KIND = {"insult_harassment": "insult", "hate_identity": "hate", "threat_violence": "threat",
         "sexual_explicit": "insult", "self_harm": "self_harm"}
_TOX_SPAN_KINDS = {"insult_harassment", "hate_identity", "threat_violence", "sexual_explicit", "self_harm"}
CRISIS_LINE = ("You don't have to go through this alone. If you are thinking about hurting yourself, please reach out "
               "now: in the US call or text 988 (Suicide & Crisis Lifeline), or call your local emergency number. "
               "Outside the US, findahelpline.com lists free, confidential lines.")
DECLINE_LINE = "I won't search for that."


def norm(w: str) -> str:
    return _PUNCT.sub("", str(w).lower()).strip()


def emit_list(base: str) -> str:
    """The tagger's emit list with the safety heads added (TAGGER_SAFETY=1)."""
    if not ENABLED:
        return base
    have = [x for x in base.split(",") if x]
    return ",".join(have + [x for x in EMIT.split(",") if x not in have])


def tagger_options() -> dict:
    return dict(TAGGER_OPTIONS) if ENABLED else {}


def _ent_idx(spans) -> set:
    return {i for s in spans or () if str(s.get("type", "")).startswith("entity.mention")
            for i in range(int(s["start"]), int(s["end"]))}


def mask_idx(words: list, spans: list | None, tagger: bool | None = None) -> dict:
    """{word index: kind} of the words to mask. kind: "profanity" | "insult" | "hate" | "threat" | "self_harm".
    tagger=True: the spans come from a tagger with the safety heads on (a Session's); otherwise (None / False, or
    TAGGER_SAFETY=0) the fallback word list: spans from a tagger without the heads say nothing about profanity."""
    toks = [norm(w) for w in words]
    use_tagger = bool(tagger) and ENABLED
    if not use_tagger:
        return {i: "profanity" for i, t in enumerate(toks) if t in WORDS}
    ent = _ent_idx(spans)
    out: dict = {}
    tox: list = []
    for s in spans or ():
        typ = str(s.get("type", ""))
        a, b = int(s.get("start", 0)), int(s.get("end", 0))
        if b > len(toks) or a >= b:
            continue
        sc = float(s.get("score") or 0.0)
        if typ.startswith("safety.profanity."):
            if sc < PROF_THR:
                continue
            for i in range(a, b):
                if i in ent and toks[i] not in WORDS:
                    continue
                out.setdefault(i, "profanity")
        elif typ.startswith("safety.toxicity."):
            k = typ.rsplit(".", 1)[-1]
            if k not in _TOX_SPAN_KINDS or sc < TOX_LO:
                continue
            if any(i in ent for i in range(a, b)):
                continue                      # a name / place the toxicity head read as an insult
            tox.append((sc, a, b, _KIND.get(k, "insult")))
    # hysteresis: a toxic turn (one span >= TOX_THR) masks its other toxicity spans down to TOX_LO ("you are a stupid
    # IDIOT": 0.76 + 0.60); a lone weak span masks nothing ("black HOLE" 0.59)
    if any(sc >= TOX_THR for sc, *_ in tox):
        for sc, a, b, kind in tox:
            for i in range(a, b):
                out.setdefault(i, kind)
    return out


def turn_flag(words: list, spans: list | None, current: dict | None, tagger: bool | None = None,
              masked: dict | None = None) -> str | None:
    """None | "insult" | "hate" | "threat" | "self_harm" for the turn (the tagger's utterance flags + toxicity spans;
    self-harm also from a few unambiguous first-person phrases)."""
    text = " ".join(norm(w) for w in words)
    flags = {x.get("tag", ""): float(x.get("p") or 0.0) for x in (current or {}).get("safety") or [] if x.get("on")}
    if _SELF_HARM_CUES.search(text) or (ENABLED and tagger and
                                        ("safety.crisis.self_harm" in flags or "safety.toxicity.self_harm" in flags)):
        return "self_harm"
    if not ENABLED or not tagger:
        return None
    m = masked if masked is not None else mask_idx(words, spans, tagger)
    kinds = set(m.values())
    if "safety.toxicity.hate_identity" in flags or "hate" in kinds:
        return "hate"
    if "safety.toxicity.threat_violence" in flags or "safety.threat.escalation" in flags or "threat" in kinds:
        return "threat"
    if "safety.toxicity.insult_harassment" in flags or "insult" in kinds:
        return "insult"
    return None


def no_search(flag: str | None) -> bool:
    """A turn that gets no web search at all (its toxic part is the whole request)."""
    return flag in ("self_harm", "hate", "threat")


def shown(word: str) -> str:
    """How a masked word shows on screen: first letter + asterisks ("f***")."""
    w = str(word)
    m = re.match(r"^(\W*)(\w)(\w*)(\W*)$", w)
    if not m:
        return "*" * max(3, len(w))
    return m.group(1) + m.group(2) + "*" * max(2, len(m.group(3))) + m.group(4)


def mask_text(words: list, idx) -> str:
    return " ".join(shown(w) if i in idx else w for i, w in enumerate(words))


class AnswerMasker:
    """Masks profanity in an answer as it streams: whole words only (the trailing partial word waits for the next delta),
    tagged by a streaming tagger handle (one per session). Toxicity in an answer is quoted material, so only the
    profanity head masks it. Falls back to the word list without a handle. Never raises: on any tagger error the text
    passes through unchanged and the masker switches to the word list."""

    def __init__(self, tagger=None):
        self.t = tagger
        self.reset()

    def reset(self):
        self.buf = ""                # text not yet emitted (the trailing partial word)
        self.words: list[str] = []   # words pushed to the tagger
        self.out_n = 0               # words already emitted
        self.prof: set = set()
        self.masked = 0
        if self.t is not None:
            try:
                self.t.reset()
            except Exception:
                self.t = None

    def _mask_ready(self, words: list[str]) -> list[str]:
        k0 = len(self.words)
        self.words += words
        if self.t is not None:
            try:
                self.t.push([w for w in words])
                r = self.t.poll()
                for s in r.get("spans") or []:
                    if str(s.get("type", "")).startswith("safety.profanity.") and float(s.get("score") or 0) >= PROF_THR:
                        self.prof.update(range(int(s["start"]), int(s["end"])))
            except Exception:
                self.t = None
        out = []
        for j, w in enumerate(words):
            i = k0 + j
            if i in self.prof or (self.t is None and norm(w) in WORDS):
                out.append(shown(w))
                self.masked += 1
            else:
                out.append(w)
        return out

    def feed(self, delta: str) -> str:
        """The masked text to show now for this delta (may be "" while a word is incomplete)."""
        self.buf += delta or ""
        m = re.search(r"\s(?=\S*$)", self.buf)
        if not m:
            return ""
        head, self.buf = self.buf[:m.end()], self.buf[m.end():]
        return self._render(head)

    def _render(self, head: str) -> str:
        parts = re.split(r"(\s+)", head)
        ws = [p for p in parts if p and not p.isspace()]
        masked = iter(self._mask_ready(ws))
        return "".join(next(masked) if (p and not p.isspace()) else p for p in parts)

    def flush(self) -> str:
        head, self.buf = self.buf, ""
        return self._render(head) if head else ""

    def mask_all(self, text: str) -> str:
        """A whole text (the answer stored in memory / Keystone / the metric), masked with the same handle state reset."""
        self.reset()
        out = self.feed(text) + self.flush()
        return out
