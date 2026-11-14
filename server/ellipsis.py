"""Noun-phrase ellipsis and relative references across turns (rules, no network, ~0.05 ms).

  "who is the president of usa" -> "you know the old one"   -> "who was the previous president of usa before the
                                                                current president donald trump"
                                -> "and the next one?"       -> "who will be the next president of usa after ..."
                                -> "the first one"           -> "who was the first president of usa"
                                -> "who was before him"      -> (same as "the old one")

A reference like this names no entity of its own and no pronoun the entity memory can resolve: "one" is a pro-form for
the previous question's HEAD noun ("president" of "president of usa"), and the modifier says which one relative to the
previous ANSWER (Donald Trump). resolve() returns the standalone question the router / web search should see, or None
(then frame.build's other memory rules apply). A new explicit entity in the turn means a new topic: never called then.

Kept out on purpose: "one of the best ...", "one day", "number one", "the one I like" (no modifier = no reference)."""
from __future__ import annotations

import os
import re

ENABLED = os.environ.get("ELLIPSIS_RULES", "1") == "1"

PREV = {"old", "older", "previous", "former", "last", "earlier", "prior", "preceding", "before"}
NEXT = {"next", "new", "newer", "following", "upcoming", "after"}
FIRST = {"first", "original"}
_DET = {"the", "that", "this", "an", "a"}
_PRON = {"him", "her", "them", "it", "that", "this", "he", "she", "they"}
_WH_AUX = re.compile(r"^(?:(?:who|what|which|where|when)(?:'s)?\s+(?:is|was|are|were|will be)?\s*)")
_HEAD_OF = re.compile(r"^(?:the |a |an )?(?P<h>[a-z][a-z'-]*(?: [a-z][a-z'-]*)?) of (?:the )?(?P<c>[a-z0-9][a-z0-9 '.-]*)$")
MAX_AGE_S = 600.0          # a reference this vague only points at a turn from the last 10 minutes


def head_of(question: str) -> tuple[str, str] | None:
    """("president", "usa") from "who is the president of usa"; None when the question has no "<head> of <x>"."""
    q = re.sub(r"[^\w' -]+", " ", (question or "").lower()).strip()
    q = re.sub(r"\s+", " ", _WH_AUX.sub("", q)).strip()
    m = _HEAD_OF.match(q)
    if not m:
        return None
    h, c = m.group("h"), m.group("c").strip()
    c = re.split(r" (?:before|after) ", c)[0].strip()       # a rewritten previous turn: "... of usa before the current"
    h = " ".join(w for w in h.split() if w not in ("current", "present", "new", "previous", "next", "first"))
    return (h, c) if h and c else None


# the kind of entity a question head asks for ("president of usa" -> a person): the answer's entity of that kind is the
# anchor "the old one" is relative to
HEAD_KIND = {w: "person" for w in """president vice-president ceo cfo cto coo chairman chairwoman chair founder king queen
monarch emperor empress pope prime minister chancellor mayor governor senator leader head director coach captain
manager owner author editor host anchor speaker secretary premier ruler sultan chief commissioner principal dean
champion winner mvp""".split()}
HEAD_KIND.update({"capital": "place", "city": "place", "largest city": "place", "headquarters": "place",
                  "parent company": "org", "owner company": "org", "team": "org", "company": "org"})
_ALIASES = [{"usa", "us", "u s", "u s a", "united states", "united states of america", "america", "the us", "the usa"},
            {"uk", "u k", "united kingdom", "britain", "great britain", "england"},
            {"uae", "united arab emirates"}, {"eu", "european union"}, {"un", "united nations"},
            {"ussr", "soviet union"}, {"nyc", "new york city", "new york"}]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def _same(a: str, b: str) -> bool:
    """Same thing under another name: equal, an alias pair, initials ("united states of america" ~ "usa"), or one name
    inside the other ("france" ~ "republic of france")."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b or any(a in g and b in g for g in _ALIASES):
        return True
    ini = lambda x: "".join(w[0] for w in x.split() if w not in ("of", "the", "and"))
    if len(a.split()) > 1 and ini(a) == b.replace(" ", "") or len(b.split()) > 1 and ini(b) == a.replace(" ", ""):
        return True
    return f" {a} " in f" {b} " or f" {b} " in f" {a} "


def anchor(prev: dict, ho: tuple | None) -> str:
    """The previous answer's entity "the old one" is relative to: never the question's own object ("usa" / "united
    states"); of the kind the question head asks for (president -> person) when the kinds are known; then the subject of
    the answer's first sentence (the earliest named), then the answer's order."""
    cands = [dict(c) for c in prev.get("answer_ents") or []] or [{"name": a, "kind": ""} for a in prev.get("answers") or []]
    objs = list(prev.get("entities") or []) + ([ho[1]] if ho else [])
    cands = [c for c in cands if c.get("name") and not any(_same(c["name"], o) for o in objs)]
    if not cands:
        return ""
    want = HEAD_KIND.get(ho[0]) if ho else None
    first = _norm(re.split(r"(?<=[.!?])\s", prev.get("answer") or "", maxsplit=1)[0])

    def key(ic):
        i, c = ic
        pos = f" {first} ".find(f" {_norm(c['name'])} ")
        return (0 if want and c.get("kind") == want else 1, pos if pos >= 0 else 10 ** 6, i)
    return sorted(enumerate(cands), key=key)[0][1]["name"].lower()


def _which(toks: list[str]) -> tuple[str, str] | None:
    """(kind, phrase): kind in prev | next | first, phrase = the reference as said ("the old one")."""
    for i, t in enumerate(toks):
        if t in ("one", "ones"):
            if i + 1 < len(toks) and toks[i + 1] in ("of", "day", "time", "more", "hundred", "thousand", "million"):
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            # "the one before" / "the one after"
            if nxt in ("before", "after") and i >= 1 and toks[i - 1] in _DET:
                return ("prev" if nxt == "before" else "next", " ".join(toks[i - 1:i + 2]))
            j = i - 1
            mods = []
            while j >= 0 and toks[j] not in _DET and len(mods) < 2:
                mods.append(toks[j])
                j -= 1
            if j < 0 and mods:
                j = 0
            phrase = " ".join(toks[max(j, 0):i + 1])
            for w in mods:
                if w in PREV:
                    return ("prev", phrase)
                if w in NEXT:
                    return ("next", phrase)
                if w in FIRST:
                    return ("first", phrase)
                if w == "other":
                    return ("other", phrase)
    # "who was before him", "who came after her", "and before that?"
    for i, t in enumerate(toks):
        if t in ("before", "after") and (i + 1 == len(toks) or toks[i + 1] in _PRON):
            if i + 1 < len(toks) or (i >= 1 and toks[i - 1] in ("was", "came", "is", "be", "comes", "and", "one")):
                return ("prev" if t == "before" else "next", " ".join(toks[i:i + 2]))
    return None


def resolve(toks: list[str], prev: dict | None) -> dict | None:
    """The standalone rewrite of a relative reference to the previous turn, or None.
    prev: frame carry "prev" (question, inline, answers = the previous answer's entity names, age_s, n_subjects)."""
    if not prev or not toks:
        return None
    if prev.get("age_s") is not None and prev["age_s"] > MAX_AGE_S:
        return None
    w = _which([t for t in toks if t])
    if not w:
        return None
    kind, phrase = w
    if kind in ("first", "other") and (prev.get("n_subjects") or 0) >= 2:
        return None                 # "go back to the first one" / "the other one" among several subjects: memory's pick
    if kind == "other":
        kind = "prev"               # one subject so far: "the other one" = the one before the answer ("the other guy")
    ho = head_of(prev.get("inline") or prev.get("question") or "")
    ans = anchor(prev, ho)
    if ho is None:
        if kind == "first" or not ans:
            return None
        base = re.sub(r"[^\w' -]+", " ", (prev.get("inline") or prev.get("question") or "").lower()).strip()
        if not base:
            return None
        rel = "before" if kind == "prev" else "after"
        q = f"{base} {rel} {ans}"
    else:
        h, c = ho
        if kind == "first":
            q = f"who was the first {h} of {c}"
        elif kind == "prev":
            # "immediately before the current one": "before donald trump" alone reads as before his FIRST term (the
            # answer model said Obama; the user meant Biden)
            q = (f"who was the previous {h} of {c} before the current {h} {ans}" if ans
                 else f"who was the previous {h} of {c}")
        else:
            q = (f"who will be the next {h} of {c} after the current {h} {ans}" if ans
                 else f"who will be the next {h} of {c}")
    return {"inline": re.sub(r"\s+", " ", q).strip(), "ref": phrase, "kind": kind, "answer": ans}


# ------------------------------------------------------------------ optional model rewrite (ELLIPSIS_LLM, default off)
# Rules first; a tiny answer-model call only for a short turn the rules left alone, right after a recent turn, with a hard
# timeout (rules / the words as said on timeout). Off by default: the answer model's measured time to first token on
# this machine (~1.0 s) is above the 400 ms budget, so most calls would time out and only add latency.
LLM = os.environ.get("ELLIPSIS_LLM", "0") == "1"
LLM_TIMEOUT_S = float(os.environ.get("ELLIPSIS_LLM_TIMEOUT_S", "0.4"))
_LLM_SYS = ("You rewrite a spoken follow-up as ONE standalone web search question, using the previous question and "
            "answer. Resolve what it refers to (\"the old one\", \"him\", \"there\", \"the other one\"). If it is already "
            "standalone or about something new, reply exactly SAME. Reply with the question only.")
_ANAPHOR = {"one", "ones", "before", "after", "other", "him", "her", "them", "that", "this", "there", "it", "he",
            "she", "they", "same"}


def needs_llm(f) -> bool:
    if f.plain or f.rewritten or f.inline or f.carried or f.followup_of:
        return False
    from .frame import NUMBER_TYPES, norm
    if any(t not in NUMBER_TYPES for t, _ in f.slots):
        return False                      # a new explicit entity: a new topic, nothing to resolve
    toks = [norm(w) for w in f.words if norm(w)]
    return 0 < len(toks) <= 6 and bool(set(toks) & _ANAPHOR)


async def llm_rewrite(f, context: str, meter: dict) -> bool:
    """Rewrite f in place from the previous turn (context = "Q: ...\\nA: ..."). True when it rewrote. Never raises."""
    import asyncio
    import time as _t
    from . import answer, config
    from .frame import STOP, norm
    t0 = _t.perf_counter()
    meter["llm"] = True
    try:
        async def call():
            r = await answer.client().messages.create(
                model=config.ANSWER_MODEL, max_tokens=40, **answer.model_opts(config.ANSWER_MODEL),
                system=[{"type": "text", "text": _LLM_SYS, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"{context[:500]}\nFollow-up: {f.clean_text}"}])
            return r
        r = await asyncio.wait_for(call(), LLM_TIMEOUT_S)
        text = "".join(getattr(b, "text", "") for b in r.content).strip().strip('"')
        u = getattr(r, "usage", None)
        meter.update(tokens_in=getattr(u, "input_tokens", None), tokens_out=getattr(u, "output_tokens", None))
    except asyncio.TimeoutError:
        meter.update(ms=round((_t.perf_counter() - t0) * 1000), result="timeout")
        return False
    except Exception as e:
        meter.update(ms=round((_t.perf_counter() - t0) * 1000), result=f"error: {type(e).__name__}"[:60])
        return False
    meter["ms"] = round((_t.perf_counter() - t0) * 1000)
    if not text or text.upper().startswith("SAME") or len(text) > 200:
        meter["result"] = "same"
        return False
    q = re.sub(r"\s+", " ", re.sub(r"[^\w' -]+", " ", text.lower())).strip()
    f.inline = f.rewritten = q
    f.mem_ref = f.mem_ref or "(rewritten)"
    m = re.match(r"Q: (.*)", context or "")
    f.followup_of = m.group(1).strip() if m else ""
    f.content = tuple(dict.fromkeys(w for w in (norm(x) for x in q.split()) if w and w not in STOP))
    meter["result"] = "rewritten"
    meter["rewrite"] = q
    return True
