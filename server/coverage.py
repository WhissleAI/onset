"""Source coverage guard (COVER_GUARD): before an answer ships, do its sources mention what the question is about?

The speculator answers at the endpoint from its pool: every source its mid-sentence searches brought back. A pool built
while the question was still being spoken can be about something else (live P0: "what are some good food places in San
Jose" after a France turn: the pool held the France article, the paid web search for San Jose was still on the wire, and
the answer shipped from France: "The sources don't cover San Jose restaurants..."). The guard is a pre-answer check:

  key_terms(f)          the question's own subject(s): its typed entity mentions (not numbers), else the referent it
                        carried from memory, else its entity phrases
  covers(doc, terms)    the source names one of them (accent / case / punctuation-insensitive, whole words)
  split(ranked, terms)  (covering sources, the rest)

No network, ~0.1 ms. The speculator decides what to do with an uncovered pool (wait for / fire the web search)."""
from __future__ import annotations

import os
import re
import unicodedata

from .frame import NUMBER_TYPES

# flags live here (not in config.py) to keep this fix off the files other branches are editing
ENABLED = os.environ.get("COVER_GUARD", "1") == "1"
# at the endpoint, how long an uncovered pool waits for the web search that can cover it (in flight or fired now)
WAIT_S = float(os.environ.get("COVER_WAIT_S", "8.0"))

# the answer model's own admission (answer.py prompt: "say in a few words that the sources don't cover it")
_ADMITS = re.compile(r"\b(?:sources?|results?) (?:don't|do not|doesn't|does not|didn't|did not) "
                     r"(?:cover|mention|include|address|say|contain)|\bnot (?:covered|mentioned) (?:in|by) the sources?\b",
                     re.I)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " " + re.sub(r"[^\w]+", " ", s).strip() + " "


def key_terms(f) -> list[str]:
    """The question's subject phrase(s), normalised; [] when the question names none (then the guard does nothing)."""
    if getattr(f, "rewritten", ""):
        return []           # a rewritten relative reference ("the old one" -> the president before X): its answer is
                            # a NEW entity the sources name, not the carried one (server/ellipsis.py)
    out = [v for t, v in (f.slots or ()) if t not in NUMBER_TYPES and v]
    if not out and getattr(f, "carried", ""):
        out = [f.carried]
    if not out:
        out = [p for p in (getattr(f, "phrases", ()) or ()) if p]
    seen, res = set(), []
    for t in out:
        k = _fold(t).strip()
        if len(k) >= 2 and k not in seen:
            seen.add(k)
            res.append(k)
    return res


def covers(doc, terms: list[str]) -> bool:
    if not terms:
        return True
    hay = _fold(f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}")
    # every word of the name, as whole words ("Albert Einstein" covers "albert einstein"; "the Einstein family"
    # alone does not; "San José" covers "san jose")
    return any(all(f" {w} " in hay for w in t.split()) for t in terms)


def split(ranked: list, terms: list[str]) -> tuple[list, list]:
    """ranked: [(doc, score)] -> (covering, not covering), order kept."""
    yes = [(d, s) for d, s in ranked if covers(d, terms)]
    no = [(d, s) for d, s in ranked if not covers(d, terms)]
    return yes, no


def admits_uncovered(answer_text: str) -> bool:
    """The answer says its sources don't cover the question (post-answer signal, logged on the metric)."""
    return bool(_ADMITS.search((answer_text or "")[:240]))
