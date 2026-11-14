"""Open-mic gate, v2 (OPENMIC_GATE=2, default): is this ear turn a request to the assistant, or room talk?

The v1 rule (session.open_mic_skip) answered anything that wasn't read as a statement: on the director's open-mic test
scenes (eval/arcvalue/textlevel.py) it answered 55 % of the non-requests (questions to another person, TV, chit-chat)
and dropped 9 % of the requests, mostly elliptical follow-ups ("And chips.", "and tomorrow").

v2 answers a turn only on positive evidence, and holds the rest:
  1. follow-up rescue: a short fragment ("and chips", "what about orlando", "how far is it") within OPENMIC_FOLLOWUP_S
     of the last answered request, by the same speaker when Arc tells speakers apart, is a request (the live path
     resolves it from memory). Checked first.
  2. no content words: held ("empty").
  3. the v1 rule's statement / report hold.
  4. Arc Tagger device_directed (the utterance-level head) below OPENMIC_DD_MIN: held ("not_directed").
  5. request evidence, any of: a confident web intent; a sentence that starts with a question / request word or an
     imperative ("set", "add", "remind", ...) and is not addressed to a person ("mom, ...", "did you ...", "are we
     ..."); a question / command act with device_directed >= OPENMIC_DD_ACT and no person-addressed sentence.
     None of these: held ("no_request").
With DIRECTOR_ADDRESSED=1 the director's addressed head is ANDed on top (session._not_request).

Measured on the 86 held-out test scenes (521 Flux turns) and checked on the val (274) and train (1,965) turns, which
were not used to pick the rules' thresholds beyond the dd bar: see docs/ARC_VALUE.md (open mic).
"""
from __future__ import annotations

import re

from . import config
from . import frame as framemod

VOCATIVE = {"mom", "mommy", "mum", "dad", "daddy", "grandma", "grandpa", "granny", "nana", "honey", "babe", "baby",
            "sweetie", "sweetheart", "dear", "guys", "kids", "buddy", "bro", "dude", "sis", "son", "mama", "papa"}
_TO_PERSON = re.compile(r"^(?:did|do|are|were|have|has|had|will|would|should|was) (?:you|we|he|she|they|ya)\b")
_REQUEST = re.compile(r"^(what|whats|what's|who|whos|who's|whom|whose|when|where|wheres|where's|why|which|how|hows|"
                      r"how's|is|are|was|were|do|does|did|can|could|will|would|should|shall|may|might|has|have|had|"
                      r"tell|show|find|search|look|give|get|play|open|define|explain|list|compare|check|any|anything|"
                      r"set|add|remind|call|convert|navigate|draft|create|text|send|turn|translate|spell|calculate|"
                      r"book|email|start|read|put|make|write|say)$")
_FOLLOW = re.compile(r"^(?:(?:okay|ok|so|um|uh) )?(?:and|or|also|plus|what about|how about)\b")
_PRON = {"it", "its", "they", "them", "that", "there", "those", "he", "she", "his", "her", "this"}


def sentences(raw: list[str]) -> list[list[str]]:
    """Split the ear's raw words (with Flux's punctuation) into normalized sentences."""
    out, cur = [], []
    for w in raw:
        n = framemod.norm(w)
        if n:
            cur.append(n)
        if re.search(r"[.?!]$", str(w).strip()) and cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def to_person(s: list[str]) -> bool:
    return bool(s) and (s[0] in VOCATIVE or bool(_TO_PERSON.match(" ".join(s))))


def followup(f, ctx: dict | None) -> bool:
    """A short elliptical turn right after an answered request: "And chips.", "what about orlando", "how far is it"."""
    ctx = ctx or {}
    since = ctx.get("since_request_s")
    if since is None or since > config.OPENMIC_FOLLOWUP_S:
        return False
    a, b = ctx.get("speaker"), ctx.get("request_speaker")
    if a is not None and b is not None and a != b:
        return False                                   # Arc heard someone else: not the asker's follow-up
    toks = [t for t in (framemod.norm(w) for w in f.words) if t]
    if not toks or len(toks) > 7:
        return False
    if _FOLLOW.match(" ".join(toks)):
        return True
    asks = f.act_question or "question" in (f.acts or ()) or bool(_REQUEST.match(toks[0]))
    return asks and bool(set(toks) & _PRON)


def skip(f, ctx: dict | None = None, rule=None) -> str | None:
    """Why this turn is heard and not answered, or None to answer it. `rule` = the v1 rule (session.open_mic_skip)."""
    if followup(f, ctx):
        return None
    if not f.has_content():
        return "empty"
    why = rule(f) if rule is not None else None
    if why:
        return why
    dd = f.directed
    if dd is not None and dd < config.OPENMIC_DD_MIN:
        return "not_directed"
    ss = sentences((ctx or {}).get("raw") or f.words)
    if f.intent in framemod.WEB_INTENTS and f.intent_p >= 0.5:
        return None
    if any(s and not to_person(s) and _REQUEST.match(s[0]) for s in ss):
        return None
    try:                                               # a question a free lane answers ("sixty kilos in pounds")
        from . import router
        if router.plan(f).lane in ("card", "facts") and router.plan(f).vertical != "leads":
            return None
    except Exception:
        pass
    acts = set(f.acts or ())
    if (f.act_question or acts & {"question", "command_request"}) and not any(to_person(s) for s in ss) \
            and (dd or 0.0) >= config.OPENMIC_DD_ACT:
        return None
    return "no_request"
