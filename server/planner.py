"""Planner: a multi-part spoken question -> sub-questions, each routed to its own lane, planned WHILE the user speaks.

  comparison : "which is taller, Everest or K2"          -> how tall is everest | how tall is k2
               "compare Nvidia and AMD revenue"          -> nvidia revenue | amd revenue
               "is the Eiffel Tower taller than ..."     -> how tall is the eiffel tower | how tall is ...
  conjunction: "flights to Denver and the weather there" -> flights to denver | what's the weather in denver
               "what's the population of France and Germany" -> ... of france | ... of germany
  multi-hop  : "how old was the founder of Amazon when it was founded"
               -> who founded amazon | when was amazon founded | when was {founder} born   (the last part waits for the
                  first: its subject is the first part's answer, read off the Wikidata fact, free)

Rules only (regex over the question as meant, with spoken corrections put back in place by router.repaired_text, plus
the Arc Tagger's entity phrases for "X and Y" splits). No model call: a decomposition costs ~0.1 ms per partial.

Every part becomes a synthetic Frame (frame.build over its words, entity spans carried from the parent), so the router
plans each part on its own (a part can be a free card / Wikidata fact while its sibling goes to the web). The speculator
(hook in speculator.py) fires a part as soon as it is stable: a CLOSED part (the speaker has moved past it: the left
side of "and", the first option before "or") at once, an open one after SPEC_SETTLE_MS or a likely end of turn.

decompose(frame) returns None for every single-part question: then nothing in the speculator changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config
from . import frame as framemod

# ------------------------------------------------------------------------------------------------- text
_LEAD = re.compile(r"^(?:(?:so|um|uh|uhm|hey|okay|ok|well|oh|hmm|and|like|quick question|tell me|can you tell me|"
                   r"do you know|i wonder|i was wondering)\s+)+")


def norm_text(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' &.-]+", " ", t)
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" '-")
    return _LEAD.sub("", t).strip()


def question_text(f) -> str:
    """The question as meant (a spoken correction spliced in place), normalized."""
    from .router import repaired_text
    try:
        return norm_text(repaired_text(f))
    except Exception:
        return norm_text(f.clean_text)


# ------------------------------------------------------------------------------------------------- plan types
@dataclass
class Part:
    id: str                 # p1, p2, ... in spoken order
    text: str               # the sub-question ("how tall is everest"); a bridge part: its template with {B}
    kind: str               # option | slot | clause | hop | bridge | attr
    closed: bool = False    # the speaker has moved past it: its words can't change any more
    entity: str = ""        # what this part is about (UI label)
    needs: str = ""         # bridge: id of the hop part whose answer fills {B}
    fallback: str = ""      # bridge: the question to search when the hop yields no entity
    role: str = ""          # hop: the role asked about ("founder", "author", "capital")

    @property
    def key(self) -> str:
        return self.text

    def resolve(self, bridge: str | None) -> str:
        if not self.needs:
            return self.text
        return self.text.replace("{B}", bridge) if bridge else self.fallback

    def to_json(self) -> dict:
        return {"id": self.id, "text": self.text, "kind": self.kind, "closed": self.closed, "entity": self.entity,
                "needs": self.needs}


@dataclass
class MPlan:
    kind: str               # compare | conj | hop
    parts: list = field(default_factory=list)
    why: str = ""
    text: str = ""
    combine: str = ""       # how the answer puts the parts together ("say which is taller", "do the arithmetic")

    @property
    def sig(self) -> tuple:
        return tuple(p.text for p in self.parts)

    @property
    def complete(self) -> bool:
        return len(self.parts) >= 2

    def part(self, pid: str) -> Part | None:
        return next((p for p in self.parts if p.id == pid), None)

    def to_json(self) -> dict:
        return {"kind": self.kind, "why": self.why, "combine": self.combine, "parts": [p.to_json() for p in self.parts]}


# ------------------------------------------------------------------------------------------------- comparisons
# comparative -> (sub-question template, what the answer must say). {x} = the option.
_CMP = [
    (r"taller|higher|shorter", "how tall is {x}", "which is taller"),
    (r"born first|born earlier|born later|born before", "when was {x} born", "who was born first"),
    (r"older|younger", None, "which is older"),               # person: born; thing: founded
    (r"founded first|founded earlier|started first", "when was {x} founded", "which came first"),
    (r"came out first|out first|released first|came first|launched first|invented first|built first|first",
     "when was {x} released",
     "which came first"),
    (r"more populous|more people|more residents|bigger population|larger population|higher population",
     "what is the population of {x}", "which has more people"),
    (r"bigger|larger|smaller", "how big is {x}", "which is bigger"),
    (r"longer", "how long is {x}", "which is longer"),
    (r"deeper", "how deep is {x}", "which is deeper"),
    (r"faster|slower|quicker", "how fast is {x}", "which is faster"),
    (r"heavier|lighter", "how heavy is {x}", "which is heavier"),
    (r"richer|wealthier", "what is the net worth of {x}", "who is richer"),
    (r"hotter|colder|warmer|cooler", "what's the weather in {x}", "where it is warmer"),
    (r"cheaper|more expensive|pricier", "how much does {x} cost", "which is cheaper"),
    (r"more valuable|worth more", "what is {x} worth", "which is worth more"),
    (r"closer|farther|further", "how far is {x}", "which is closer"),
]
_CMP_ALT = "|".join(p for p, _, _ in _CMP)
_COPULA = r"(?:is|was|are|were|s|'s|has|had|have|got|came|come|comes|were|did)"
_WHICH = r"(?:which|who|what)(?: one| ones)?(?:'s| (?:[a-z]+ ){0,2}?" + _COPULA + r")"
# "which is taller everest or k2", "which mountain is taller ...", "who was born first ...", "which one's bigger ..."
_CHOICE = re.compile(rf"^{_WHICH} (?P<cmp>{_CMP_ALT})(?: (?:one|of the two|than the other))?"
                     rf"(?: (?P<rest>.*))?$")
_CHOICE_HAS = re.compile(rf"^(?:which|who|what)(?: [a-z]+)?(?: one)? (?:has|had) (?:the |a )?(?P<cmp>more|bigger|larger|"
                         rf"higher|smaller|fewer|lower) (?P<noun>population|people|residents)(?: (?P<rest>.*))?$")
# "everest or k2 which is taller", "python versus java which came out first"
_CHOICE_AFTER = re.compile(rf"^(?P<opts>.+?) (?:which|who)(?: one)?(?:'s| [a-z]+)? (?:(?:is|was|s|'s|came|has) )?"
                           rf"(?:the )?(?P<cmp>{_CMP_ALT})$")
# "is everest taller than k2"
_THAN = re.compile(rf"^(?:is|are|was|were) (?P<a>.+?) (?P<cmp>{_CMP_ALT}) than (?P<b>.+)$")
_OR = re.compile(r" (?:or|versus|vs) ")
# comparatives that also open ordinary questions ("who was the first person to ...", "which is better for ..."): split
# only once the options are said ("... X or Y")
_NEEDS_OR = {"first", "out first", "came first", "better", "closer", "farther", "further", "older", "younger",
             "bigger", "larger", "smaller", "longer", "faster", "cheaper", "hotter", "colder", "warmer", "cooler"}
_PERSON_WH = re.compile(r"^who\b")
_PRONOUNS = {"it", "he", "she", "they", "this", "that", "these", "those", "them"}


def _cmp_template(cmp: str, x: str, person: bool, place: bool = True) -> str:
    if re.fullmatch(r"hotter|colder|warmer|cooler", cmp) and not place:
        return f"how hot is {x}"
    for pat, tpl, _ in _CMP:
        if re.fullmatch(pat, cmp):
            if tpl is None:                                   # older / younger
                tpl = "when was {x} born" if person else "when was {x} founded"
            return tpl.format(x=x)
    return f"{x} {cmp}"


def _cmp_combine(cmp: str) -> str:
    for pat, _, say in _CMP:
        if re.fullmatch(pat, cmp):
            return f"then say {say} (compare the two values)"
    return "then compare them"


_CUE_W = {"make", "that", "actually", "sorry", "mean", "meant", "wait", "no", "scratch", "rather", "i"}


def _opt(s: str) -> str:
    s = s.strip(" ,'")
    s = re.sub(r"^(?:or|and|vs|versus) ", "", s)
    s = re.sub(r" (?:or|and|vs|versus)$", "", s)          # "mozart or": the second option is still coming
    if set(s.split()) & _CUE_W:
        return ""                                        # mid-correction ("... or actually make"): not an option yet
    return s.strip()


def _is_person(x: str, f) -> bool:
    for t, s in getattr(f, "slots", ()) or ():
        if t == "person" and (s in x or x in s):
            return True
    return False


def _choice_parts(cmp: str, opts: list[str], closed_upto: int, f, person_q: bool) -> list[Part]:
    parts = []
    for i, x in enumerate(opts[:config.PLAN_MAX_PARTS]):
        x = _opt(x)
        if not x or len(x.split()) > 6:
            continue
        person = person_q                         # "who is older" -> born; "which is older" -> founded
        place = any(t == "place" and (s_ in x or x in s_) for t, s_ in (getattr(f, "slots", ()) or ()))
        parts.append(Part(id=f"p{len(parts) + 1}", text=_cmp_template(cmp, x, person, place), kind="option",
                          closed=i < closed_upto, entity=x))
    return parts


def _comparison(t: str, f) -> MPlan | None:
    m = _THAN.match(t)
    if m:
        cmp, a, b = m.group("cmp"), m.group("a"), m.group("b")
        if len(a.split()) > 4 or len(b.split()) > 4 or re.search(r"\b(?:right now|today|this week|now)\b", b) \
                or {a, b} & _PRONOUNS or re.search(r"\d|\b(?:hundred|thousand|million|billion)\b", a + " " + b):
            return None                          # "is it heavier than the 13", "worth more than one billion dollars"
        parts = _choice_parts(cmp, [a, b], 1, f, False)
        return MPlan("compare", parts, f"comparison: {cmp} than", t, _cmp_combine(cmp)) if parts else None
    m = _CHOICE.match(t) or None
    cmp = rest = None
    if m:
        cmp, rest = m.group("cmp"), m.group("rest") or ""
    else:
        m2 = _CHOICE_HAS.match(t)
        if m2:
            cmp, rest = "more people", m2.group("rest") or ""
    if cmp is not None:
        rest = re.sub(r"^(?:between |out of |of )", "", rest.strip())
        if not rest:
            return None                          # "which is taller" and nothing yet: nothing to search
        if rest.split()[0] in ("to", "for", "than", "in", "at", "on", "with", "from"):
            return None                          # "which airport is closer to manhattan, jfk or ...": not parsed
        if cmp in _NEEDS_OR and not _OR.search(" " + rest + " "):
            return None                          # "who was the first woman to ...": a superlative, not a choice
        opts = _OR.split(rest)
        parts = _choice_parts(cmp, opts, len(opts) - 1, f, bool(_PERSON_WH.match(t)))
        return MPlan("compare", parts, f"comparison: {cmp}", t, _cmp_combine(cmp)) if parts else None
    m = _CHOICE_AFTER.match(t)
    if m and _OR.search(m.group("opts")):
        cmp = m.group("cmp")
        opts = _OR.split(m.group("opts"))
        parts = _choice_parts(cmp, opts, len(opts), f, t.split(" or ")[-1].startswith("who"))
        return MPlan("compare", parts, f"comparison: {cmp}", t, _cmp_combine(cmp)) if len(parts) >= 2 else None
    return None


# "compare nvidia and amd revenue", "compare the populations of japan and germany", "nvidia vs amd revenue"
_COMPARE = re.compile(r"^(?:can you |could you |please )?(?:compare|comparison of|contrast)(?: the)? (?P<body>.+)$")
_ATTR_OF = re.compile(r"^(?P<attr>.+?) (?:of|for|between|in) (?P<ents>.+)$")
_ATTR_Q = {"population": "what is the population of {x}", "populations": "what is the population of {x}",
           "capital": "what is the capital of {x}", "capitals": "what is the capital of {x}",
           "height": "how tall is {x}", "heights": "how tall is {x}", "founder": "who founded {x}",
           "founders": "who founded {x}", "ceo": "who is the ceo of {x}", "ceos": "who is the ceo of {x}",
           "area": "how big is {x}", "size": "how big is {x}", "sizes": "how big is {x}",
           "age": "how old is {x}", "ages": "how old is {x}", "weather": "what's the weather in {x}",
           "currency": "what is the currency of {x}", "currencies": "what is the currency of {x}",
           "headquarters": "where is the headquarters of {x}", "languages": "what language is spoken in {x}",
           "language": "what language is spoken in {x}"}
_SEP = re.compile(r" (?:and|with|to|vs|versus|or) ")
_SEP_NOVERB = re.compile(r" (?:vs|versus) ")
# attribute words that may trail the last option ("compare nvidia and amd revenue")
_ATTR_WORDS = set(_ATTR_Q) | {"revenue", "revenues", "price", "prices", "salary", "salaries", "specs", "battery", "camera",
                              "cameras", "sales", "profit", "profits", "earnings", "stock", "stocks", "valuation",
                              "market", "cap", "gdp", "economy", "economies", "weight", "speed", "range", "mileage",
                              "cost", "costs", "rent", "taxes", "crime", "climate", "rating", "ratings", "reviews",
                              "share", "shares", "users", "subscribers"}


def _attr_part(attr: str, x: str) -> str:
    a = attr.strip()
    if a in _ATTR_Q:
        return _ATTR_Q[a].format(x=x)
    return f"{x} {a}".strip()


def _split_ents(s: str, f, sep=_SEP) -> tuple[list[str], str]:
    """'nvidia and amd revenue' -> (['nvidia', 'amd'], 'revenue'): the options, then what's left after the last one
    (the attribute: a tagged entity phrase ends the last option, else trailing attribute words do)."""
    pieces = sep.split(s)
    if len(pieces) < 2:
        return [], ""
    *heads, last = pieces
    phrases = sorted((p for p in getattr(f, "phrases", ()) or () if p), key=len, reverse=True)
    tail = ""
    for p in phrases:
        if last == p or last.startswith(p + " "):
            tail = last[len(p):].strip()
            last = p
            break
    else:
        w = last.split()
        k = len(w)
        while k > 1 and w[k - 1] in _ATTR_WORDS:
            k -= 1
        if 0 < k < len(w):
            last, tail = " ".join(w[:k]), " ".join(w[k:])
    return [h for h in heads if h] + [last], tail


def _compare_verb(t: str, f) -> MPlan | None:
    m = _COMPARE.match(t)
    body = m.group("body") if m else None
    if body is None:
        if not re.search(r" (?:vs|versus) ", t) or t.split()[0] in _WH | _REQ | {"i", "my", "do", "does"} \
                or len(t.split()) > 8:
            return None                           # "how do i open the terminal in vs code"
        opts, attr = _split_ents(t, f, _SEP_NOVERB)
        opts = [_opt(o) for o in opts if _opt(o)][:config.PLAN_MAX_PARTS]
        if len(opts) < 2 or any(len(o.split()) > 4 for o in opts):
            return None
        return MPlan("compare", [Part(f"p{i + 1}", _attr_part(attr, x) if attr else x, "attr", False, x)
                                 for i, x in enumerate(opts)], "versus " + (attr or "options"), t, "then compare them")
    attr = ""
    ma = _ATTR_OF.match(body)
    if ma and _SEP.search(ma.group("ents")) and len(ma.group("attr").split()) <= 3:
        attr, ents = ma.group("attr"), ma.group("ents")
        opts, tail = _split_ents(ents, f)
        if tail:
            opts[-1] = f"{opts[-1]} {tail}"
    else:
        opts, attr = _split_ents(body, f)
    opts = [_opt(o) for o in opts if _opt(o)][:config.PLAN_MAX_PARTS]
    if len(opts) < 2:
        return None
    parts = [Part(id=f"p{i + 1}", text=_attr_part(attr, x) if attr else x, kind="attr", closed=False, entity=x)
             for i, x in enumerate(opts)]
    return MPlan("compare", parts, "compare " + (attr or "options"), t, "then compare them")


# ------------------------------------------------------------------------------------------------- conjunctions
_CONJ = re.compile(r" (?:and also|as well as|and then|and) ")
_WH = {"who", "what", "what's", "whats", "when", "where", "which", "why", "how", "is", "are", "does", "do", "did",
       "can", "will", "was", "were", "whose"}
_REQ = {"tell", "show", "find", "give", "get", "search", "look", "check", "how", "should", "can", "could", "would",
        "will", "whats", "what's", "who's", "where's", "when's", "how's", "flights", "weather", "directions"}
_DET = {"the", "a", "an", "its", "their", "his", "her", "my", "your"}
_PREP = {"in", "of", "for", "to", "at", "on", "from", "about", "near", "by", "with"}
_PRON_THING = {"it", "its", "it's", "they", "them", "their", "that"}
_PRON_PERSON = {"he", "she", "his", "her", "him"}
_BINOMIAL = {("salt", "pepper"), ("rock", "roll"), ("black", "white"), ("bread", "butter"), ("pros", "cons"),
             ("ups", "downs"), ("mac", "cheese"), ("fish", "chips"), ("peanut", "butter"), ("research", "development"),
             ("pride", "prejudice"), ("war", "peace"), ("crime", "punishment"), ("romeo", "juliet"),
             ("sense", "sensibility"), ("johnson", "johnson"), ("procter", "gamble"), ("at", "t"), ("b", "b"),
             ("q", "a"), ("r", "b"), ("arts", "crafts"), ("law", "order"), ("trial", "error"), ("give", "take"),
             ("back", "forth"), ("more", "more"), ("over", "over"), ("bits", "pieces"), ("lord", "rings")}
_AND_NOT = re.compile(r"\b(?:between|both|difference|together|relationship|married|versus|vs)\b")


def _phrase_spans(toks: list[str], f) -> list[tuple[int, int, str]]:
    """(start, end, type) token spans of the parent frame's entity phrases inside `toks`."""
    out = []
    types = {s: t for t, s in getattr(f, "slots", ()) or () if t not in framemod.NUMBER_TYPES}
    for p in sorted({p for p in (getattr(f, "phrases", ()) or ()) if p} | set(types), key=len, reverse=True):
        pw = p.split()
        if not pw:
            continue
        for i in range(len(toks) - len(pw) + 1):
            if toks[i:i + len(pw)] == pw and not any(a < i + len(pw) and i < b for a, b, _ in out):
                out.append((i, i + len(pw), types.get(p, "thing")))
    return sorted(out)


def _last_entity(spans, toks, person: bool | None = None) -> str:
    for a, b, t in reversed(spans):
        if person is None or (t == "person") == person:
            return " ".join(toks[a:b])
    return ""


def _resolve(right: list[str], left: list[str], spans_left) -> tuple[list[str], bool]:
    """Pronouns in the right clause -> the left clause's entity ("the weather there" -> "the weather in denver")."""
    out, changed = [], False
    place = _last_entity([s for s in spans_left if s[2] == "place"], left) or _last_entity(spans_left, left)
    thing = _last_entity([s for s in spans_left if s[2] != "person"], left) or _last_entity(spans_left, left)
    person = _last_entity(spans_left, left, person=True) or thing
    for w in right:
        if w == "there" and place:
            out += ["in"] + place.split() if (not out or out[-1] not in _PREP) else place.split()
            changed = True
        elif w in _PRON_THING and thing and w != "that":
            out += thing.split() if w not in ("its", "their") else (thing + "'s").split()
            changed = True
        elif w in _PRON_PERSON and person:
            out += person.split() if w not in ("his", "her") else (person + "'s").split()
            changed = True
        else:
            out.append(w)
    return out, changed


_AUX_FRONT = re.compile(r"^(when|where|how \w+|what year|what) (it|he|she|they) (was|is|were|are) (\w+)(.*)$")


def _wh_order(s: str) -> str:
    """'when it was founded' -> 'when was it founded' (a subordinate clause read as a question)."""
    m = _AUX_FRONT.match(s)
    return f"{m.group(1)} {m.group(3)} {m.group(2)} {m.group(4)}{m.group(5)}" if m else s


def _conjunction(t: str, f, depth: int = 0) -> MPlan | None:
    toks = t.split()
    spans = _phrase_spans(toks, f)
    padded = " " + t + " "
    for m in _CONJ.finditer(padded):
        i = len(padded[:m.start()].split())       # token index of the conjunction
        n_conj = len(m.group(0).split())
        explicit = n_conj > 1                     # "and also" / "as well as": the speaker marks a second question
        left, right = toks[:i], toks[i + n_conj:]
        if not explicit and _AND_NOT.search(" ".join(left[-6:] + right[:2])):
            continue                              # "difference between X and Y", "both X and Y": one question
        if not left or len(left) < 2 or len(left) > 14 or len(right) > 10:
            continue
        if left[0] not in _WH | _REQ and len(left) > 4:
            continue                              # a story, not a request ("my mom loves X and her birthday's ...")
        if any(a < i < b for a, b, _ in spans):
            continue                              # inside an entity phrase ("pride and prejudice")
        if right and (left[-1], right[0]) in _BINOMIAL:
            continue
        if left[-1] in framemod._NUMW or left[-1].isdigit() or (right and (right[0].isdigit() or right[0] in
                                                                             framemod._NUMW)):
            continue                              # "two and a half", "68 at 10 pm and 72 at 6 am"
        lspans = [s for s in spans if s[1] <= i]
        rspans = [(a - i - n_conj, b - i - n_conj, ty) for a, b, ty in spans if a >= i + n_conj]
        lcontent = [w for w in left if w not in framemod.STOP]
        if not right:
            continue                              # "... and" with nothing after it yet
        r0 = right[0]
        rcontent = [w for w in right if w not in framemod.STOP]
        has_pron = any(w in _PRON_THING | _PRON_PERSON | {"there"} for w in right)
        # (a) a new question: "... and when was it founded", "... and what's the weather there"
        if r0 in _WH and (rcontent or has_pron):
            rr, changed = _resolve(right, left, lspans)
            if has_pron and not changed:
                if not explicit or not lcontent:
                    continue                      # "... and how does it work" with no entity for "it": one question
                rr, changed = _resolve(right, left, [(left.index(lcontent[-1]), left.index(lcontent[-1]) + 1, "thing")])
            r_text = _wh_order(" ".join(rr))
            plan = _two("clause", left, r_text, lspans, rspans, rr, t, "conjunction: a second question")
            if depth == 0 and config.PLAN_MAX_PARTS > 2:
                more = _conjunction(r_text, f, depth + 1)     # "... and what's the range and also is there a waitlist"
                if more is not None and len(more.parts) >= 2:
                    plan.parts = plan.parts[:1] + [Part(f"p{k + 2}", p.text, p.kind, p.closed or k == 0, p.entity)
                                                   for k, p in enumerate(more.parts[:config.PLAN_MAX_PARTS - 1])]
            return plan
        # (b) a noun phrase about the same thing: "... and the weather there", "... and its population"
        if has_pron and rcontent and r0 in _DET and len(right) <= 6:
            rr, changed = _resolve(right, left, lspans)
            if not changed:
                continue
            r_text = " ".join(rr)
            if rr[0] in _DET or rr[0] not in _WH:
                r_text = "what's " + r_text
            return _two("clause", left, r_text, lspans, rspans, rr, t, "conjunction: a second request")
        # (c) the same question about a second thing: "population of france and germany", "in paris and in london",
        #     "calories in an apple and a banana" (both sides tagged entities, or both determiner phrases)
        sub = _slot_swap(left, right, lspans, rspans)
        if sub:
            return MPlan("conj", [Part("p1", " ".join(left), "slot", True, sub[1]),
                                  Part("p2", sub[0], "slot", False, sub[2])], "conjunction: same question, two things",
                         t, "answer it for each")
    return None


def _two(kind, left, r_text, lspans, rspans, rr, t, why) -> MPlan:
    return MPlan("conj", [Part("p1", " ".join(left), kind, True, _last_entity(lspans, left)),
                          Part("p2", r_text, kind, False, _last_entity(rspans, rr) if rspans else "")],
                 why, t, "answer both parts")


def _slot_swap(left, right, lspans, rspans):
    """The right side replaces the left side's last noun phrase: (new question, left entity, right entity) or None."""
    if not right or len(right) > 6 or any(w.endswith("'s") for w in right):
        return None                                # "tom holland and zendaya's wedding": joint, not two questions
    r = list(right)
    prep = None
    if r[0] in _PREP and len(r) > 1:
        prep, r = r[0], r[1:]
    # "the capital of japan and the capital of brazil": the right side repeats the left's head noun -> replace from it
    head = r[1] if r[0] in _DET and len(r) > 1 else r[0]
    if head not in framemod.STOP and head in left[1:]:
        k = len(left) - 1 - left[::-1].index(head)
        if k > 0 and left[k - 1] in _DET and r[0] in _DET:
            k -= 1
        return " ".join(left[:k] + r), " ".join(left[k:]), " ".join(r)
    # left tail = a tagged entity ending the left side, or the determiner phrase ending it
    tail_a = None
    if lspans and lspans[-1][1] == len(left):
        tail_a = lspans[-1][0]
        if tail_a > 0 and left[tail_a - 1] in _DET:
            tail_a -= 1
        off = len(right) - len(r)
        whole = [x for x in rspans if x[0] - off <= (1 if r[0] in _DET else 0) and x[1] == len(right)]
        if not whole or whole[0][2] != lspans[-1][2]:
            return None                            # "population of france and germany": two entities of one type
    elif r[0] in ("a", "an", "the") and len(r) >= 2:
        for k in range(len(left) - 1, max(-1, len(left) - 5), -1):
            if left[k] in ("a", "an", "the") and k > 0:
                tail_a = k
                break
        if tail_a is None:
            return None
    else:
        return None
    if prep and (tail_a == 0 or left[tail_a - 1] != prep):
        return None                                # "in paris and in london": the same preposition on both sides
    new = left[:tail_a] + r
    return " ".join(new), " ".join(left[tail_a:]), " ".join(r)


# ------------------------------------------------------------------------------------------------- multi-hop
_ROLE_Q = {"founder": "who founded {e}", "founders": "who founded {e}", "co-founder": "who founded {e}",
           "cofounder": "who founded {e}", "ceo": "who is the ceo of {e}", "author": "who wrote {e}",
           "writer": "who wrote {e}", "director": "who directed {e}", "creator": "who created {e}",
           "inventor": "who invented {e}", "capital": "what is the capital of {e}",
           "president": "who is the president of {e}", "prime minister": "who is the prime minister of {e}",
           "leader": "who is the leader of {e}", "painter": "who painted {e}", "architect": "who designed {e}",
           "composer": "who composed {e}", "discoverer": "who discovered {e}", "owner": "who owns {e}",
           "singer": "who sang {e}", "manufacturer": "which company makes {e}", "maker": "which company makes {e}"}
_ROLE = "|".join(sorted((re.escape(k) for k in _ROLE_Q), key=len, reverse=True))
_HOP = re.compile(rf"\bthe (?P<role>{_ROLE}) of (?P<e>.+?)(?= (?:when|where|who|in|at|born|die|died|before|after)\b|$)")
_SINGLE_HEAD = re.compile(r"^(?:(?:who|what|which) (?:is|was|s|'s|are|were)|who's|what's|whats|tell me|name)?$")
_PERSON_ROLE = {"founder", "founders", "co-founder", "cofounder", "ceo", "author", "writer", "director", "creator",
                "inventor", "president", "prime minister", "leader", "painter", "architect", "composer",
                "discoverer", "owner", "singer"}


def _multihop(t: str, f) -> MPlan | None:
    m = None
    for m in _HOP.finditer(t):
        pass                                      # the last "the <role> of X" (innermost)
    if m is None:
        return None
    role, e = m.group("role"), m.group("e").strip()
    e = re.sub(r"^(?:the )", "", e) if role not in ("capital",) else e
    if not e or len(e.split()) > 6:
        return None
    head, tail = t[:m.start()].strip(), t[m.end():].strip()
    if len(head.split()) > 5 or len(tail.split()) > 6:
        return None
    if _SINGLE_HEAD.match(head) and (not tail or tail.split()[0] in ("in", "at", "of", "for", "on")):
        return None                               # "who is the founder of tesla": one hop, the router handles it
    hop = Part("p1", _ROLE_Q[role].format(e=e), "hop", bool(tail), e, role=role)
    parts = [hop]
    combine = "then answer the question as asked"
    person = role in _PERSON_ROLE
    tmpl = None
    clause = None
    if tail.startswith(("when ", "where ", "before ", "after ")) and re.match(r"^(?:how old|what age)\b", head):
        # "how old was the founder of tesla when it was founded": his birth date + the clause's date
        tmpl = "when was {B} born"
        clause = re.sub(r"\b(?:it|they|that)\b", e, _wh_order(tail), count=1)
        combine = "then do the arithmetic (age = year of the event - birth year) and give the age"
    elif re.match(r"^(?:how old|what age) (?:is|was)$", head) and not tail:
        tmpl = "when was {B} born" if person else "when was {B} founded"
        combine = "then give the age"
    elif head:
        tmpl = (head + " {B} " + tail).strip()
    elif tail:
        tmpl = ("{B} " + tail).strip()
    if tmpl:
        fb = tmpl.replace("{B}", f"the {role} of {e}")
        parts.append(Part("p2", tmpl, "bridge", False, "", needs="p1", fallback=fb))
    if clause:
        parts.append(Part(f"p{len(parts) + 1}", clause, "clause", False, e))
    if len(parts) < 2:
        return None
    return MPlan("hop", parts, f"multi-hop: the {role} of {e}", t, combine)


# ------------------------------------------------------------------------------------------------- entry points
_memo: dict = {}


def decompose(f) -> MPlan | None:
    """The plan for this frame, or None (a single-part question: nothing changes)."""
    if not config.PLAN or f is None or getattr(f, "route", "web") != "web" or getattr(f, "jailbreak", False):
        return None
    if getattr(f, "followup_of", "") or getattr(f, "convert", ()):
        return None
    t = question_text(f)
    if len(t.split()) < 3:
        return None
    key = (t, getattr(f, "phrases", ()), getattr(f, "slots", ()))
    if key in _memo:
        return _memo[key]
    p = None
    rules = (_comparison, _multihop, _compare_verb, _conjunction)
    if getattr(f, "corrected", False):
        rules = (_comparison, _compare_verb)     # a correction re-plans a comparison; hop / "and" splits of a repaired
        #                                          sentence are too often splices of the two versions
    for rule in rules:
        try:
            p = rule(t, f)
        except Exception:
            p = None
        if p is not None and p.parts:
            break
        p = None
    if len(_memo) > 512:
        _memo.clear()
    _memo[key] = p
    return p


def part_frame(text: str, parent, bridge_type: str = "") -> framemod.Frame:
    """A synthetic Frame for one part: its own words, the parent's entity spans re-found in them (+ the bridge
    entity), no intent (the router plans the part from its words). Every lane takes it like a spoken frame."""
    words = text.split()
    toks = [framemod.norm(w) for w in words]
    spans = []
    types = {s: ty for ty, s in (getattr(parent, "slots", ()) or ()) if ty not in framemod.NUMBER_TYPES}
    for p in getattr(parent, "phrases", ()) or ():
        types.setdefault(p, "thing")
    taken: set = set()
    for p in sorted(types, key=len, reverse=True):
        pw = p.split()
        for i in range(len(toks) - len(pw) + 1):
            if toks[i:i + len(pw)] == pw and not taken & set(range(i, i + len(pw))):
                ty = types[p] if types[p] != "thing" else "organization"
                spans.append({"family": "entity", "type": f"entity.mention.{ty}", "start": i, "end": i + len(pw),
                              "text": " ".join(words[i:i + len(pw)])})
                taken |= set(range(i, i + len(pw)))
    if bridge_type:
        # the bridge entity: a capitalized run the parent never said ("Jeff Bezos")
        caps = [i for i, w in enumerate(words) if w[:1].isupper() and i not in taken]
        if caps:
            a = caps[0]
            b = a
            while b + 1 < len(words) and words[b + 1][:1].isupper():
                b += 1
            spans.append({"family": "entity", "type": f"entity.mention.{bridge_type}", "start": a, "end": b + 1,
                          "text": " ".join(words[a:b + 1])})
    return framemod.build(words, spans, None, local_ok=False)


_NAME = r"((?:[A-Z][\w.'’-]+ ){0,2}[A-Z][\w'’-]+)"
_ROLE_RX = {
    "founder": [rf"(?:co-?)?founded (?:in \d{{4}} )?(?:by|together with) {_NAME}", rf"{_NAME} (?:co-?)?founded\b"],
    "author": [rf"(?:written|novel|book|play|poem|authored) by {_NAME}", rf"\bby (?:the )?(?:English |American )?"
               rf"(?:author |writer |novelist |poet )?{_NAME}"],
    "director": [rf"directed by {_NAME}", rf"{_NAME} directed\b"],
    "ceo": [rf"(?:CEO|chief executive officer|Chief Executive Officer)(?: is)?,? {_NAME}",
            rf"{_NAME},? (?:is |has been |serves as )?(?:the |its )?(?:co-?founder,? )?(?:president and )?(?:CEO|chief executive)"],
    "capital": [rf"capital(?: city)?(?: and largest city)? (?:is|of [A-Z]\w+ is) {_NAME}", rf"{_NAME} is the capital"],
    "inventor": [rf"invented by {_NAME}", rf"{_NAME} invented\b"],
    "creator": [rf"created by {_NAME}", rf"{_NAME} created\b"],
    "painter": [rf"painted by {_NAME}", rf"\bby (?:the )?(?:Italian |Dutch |French |Spanish )?(?:painter |artist )?{_NAME}"],
    "composer": [rf"composed by {_NAME}"],
    "president": [rf"(?:President|president) {_NAME}"],
}
_ROLE_RX["writer"] = _ROLE_RX["author"]
_ROLE_RX["founders"] = _ROLE_RX["co-founder"] = _ROLE_RX["cofounder"] = _ROLE_RX["founder"]
_NOT_NAME = {"The", "In", "It", "Its", "He", "She", "They", "This", "Inc", "Company", "Corporation", "American",
             "English", "British", "Wikipedia", "Wikidata"}


def bridge_of(docs, role: str = "") -> str | None:
    """The entity a hop part found: the first value of a Wikidata fact (free, typed), else a name the top results
    state in the role's own wording ("founded by Jeff Bezos", "written by Jane Austen", "CEO Jensen Huang"), else
    None (the bridge part then searches its unresolved wording)."""
    for d in docs or []:
        fact = (getattr(d, "meta", None) or {}).get("fact")
        if fact and fact.get("values"):
            v = str(fact["values"][0]).split(" (")[0].strip()
            if v and not re.search(r"\d{3,}", v):
                return v
    for d in (docs or [])[:3]:
        text = f"{getattr(d, 'text', '')}"
        for rx in _ROLE_RX.get(role, []):
            for mm in re.finditer(rx, text):
                name = mm.group(1).strip(" .,'’")
                ws = [w for w in name.split() if w not in _NOT_NAME]
                if ws and ws[0][:1].isupper():
                    return " ".join(ws)
    return None


def answer_hint(mp: MPlan, groups: list[tuple[str, list[int]]], question: str) -> str:
    """The part-aware instruction for the answer model: the parts, which sources belong to each, how to combine."""
    lines = []
    for i, (text, nums) in enumerate(groups, 1):
        src = ", ".join(f"[{n}]" for n in nums) if nums else "no source found"
        lines.append(f"({i}) {text}: {src}")
    return (f"This spoken question has {len(groups)} parts; the sources are grouped by part: " + "; ".join(lines) +
            f". Answer each part in order, each with its own citation, {mp.combine or 'then combine them'}. "
            f"If a part has no supporting source, say so briefly for that part only. Up to 60 words.")
