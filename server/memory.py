"""Session memory v2: an entity memory across the turns of a conversation, user facts, and a local store.

  Memory.observe(...)        after each answered turn: the question's entities, the answer's new entities (typed by the
                             Arc Tagger, corrected by what the question asked for), Wikidata facts the answer used
                             (QID + property edges), and a "said" edge between the question's subject and the answer
  Memory.carry_for(words)    what a pronoun / reference in these words points at, as the frame's carry dict:
                             "it" / "he" / "there" / "its headquarters" / "that company" / "the first one"
  Memory.learn_user(words)   simple facts the user says about themselves ("I live in Denver", "I'm vegetarian")
  Memory.graph()             nodes + edges for the knowledge-graph panel (web/graph.js)
  Store                      runs/memory/<hash of client id>.json: the same browser gets its memory back after a
                             reconnect (TTL, "forget" clears it). Local only: user facts never leave the machine except
                             as the resolved word of a query ("weather tomorrow" -> "... denver").

Resolution order (multi-turn, newest turn first): a reference names the KIND it wants (he/she -> person, there ->
place, "its headquarters" -> organization, "that city" -> place, ...). In each turn, the entity the question ASKED for
(the answer's new entity: "capital of Australia?" -> Canberra; "which company makes the PlayStation?" -> Sony) comes
first when it is of that kind, then the question's own entities. "the first one" / "the other one" pick among the
conversation's subjects in turn order."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
from pathlib import Path

from . import config

# Arc Tagger mention type -> coarse kind
KIND = {"person": "person", "place": "place", "organization": "org", "team": "org"}
# nouns that name a kind ("that company", "which city ...")
TYPE_NOUNS = {
    "company": "org", "companies": "org", "firm": "org", "brand": "org", "manufacturer": "org", "maker": "org",
    "corporation": "org", "business": "org", "team": "org", "band": "org", "organization": "org", "startup": "org",
    "city": "place", "country": "place", "town": "place", "state": "place", "place": "place", "museum": "place",
    "capital": "place", "island": "place", "river": "place", "mountain": "place", "continent": "place",
    "person": "person", "guy": "person", "man": "person", "woman": "person", "lady": "person", "artist": "person",
    "author": "person", "writer": "person", "painter": "person", "scientist": "person", "singer": "person",
    "actor": "person", "actress": "person", "player": "person", "inventor": "person",
    "book": "thing", "novel": "thing", "movie": "thing", "film": "thing", "painting": "thing", "song": "thing",
    "album": "thing", "game": "thing", "console": "thing", "theory": "thing", "product": "thing", "show": "thing",
}
# what a follow-up asks about says which kind "it" is ("where is its headquarters" -> the company, not the console)
AFFINITY = {
    "headquarters": "org", "headquartered": "org", "hq": "org", "ceo": "org", "employees": "org", "revenue": "org",
    "stock": "org", "shares": "org", "founder": "org", "founders": "org", "subsidiary": "org", "owner": "org",
    "population": "place", "weather": "place", "mayor": "place", "climate": "place", "temperature": "place",
    "currency": "place", "language": "place", "languages": "place", "governor": "place", "elevation": "place",
    "born": "person", "died": "person", "wife": "person", "husband": "person", "married": "person",
    "spouse": "person", "children": "person", "kids": "person", "parents": "person", "nationality": "person",
}
PERSONAL = {"he", "she", "his", "her", "him", "hers", "himself", "herself"}
FEMALE = {"she", "her", "hers", "herself"}
MALE = {"he", "his", "him", "himself"}
# a relation noun in the question says the answer's person's gender ("who was her husband" -> Pierre is "he")
REL_MALE = {"husband", "father", "son", "brother", "king", "boyfriend", "dad", "grandfather", "uncle", "nephew"}
REL_FEMALE = {"wife", "mother", "daughter", "sister", "queen", "girlfriend", "mom", "grandmother", "aunt", "niece"}
SUBJ = {"it", "he", "she", "him", "they", "them", "this", "that", "its", "his", "her", "their", "there", "it's"}
POSSESSIVE = {"its", "his", "their", "her"}
GENERIC_REL = {"company", "city", "country", "theory", "what", "which", "who", "when", "where", "general"} | set(TYPE_NOUNS)
_ORD = {"first": 0, "original": 0, "second": 1, "third": 2}
_REF_ORD = re.compile(r"\b(?:(?:go|going|let's go|lets go|come) )?(?:back )?(?:to )?(?:the )?"
                      r"(first|second|third|original|other|earlier|previous) one\b|"
                      r"\b(?:go|going|let's go|lets go|come) back to (?:the )?(first|second|third|original|other|earlier|previous)\b")
_REF_DEM = re.compile(r"\b(?:that|this|the same) (" + "|".join(sorted(TYPE_NOUNS, key=len, reverse=True)) + r")\b")

# ---------------------------------------------------------------------------------------------- user facts
_CUT = r"(?=$| and | but | so | because | what| how| where| when| who| is | can | could | will | tell | please)"
USER_PATTERNS = [
    ("home", re.compile(r"\bi (?:live|am living|m living|'m living) in (?P<v>[a-z][a-z .'-]{1,40}?)" + _CUT)),
    ("home", re.compile(r"\bi(?: am|'m| m) based in (?P<v>[a-z][a-z .'-]{1,40}?)" + _CUT)),
    ("home", re.compile(r"\bi (?:just )?moved to (?P<v>[a-z][a-z .'-]{1,40}?)" + _CUT)),
    ("home", re.compile(r"\bmy (?:home|hometown|city) is (?P<v>[a-z][a-z .'-]{1,40}?)" + _CUT)),
    ("diet", re.compile(r"\bi(?: am|'m| m)(?: a)? (?P<v>vegetarian|vegan|pescatarian)\b")),
    ("diet", re.compile(r"\bi (?:don't|do not|dont) eat (?:any )?meat\b")),
    ("diet", re.compile(r"\bi (?:eat|keep) (?P<v>kosher|halal)\b")),
    ("allergy", re.compile(r"\bi(?: am|'m| m) allergic to (?P<v>[a-z][a-z -]{1,30}?)" + _CUT)),
]
USER_LABEL = {"home": "you live in {v}", "diet": "you're {v}", "allergy": "you're allergic to {v}"}
_WEATHER = {"weather", "forecast", "rain", "raining", "snow", "snowing", "temperature", "umbrella", "sunny", "cold",
            "hot", "warm", "windy", "degrees", "humid", "sunset", "sunrise"}
_NEAR = re.compile(r"\b(near me|nearby|around here|close to me|in my area|in town|around me|local)\b")
_FOOD = {"restaurant", "restaurants", "recipe", "recipes", "dinner", "lunch", "breakfast", "eat", "food", "meal",
         "meals", "cook", "cooking", "dish", "dishes", "snack", "snacks", "brunch"}
_TIME_PLACE = re.compile(r"\b(what time is it|time is it|sunset|sunrise)\b")


def norm(s: str) -> str:
    return re.sub(r"[^\w'\- ]+", "", (s or "").lower()).strip()


def kind_of(t: str) -> str:
    return KIND.get(t, "thing")


def names_thing(toks: list[str]) -> bool:
    """The question asks for a THING it names ("which company makes ...", "what is the capital of ...", "who makes
    ..."): a later "it" means that answer. A "where ..." question asks for a location of its subject: "it" stays the
    subject ("where is the Eiffel Tower? how tall is it?")."""
    s = " ".join(toks)
    m = re.search(r"\b(?:which|what) (\w+)", s)
    return bool((m and m.group(1) in TYPE_NOUNS and TYPE_NOUNS[m.group(1)] != "person")
                or re.search(r"\bwho (?:makes|made|manufactures|produces|owns)\b|\bcapital (?:city )?of\b", s)
                # "what is the tallest mountain in the world": the answer is the thing a later "it" means
                or re.search(r"\b(?:what|which)(?: is| was|'s)? the (?:\w+est|biggest|most \w+|first|largest) \b", s))


def asked_kind(toks: list[str]) -> str | None:
    """What kind of entity the question asks for: its answer's new entity is that kind."""
    s = " ".join(toks)
    m = re.search(r"\b(?:which|what) (\w+)", s)
    if m and m.group(1) in TYPE_NOUNS:
        return TYPE_NOUNS[m.group(1)]
    if re.search(r"\bwho (?:makes|made|manufactures|produces|owns)\b", s):
        return "org"
    if re.search(r"\bcapital (?:city )?of\b", s) or toks[:1] == ["where"]:
        return "place"
    if toks[:1] == ["who"] or re.search(r"\bwho (?:is|was|were|are)\b", s):
        return "person"
    return None


def learn_facts(toks: list[str]) -> list[tuple[str, str]]:
    s = " ".join(toks)
    out = []
    for key, rx in USER_PATTERNS:
        m = rx.search(s)
        if not m:
            continue
        v = (m.groupdict().get("v") or ("vegetarian" if key == "diet" else "")).strip(" .'-")
        if not v or len(v.split()) > 4:
            continue
        if key == "home" and v in ("the city", "a city", "the country", "a house", "an apartment", "the area"):
            continue
        if not any(k == key for k, _ in out):
            out.append((key, v))
    return out


class Memory:
    def __init__(self):
        self.ents: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.turns: list[dict] = []
        self.user: dict[str, dict] = {}
        self.focus: str | None = None          # the entity the last follow-up was resolved to
        self.updated = time.time()

    # ------------------------------------------------------------------ state
    def __bool__(self):
        return bool(self.ents or self.user)

    def clear(self, user: bool = True):
        self.ents, self.edges, self.turns, self.focus = {}, [], [], None
        if user:
            self.user = {}
        self.updated = time.time()

    def to_json(self) -> dict:
        return {"ents": self.ents, "edges": self.edges, "turns": self.turns, "user": self.user, "focus": self.focus,
                "updated": self.updated, "v": 2}

    @classmethod
    def from_json(cls, d: dict | None, conv_ttl_s: float | None = None, user_ttl_s: float | None = None) -> "Memory":
        m = cls()
        if not d or d.get("v") != 2:
            return m
        now = time.time()
        conv_ttl = config.MEM_CONV_TTL_S if conv_ttl_s is None else conv_ttl_s
        user_ttl = config.MEM_USER_TTL_S if user_ttl_s is None else user_ttl_s
        m.user = {k: v for k, v in (d.get("user") or {}).items() if now - float(v.get("at", 0)) <= user_ttl}
        if now - float(d.get("updated", 0)) <= conv_ttl:
            m.ents = d.get("ents") or {}
            m.edges = d.get("edges") or []
            m.turns = d.get("turns") or []
            m.focus = d.get("focus")
            m.updated = float(d.get("updated", now))
        return m

    def snapshot(self) -> dict:
        return copy.deepcopy(self.to_json())

    def restore(self, snap: dict):
        self.ents, self.edges, self.turns = snap["ents"], snap["edges"], snap["turns"]
        self.user, self.focus, self.updated = snap["user"], snap["focus"], snap["updated"]

    # ------------------------------------------------------------------ observe a turn
    def _ent(self, name: str, typ: str, turn: int, role: str, w: float) -> str | None:
        key = norm(name)
        if not key or len(key) < 2 or key in SUBJ:
            return None
        # "nvidia" said, "nvidia corporation" in the answer: one node (the shorter name wins as key if already there)
        for k in list(self.ents):
            if key != k and (key.startswith(k + " ") or k.startswith(key + " ")
                             # "Huang" after "Jensen Huang": the same person by surname
                             or (" " not in key and k.endswith(" " + key) and self.ents[k].get("kind") == "person")):
                key = k
                break
        e = self.ents.get(key)
        if e is None:
            e = self.ents[key] = {"key": key, "name": name.strip(" .,;:!?"), "type": typ, "votes": {}, "qid": None,
                                  "desc": "", "turns": [], "role": {}, "facts": {}}
        kd = kind_of(typ)
        e["votes"][kd] = e["votes"].get(kd, 0.0) + w
        if kd == "thing" and typ != "thing":
            e["thing_type"] = typ                 # keep the tagger's own type (work_product, term, ...) for slots
        _settle_kind(e)
        if turn not in e["turns"]:
            e["turns"].append(turn)
        e["role"].setdefault(str(turn), role)
        return key

    def _edge(self, a: str, b: str, rel: str, kind: str, turn: int):
        if not a or not b or a == b:
            return
        for x in self.edges:
            if x["a"] == a and x["b"] == b and x["rel"] == rel:
                return
        self.edges.append({"a": a, "b": b, "rel": rel, "kind": kind, "turn": turn})

    def observe(self, f, answer: str, ans_spans: list[dict], docs: list | None = None) -> int:
        """One answered turn. f = its frame; ans_spans = Arc Tagger entity spans of the answer [(type, text)]; docs =
        the sources the answer used (Wikidata facts carry QIDs + property values). Returns the turn number."""
        turn = (self.turns[-1]["t"] + 1) if self.turns else 1
        # the words that still mean something: a corrected-away entity ("Microsoft, sorry, Amazon") or a redacted
        # span must not become the fallback subject / asked kind (getattr: tests pass bare frames)
        gone = set(getattr(f, "dropped", ()) or ()) | set(getattr(f, "redact_tok", ()) or ())
        toks = [norm(w) for i, w in enumerate(f.words) if i not in gone and w]
        asked = asked_kind(toks)
        q_ents = []
        for t, txt in f.slots:
            if t in NUMBER_TYPES:
                continue
            k = self._ent(txt, t, turn, "q", 2.0)
            if k and k not in q_ents:
                q_ents.append(k)
        subject = None
        if f.carried:
            subject = norm(f.carried)
            if subject in self.ents and turn not in self.ents[subject]["turns"]:
                self.ents[subject]["turns"].append(turn)
        if subject is None and not q_ents:
            # the tagger marked nothing ("who painted the starry night"): the fact parser's subject is the entity
            try:
                from . import knowledge
                fq = knowledge.parse_fact(" ".join(toks))
                sub = fq.subject if fq else knowledge.parse_whois(" ".join(toks))
            except Exception:
                sub = None
            if sub and len(sub.split()) <= 5:
                k = self._ent(sub, "thing", turn, "q", 1.0)
                if k:
                    q_ents.append(k)
                    subject = k
        if subject is None and f.phrases:
            subject = next((k for k in q_ents if k == norm(f.phrases[-1])), None) or (q_ents[-1] if q_ents else None)
        elif subject is None and q_ents:
            subject = q_ents[-1]
        a_new = []
        for t, txt in ans_spans[:8]:
            key = norm(txt)
            if not key or any(key == q or key in q.split() or q in key.split() for q in q_ents + [subject or ""]):
                continue
            k = self._ent(txt, t, turn, "a", 1.0)
            if k and k not in a_new and k not in q_ents:
                a_new.append(k)
        g = _gender(toks)
        if g and subject in self.ents and f.carried:
            self.ents[subject]["gender"] = g       # "who was her husband" resolved to Marie Curie: she is "she"
        rg = "m" if set(toks) & REL_MALE else "f" if set(toks) & REL_FEMALE else None
        if rg and a_new:
            self.ents[a_new[0]]["gender"] = rg
        if asked and a_new:
            # the entity the question asked for is the answer's first new one: its kind is known from the question
            e = self.ents[a_new[0]]
            e["votes"][asked] = e["votes"].get(asked, 0.0) + 3.0
            _settle_kind(e)
        rel = next((w for w in f.content if w not in GENERIC_REL and not any(w in (k or "").split()
                                                                            for k in q_ents + a_new)), "")
        if subject and a_new:
            self._edge(a_new[0], subject, rel or "answer", "said", turn)
        for k in a_new[1:4]:
            if subject:
                self._edge(subject, k, "mentioned", "co", turn)
        vals = self._facts(docs or [], turn)
        # a Wikidata fact the answer used names the entity asked for ("made by: Apple Inc."): it leads
        if asked and vals and not a_new:
            # the answer named nothing the tagger caught: the Wikidata value is the entity asked for
            v = next((k for k in vals if self._kind(k) == asked), None)
            if v:
                a_new = [v]
        self.turns.append({"t": turn, "q": getattr(f, "kept_text", None) or f.text, "inline": getattr(f, "inline", "") or "", "content": list(f.content),
                           "intent": f.intent, "q_ents": q_ents, "a_new": a_new, "asked": asked, "subject": subject,
                           "names": names_thing(toks),
                           "answer": re.sub(r"\[\d+\]", "", answer or "")[:300], "at": time.time()})
        self.turns = self.turns[-config.MEM_MAX_TURNS:]
        self.updated = time.time()
        return turn

    def _facts(self, docs: list, turn: int) -> list[str]:
        """Wikidata docs (id wikidata:<QID>:<PID>, meta fact): the item's QID and an edge per value. Returns the value
        entities (keys)."""
        vals_out = []
        for d in docs:
            m = re.match(r"wikidata:(Q\d+):(P\d+)", getattr(d, "id", "") or "")
            fact = (getattr(d, "meta", None) or {}).get("fact")
            if not m or not fact:
                continue
            qid, item = m.group(1), norm(fact.get("item", ""))
            key = next((k for k in self.ents if k == item or item.startswith(k + " ") or k.startswith(item + " ")),
                       None) or self._ent(fact.get("item", ""), "thing", turn, "fact", 0.5)
            if not key:
                continue
            e = self.ents[key]
            e["qid"] = e["qid"] or qid
            title = getattr(d, "text", "") or ""
            dm = re.match(r"[^(]*\(([^)]*)\)", title)
            if dm and not e["desc"]:
                e["desc"] = dm.group(1)[:80]
            phrase = _PROP_PHRASE.get(fact.get("prop"), fact.get("prop", ""))
            vk = _PROP_KIND.get(fact.get("prop"))
            vals = fact.get("values") or []
            if vk is None:                       # a date / quantity: an attribute, not a node
                e["facts"][phrase] = ", ".join(vals[:2])
                continue
            for v in vals[:3]:
                name = re.sub(r"\s*\(.*\)$", "", v)
                k = self._ent(name, {"person": "person", "place": "place", "org": "organization"}[vk], turn, "fact", 3.0)
                if k:
                    self._edge(key, k, phrase, "wikidata", turn)
                    vals_out.append(k)
        return vals_out

    # ------------------------------------------------------------------ user facts
    def learn_user(self, toks: list[str]) -> list[dict]:
        """Facts the user said about themselves, in these words. Returns the ones that are new or changed."""
        new = []
        for key, v in learn_facts([norm(t) for t in toks if norm(t)]):
            old = self.user.get(key)
            if old and old["v"] == v:
                old["at"] = time.time()
                continue
            self.user[key] = {"v": v, "at": time.time(), "label": user_label(key, v)}
            new.append({"key": key, **self.user[key]})
        if new:
            self.updated = time.time()
        return new

    def user_terms(self) -> dict:
        return {k: x["v"] for k, x in self.user.items()}

    # ------------------------------------------------------------------ resolution
    def _pick(self, want: str | None, personal: bool, gender: str | None = None) -> str | None:
        """The entity a reference of kind `want` points at, newest turn first. A gendered pronoun skips persons known
        to be the other gender."""
        for strict in (True, False):
            for tr in reversed(self.turns[-config.MEM_LOOKBACK:]):
                asked, qs, an = tr.get("asked"), list(tr.get("q_ents") or []), list(tr.get("a_new") or [])
                if tr.get("subject") and tr["subject"] not in qs:
                    qs = [tr["subject"]] + qs
                if personal:
                    # a person the answer merely mentioned in a turn that did not ask "who" is not "he"
                    order = (an + qs) if asked == "person" else qs
                    ok = [k for k in order if self._kind(k) == "person"
                          and not (gender and self.ents[k].get("gender") not in (None, gender))]
                elif want:
                    order = (an if asked == want else []) + qs + an
                    ok = [k for k in order if self._kind(k) == want or (not strict and self._kind(k) != "person")]
                else:
                    order = (an if tr.get("names") else []) + qs + an
                    ok = [k for k in order if self._kind(k) != "person"]
                if ok:
                    return ok[0]
            if personal or not want:
                break
        return None

    def _kind(self, k: str) -> str:
        e = self.ents.get(k)
        return e.get("kind", "thing") if e else "thing"

    def subjects(self) -> list[str]:
        out = []
        for tr in self.turns:
            s = tr.get("subject")
            if s and s in self.ents and s not in out:
                out.append(s)
        return out

    def carry_for(self, words: list[str], base: dict | None = None) -> dict:
        """The frame's carry for these words: the entity each kind of reference points at (best first), an explicit
        reference ("that company", "the first one") as "pick", and the user's facts. With no turns in memory it is
        the session's one-turn carry (base) plus the user's facts."""
        base = dict(base or {})
        if self.user:
            base["user"] = self.user_terms()
        if not self.turns:
            return base
        toks = [norm(w) for w in words if norm(w)]
        s = " ".join(toks)
        want = None
        for t in toks:
            if t in AFFINITY:
                want = AFFINITY[t]
        personal = any(t in PERSONAL for t in toks)
        if "there" in toks and not personal:
            want = "place"
        p = self._pick(None if personal else want, personal, _gender(toks))
        if p:
            e = self.ents[p]
            item = (e["type"], p)
            if personal:
                base["person"] = [item] + [x for x in base.get("person") or [] if x[1] != p]
            else:
                base["thing"] = [item] + [x for x in base.get("thing") or [] if x[1] != p]
        # a person reference needs a person even when "it"-things exist (and vice versa): fill the other list too
        if not personal:
            pp = self._pick(None, True)
            if pp and not base.get("person"):
                base["person"] = [(self.ents[pp]["type"], pp)]
        # explicit references: "the first one", "the other one", "that company"
        m = _REF_ORD.search(s)
        if m:
            which = m.group(1) or m.group(2)
            subs = self.subjects()
            k = None
            if which in _ORD and len(subs) > _ORD[which]:
                k = subs[_ORD[which]]
            elif which in ("other", "earlier", "previous") and len(subs) >= 2:
                last = self.turns[-1].get("subject")
                cur = last if last in subs else (self.focus if self.focus in subs else subs[-1])
                k = next((x for x in reversed(subs) if x != cur), None)
            if k:
                base["pick"] = {"ent": (self.ents[k]["type"], k), "span": _span(toks, m.group(0)), "why": m.group(0)}
        else:
            m = _REF_DEM.search(s)
            if m:
                k = self._pick(TYPE_NOUNS[m.group(1)] if TYPE_NOUNS[m.group(1)] != "thing" else None,
                               TYPE_NOUNS[m.group(1)] == "person")
                if k:
                    base["pick"] = {"ent": (self.ents[k]["type"], k), "span": _span(toks, m.group(0)), "why": m.group(0)}
        prev = self.turns[-1]
        if "prev" not in base:
            base["prev"] = {"content": prev["content"], "entities": prev["q_ents"], "intent": prev["intent"],
                            "question": prev["q"], "inline": prev.get("inline", "")}
        # what a relative reference needs ("the old one" -> server/ellipsis.py): the previous answer's entities (the
        # thing it named first), how long ago, how many subjects the conversation has
        base["prev"] = {**base["prev"],
                        "answers": [self.ents[k]["name"].lower() for k in prev.get("a_new") or [] if k in self.ents][:2],
                        "answer_ents": [{"name": self.ents[k]["name"], "kind": self.ents[k].get("kind", "")}
                                        for k in prev.get("a_new") or [] if k in self.ents][:6],
                        "answer": prev.get("answer", ""),
                        "age_s": max(0.0, time.time() - prev["at"]) if prev.get("at") else None,
                        "n_subjects": len(self.subjects())}
        return base

    def note_used(self, key: str | None):
        if key and key in self.ents:
            self.focus = key

    def entity(self, name: str) -> dict | None:
        return self.ents.get(norm(name))

    # ------------------------------------------------------------------ pre-warm
    def prewarm_subjects(self, n: int = 2) -> list[str]:
        """The entities a follow-up most likely asks about: the last turn's answer entity and subject."""
        if not self.turns:
            return []
        tr = self.turns[-1]
        out = []
        for k in (tr.get("a_new") or [])[:1] + [tr.get("subject")] + (tr.get("q_ents") or []):
            if k and k in self.ents and k not in out:
                out.append(k)
        return out[:n]

    # ------------------------------------------------------------------ graph
    def graph(self, used: str | None = None) -> dict:
        nodes = []
        for k, e in self.ents.items():
            nodes.append({"id": k, "label": _title(e["name"]), "kind": e.get("kind", "thing"), "qid": e.get("qid"),
                          "desc": e.get("desc", ""), "turns": e["turns"], "facts": e.get("facts", {})})
        edges = [dict(x) for x in self.edges]
        # co-mention: the question's entities of one turn are related
        for tr in self.turns:
            qs = tr.get("q_ents") or []
            for a, b in zip(qs, qs[1:]):
                if not any({x["a"], x["b"]} == {a, b} for x in edges):
                    edges.append({"a": a, "b": b, "rel": "", "kind": "co", "turn": tr["t"]})
        user = [{"key": k, "label": x["label"], "v": x["v"]} for k, x in self.user.items()]
        return {"nodes": nodes, "edges": edges, "user": user, "focus": used or self.focus,
                "turns": [{"t": t["t"], "q": t["q"], "subject": t.get("subject")} for t in self.turns]}


NUMBER_TYPES = {"date", "time", "time_of_day", "duration", "money", "percent", "cardinal", "ordinal", "quantity",
                "number", "numeral", "age", "year"}
_PROP_PHRASE = {"founder": "founded by", "creator": "created by", "author": "written by", "director": "directed by",
                "discoverer": "invented by", "manufacturer": "made by", "ceo": "CEO", "spouse": "spouse",
                "born_place": "born in", "hq": "headquarters", "capital": "capital", "country": "country",
                "river": "river", "head_state": "head of state", "head_gov": "led by", "language": "language", "currency": "currency",
                "inception": "founded", "born_date": "born", "died_date": "died", "height": "height",
                "population": "population", "area": "area", "symbol": "symbol"}
_PROP_KIND = {"founder": "person", "creator": "person", "author": "person", "director": "person",
              "discoverer": "person", "ceo": "person", "spouse": "person", "head_state": "person", "head_gov": "person",
              "manufacturer": "org", "born_place": "place", "hq": "place", "capital": "place", "country": "place",
              "river": "place"}


DEMONYMS = set("""american british english french german italian spanish chinese japanese korean indian russian dutch
swedish polish canadian australian mexican brazilian irish scottish welsh greek turkish israeli egyptian european
african asian swiss austrian belgian danish norwegian finnish portuguese""".split())
_CAP_STOP = {"the", "a", "an", "it", "its", "this", "that", "in", "on", "of", "and", "he", "she", "they", "his", "her"}


def answer_entities(words: list[str], spans: list[dict], q_toks: list[str]) -> list[tuple[str, str]]:
    """The answer's entity mentions as (type, text): each Arc Tagger span grown over the capitalised words around it
    (the tagger marks "Interactive Entertainment" of "Sony Interactive Entertainment"), mentions that only repeat
    words of the question dropped ("Prejudice" when the question named "Pride and Prejudice")."""
    out, seen = [], set()
    qset = set(q_toks)
    cap = lambda w: bool(w) and w[0].isupper() and norm(w) not in _CAP_STOP
    for sp in sorted(spans, key=lambda x: x.get("start", 0)):
        if sp.get("family") != "entity" or not sp["type"].startswith("entity.mention"):
            continue
        a, b = sp.get("start"), sp.get("end")
        if a is None or b is None or not words:
            text = sp["text"]
        else:
            while a > 0 and cap(words[a - 1]) and not words[a - 1].rstrip().endswith((".", ",", ";", ":")):
                a -= 1
            while b < len(words) and cap(words[b]) and not words[b - 1].rstrip().endswith((".", ",", ";", ":")):
                b += 1
            text = " ".join(words[a:b])
        key = norm(text)
        toks = key.split()
        if not key or key in seen or (toks and all(t in qset for t in toks)) or key in DEMONYMS:
            continue
        seen.add(key)
        out.append((sp["type"].split(".")[-1], text.strip(" .,;:!?")))
    return out


def _gender(toks) -> str | None:
    t = set(toks)
    return "f" if t & FEMALE else "m" if t & MALE else None


def user_label(key: str, v: str) -> str:
    return USER_LABEL[key].format(v=_title(v) if key == "home" else v)


def _settle_kind(e: dict):
    """The entity's kind = its best-voted one (question spans 2, answer spans 1, what the question asked for 3,
    Wikidata property values 3); its slot type follows the kind."""
    e["kind"] = max(e["votes"].items(), key=lambda kv: kv[1])[0]
    e["type"] = {"person": "person", "place": "place", "org": "organization"}.get(e["kind"],
                                                                                   e.get("thing_type") or "thing")


def _title(s: str) -> str:
    return " ".join(w[:1].upper() + w[1:] if w.islower() else w for w in (s or "").split())


def _span(toks: list[str], phrase: str) -> tuple[int, int]:
    p = phrase.split()
    for i in range(len(toks) - len(p) + 1):
        if toks[i:i + len(p)] == p:
            return (i, i + len(p))
    return (0, 0)


# ------------------------------------------------------------------------------------------------- store
class Store:
    """runs/memory/<sha256(client)[:24]>.json, one file per browser. The client id is a random id the browser made
    (localStorage); nothing else identifies the user. TTLs: conversation MEM_CONV_TTL_S, user facts MEM_USER_TTL_S."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or config.MEM_DIR)

    @staticmethod
    def valid(client) -> bool:
        return isinstance(client, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{8,64}", client))

    def path(self, client: str) -> Path:
        return self.root / (hashlib.sha256(client.encode()).hexdigest()[:24] + ".json")

    def load(self, client: str) -> Memory:
        if not config.MEM_PERSIST or not self.valid(client):
            return Memory()
        try:
            return Memory.from_json(json.loads(self.path(client).read_text()))
        except Exception:
            return Memory()

    def save(self, client: str, mem: Memory):
        if not config.MEM_PERSIST or not self.valid(client):
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            p = self.path(client)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(mem.to_json()))
            tmp.replace(p)
        except Exception:
            pass

    def forget(self, client: str):
        if not self.valid(client):
            return
        try:
            self.path(client).unlink()
        except FileNotFoundError:
            pass


async def prewarm(mem: Memory, timeout_s: float | None = None) -> list[str]:
    """Free pre-warm after an answer: fetch the likely follow-up subjects' Wikidata facts (entity search + one SPARQL
    query for every property the facts lane knows) into the knowledge cache, so a follow-up's fact lookup is local.
    Never raises; returns the subjects warmed."""
    if not config.MEM_PREWARM or not config.ROUTER:
        return []
    from . import knowledge
    subs = mem.prewarm_subjects()
    if not subs:
        return []
    tasks = [asyncio.ensure_future(knowledge.prefetch(x)) for x in subs]
    try:
        await asyncio.wait(tasks, timeout=timeout_s or config.MEM_PREWARM_TIMEOUT_S)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)     # (retrieve every outcome: no stray warnings)
    return [x for x, t in zip(subs, tasks) if t.done() and not t.cancelled() and t.result() is True]
