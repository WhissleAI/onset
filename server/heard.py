"""Retrieval-in-the-loop speech understanding: results feed back into what was heard (LOOP_ASR_FIX).

Misrecognised names cause most of the remaining errors ("who was mari guri" -> Marie Curie, "how tall is k two" -> K2).
While the user is still speaking, the session already holds names it could have meant:
  - spec    titles + names in the passages of this question's speculative results (server/speculator.py pool)
  - memory  entities of the conversation (server/memory.py: questions AND answers)
  - facts   Wikidata labels of lookups already made (server/knowledge.py cache)
  - leads   the local Wikipedia leads index (index/data/knowledge: 11K article titles, offline)
A *suspect* span is 1-3 words that the ear was unsure of (fused per-word confidence, server/confidence.py, below
LOOP_SUSPECT_CONF) or an Arc Tagger entity mention no candidate spells exactly. Each suspect span is re-scored against
the candidates with a phonetic key (consonant classes, voicing merged, vowels dropped: "mari guri" and "marie curie"
are both MRKR) plus orthographic similarity of the letters, conditioned on the entity type Arc gave the span. A
confident, unambiguous match rewrites the frame (query, slots, entity phrase, what the answer model reads); the UI
shows it ("heard 'mari guri' -> Marie Curie").

Safeguard: a word with fused confidence >= LOOP_KEEP_CONF is never overridden, and the candidate must beat the runner-
up by LOOP_FIX_MARGIN. Pure functions, no network; ~1 ms per frame (the leads gazetteer is matched with rapidfuzz).
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

from . import config

try:                                    # C-speed matching over the 11K-title gazetteer (in macapp/requirements.txt)
    from rapidfuzz import fuzz as _fuzz, process as _process
    from rapidfuzz.distance import Levenshtein as _Lev
except Exception:                       # pragma: no cover - pure-Python fallback
    _fuzz = _process = _Lev = None

# ------------------------------------------------------------------------------------------------ text normalisation
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("" if o == 0 else " " + _ONES[o])


def spoken(name: str) -> str:
    """A name as it would be said: lower case, letters/digits split ("K2" -> "k two"), small numbers spelled
    ("Apollo 11" -> "apollo eleven"), punctuation and parentheticals dropped ("Mercury (planet)" -> "mercury")."""
    import unicodedata
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))            # "André" -> "andre"
    s = re.sub(r"\s*\(.*?\)", "", s)
    s = re.sub(r"\s*·.*$", "", s)
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)
    out = []
    for w in re.findall(r"[a-z0-9']+", s.lower().replace("-", " ").replace("&", " and ")):
        w = w.strip("'")
        if w.isdigit() and len(w) <= 2:
            out.append(_num_words(int(w)))
        elif w:
            out.append(w)
    return " ".join(out)


def letters(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s)


# consonant classes: voicing merged (ASR confuses b/p, d/t, g/k, v/f, z/s), vowels dropped except a leading one
_CLS = {"b": "P", "p": "P", "d": "T", "t": "T", "g": "K", "k": "K", "q": "K", "c": "K", "f": "F", "v": "F",
        "s": "S", "z": "S", "x": "KS", "m": "M", "n": "N", "l": "L", "r": "R", "j": "J"}
_PRE = [("tch", "ch"), ("sch", "sk"), ("ph", "f"), ("ck", "k"), ("qu", "kw"), ("gh", ""), ("th", "t"), ("dg", "j"),
        ("ch", "J"), ("sh", "J"), ("ge", "je"), ("gi", "ji"), ("ce", "se"), ("ci", "si"), ("cy", "sy")]


def pkey(text: str) -> str:
    """Phoneme-ish key of a phrase (word boundaries removed: the ear splits / merges names freely)."""
    out = []
    for w in spoken(text).split():
        if w.startswith(("kn", "gn", "wr", "ps")):
            w = w[1:]
        for a, b in _PRE:
            w = w.replace(a, b)
        k = []
        for i, ch in enumerate(w):
            if ch == "J":
                c = "J"
            elif ch in "aeiouy":
                c = "A" if i == 0 else ""
            elif ch in "wh":
                c = ""
            else:
                c = _CLS.get(ch, "")
            if c and not (k and k[-1] == c):
                k.append(c)
        out.append("".join(k))
    key = "".join(out)
    return re.sub(r"(.)\1+", r"\1", key)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _Lev is not None:
        return _Lev.normalized_similarity(a, b)
    import difflib
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def _indel(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _fuzz is not None:
        return _fuzz.ratio(a, b) / 100.0
    import difflib
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def similarity(said: str, cand: str) -> tuple[float, float, float]:
    """(score, phonetic, orthographic) of a heard phrase vs a candidate name."""
    a, b = spoken(said), spoken(cand)
    pa, pb = pkey(a), pkey(b)
    ph = _ratio(pa, pb)
    if min(len(pa), len(pb)) <= 2 and pa != pb:
        ph *= 0.5                     # two-consonant keys collide easily ("kt" = cat, kit, kate)
    orth = _indel(letters(a), letters(b))
    return 0.55 * ph + 0.45 * orth, ph, orth


# ------------------------------------------------------------------------------------------------------ candidates
# Arc Tagger mention types -> the kind of thing a candidate must be (unknown candidate kinds are neutral)
TYPE_KIND = {"person": "person", "place": "place", "city": "place", "country": "place", "location": "place",
             "organization": "org", "company": "org", "team": "org", "brand": "org", "product": "thing",
             "event_name": "thing", "title": "thing", "work_of_art": "thing", "org": "org"}
SRC_BONUS = {"spec": 0.03, "memory": 0.04, "answer": 0.05, "facts": 0.02, "keystone": 0.04, "leads": 0.0}
_PERSON_LEAD = re.compile(r"^[^.]{0,120}\((?:[^)]*\bborn\b|[^)]*\d{3,4}\s*[–-]\s*\d{1,2}?\s*\w*\s*\d{3,4})")


@dataclass
class Cand:
    name: str               # display form ("Marie Curie")
    src: str                # spec | memory | answer | facts | leads
    kind: str = ""          # person | place | org | thing | "" (unknown)


class Pool:
    """A session's candidate names (spec / memory / answer / facts); the leads gazetteer is shared (see leads())."""

    def __init__(self):
        self.c: dict[str, Cand] = {}
        self._seen_docs: set = set()

    def add(self, name: str, src: str, kind: str = ""):
        name = re.sub(r"\s*·.*$", "", (name or "").strip())
        name = re.sub(r"\s*\((?:disambiguation|company|band|film|album|song)\)", "", name)
        k = spoken(name)
        if not k or len(letters(k)) < 2 or len(k.split()) > 5:
            return
        old = self.c.get(k)
        if old is None or SRC_BONUS.get(src, 0) > SRC_BONUS.get(old.src, 0):
            self.c[k] = Cand(name, src, kind or (old.kind if old else ""))

    def add_docs(self, docs):
        """Titles of speculative results + capitalised name runs at the start of their passages."""
        for d in docs:
            did = getattr(d, "id", None) or id(d)
            if did in self._seen_docs:
                continue
            self._seen_docs.add(did)
            title = getattr(d, "title", "") or ""
            if title and "wikidata" not in title.lower():
                self.add(title, "spec")
            text = (getattr(d, "text", "") or "")[:400]
            for m in _NAMES.finditer(text):
                self.add(m.group(0), "spec")

    def __len__(self):
        return len(self.c)


_NAMES = re.compile(r"\b[A-Z][a-zà-ÿ'\-]+(?:\s+(?:of|de|da|van|von)?\s*[A-Z][a-zà-ÿ'\-]+){1,3}\b|\b[A-Z]\d{1,2}\b")


class Gazetteer:
    """The local Wikipedia leads titles (index/data/knowledge/leads.jsonl), keyed by phonetic key for fast lookup."""

    def __init__(self, docs: list[dict] | None = None):
        self.names: list[str] = []
        self.kinds: list[str] = []
        self.keys: list[str] = []
        self.spoken: dict[str, int] = {}
        self.by_key: dict[str, list[int]] = {}
        for d in docs or []:
            t = d.get("title") or ""
            sp = spoken(t)
            if not sp or len(sp.split()) > 5 or sp in self.spoken:
                continue
            self.spoken[sp] = len(self.names)
            self.names.append(re.sub(r"\s*\(.*?\)", "", t))
            self.kinds.append("person" if _PERSON_LEAD.search(d.get("text") or "") else "")
            self.keys.append(pkey(sp))
            self.by_key.setdefault(self.keys[-1], []).append(len(self.names) - 1)

    def __len__(self):
        return len(self.names)

    def near(self, said: str, n: int = 12) -> list[Cand]:
        k = pkey(said)
        if not self.keys or len(k) < 2:
            return []
        if _process is not None:
            hits = _process.extract(k, self.keys, scorer=_Lev.normalized_similarity, limit=n, score_cutoff=0.6)
            idx = [i for _, _, i in hits]
        else:                                                   # pragma: no cover
            idx = sorted(range(len(self.keys)), key=lambda i: -_ratio(k, self.keys[i]))[:n]
        # every exact key match too (short keys tie with dozens of names: "KT" = Cat, Cod, Goat, K2)
        idx = list(dict.fromkeys(self.by_key.get(k, [])[:40] + idx))
        return [Cand(self.names[i], "leads", self.kinds[i]) for i in idx]


_gaz: Gazetteer | None = None
_gaz_lock = threading.Lock()


def leads() -> Gazetteer:
    global _gaz
    with _gaz_lock:
        if _gaz is None:
            docs = []
            p = config.KNOWLEDGE_DIR / "leads.jsonl"
            try:
                docs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            except (OSError, ValueError):
                pass
            _gaz = Gazetteer(docs)
        return _gaz


def set_leads(g: Gazetteer | None):
    """Tests: install a gazetteer (None = reload from disk on next use)."""
    global _gaz
    _gaz = g


def facts_labels() -> list[str]:
    """Wikidata labels of the entity searches already made this process (server/knowledge.py cache)."""
    try:
        from . import knowledge
        out = []
        for key, val in list(getattr(knowledge._cache, "d", {}).items())[-200:]:
            if isinstance(key, tuple) and key and key[0] == "s":
                v = val[1] if isinstance(val, tuple) and len(val) == 2 else val
                for x in (v or [])[:2]:
                    if isinstance(x, dict) and x.get("label"):
                        out.append(x["label"])
        return out
    except Exception:
        return []


def keystone_names(k: int = 60) -> list[tuple[str, str]]:
    """(name, kind) of the Keystone graph's most active nodes (session + user scopes, server/keystone/graph.py) and
    the values of their typed edges: what this user has been asking about, and what those answers named."""
    try:
        if not config.INDEX:
            return []
        from .keystone import hook as ks
        g = ks.store().g
        scopes = [f"s:{ks.SCOPE['session']}", f"u:{ks.SCOPE['user']}"]
        act = sorted(g.activation(scopes, spread=0.3).items(), key=lambda x: -x[1])[:k]
        out, live = [], set()
        for n, _ in act:
            node = g.nodes.get(n)
            if node and node.get("name"):
                live.add(n)
                out.append((node["name"], node.get("kind") or ""))
        for (subj, _rel), vals in list(g.edges.items()):          # one pass: the active nodes' edge values
            if subj in live:
                out += [(v["value"], "") for v in vals[:3] if isinstance(v, dict) and v.get("value")]
        return out[:3 * k]
    except Exception:
        return []


# ---------------------------------------------------------------------------------------------------------- fixing
def suspect_conf(ear: list, fused: list | None = None) -> list[float]:
    """Per-word "is this word right" signal: min(ear confidence, fused confidence). The fused confidence
    (server/confidence.py) predicts whether a word SURVIVES into the final transcript: on final words it saturates
    (~1.0 for 97 % of misheard name words on the Flux traces), while Flux's own word confidence still separates them
    (39 % of misheard name words < 0.9 vs 7 % of correctly heard names). [] when the ear gives no confidence."""
    if not ear or all(c is None for c in ear):
        return []
    out = []
    for i, c in enumerate(ear):
        c = 1.0 if c is None else float(c)
        if fused and i < len(fused):
            c = min(c, float(fused[i]))
        out.append(c)
    return out


@dataclass
class Fix:
    a: int                  # token span [a, b) in the frame's words
    b: int
    said: str               # as heard ("mari guri")
    name: str               # what it meant ("Marie Curie")
    score: float
    src: str
    kind: str = ""          # Arc's entity type for the span, if any
    runner: float = 0.0     # best other candidate's score
    ckind: str = ""         # the candidate's own kind (person / place / org), when its source knows it

    def to_json(self) -> dict:
        return {"said": self.said, "as": self.name, "src": self.src, "score": round(self.score, 3),
                "a": self.a, "b": self.b, "type": self.kind}


def _mention_spans(f) -> list[tuple[int, int, str]]:
    out = []
    for s in f.spans or []:
        if str(s.get("type", "")).startswith("entity.mention") and s["end"] <= len(f.words):
            out.append((s["start"], s["end"], s["type"].split(".")[-1]))
    return out


def suspects(f, wconf) -> list[tuple[int, int, str, bool]]:
    """Candidate windows (a, b, Arc type, in a tagger mention) to re-score: 1-3 non-stopword-edged words, none of
    them high-confidence, at least one low-confidence; or an Arc entity mention (any confidence below KEEP)."""
    from .frame import STOP, norm
    toks = [norm(w) for w in f.words]
    n = len(toks)
    skip = set(f.dropped) | set(f.redact_tok) | set(f.cue)
    have = bool(wconf) and len(wconf) == n
    keep, low = config.LOOP_KEEP_CONF, config.LOOP_SUSPECT_CONF
    ments = _mention_spans(f)
    out = []
    seen = set()

    def ok(a, b, pin_ok=False):
        if any(i in skip or not toks[i] for i in range(a, b)):
            return False
        if toks[a] in STOP or toks[b - 1] in STOP:
            return False
        if have and any(wconf[i] >= keep for i in range(a, b)):
            # never override a high-confidence word; inside an Arc mention it may stay as an anchor ("jensen" of
            # "jensen hwang"): the candidate must then contain it unchanged (propose checks). LOOP_KEEP_EXACT (off by
            # default) lets an Arc mention through: propose then needs an exact sound match from the context
            return (pin_ok and any(wconf[i] < keep for i in range(a, b))) or (pin_ok and config.LOOP_KEEP_EXACT)
        return True

    for a, b, t in ments:
        # the tagger's span, grown by a word on either side when that word is unsure too ("mari" | "guri")
        for aa, bb in sorted({(a, b), (max(0, a - 1), b), (a, min(n, b + 1))}):
            if bb - aa <= 3 and ok(aa, bb, pin_ok=(aa, bb) == (a, b)) and (aa, bb) not in seen:
                seen.add((aa, bb))
                out.append((aa, bb, t, True))
    if have:
        for a in range(n):
            for b in range(a + 1, min(n, a + 3) + 1):
                if (a, b) in seen or not ok(a, b):
                    continue
                if min(wconf[i] for i in range(a, b)) < low:
                    seen.add((a, b))
                    t = next((tt for ma, mb, tt in ments if ma <= a and b <= mb), "")
                    out.append((a, b, t, False))
    return out


def _prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _kind_adj(span_type: str, c: Cand) -> float:
    want = TYPE_KIND.get(span_type, "")
    if not want or not c.kind:
        return 0.0
    return 0.03 if want == c.kind else -0.08


def propose(f, wconf, pool: Pool | None = None, use_leads: bool = True) -> list[Fix]:
    """Confident rewrites of suspect spans, non-overlapping, best first."""
    from .frame import norm
    toks = [norm(w) for w in f.words]
    wins = suspects(f, wconf)
    if not wins:
        return []
    ctx = list((pool.c if pool else {}).values())
    gaz = leads() if use_leads else None
    have = bool(wconf) and len(wconf) == len(toks)
    found = []
    for a, b, t, ment in wins:
        said = " ".join(toks[a:b])
        sp = spoken(said)
        if len(letters(sp)) < 3 and not ment:
            continue
        cands = list(ctx)
        if gaz is not None and ment:
            # the 11K-title index only for spans Arc calls a name: on plain low-confidence words it rewrote ordinary
            # words into titles ("content" -> Continent, "whether" -> Weather) on the Flux traces
            cands += gaz.near(sp)
        if gaz is not None and sp in gaz.spoken:
            nm = gaz.names[gaz.spoken[sp]]
            if re.search(r"[A-Za-z]\d|\d[A-Za-z]", nm) and letters(nm.lower()) != letters(said) and \
                    (ment or (have and min(wconf[a:b]) < config.LOOP_SUSPECT_CONF)):
                found.append(Fix(a, b, said, nm, 1.0, "leads", t, 0.0))   # "k two" -> K2: the same name, spelled
            continue                                     # heard as an existing name: nothing to fix
        if any(spoken(c.name) == sp for c in ctx):
            continue
        hi = have and all(wconf[i] >= config.LOOP_KEEP_CONF for i in range(a, b))
        pinned = [] if hi else [toks[i] for i in range(a, b) if have and wconf[i] >= config.LOOP_KEEP_CONF]
        scored = {}
        for c in cands:
            cs = spoken(c.name)
            if pinned and not all(p in cs.split() for p in pinned):
                continue                                 # a high-confidence word must survive as said
            nd = abs(len(cs.split()) - len(sp.split()))
            if nd > 1 or cs == sp:
                continue
            cw, sw = set(cs.split()), set(sp.split())
            lc, ls_ = letters(cs), letters(sp)
            if cw <= sw or sw <= cw or lc.rstrip("s") == ls_.rstrip("s") or \
                    (min(len(lc), len(ls_)) >= 7 and _prefix(lc, ls_) >= 0.85 * min(len(lc), len(ls_))):
                continue                                 # "nasdaq" vs Nasdaq-100, "birch way" vs Birch: not misheard
            if len(re.sub(r"[^a-z]", "", c.name.lower())) < 3:
                continue                                 # "8½", "C": nothing to spell against
            if c.src == "leads" and len(cs.split()) == 1 and len(sp.split()) == 1 and len(letters(sp)) < 7:
                continue                                 # short word vs short title: "cost" Coast, "meta" Meat
            s, ph, orth = similarity(sp, cs)
            if ph < 0.6:
                continue
            if hi and (c.src == "leads" or pkey(sp) != pkey(cs) or orth < 0.7):
                continue                                 # LOOP_KEEP_EXACT: sure words change only to a same-sound name
                                                         # the conversation / results already hold
            if c.src == "leads":
                s -= config.LOOP_LEADS_PENALTY           # 11K titles: only a near-exact sound match counts
                if len(cs.split()) == 1 and len(sp.split()) == 1:
                    s -= 0.06                            # one word vs one title: "danielle" Daniel, "thunder" Tundra
            s += SRC_BONUS.get(c.src, 0.0) + _kind_adj(t, c)
            if not ment:
                s -= 0.03                                # a span Arc didn't call a name needs a better match
            if cs not in scored or s > scored[cs][0]:
                scored[cs] = (s, c)
        if not scored:
            continue
        ranked = sorted(scored.values(), key=lambda x: -x[0])
        best, c = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else 0.0
        thr = config.LOOP_FIX_MIN + (0.0 if have else 0.04)
        if best >= thr and best - runner >= config.LOOP_FIX_MARGIN:
            found.append(Fix(a, b, said, c.name, best, c.src, t, runner, c.kind))
    found.sort(key=lambda x: -x.score)
    out, used = [], set()
    for fx in found:
        if used & set(range(fx.a, fx.b)):
            continue
        used |= set(range(fx.a, fx.b))
        out.append(fx)
    return sorted(out, key=lambda x: x.a)


def apply(f, fixes: list[Fix]):
    """Rewrite the frame in place: the span's words become the name (as one token; the others in the span become empty
    so every index stays aligned), the content words / slots / entity phrases / inline question follow, and the fix is
    recorded in f.resolved for the UI and the metric."""
    if not fixes:
        return f
    from .frame import norm
    words = list(f.words)
    content, cidx = list(f.content), list(f.content_idx)
    slots = set(f.slots)
    phrases = list(f.phrases)
    inline = f.inline
    for fx in fixes:
        said, nn = fx.said, norm(fx.name)
        words[fx.a] = fx.name
        for i in range(fx.a + 1, fx.b):
            words[i] = ""
        span = set(range(fx.a, fx.b))
        keep = [(w, i) for w, i in zip(content, cidx) if i not in span]
        pos = next((k for k, (_, i) in enumerate(keep) if i > fx.a), len(keep))
        new = [(w, fx.a) for w in nn.split() if w not in {x for x, _ in keep}]
        keep = keep[:pos] + new + keep[pos:]
        # a word of the span that wasn't a content word but whose replacement now is: content changes either way
        content, cidx = [w for w, _ in keep], [i for _, i in keep]
        hit = [s for s in slots if s[1] and (s[1] in said or said in s[1])]
        for s in hit:
            slots.discard(s)
        # the name's own kind wins over the misheard span's type ("mari guri" typed as a thing; Marie Curie is a
        # person, so memory resolves "she" / "her" to her)
        own = {"person": "person", "place": "place", "org": "organization"}.get(fx.ckind, "")
        slots.add((own or (hit[0][0] if hit else (fx.kind or "thing")), nn))
        phrases = [nn if (p in said or said in p) else p for p in phrases]
        if nn not in phrases:
            phrases.append(nn)
        if inline and f" {said} " in f" {inline} ":
            inline = f" {inline} ".replace(f" {said} ", f" {nn} ").strip()
    f.words = words
    f.content, f.content_idx = tuple(content), tuple(cidx)
    f.slots = tuple(sorted(slots))
    f.phrases = tuple(phrases)
    f.inline = inline
    f.ent_types = tuple(dict.fromkeys(list(f.ent_types) + [fx.kind for fx in fixes if fx.kind]))
    f.resolved = tuple(fx.to_json() for fx in fixes)
    return f
