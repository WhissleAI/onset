"""Verify: check the answer's key claims against the sources it cites, cheaply, AFTER the answer is shown.

Claims = numbers (with their scale: "60.9 billion"), years, and proper names in the answer, minus anything already in
the question. Each claim is looked for in the cited sources first, then in every source (string / number matching,
relative tolerance VERIFY_TOL for non-year numbers, rounding allowed). Cross-source agreement: how many DISTINCT sources
(by URL / title) support every claim. Disagreement: a year claim whose context word ("born", "founded", ...) appears in
another source next to a different year, while that source never states the claimed one.

Result (badge text in `label`):
  agree      "✓ 2 sources agree"          every claim found, in >= 2 distinct sources
  supported  "✓ matches 1 source"         every claim found, one source
  disagree   "⚠ sources disagree: 1993 vs 1991"
  partial    "◐ 2 of 3 claims found in sources"
  unverified "unverified"                 no checkable claim, or none found

No model call, no network: ~1 ms. Honest scope: it checks that the sources SAY it, not that it is true.
"""
from __future__ import annotations

import re
import time
import unicodedata

from . import config

_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12, "k": 1e3, "m": 1e6, "bn": 1e9}
_NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?:\s*(thousand|million|billion|trillion|bn)\b)?",
                  re.I)
_CITE = re.compile(r"\[\d+(?:\s*[,–-]\s*\d+)*\]")
_PRED = ("born", "founded", "established", "died", "released", "launched", "built", "opened", "completed", "written",
         "published", "created", "started", "invented", "discovered", "elected", "became", "formed", "incorporated")
_NAME = re.compile(r"\b([A-Z][a-zà-ÿ'’\-]+(?:\s+(?:[A-Z][a-zà-ÿ'’\-]+|de|da|van|von|of|the|bin|al))*\s+[A-Z][a-zà-ÿ'’\-]+"
                   r"|[A-Z][a-zà-ÿ'’\-]{2,})\b")
_COMMON = set("""The A An It Its In On At Of For And But Or So As By To From With This That These Those He She They
His Her Their There Here Yes No About Around Over Under After Before Since Both Each Some Most Many According However
Also Today Tomorrow Yesterday Sources Source Note January February March April May June July August September October
November December Monday Tuesday Wednesday Thursday Friday Saturday Sunday Mount Lake River""".split())


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _val(m) -> tuple[float, bool, int]:
    """(value, is_year, decimals) of a number match."""
    whole, dec, scale = m.group(1), m.group(2), (m.group(3) or "").lower()
    v = float(whole.replace(",", "") + ("." + dec if dec else ""))
    year = not dec and not scale and "," not in whole and 1000 <= v <= 2100 and len(whole) == 4
    if scale:
        v *= _SCALE[scale]
    return v, year, len(dec or "")


def numbers(text: str) -> list[tuple[float, bool, int, int, str]]:
    """(value, is_year, decimals, position, as written) of every number in the text."""
    out = []
    for m in _NUM.finditer(_CITE.sub(" ", text)):
        v, y, d = _val(m)
        out.append((v, y, d, m.start(), m.group(0)))
    return out


def _num_match(a: float, ay: bool, ad: int, b: float, by: bool) -> bool:
    if ay or by:
        return ay == by and a == b
    if a == b:
        return True
    if round(b, ad) == a or round(b) == a:
        return True
    return abs(a - b) <= config.VERIFY_TOL * max(abs(a), abs(b))


def _has_name(name: str, text: str) -> bool:
    toks = [t for t in re.findall(r"[a-z0-9]+", fold(name)) if len(t) > 1]
    if not toks:
        return False
    low = " " + " ".join(re.findall(r"[a-z0-9]+", fold(text))) + " "
    return all(f" {t} " in low or f" {t}s " in low for t in toks)


def claims(answer: str, question: str = "") -> list[dict]:
    """Checkable claims of the answer, each with the source numbers its sentence cites."""
    out = []
    qnums = {v for v, *_ in numbers(question)}
    qlow = fold(question)
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        cites = sorted({int(n) for g in re.findall(r"\[([\d,\s–-]+)\]", sent) for n in re.findall(r"\d+", g)})
        clean = _CITE.sub(" ", sent)
        for v, y, d, pos, raw in numbers(clean):
            if v in qnums or (not y and v < 10 and not d):
                continue                                   # said in the question, or a tiny count ("2 sources")
            # the predicate this number belongs to: the nearest one said before it ("founded by X, who was born on
            # January 12, 1964": born, not founded)
            ctx = fold(clean[max(0, pos - 60):pos])
            near = max(((ctx.rfind(p), p) for p in _PRED if p in ctx), default=None)
            out.append({"kind": "year" if y else "number", "text": raw.strip(), "value": v, "year": y, "dec": d,
                        "cites": cites, "pred": [near[1]] if near else []})
        for i, mm in enumerate(_NAME.finditer(clean)):
            name = mm.group(1)
            first = clean[:mm.start()].strip() == ""
            if (first and " " not in name) or name in _COMMON or _has_name(name, qlow):
                continue
            if all(w in _COMMON for w in name.split()):
                continue
            out.append({"kind": "name", "text": name, "cites": cites})
    seen, uniq = set(), []
    for c in out:
        k = (c["kind"], fold(c["text"]))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq[:8]


def _supports(c: dict, text: str) -> bool:
    if c["kind"] == "name":
        return _has_name(c["text"], text)
    return any(_num_match(c["value"], c["year"], c["dec"], v, y) for v, y, *_ in numbers(text))


def _about(title: str, context: str) -> bool:
    """Is this source about the claim's subject? Every word of its title (minus "(company)", "· Wikidata") is said in
    the claim's sentence or the question: "Amazon (company)" yes, "Amazon Web Services" no (another 2010 is fine)."""
    t = re.sub(r"\s*\(.*?\)|\s*·.*$", "", title or "")
    toks = [w for w in re.findall(r"[a-z0-9]+", fold(t)) if len(w) > 1]
    ctx = " " + " ".join(re.findall(r"[a-z0-9]+", fold(context))) + " "
    return bool(toks) and all(f" {w} " in ctx for w in toks)


def _conflict(c: dict, text: str, title: str, context: str, own: set, subject: str) -> str | None:
    """A different year for the same fact in a source about the same subject that never states the claim: the year
    sits right after the claim's predicate ("born ... 1991"), in a sentence naming the claim's subject, and is not
    another year the answer itself states (Harvard 1636 vs Yale 1701 is two facts, not a disagreement)."""
    if c["kind"] != "year" or not c["pred"] or not _about(title, context):
        return None
    for sent in re.split(r"(?<=[.!?;])\s+", text):
        low = fold(sent)
        if subject and subject not in low:
            continue
        for v, y, _, pos, raw in numbers(sent):
            if not y or v == c["value"] or v in own:
                continue
            win = low[max(0, pos - 40):pos]
            if any(p in win for p in c["pred"]):
                return raw.strip()
    return None


def _src_key(d) -> str:
    return (getattr(d, "url", "") or getattr(d, "title", "") or getattr(d, "id", "")).split("#")[0]


def _subject_of(c: dict, sents: dict) -> str:
    """The name the claim is about: the last capitalized name said before it in its sentence (its last word, folded)."""
    sent = sents.get(c["text"], "")
    head = sent[:sent.find(c["text"])] if c["text"] in sent else sent
    names = [m.group(1) for m in _NAME.finditer(_CITE.sub(" ", head)) if m.group(1) not in _COMMON]
    return fold(names[-1].split()[-1]) if names else ""


def check(answer: str, docs, question: str = "") -> dict:
    t0 = time.perf_counter()
    texts = [f"{getattr(d, 'title', '')}. {getattr(d, 'text', '')}" for d in docs]
    cl = claims(answer or "", question)
    sents = {c["text"]: s_ for s_ in re.split(r"(?<=[.!?])\s+", answer or "") for c in cl if c["text"] in s_}
    own_years = {c["value"] for c in cl if c["kind"] == "year"}
    found, conflicts = 0, []
    agree_counts = []
    for c in cl:
        sup = [i for i, t in enumerate(texts) if _supports(c, t)]
        c["support"] = [i + 1 for i in sup]
        c["cited_ok"] = bool(set(c["support"]) & set(c["cites"])) if c["cites"] else bool(sup)
        if sup:
            found += 1
            agree_counts.append(len({_src_key(docs[i]) for i in sup}))
        for i, t in enumerate(texts):
            if i in sup:
                continue
            other = _conflict(c, getattr(docs[i], "text", ""), getattr(docs[i], "title", ""),
                              sents.get(c["text"], "") + " " + question, own_years, _subject_of(c, sents))
            if other:
                c["conflict"] = {"source": i + 1, "value": other}
                conflicts.append((c["text"], other))
                break
    # arithmetic on the answer's own years ("born 1964, founded 1994: 30 years old") is computed, not a source claim
    years = [c["value"] for c in cl if c["kind"] == "year"]
    for c in cl:
        if c["kind"] == "number" and not c["support"] and any(abs(a - b) in (c["value"], c["value"] + 1, c["value"] - 1)
                                                             for a in years for b in years if a != b):
            c["kind"] = "computed"
    computed = [c for c in cl if c["kind"] == "computed"]
    cl_checked = [c for c in cl if c["kind"] != "computed"]
    found = sum(1 for c in cl_checked if c["support"])
    agree_counts = [len({_src_key(docs[i - 1]) for i in c["support"]}) for c in cl_checked if c["support"]]
    n = len(cl_checked)
    if n == 0:
        status, label = "unverified", "unverified"
    elif conflicts:
        a, b = conflicts[0]
        status, label = "disagree", f"⚠ sources disagree: {a} vs {b}"
    elif found == n:
        k = min(agree_counts) if agree_counts else 0
        status, label = ("agree", f"✓ {k} sources agree") if k >= 2 else ("supported", "✓ matches 1 source")
    elif found:
        status, label = "partial", f"◐ {found} of {n} claims found in sources"
    else:
        status, label = "unverified", "unverified"
    if computed and status in ("agree", "supported"):
        label += " · " + ", ".join(c["text"] for c in computed) + " computed"
    return {"status": status, "label": label, "n_claims": n, "found": found, "computed": len(computed),
            "claims": [{k: v for k, v in c.items() if k not in ("value", "dec", "year")} for c in cl],
            "ms": round((time.perf_counter() - t0) * 1000, 2)}
