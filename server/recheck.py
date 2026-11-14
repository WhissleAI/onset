"""Verify-triggered re-search (VERIFY_RESEARCH): when server/verify.py could not confirm the answer's key claim
("⚠ sources disagree", "unverified", or a partial check that missed the key claim), ONE targeted lookup on that claim,
AFTER the answer is shown (it never delays the first token: the speculator starts it once the answer is done).

Lanes, cheapest first:
  facts   Wikidata property lookup (server/knowledge.py) when the question is a fact question it can parse: free,
          structured. A different value for the asked property CORRECTS a year / number claim.
  leads   the local Wikipedia leads index (offline, free): the claim's subject article.
  web     the configured web lane, only with VERIFY_RESEARCH_PAID=1 and within the question's paid budget (COST_MODE
          parity counts it: one more paid call). Off by default.
Outcome (badge): "re-checked ✓ <source>" (the new source states the claim), "corrected: A → B (<source>)" (Wikidata
gives another value; the answer text is patched and labelled), or "still unconfirmed" (nothing new either way).
"""
from __future__ import annotations

import asyncio
import re
import time

from . import config, verify


def key_claim(v: dict, loose: bool | None = None) -> dict | None:
    """The claim worth re-checking: the disputed one, else the first unsupported year / number, else name.
    `loose` (default: VERIFY_EXTRACT on) also takes "The Milwaukee Brewers"-style names (a leading "The")."""
    if loose is None:
        from . import extract
        loose = extract.VERIFY_EXTRACT
    cl = v.get("claims") or []
    for c in cl:
        if c.get("conflict"):
            return c
    if v.get("status") not in ("unverified", "partial"):
        return None
    for kinds in (("year", "number"), ("name",)):
        for c in cl:
            if c.get("kind") in kinds and not c.get("support"):
                if c["kind"] == "name" and not _real_name(_strip_the(c["text"]) if loose else c["text"]):
                    continue
                return c
    return None


_NOT_NAME = {"its", "his", "her", "their", "i'm", "i", "command", "shift", "control", "option", "the", "a"}


def _strip_the(t: str) -> str:
    return re.sub(r"^(?:the)\s+", "", t, flags=re.I)


def _real_name(t: str) -> bool:
    """A name claim worth a lookup: two+ words, not opening with a pronoun / key name ("Its North American",
    "I'm", "Command" were verify's name claims on the eval sets)."""
    ws = t.split()
    return len(ws) >= 2 and ws[0].lower() not in _NOT_NAME


# the claim's predicate -> the Wikidata property that states it (knowledge.PROPS)
_PRED_PROP = {"born": "born_date", "founded": "inception", "established": "inception", "incorporated": "inception",
              "formed": "inception", "created": "inception", "started": "inception", "died": "died_date"}


def claim_fact(claim: dict, answer: str):
    """A targeted fact lookup for THIS claim: "Jane Austen ... was born in 1775" -> born_date(jane austen). The
    subject is the full name said before the claim in its sentence."""
    from . import knowledge
    if claim.get("kind") != "year":
        return None
    prop = next((_PRED_PROP[p] for p in claim.get("pred") or [] if p in _PRED_PROP), None)
    if not prop:
        return None
    sent = next((s for s in re.split(r"(?<=[.!?])\s+", answer or "") if claim["text"] in s), "")
    head = verify._CITE.sub(" ", sent[:sent.find(claim["text"])])
    pw = next((p for p in claim.get("pred") or [] if p in _PRED_PROP), "")
    k = verify.fold(head).rfind(pw) if pw else -1
    head = head[:k] if k > 0 else head                     # the subject is said before its predicate
    names = [m.group(1) for m in verify._NAME.finditer(head) if m.group(1) not in verify._COMMON]
    if not names:
        # "Harvard is older. It was founded in 1636": the pronoun's referent is the previous sentence's first name
        before = verify._CITE.sub(" ", (answer or "")[:max(0, (answer or "").find(sent))])
        prev = re.split(r"(?<=[.!?])\s+", before.strip())[-1] if before.strip() else ""
        names = [m.group(1) for m in verify._NAME.finditer(prev) if m.group(1) not in verify._COMMON]
        if not names or not re.match(r"\s*(it|he|she|they)\b", head, re.I):
            return None
    return knowledge.FactQ(prop, names[0].lower())          # the sentence's subject: its first name


def _full(c: dict) -> dict:
    """verify.check's public claims drop value / year / dec: rebuild them from the claim text."""
    c = dict(c)
    if c.get("kind") in ("year", "number", "computed") and "value" not in c:
        ns = verify.numbers(c.get("text", ""))
        if ns:
            v, y, d, *_ = ns[0]
            c.update(value=v, year=y, dec=d)
        else:
            c.update(value=float("nan"), year=False, dec=0)
    c.setdefault("pred", [])
    return c


def needs(v: dict) -> bool:
    return bool(config.VERIFY_RESEARCH and v and v.get("status") in ("disagree", "unverified", "partial")
                and key_claim(v))


def _years(vals) -> set:
    out = set()
    for x in vals or []:
        out |= {int(y) for y in re.findall(r"\b(1\d{3}|20\d{2})\b", str(x))}
    return out


def _nums(vals) -> list[float]:
    out = []
    for x in vals or []:
        for v, y, *_ in verify.numbers(str(x)):
            out.append(v)
    return out


async def _facts(question: str, claim: dict | None = None, answer: str = ""):
    from . import knowledge
    fq = (claim_fact(claim, answer) if claim else None) or knowledge.parse_fact(question)
    if not fq:
        return None, []
    return fq, await knowledge.wikidata(fq)


async def _leads(subject: str, question: str):
    from . import knowledge
    if not subject:
        return []
    docs = await knowledge.leads(question, k=2, subject=subject)
    return [d for d in docs if verify._about(d.title.replace(" · local Wikipedia", ""), question + " " + subject)]


def _subject(claim: dict, answer: str, question: str) -> str:
    sents = {claim["text"]: s for s in re.split(r"(?<=[.!?])\s+", answer or "") if claim["text"] in s}
    sub = verify._subject_of(claim, sents)
    from . import knowledge
    fq = knowledge.parse_fact(question)
    return (fq.subject if fq else "") or sub


async def run(answer: str, docs: list, question: str, v: dict, web=None, paid_ok: bool = False) -> dict:
    """One targeted re-check of the key claim. `web(query) -> docs` is the paid lane (None = not allowed)."""
    t0 = time.perf_counter()
    claim = key_claim(v)
    out = {"status": "skip", "claim": claim and claim["text"], "lanes": [], "paid": 0}
    if not claim:
        return out
    claim = _full(claim)
    tried = []
    new_docs, src = [], ""
    try:
        fq, fdocs = await asyncio.wait_for(_facts(question, claim, answer), config.VERIFY_RESEARCH_TIMEOUT_S)
        tried.append("facts")
        if fdocs:
            fact = (fdocs[0].meta or {}).get("fact") or {}
            vals = fact.get("values") or []
            if claim["kind"] == "year" and fq and fq.prop in ("inception", "born_date", "died_date"):
                ys = _years(vals)
                cy = int(float(re.sub(r"[^\d.]", "", claim["text"]) or 0))
                if ys and cy not in ys:
                    new = sorted(ys)[0]
                    out.update(status="corrected", source="Wikidata", value=str(new),
                               label=f"corrected: {claim['text']} → {new} (Wikidata)",
                               patched=_patch(answer, claim["text"], str(new)))
                elif ys and cy in ys:
                    out.update(status="rechecked", source="Wikidata", label="re-checked ✓ Wikidata")
            elif claim["kind"] == "number" and fq and fq.prop in ("height", "population", "area"):
                ns = _nums(vals)
                if ns:
                    c = verify.numbers(claim["text"])
                    if c and any(verify._num_match(c[0][0], False, c[0][2], n, False) for n in ns):
                        out.update(status="rechecked", source="Wikidata", label="re-checked ✓ Wikidata")
            if out["status"] == "skip":
                new_docs, src = fdocs, "Wikidata"
    except Exception as e:                          # a re-check must never surface an error to the answer
        out["error"] = f"{type(e).__name__}: {e}"[:120]
    if out["status"] == "skip" and not new_docs:
        try:
            sub = _subject(claim, answer, question)
            ld = await asyncio.wait_for(_leads(sub, question), config.VERIFY_RESEARCH_TIMEOUT_S)
            tried.append("leads")
            if ld:
                new_docs, src = ld, "local Wikipedia"
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"[:120]
    if out["status"] == "skip" and not new_docs and web is not None and paid_ok:
        try:
            sub = _subject(claim, answer, question)
            q = " ".join(x for x in (sub, (claim.get("pred") or [""])[0], claim["text"] if claim["kind"] == "name"
                                     else "") if x) or question
            wd = await asyncio.wait_for(web(q), config.VERIFY_RESEARCH_TIMEOUT_S)
            tried.append("web")
            out["paid"] = 1
            if wd:
                new_docs, src = list(wd), "web"
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"[:120]
    if out["status"] == "skip" and new_docs:
        texts = [f"{getattr(d, 'title', '')}. {getattr(d, 'text', '')}" for d in new_docs]
        if any(verify._supports(claim, t) for t in texts):
            out.update(status="rechecked", source=src, label=f"re-checked ✓ {src}")
        else:
            out.update(status="unconfirmed", source=src, label="still unconfirmed")
    elif out["status"] == "skip":
        out.update(status="unconfirmed", label="still unconfirmed")
    from . import extract
    if out["status"] == "unconfirmed" and extract.VERIFY_EXTRACT and docs:
        # VERIFY_EXTRACT: read the cited page itself (Tavily /extract, claim sentence as the query)
        pg = await extract.recheck_page(claim, answer, docs)
        if pg["status"] != "skip":
            tried.append("extract")
            out["extract"] = {k: pg.get(k) for k in ("status", "url", "credits", "ms", "error", "why")}
            out["extract_credits"] = pg.get("credits") or 0.0
            if pg["status"] == "rechecked":
                out.update(status="rechecked", source="page", label="re-checked ✓ page")
    out["lanes"] = tried
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def _patch(answer: str, old: str, new: str) -> str:
    """The answer with the corrected value, labelled where it changed (never silently)."""
    return re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w])", f"{new} (corrected from {old})", answer, count=1)
