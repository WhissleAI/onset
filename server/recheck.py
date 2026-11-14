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


def key_claim(v: dict) -> dict | None:
    """The claim worth re-checking: the disputed one, else the first unsupported year / number, else name."""
    cl = v.get("claims") or []
    for c in cl:
        if c.get("conflict"):
            return c
    if v.get("status") not in ("unverified", "partial"):
        return None
    for kinds in (("year", "number"), ("name",)):
        for c in cl:
            if c.get("kind") in kinds and not c.get("support"):
                return c
    return None


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


async def _facts(question: str):
    from . import knowledge
    fq = knowledge.parse_fact(question)
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
    tried = []
    new_docs, src = [], ""
    try:
        fq, fdocs = await asyncio.wait_for(_facts(question), config.VERIFY_RESEARCH_TIMEOUT_S)
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
    out["lanes"] = tried
    out["ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def _patch(answer: str, old: str, new: str) -> str:
    """The answer with the corrected value, labelled where it changed (never silently)."""
    return re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w])", f"{new} (corrected from {old})", answer, count=1)
