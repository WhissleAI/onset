"""Verify-triggered re-search (server/recheck.py, VERIFY_RESEARCH): one targeted lookup on the key claim after the
answer, free lanes first, paid web only when allowed and counted."""
import asyncio

import pytest

from server import config, knowledge, lanes, recheck, speculator, verify


def D(title, text, url=""):
    return lanes.Doc(id=url or title, title=title, text=text, url=url or f"https://x/{title}", source="wikipedia")


def WD(item, prop, vals):
    return lanes.Doc(id=f"wikidata:{item}", title=f"{item} · Wikidata", text=f"{item}: {prop}: {', '.join(vals)}.",
                     url="https://www.wikidata.org/wiki/Q1", source="Wikidata", score=1.0,
                     meta={"fact": {"prop": prop, "values": vals, "item": item}})


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setattr(config, "VERIFY_RESEARCH", True)


def _facts(monkeypatch, docs):
    async def wikidata(q):
        return docs
    monkeypatch.setattr(knowledge, "wikidata", wikidata)


def _leads(monkeypatch, docs):
    async def leads(query, k=3, subject=""):
        return docs
    monkeypatch.setattr(knowledge, "leads", leads)


Q = "when was nvidia founded"
ANS = "Nvidia was founded in 1991 [1]."
SRC = [D("Nvidia", "Nvidia was founded on April 5, 1993, by Jensen Huang."),
       D("Nvidia history", "The company Nvidia, founded 1991 in a Denny's, ...")]


def test_disagreement_is_corrected_from_wikidata(monkeypatch):
    v = verify.check(ANS, [SRC[0]], Q)
    assert v["status"] in ("disagree", "unverified") and recheck.needs(v)
    _facts(monkeypatch, [WD("Nvidia", "inception", ["5 April 1993"])])
    r = asyncio.run(recheck.run(ANS, [SRC[0]], Q, v))
    assert r["status"] == "corrected" and r["value"] == "1993" and r["paid"] == 0 and r["lanes"] == ["facts"]
    assert "1993 (corrected from 1991)" in r["patched"]


def test_supported_by_wikidata_is_rechecked(monkeypatch):
    ans = "Nvidia was founded in 1993 [1]."
    v = verify.check(ans, [D("GPUs", "Graphics cards render images.")], Q)
    assert v["status"] == "unverified"
    _facts(monkeypatch, [WD("Nvidia", "inception", ["5 April 1993"])])
    r = asyncio.run(recheck.run(ans, [], Q, v))
    assert r["status"] == "rechecked" and r["label"] == "re-checked ✓ Wikidata"


def test_name_claim_rechecked_from_local_leads(monkeypatch):
    q = "who painted the starry night"
    ans = "The Starry Night was painted by Vincent van Gogh [1]."
    v = verify.check(ans, [D("Starry", "A night sky painting.")], q)
    assert v["status"] == "unverified"
    _facts(monkeypatch, [])
    _leads(monkeypatch, [D("Vincent van Gogh · local Wikipedia", "Vincent van Gogh painted The Starry Night in 1889.")])
    monkeypatch.setattr(recheck, "_leads", lambda sub, q: knowledge.leads(q))
    r = asyncio.run(recheck.run(ans, [], q, v))
    assert r["status"] == "rechecked" and r["source"] == "local Wikipedia" and r["paid"] == 0


def test_paid_web_only_when_allowed_and_counted(monkeypatch):
    q = "who painted the starry night"
    ans = "The Starry Night was painted by Vincent van Gogh [1]."
    v = verify.check(ans, [D("Starry", "A night sky painting.")], q)
    _facts(monkeypatch, [])
    _leads(monkeypatch, [])
    calls = []

    async def web(query):
        calls.append(query)
        return [D("Van Gogh", "The Starry Night is an oil painting by Vincent van Gogh.")]
    r = asyncio.run(recheck.run(ans, [], q, v, web=web, paid_ok=False))
    assert not calls and r["paid"] == 0 and r["status"] == "unconfirmed"
    r = asyncio.run(recheck.run(ans, [], q, v, web=web, paid_ok=True))
    assert len(calls) == 1 and r["paid"] == 1 and r["status"] == "rechecked"


def test_agreeing_answer_is_not_rechecked(monkeypatch):
    v = verify.check("Nvidia was founded in 1993 [1].", [SRC[0]], Q)
    assert v["status"] in ("agree", "supported") and not recheck.needs(v)
    monkeypatch.setattr(config, "VERIFY_RESEARCH", False)
    assert not recheck.needs({"status": "disagree", "claims": [{"text": "1991", "kind": "year", "conflict": {}}]})


def test_speculator_recheck_updates_metric_and_counts_paid(monkeypatch):
    _facts(monkeypatch, [WD("Nvidia", "inception", ["5 April 1993"])])
    ev = []

    async def go():
        sp = speculator.Speculator(ev.append, "headstart")
        m = {"paid_calls": 1}
        v = verify.check(ANS, [SRC[0]], Q)
        await sp._recheck(ANS, [SRC[0]], Q, v, m)
        sp.close()
        return m
    m = asyncio.run(go())
    assert m["recheck"]["status"] == "corrected" and m["paid_calls"] == 1 and "answer_patched" in m
    assert [e for e in ev if e.get("t") == "recheck"]


def test_public_verify_claims_are_rechecked_without_crashing(monkeypatch):
    """verify.check's output claims carry no value / year fields: the re-check rebuilds them (a leads source states
    the year -> re-checked)."""
    ans = "Harvard was founded in 1636 [1]."
    q = "when was harvard founded"
    v = verify.check(ans, [D("Ivy League", "Eight private universities.")], q)
    assert v["status"] == "unverified" and "value" not in v["claims"][0]
    _facts(monkeypatch, [])
    _leads(monkeypatch, [D("Harvard University · local Wikipedia", "Harvard University was founded in 1636.")])
    monkeypatch.setattr(recheck, "_leads", lambda sub, q: knowledge.leads(q))
    r = asyncio.run(recheck.run(ans, [], q, v))
    assert r["status"] == "rechecked" and "error" not in r


def test_targeted_lookup_is_the_claims_own_fact(monkeypatch):
    """A multi-part question no fact parser reads: the claim's own sentence names the subject and predicate
    ("Jane Austen ... was born ... in 1775" -> born_date(jane austen)); a different Wikidata year corrects it."""
    ans = "Jane Austen, the author of Pride and Prejudice, was born in Steventon, England, in 1776 [1]."
    q = "where was the author of pride and prejudice born"
    v = verify.check(ans, [D("Novels", "Pride and Prejudice is a novel.")], q)
    seen = []

    async def wikidata(fq):
        seen.append((fq.prop, fq.subject))
        return [WD("Jane Austen", "born_date", ["16 December 1775"])]
    monkeypatch.setattr(knowledge, "wikidata", wikidata)
    r = asyncio.run(recheck.run(ans, [], q, v))
    assert seen == [("born_date", "jane austen")]
    assert r["status"] == "corrected" and r["value"] == "1775"


def test_paid_recheck_only_on_disagree_within_parity_budget(monkeypatch):
    """Parity: free lanes first; the one paid call only for "sources disagree" and only while the question's paid
    budget has room."""
    monkeypatch.setattr(config, "VERIFY_RESEARCH_PAID", True)
    _facts(monkeypatch, [])
    _leads(monkeypatch, [])
    calls = []

    async def fake_web(q, **kw):
        calls.append(q)
        return [D("Nvidia", "Nvidia was founded in 1991 in a Denny's.")]
    monkeypatch.setattr(lanes, "web", fake_web)

    async def go(v, paid_before):
        sp = speculator.Speculator(lambda e: None, "headstart", cost_mode="parity")
        sp._paid = lambda: paid_before
        m = {"paid_calls": paid_before}
        await sp._recheck(ANS, [SRC[0]], Q, v, m)
        sp.close()
        return m
    unver = {"status": "unverified", "claims": [{"kind": "year", "text": "1991", "cites": [1], "pred": ["founded"]}]}
    m = asyncio.run(go(unver, 1))
    assert not calls and m["recheck"]["paid"] == 0                     # not a disagreement: free lanes only
    dis = {"status": "disagree", "claims": [{"kind": "year", "text": "1991", "cites": [1], "pred": ["founded"],
                                             "conflict": {"source": 1, "value": "1993"}}]}
    m = asyncio.run(go(dis, 1))
    assert len(calls) == 1 and m["recheck"]["paid"] == 1 and m["paid_calls"] == 2   # counted
    calls.clear()
    m = asyncio.run(go(dis, 2))                                         # parity budget (1 + 1) already spent
    assert not calls and m["recheck"]["paid"] == 0
