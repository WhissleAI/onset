"""Site brain (server/sitebrain.py): a site read once into the local tier; questions that name it are answered from it
at 0 paid calls, everything else goes to the paid lane as before."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from server import config, embed, sitebrain
from server.keystone import hook


def _site(vecs):
    ps = [{"url": "https://docs.tavily.com/documentation/api-credits", "title": "API Credits · docs.tavily.com",
           "i": 0, "text": "Search: 1 credit for basic depth; 2 credits for advanced."},
          {"url": "https://docs.tavily.com/documentation/rate-limits", "title": "Rate limits · docs.tavily.com",
           "i": 0, "text": "Crawl has its own limit of 100 requests per minute."}]
    return sitebrain.Site("docs.tavily.com", ["tavily"], "2026-11-14", ps, np.asarray(vecs, np.float32))


@pytest.fixture
def qv(monkeypatch):
    """The query embedding: a fixed vector per call (set .v)."""
    box = SimpleNamespace(v=np.array([1.0, 0.0], np.float32), calls=0)

    async def arun(fn, *args):
        box.calls += 1
        return np.asarray([box.v])
    monkeypatch.setattr(embed, "arun", arun)
    monkeypatch.setattr(sitebrain, "SITE_BRAIN", True)
    sitebrain.set_sites([_site([[1, 0], [0, 1]])])
    yield box
    sitebrain.set_sites(None)


def F(text):
    return SimpleNamespace(resolved_text=text, inline="", words=text.split())


def test_scoped_question_is_served_locally_and_labelled(qv):
    docs = asyncio.run(sitebrain.lookup(F("how many credits is an advanced search on tavily"), "advanced search tavily"))
    assert docs and docs[0].url.endswith("api-credits") and docs[0].source == "Onset index · docs.tavily.com"
    assert docs[0].meta["tier"] == "index" and "as of 2026-11-14" in docs[0].meta["snapshot"]


def test_unscoped_question_never_embeds(qv):
    assert asyncio.run(sitebrain.lookup(F("how many credits does a basic search cost"), "credits")) is None
    assert qv.calls == 0


def test_low_similarity_goes_to_paid(qv):
    qv.v = np.array([0.3, 0.3], np.float32)
    m = {}
    assert asyncio.run(sitebrain.lookup(F("who founded tavily"), "founded tavily", m)) is None
    assert m["site_sim"] < sitebrain.SITE_MIN_SIM


def test_wrap_web_serves_site_before_paid(qv, monkeypatch):
    monkeypatch.setattr(config, "INDEX", True)

    async def none(*a, **k):
        return None
    monkeypatch.setattr(hook, "local_first", none)
    paid = []

    async def web():
        paid.append(1)
        return []
    m = {}
    res = asyncio.run(hook.wrap_web(web, F("tavily crawl rate limit"), "tavily crawl rate limit", m)())
    assert not paid and res.backend == "index" and m["credits"] == 0.0 and m["index_path"] == "site"
    res = asyncio.run(hook.wrap_web(web, F("crawl rate limit"), "crawl rate limit", {})())
    assert paid == [1]


def test_load_reads_built_site(tmp_path, monkeypatch):
    d = tmp_path / "docs.tavily.com"
    d.mkdir()
    (d / "site.json").write_text(json.dumps({"name": "docs.tavily.com", "terms": ["Tavily"], "as_of": "2026-11-14"}))
    (d / "passages.jsonl").write_text(json.dumps({"url": "https://x", "title": "t", "text": "hello"}) + "\n")
    np.savez(d / "vectors.npz", vecs=np.ones((1, 4), np.float16))
    monkeypatch.setattr(sitebrain, "SITE_DIR", tmp_path)
    s = sitebrain.load(force=True)
    assert len(s) == 1 and s[0].terms == ["tavily"] and s[0].vecs.dtype == np.float32
    monkeypatch.setattr(sitebrain, "SITE_DIR", tmp_path / "missing")
    assert sitebrain.load(force=True) == []
    sitebrain.set_sites(None)


def test_chunk_splits_at_boundaries():
    text = "# Title\n\n" + "A sentence about credits. " * 60 + "\n\nShort tail paragraph that is long enough."
    cs = sitebrain.chunk(text, 300)
    assert all(len(c) <= 300 for c in cs) and len(cs) >= 5


def test_scope_takes_the_ears_near_misses_only():
    s = _site([[1, 0], [0, 1]])
    assert s.scoped("how much does tavali charge to extract")          # Flux heard "tavali"
    assert not s.scoped("travel to italy") and not s.scoped("tally the votes") and not s.scoped("tavern hours")
    import json as _j
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for f in ("eval/questions.jsonl", "eval/showcase/questions.jsonl", "eval/current/questions.jsonl",
              "index/questions_local.jsonl"):
        for ln in (root / f).read_text().splitlines():
            if ln.strip():
                assert not s.scoped(_j.loads(ln)["text"]), ln       # no other eval question reaches the site tier
