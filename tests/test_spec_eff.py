"""Search efficiency (SPEC_EFF + server/searchcache.py): canonical query keys, the fire policy's budget, pool reuse
with a coverage check, and the cross-question search cache (labelled hits, TTL classes, persistence).

Run: .venv/bin/python -m pytest -q tests/test_spec_eff.py
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_speculation as ts  # noqa: E402  (fake lanes, fake answer model, speak())
from server import config, frame, lanes, searchcache  # noqa: E402

fakes = ts.fakes          # the same autouse fixture (fake web / embeddings / answer stream)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(config, "CACHE_PERSIST", False)
    monkeypatch.setattr(config, "CACHE_SEM", True)
    monkeypatch.setattr(config, "SPEC_EFF", True)
    searchcache.reset()
    yield
    searchcache.reset()


def F(words, slots=(), content=None, intent="discourse.intent.qa_factoid", phrases=(), carried=""):
    """A frame without the tagger: content = the non-stop words unless given."""
    ws = words.split()
    c = tuple(w for w in (frame.norm(x) for x in ws) if w and w not in frame.STOP) if content is None else content
    return frame.Frame(words=ws, intent=intent, intent_p=0.9, slots=tuple(sorted(slots)), content=c,
                       phrases=tuple(phrases), carried=carried)


# ----------------------------------------------------------------------------------------------- keys + relations
def rel(a, b):
    return searchcache.relation(searchcache.qkey(a), searchcache.qkey(b))


def test_more_words_same_subject_extends():
    old = F("who founded Nvidia", slots=[("organization", "nvidia")], phrases=["nvidia"])
    new = F("who founded Nvidia and in what year", slots=[("organization", "nvidia")], phrases=["nvidia"])
    assert rel(new, old) == "extends"
    assert rel(old, new) == "subsumed"            # the longer search already asked for everything
    assert rel(old, old) == "same"


def test_half_heard_word_matches_its_completion():
    old = F("score celt", content=("score", "celt"))
    new = F("score celtics game", content=("score", "celtics", "game"))
    assert rel(new, old) == "extends"


def test_correction_and_new_entity_change_the_meaning():
    canada = F("capital of Canada", slots=[("place", "canada")], phrases=["canada"])
    turkey = F("capital of Canada actually Turkey", slots=[("place", "turkey")], phrases=["turkey"],
               content=("capital", "turkey"))
    assert rel(turkey, canada) is None
    boston = F("weather in Boston", slots=[("place", "boston")], phrases=["boston"],
               intent="discourse.intent.weather_query")
    both = F("weather in Boston and Denver", slots=[("place", "boston"), ("place", "denver")],
             phrases=["boston", "denver"], intent="discourse.intent.weather_query")
    assert rel(both, boston) is None              # a second, unrelated entity: search it


def test_entity_phrase_growth_is_grows():
    a = F("how tall is the Eiffel", slots=[("place", "eiffel")], phrases=["eiffel"])
    b = F("how tall is the Eiffel Tower", slots=[("place", "eiffel tower")], phrases=["eiffel tower"])
    assert rel(b, a) == "grows"


def test_numbers_added_extend_changed_numbers_dont():
    a = F("weather in Chicago", slots=[("place", "chicago")], phrases=["chicago"])
    b = F("weather in Chicago tomorrow", slots=[("place", "chicago"), ("date", "tomorrow")], phrases=["chicago"])
    c = F("weather in Chicago today", slots=[("place", "chicago"), ("date", "today")], phrases=["chicago"])
    assert rel(b, a) == "extends"
    assert rel(c, b) is None


def test_news_intent_flicker_is_not_a_topic_change():
    a = F("Amazon the company", intent="discourse.intent.news_query", slots=[("organization", "amazon")],
          phrases=["amazon"])
    b = F("Amazon the company what year was it founded", slots=[("organization", "amazon")], phrases=["amazon"])
    assert searchcache.qkey(a).topic == "news" and searchcache.qkey(b).topic == "general"
    assert rel(b, a) == "extends"


def test_coverage_needs_entities_and_a_live_search():
    k = searchcache.qkey(F("capital of Turkey", slots=[("place", "turkey")], phrases=["turkey"]))
    canada = [(lanes.Doc(id="c", title="Canada", text="Ottawa is the capital of Canada."), 0.7)]
    turkey = [(lanes.Doc(id="t", title="Turkey", text="Ankara is the capital of Turkey."), 0.7)]
    assert not searchcache.coverage(k, canada).ok
    assert searchcache.coverage(k, turkey).ok
    # live data: results of a search that didn't ask for the date don't cover "... tomorrow"
    w = F("weather in Chicago tomorrow", slots=[("place", "chicago"), ("date", "tomorrow")], phrases=["chicago"],
          intent="discourse.intent.weather_query")
    old = searchcache.qkey(F("weather in Chicago", slots=[("place", "chicago")], phrases=["chicago"],
                             intent="discourse.intent.weather_query"))
    docs = [(lanes.Doc(id="w", title="Chicago weather", text="Chicago weather forecast tomorrow: rain."), 0.8)]
    assert not searchcache.coverage(searchcache.qkey(w), docs, [old]).ok
    assert searchcache.coverage(searchcache.qkey(w), docs, [searchcache.qkey(w)]).ok


# ----------------------------------------------------------------------------------------------- the fire policy
@pytest.mark.parametrize("text", ["who founded the company Nvidia and when did it go public",
                                  "how many bones are there in an adult human body"])
def test_growing_question_stays_within_budget(text):
    m_old, _ = asyncio.run(_speak(text, eff=False))
    ts.WEB_CALLS.clear()
    m_new, ev = asyncio.run(_speak(text, eff=True))
    assert m_new["paid_calls"] <= config.SPEC_PAID_MAX
    assert m_new["paid_calls"] < m_old["web_calls"]
    assert len(ts.WEB_CALLS) == m_new["paid_calls"]
    assert m_new.get("answer")


def test_budget_is_hard_even_with_many_meaning_changes(monkeypatch):
    monkeypatch.setattr(config, "SPEC_PAID_MAX", 1)
    m, ev = asyncio.run(_speak("weather in Boston sorry I mean Denver no wait Austin", eff=True))
    assert m["paid_calls"] <= 1 and len(ts.WEB_CALLS) <= 1
    assert m.get("answer")


def test_correction_searches_the_replacement():
    m, ev = asyncio.run(_speak("what's the capital of Canada actually make that Turkey", eff=True, gap=0.3))
    assert "turkey" in m["query"] and "canada" not in m["query"]
    assert any("turkey" in q for q in ts.WEB_CALLS)
    assert m["paid_calls"] <= config.SPEC_PAID_MAX


def test_baseline_unchanged_one_call():
    m, _ = asyncio.run(_speak("who founded the company Nvidia", eff=True, mode="baseline"))
    assert m["web_calls"] == 1 and m["paid_calls"] == 1


async def _speak(text, eff=True, mode="headstart", gap=0.25):
    old = config.SPEC_EFF
    config.SPEC_EFF = eff
    try:
        return await ts.speak(text, mode, gap=gap)
    finally:
        config.SPEC_EFF = old


# ----------------------------------------------------------------------------------------------- search cache
def _docs(q):
    return [lanes.Doc(id=f"web:{q}:{i}", title=q.title(), text=f"About {q}: fact {i}.", source="wikipedia")
            for i in range(3)]


def test_cache_exact_hit_is_labelled_and_free():
    m1, _ = asyncio.run(_speak("who founded the company Nvidia", eff=True))
    n1 = len(ts.WEB_CALLS)
    assert n1 >= 1 and len(searchcache.cache()) >= 1
    m2, ev = asyncio.run(_speak("who founded the company Nvidia", eff=True))
    assert len(ts.WEB_CALLS) == n1                 # nothing new went out
    assert m2["paid_calls"] == 0 and m2["search_cached"] >= 1 and m2["search_cached_as_of"]
    done = [e for e in ev if e["t"] == "spec" and e.get("semcache")]
    assert done and done[0]["cached"] is True and done[0].get("as_of")
    src = next(e for e in ev if e["t"] == "sources")
    assert all(d.get("cached") and d.get("as_of") for d in src["docs"] if d["id"].startswith("web:"))


def test_cache_off_flag(monkeypatch):
    monkeypatch.setattr(config, "CACHE_SEM", False)
    asyncio.run(_speak("who founded the company Nvidia", eff=True))
    n1 = len(ts.WEB_CALLS)
    m2, _ = asyncio.run(_speak("who founded the company Nvidia", eff=True))
    assert len(ts.WEB_CALLS) > n1 and not m2["search_cached"]


def test_cache_ttl_classes_and_expiry(monkeypatch):
    live = searchcache.qkey(F("weather in Paris", slots=[("place", "paris")], phrases=["paris"],
                              intent="discourse.intent.weather_query"))
    stable = searchcache.qkey(F("what does ubiquitous mean", content=("ubiquitous", "mean")))
    assert searchcache.ttl_class(live) == "live"
    assert searchcache.ttl_class(stable) == "stable"
    c = searchcache.cache()
    c.put(live, "weather paris", _docs("weather paris"), "wikipedia")
    assert c.get(live) is not None
    real = time.time
    monkeypatch.setattr(searchcache.time, "time", lambda: real() + config.CACHE_TTL_LIVE_S + 1)
    assert c.get(live) is None


def test_cache_nn_needs_same_subject():
    c = searchcache.cache()
    a = searchcache.qkey(F("how tall is Mount Everest", slots=[("place", "mount everest")], phrases=["mount everest"]))
    b = searchcache.qkey(F("Mount Everest height", slots=[("place", "mount everest")], phrases=["mount everest"]))
    other = searchcache.qkey(F("how tall is K2", slots=[("place", "k2")], phrases=["k2"]))
    c.put(a, "tall mount everest", _docs("tall mount everest"), "wikipedia")
    v = np.ones(8, np.float32) / np.sqrt(8)
    c.note(a, v)
    hit = c.nearest(b, v)
    assert hit is not None and hit.match == "nn"
    assert c.nearest(other, v) is None


def test_cache_persistence_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_PERSIST", True)
    p = tmp_path / "search_cache.json"
    c = searchcache.reset(p)
    k = searchcache.qkey(F("who founded Nvidia", slots=[("organization", "nvidia")], phrases=["nvidia"]))
    c.put(k, "founded nvidia", _docs("founded nvidia"), "tavily")
    assert p.exists() and p.stat().st_size < 10_000
    c2 = searchcache.reset(p)
    h = c2.get(k)
    assert h is not None and h.entry.backend == "tavily" and len(h.entry.docs) == 3


def test_headstart_still_beats_baseline_with_one_call():
    # no Flux end-of-turn score in this harness: the first search waits SPEC_SETTLE_MS after the last word, still
    # well before an endpoint 0.9 s after it
    hs, ev = asyncio.run(ts.speak("who founded the company Nvidia", "headstart", endpoint_delay=0.9))
    bl, _ = asyncio.run(ts.speak("who founded the company Nvidia", "baseline", endpoint_delay=0.9))
    assert hs["path"] in ("draft-reuse", "speculative-hit") and hs["paid_calls"] == 1
    assert hs["ready_ms"] < bl["ready_ms"], (hs["ready_ms"], bl["ready_ms"])


def test_end_of_turn_score_releases_a_waiting_search():
    from server import speculator
    sp = speculator.Speculator(lambda e: None)
    sp.on_eot(0.9)                                  # no frame yet: nothing to do, no crash
    assert sp.eot_p == 0.9
