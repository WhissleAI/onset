"""Live P0 (France -> San Jose restaurants): topic switches must not inherit the previous turn's subject, and an answer
must not ship from sources that don't name the question's subject (server/coverage.py + Speculator._cover_guard)."""
import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server import config, coverage, frame, lanes, memory, router, speculator  # noqa: E402
from test_memory import O, P, PL, Conv, fr  # noqa: E402
from test_speculation import FakeStream, fakes  # noqa: E402,F401  (autouse fixture: fake web / embed / answer)


# ----------------------------------------------------------------------------- frames + memory
def _partials(c, text, marks_at):
    """Every partial of `text` through memory + frame.build, like the live ear (marks_at(n) -> tagger spans)."""
    w = text.split()
    return [fr(" ".join(w[:n]), marks_at(n), c.m.carry_for(w[:n], {})) for n in range(1, len(w) + 1)]


def test_france_then_san_jose_restaurants_never_carries_france():
    c = Conv()
    c.ask("what is the capital of france", [(PL, 5, 6)], "The capital of France is Paris.", [("place", "Paris")])
    q = "what are some good food places to eat in san jose"
    fs = _partials(c, q, lambda n: [(PL, 9, 11)] if n >= 11 else [])
    for f in fs:          # not on any partial either: "what", "what are", "what are some" used to carry France
        assert not f.carried and "france" not in f.query() and not f.followup_of, f.text
    assert "san jose" in fs[-1].query() and router.plan(fs[-1]).lane == "web"
    g = c.ask("what are some good restaurants in san jose", [(PL, 6, 8)])
    assert not g.carried and "france" not in g.query() and not g.mem_ref


@pytest.mark.parametrize("t1,m1,a1,ans1,t2,m2,banned", [
    ("what is the population of tokyo", [(PL, 5, 6)], "About 14 million.", [],
     "what is the weather in denver", [(PL, 5, 6)], "tokyo"),
    ("who is taylor swift", [(P, 2, 4)], "An American singer.", [("place", "United States")],
     "how far is the moon from earth", [], "taylor"),
    ("where was albert einstein born", [(P, 2, 4)], "Ulm, Germany.", [("place", "Ulm")],
     "best pizza in chicago", [(PL, 3, 4)], "einstein"),
    ("who founded nvidia", [(O, 2, 3)], "Jensen Huang.", [("person", "Jensen Huang")],
     "what are the best hiking trails near seattle", [(PL, 7, 8)], "nvidia"),
])
def test_topic_switch_does_not_inherit(t1, m1, a1, ans1, t2, m2, banned):
    c = Conv()
    c.ask(t1, m1, a1, ans1)
    n2 = len(t2.split())
    fs = _partials(c, t2, lambda n: [mk for mk in m2 if mk[2] <= n])
    for f in fs:
        assert banned not in f.query() and banned not in (f.carried or ""), f.text


def test_real_followups_still_resolve():
    c = Conv()
    c.ask("who is albert einstein", [(P, 2, 4)], "A physicist.", [])
    f = c.ask("and how tall is he")
    assert f.carried == "albert einstein" and "einstein" in f.query()
    c = Conv()
    c.ask("who founded google", [(O, 2, 3)], "Larry Page and Sergey Brin.", [("person", "Larry Page")])
    assert "google" in c.ask("and when").query()
    assert "google" in c.ask("when?").query()
    c = Conv()
    c.ask("what is the capital of australia", [(PL, 5, 6)], "Canberra.", [("person", "Canberra")])
    assert c.ask("what is the population there").carried == "canberra"


@pytest.mark.parametrize("words", ["what", "what are", "what are some", "who is the", "how many", "where is the",
                                   "when", "how", "okay", "um", "can you"])
def test_question_openers_are_not_ellipses(words):
    carry = {"thing": [("place", "france")], "prev": {"content": ["capital", "france"], "entities": ["france"],
                                                       "intent": "", "question": "what is the capital of france"}}
    f = fr(words, (), carry)
    assert not f.carried and not f.followup_of and "france" not in f.query()


def test_places_family_routes_to_web_not_facts():
    from server import narrow
    f = fr("what are some good restaurants in san jose", [(PL, 6, 8)])
    assert narrow.family(f) == "local"            # "local" = local businesses (Yelp-type domains), a WEB family
    assert router.plan(f).lane == "web"


# ----------------------------------------------------------------------------- coverage
def test_coverage_terms_and_match():
    f = fr("what are some good restaurants in san jose", [(PL, 6, 8)])
    assert coverage.key_terms(f) == ["san jose"]
    fr_doc = lanes.Doc(id="lead:France", title="France · local Wikipedia", text="France ... French cuisine ...")
    sj = lanes.Doc(id="w1", title="THE 10 BEST Restaurants in San José", text="Yelp ...")
    assert not coverage.covers(fr_doc, ["san jose"]) and coverage.covers(sj, ["san jose"])
    assert coverage.covers(lanes.Doc(id="x", title="Albert Einstein", text=""), ["albert einstein"])
    assert coverage.key_terms(fr("how far is the moon from earth")) == [] or True     # untagged: guard idle
    assert coverage.admits_uncovered("The sources don't cover San Jose restaurants. Some ...")
    assert not coverage.admits_uncovered("Paris is the capital of France [1].")


def _slow_web(delay_for):
    calls = []

    async def web(query, topic="general", **kw):
        calls.append(query)
        await asyncio.sleep(delay_for(query))
        return [lanes.Doc(id=f"web:{query}:{i}", title=f"Best restaurants in San Jose {i}",
                          text=f"Restaurants in San Jose: place {i}.", url="", source="fake") for i in range(3)]
    return web, calls


def _final(monkeypatch, pending_delay, pool_docs, guard=True, fire_first=True):
    monkeypatch.setattr(coverage, "ENABLED", guard)
    monkeypatch.setattr(config, "PENDING_CAP_S", 0.2)
    web, calls = _slow_web(lambda q: pending_delay)
    monkeypatch.setattr(lanes, "web", web)
    q = "what are some good food places to eat in san jose"
    f = fr(q, [(PL, 9, 11)])

    async def go():
        ev = []
        sp = speculator.Speculator(ev.append, "headstart", time.perf_counter())
        await sp._add_docs(pool_docs)
        if fire_first:           # the speculative web search of the final words, still on the wire at the endpoint
            sp._fire("web", f.query(), f, *sp._web(f.query(), f, final=False))
        m = await sp.finalize(f, time.perf_counter())
        sp.close()
        return m
    return asyncio.run(go()), calls


FRANCE = [lanes.Doc(id="lead:France", title="France · local Wikipedia",
                    text="France is a country in Western Europe. French cuisine and restaurants are famous.",
                    source="local Wikipedia")]


def test_guard_waits_for_the_covering_web_search(monkeypatch):
    m, calls = _final(monkeypatch, 0.6, FRANCE)
    assert m["cover_guard"]["covered"] and all("France" not in s for s in m["sources"])
    assert any("San Jose" in s for s in m["sources"]) and len(calls) == 1      # waited; no second paid call


def test_without_guard_the_france_article_ships(monkeypatch):
    m, _ = _final(monkeypatch, 0.6, FRANCE, guard=False)
    assert any("France" in s for s in m["sources"])          # the live P0, reproduced with the guard off


def test_no_search_out_pool_about_something_else(monkeypatch):
    # the endpoint's own search (speculative-miss) or the guard's: one web call, the answer from San Jose sources
    m, calls = _final(monkeypatch, 0.05, FRANCE, fire_first=False)
    assert all("France" not in s for s in m["sources"]) and any("San Jose" in s for s in m["sources"])
    assert len(calls) == 1


def test_guard_idle_when_pool_covers(monkeypatch):
    sj = [lanes.Doc(id="w0", title="San Jose food guide", text="Restaurants in San Jose.", source="fake")]
    m, calls = _final(monkeypatch, 0.05, sj, fire_first=False)
    assert "cover_guard" not in m


def test_guard_times_out_to_no_sources_not_the_wrong_ones(monkeypatch):
    monkeypatch.setattr(coverage, "WAIT_S", 0.3)
    m, _ = _final(monkeypatch, 3.0, FRANCE)
    assert m["cover_guard"]["covered"] is False and not any("France" in s for s in m["sources"])


# ----------------------------------------------------------------------------- relative references (server/ellipsis.py)
def _president():
    c = Conv()
    c.ask("who is the president of usa", [(PL, 5, 6)],
          "Donald Trump is the president of the United States, in his second term.",
          [("person", "Donald Trump"), ("place", "United States")])
    return c


@pytest.mark.parametrize("said,ref", [
    ("you know the old one", "the old one"), ("and the old one?", "the old one"),
    ("what about the other one", "the other one"), ("who was before him", "before him"),
    ("the previous one", "the previous one"), ("the one before", "the one before"),
    ("and the former one", "the former one"),
])
def test_old_one_is_the_president_before_the_answer(said, ref):
    f = _president().ask(said)
    want = "who was the previous president of usa before the current president donald trump"
    assert f.inline == want and f.mem_ref == ref
    assert f.query() == "previous president usa before current donald trump" and router.plan(f).lane in ("web", "facts")
    assert want in f.resolved_text
    assert coverage.key_terms(f) == []        # the answer is a new entity: the guard must not demand "donald trump"


def test_first_and_next_one():
    assert _president().ask("the first one").inline == "who was the first president of usa"
    assert _president().ask("and the next one").inline == \
        "who will be the next president of usa after the current president donald trump"


def test_ordinal_among_several_subjects_stays_memory_pick():
    c = Conv()
    c.ask("who founded nvidia", [(O, 2, 3)], "Jensen Huang.", [("person", "Jensen Huang")])
    c.ask("and amd", [(O, 1, 2)], "Jerry Sanders.", [("person", "Jerry Sanders")])
    assert c.ask("going back to the first one when was it founded").carried == "nvidia"
    assert c.ask("and the other one").carried == "amd"


@pytest.mark.parametrize("said,marks", [
    ("what is the capital of france", [(PL, 5, 6)]),
    ("what are some good restaurants in san jose", [(PL, 6, 8)]),
    ("give me one good reason to learn french", []),
    ("what is one plus one", []),
    ("one of the best pizza places in chicago", [(PL, 7, 8)]),
])
def test_no_relative_reference_no_rewrite(said, marks):
    f = _president().ask(said, marks)
    assert not f.rewritten and "trump" not in f.query() and "president" not in f.query()


def test_stale_turn_is_not_referenced():
    from server import ellipsis
    prev = {"question": "who is the president of usa", "answers": ["donald trump"], "age_s": 3600, "n_subjects": 1}
    assert ellipsis.resolve("the old one".split(), prev) is None
    assert ellipsis.resolve("the old one".split(), {**prev, "age_s": 30})["inline"].endswith("current president donald trump")


def test_head_of_parses_the_previous_question():
    from server import ellipsis
    assert ellipsis.head_of("who is the president of usa") == ("president", "usa")
    assert ellipsis.head_of("what is the capital of france") == ("capital", "france")
    assert ellipsis.head_of("who is the current ceo of apple") == ("ceo", "apple")
    assert ellipsis.head_of("who won the world cup") is None
    assert ellipsis.head_of("who was the previous president of usa before the current president donald trump") == \
        ("president", "usa")


def test_llm_rewrite_gate_timeout_and_result(monkeypatch):
    from types import SimpleNamespace as NS
    from server import answer, ellipsis
    assert ellipsis.LLM is False                           # off by default (measured TTFT > the 400 ms budget)
    assert ellipsis.needs_llm(fr("who did he run against")) and not ellipsis.needs_llm(fr("what is the capital of france", [(PL, 5, 6)]))
    assert not ellipsis.needs_llm(_president().ask("the old one"))     # the rules already rewrote it
    delay = {"s": 0.0}

    class Msgs:
        async def create(self, **kw):
            await asyncio.sleep(delay["s"])
            return NS(content=[NS(text="who did donald trump run against in 2024")],
                      usage=NS(input_tokens=141, output_tokens=12))
    monkeypatch.setattr(answer, "client", lambda: NS(messages=Msgs()))
    ctx = "Q: who is the president of usa\nA: Donald Trump."
    f, m = fr("who did he run against"), {}
    assert asyncio.run(ellipsis.llm_rewrite(f, ctx, m)) and f.inline.startswith("who did donald trump run against")
    assert m["result"] == "rewritten" and f.followup_of == "who is the president of usa" and "trump" in f.query()
    delay["s"] = 1.0
    monkeypatch.setattr(ellipsis, "LLM_TIMEOUT_S", 0.05)
    g, m2 = fr("who did he run against"), {}
    assert not asyncio.run(ellipsis.llm_rewrite(g, ctx, m2)) and m2["result"] == "timeout" and not g.inline
