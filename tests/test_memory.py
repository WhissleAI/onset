"""Memory v2 (server/memory.py): multi-turn resolution, in-place resolved questions for the router, user facts,
persistence / TTL / forget, the speculation hold on an unresolved pronoun, the free Wikidata pre-warm."""
import asyncio
import json
import time

import pytest

from server import config, frame, knowledge, memory, router, speculator
from server.session import Session


def spans(text, marks):
    toks = text.split()
    out = []
    for typ, a, b in marks:
        fam = "number" if typ.startswith("entity.number") else "entity"
        out.append({"type": typ, "start": a, "end": b, "text": " ".join(toks[a:b]), "family": fam})
    return out


def fr(text, marks=(), carry=None):
    return frame.build(text.split(), spans(text, marks), None, local_ok=False, carry=carry)


P, PL, O, W = "entity.mention.person", "entity.mention.place", "entity.mention.organization", "entity.mention.work_product"


class Conv:
    """A scripted conversation through Memory + frame.build (the same path the session uses)."""

    def __init__(self):
        self.m = memory.Memory()

    def ask(self, text, marks=(), answer="", ans=(), docs=()):
        f = fr(text, marks, self.m.carry_for(text.split(), {}))
        self.m.observe(f, answer, list(ans), list(docs))
        return f


def test_it_and_he_across_turns():
    c = Conv()
    c.ask("who founded nvidia", [(O, 2, 3)], "Jensen Huang co-founded Nvidia.", [("person", "Jensen Huang")])
    f = c.ask("when was it founded")
    assert f.carried == "nvidia" and f.inline == "when was nvidia founded"
    assert router.plan(f).vertical == "wikidata"                     # the follow-up gets the free facts lane
    g = c.ask("when was he born")
    assert g.carried == "jensen huang" and g.inline == "when was jensen huang born"


def test_asked_for_entity_wins_for_its_kind():
    c = Conv()
    # the tagger reads "Canberra" as a person in the answer: what the question asked for (a capital) makes it a place
    c.ask("what is the capital of australia", [(PL, 5, 6)], "Canberra is the capital of Australia.",
          [("person", "Canberra"), ("place", "Australia")])
    f = c.ask("what is the population there")
    assert f.carried == "canberra" and f.inline == "what is the population in canberra"
    assert router.plan(f).vertical == "wikidata"
    c = Conv()
    c.ask("which company makes the playstation", [(W, 4, 5)], "Sony makes the PlayStation.",
          [("organization", "Sony"), ("organization", "PlayStation")])
    g = c.ask("where is its headquarters")
    assert g.carried == "sony" and g.inline == "where is the headquarters of sony"


def test_references_beyond_one_turn():
    c = Conv()
    c.ask("who founded nvidia", [(O, 2, 3)], "Jensen Huang.", [("person", "Jensen Huang")])
    f = c.ask("and amd", [(O, 1, 2)], "Jerry Sanders.", [("person", "Jerry Sanders")])
    assert f.inline == "who founded amd" and "nvidia" not in f.query()
    g = c.ask("going back to the first one when was it founded")
    assert g.carried == "nvidia" and g.mem_ref == "going back to the first one"
    assert "amd" not in g.query() and router.plan(g).vertical == "wikidata"
    h = c.ask("who founded that company")
    assert h.carried == "nvidia" and h.inline == "who founded nvidia"
    k = c.ask("and the other one")
    assert k.carried == "amd"


def test_gendered_pronouns_and_relations():
    c = Conv()
    c.ask("who was marie curie", [(P, 2, 4)], "Marie Curie was a physicist.", [("person", "Marie Curie")])
    f = c.ask("who was her husband", (), "Pierre Curie.", [("person", "Pierre Curie")])
    assert f.carried == "marie curie" and f.inline == "who was the husband of marie curie"
    assert c.ask("where was she born").carried == "marie curie"      # Pierre is "he" (a husband)
    assert c.ask("when did he die").carried == "pierre curie"


def test_ellipsis_time_swap_and_time_word_mentions():
    c = Conv()
    c.ask("what is the weather in chicago today", [(PL, 5, 6)], "Sunny in Chicago.", [("place", "Chicago")])
    f = c.ask("what about tomorrow")
    assert f.inline == "what is the weather in chicago tomorrow" and "today" not in f.query()
    assert router.plan(f).vertical == "weather"
    # "winter" read as a place is not a subject: "there" still resolves to Chicago
    g = c.ask("how cold does it get there in winter", [(PL, 7, 8)])
    assert g.carried == "chicago"


def test_wikidata_qids_and_edges():
    from server.lanes import Doc
    c = Conv()
    d = Doc(id="wikidata:Q182477:P112", title="Nvidia · Wikidata",
            text="Nvidia (American technology company): founded by: Jensen Huang, Chris Malachowsky.",
            meta={"tier": "facts", "fact": {"prop": "founder", "pid": "P112", "item": "Nvidia",
                                            "values": ["Jensen Huang", "Chris Malachowsky"]}})
    c.ask("who founded nvidia", [(O, 2, 3)], "Jensen Huang and Chris Malachowsky.", [("person", "Jensen Huang")], [d])
    g = c.m.graph()
    nv = next(n for n in g["nodes"] if n["id"] == "nvidia")
    assert nv["qid"] == "Q182477" and nv["kind"] == "org"
    assert {"a": "nvidia", "b": "chris malachowsky", "rel": "founded by"} in [
        {k: e[k] for k in ("a", "b", "rel")} for e in g["edges"]]
    assert any(e["kind"] == "said" and e["a"] == "jensen huang" for e in g["edges"])


def test_no_memory_no_change():
    f = fr("when was it founded", carry=memory.Memory().carry_for("when was it founded".split(), {}))
    assert f.carried == "" and f.inline == ""
    # a standalone question with its own entity is never rewritten
    c = Conv()
    c.ask("who founded nvidia", [(O, 2, 3)], "Jensen Huang.", [("person", "Jensen Huang")])
    g = c.ask("what is the stock price of apple", [(O, 6, 7)])
    assert g.carried == "" and g.inline == ""


def test_dangling_preposition_after_pronoun_is_a_full_question():
    assert not frame.dangling_end("which museum is it in".split())
    assert not frame.dangling_end("who is it for".split())
    assert frame.dangling_end("is it bad to take a".split())


# ----------------------------------------------------------------------------- user facts
@pytest.mark.parametrize("text,facts", [
    ("i live in denver", [("home", "denver")]),
    ("i live in new york and i like jazz", [("home", "new york")]),
    ("i'm vegetarian", [("diet", "vegetarian")]),
    ("i am a vegan", [("diet", "vegan")]),
    ("i don't eat meat", [("diet", "vegetarian")]),
    ("i'm allergic to peanuts", [("allergy", "peanuts")]),
    ("what is the weather in denver", []),
    ("where do penguins live", []),
])
def test_learn_facts(text, facts):
    assert memory.learn_facts(text.split()) == facts


def test_user_facts_fill_the_query():
    m = memory.Memory()
    m.learn_user("i live in denver".split())
    m.learn_user("i'm vegetarian".split())
    f = fr("what's the weather tomorrow", carry=m.carry_for("what's the weather tomorrow".split(), {}))
    assert ("place", "denver") in f.slots and f.user_used == (("home", "denver"),)
    p = router.plan(f)
    assert p.vertical == "weather" and p.args["place"] == "denver"
    assert f.to_json()["memory"]["user"] == ["you live in Denver"]
    g = fr("give me a recipe for lasagna", carry=m.carry_for("give me a recipe for lasagna".split(), {}))
    assert "vegetarian" in g.query()
    h = fr("what's the weather in boston", [(PL, 4, 5)], carry=m.carry_for("what's the weather in boston".split(), {}))
    assert h.user_used == () and "denver" not in h.query()            # a place said wins
    k = fr("who founded nvidia", [(O, 2, 3)], carry=m.carry_for("who founded nvidia".split(), {}))
    assert "denver" not in k.query() and "vegetarian" not in k.query()   # never sent where not needed


# ----------------------------------------------------------------------------- persistence
def test_store_roundtrip_ttl_forget(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEM_PERSIST", True)
    st = memory.Store(tmp_path)
    c = Conv()
    c.ask("who founded nvidia", [(O, 2, 3)], "Jensen Huang.", [("person", "Jensen Huang")])
    c.m.learn_user("i live in denver".split())
    st.save("client-abc123", c.m)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and "client-abc123" not in files[0].name     # the id is hashed
    back = st.load("client-abc123")
    assert set(back.ents) == set(c.m.ents) and back.user_terms() == {"home": "denver"}
    assert back.carry_for("when was it founded".split(), {})["thing"][0][1] == "nvidia"
    # conversation TTL passed, user facts kept
    d = json.loads(files[0].read_text())
    d["updated"] = time.time() - config.MEM_CONV_TTL_S - 5
    files[0].write_text(json.dumps(d))
    old = st.load("client-abc123")
    assert not old.ents and old.user_terms() == {"home": "denver"}
    # user facts TTL
    d["user"]["home"]["at"] = time.time() - config.MEM_USER_TTL_S - 5
    files[0].write_text(json.dumps(d))
    assert st.load("client-abc123").user == {}
    st.forget("client-abc123")
    assert not list(tmp_path.glob("*.json")) and not st.load("client-abc123")
    # bad ids are never used as file names
    st.save("../../etc", c.m)
    assert not list(tmp_path.glob("*.json")) and not memory.Store.valid("a/b")


def test_session_export_import_carries_memory():
    async def go():
        a = Session(lambda e: None, mode="headstart")
        a.mem.learn_user("i live in denver".split())
        mem = a.export_memory()
        a.close()
        b = Session(lambda e: None, mode="headstart")
        b.import_memory(json.loads(json.dumps(mem)))
        ok = b.mem.user_terms() == {"home": "denver"}
        b.close()
        return ok
    assert asyncio.run(go())


# ----------------------------------------------------------------------------- session: ack, hold, refresh
async def _speak(s, text, t0=0.2):
    words = text.split()
    times = [(t0 + i * 0.25, t0 + i * 0.25 + 0.22) for i in range(len(words))]
    for k in range(1, len(words) + 1):
        s.on_partial([dict(text=w, start=a, end=b) for w, (a, b) in zip(words[:k], times[:k])])
        await asyncio.sleep(0)
    return await s.on_final([dict(text=w, start=a, end=b) for w, (a, b) in zip(words, times)])


def test_user_fact_statement_is_acknowledged_not_searched(monkeypatch):
    monkeypatch.setattr(config, "SEMANTIC_ENDPOINT", False)
    calls = []

    async def finalize(self, f, t_end, **kw):
        calls.append(f.text)
        return {"mode": self.mode, "question": f.text, "path": "stub", "answer": ""}
    monkeypatch.setattr(speculator.Speculator, "finalize", finalize)
    fired = []
    monkeypatch.setattr(speculator.Speculator, "on_frame", lambda self, f, now=None: fired.append(f.text))
    ev = []

    async def go():
        s = Session(ev.append, mode="headstart", t0=time.perf_counter())
        m = await _speak(s, "i live in denver")
        s.close()
        return s, m
    s, m = asyncio.run(go())
    assert m["path"] == "memory" and "Denver" in m["answer"] and not calls
    assert "i live in denver" not in fired             # held: no speculative search for a user fact
    assert s.mem.user_terms() == {"home": "denver"}
    assert any(e["t"] == "memory" and e.get("learned") for e in ev)
    assert any(e["t"] == "spec_hold" and e["why"] == "user fact" for e in ev)


def test_hold_while_referent_is_being_answered_then_refresh(monkeypatch):
    fired = []
    monkeypatch.setattr(speculator.Speculator, "on_frame", lambda self, f, now=None: fired.append(f))
    ev = []

    async def go():
        s = Session(ev.append, mode="headstart", t0=time.perf_counter())
        s.busy = True                                   # the previous answer is still streaming
        ws = [dict(text=w, start=0.2 + i * 0.25, end=0.4 + i * 0.25) for i, w in enumerate("when was it founded".split())]
        s.on_partial(ws)
        held = list(fired)
        # the answer lands: memory now knows Nvidia; the question being heard is rebuilt and speculated on
        s.busy = False
        f0 = frame.build("who founded nvidia".split(), spans("who founded nvidia", [(O, 2, 3)]), None, local_ok=False)
        s.mem.observe(f0, "Jensen Huang.", [("person", "Jensen Huang")], [])
        s.carry = {"thing": [("organization", "nvidia")], "prev": {"content": list(f0.content),
                   "entities": ["nvidia"], "intent": "", "question": f0.text}}
        s._refresh()
        s.close()
        return held
    held = asyncio.run(go())
    assert not held and any(e["t"] == "spec_hold" for e in ev)
    assert fired and fired[-1].carried == "nvidia" and fired[-1].inline == "when was nvidia founded"
    assert any(e["t"] == "memory" and e.get("used") == "nvidia" for e in ev)


def test_memory_resolved_followup_lowers_end_of_turn_bar(monkeypatch):
    f = fr("when was it founded", carry={"thing": [("organization", "nvidia")]})
    monkeypatch.setattr(config, "MEM_EOT", True)
    assert Session._mem_thr(f, "EagerEndOfTurn", 0.6) == config.MEM_EOT_THR[0]
    assert Session._mem_thr(fr("who founded nvidia", [(O, 2, 3)]), "EagerEndOfTurn", 0.6) == 0.6
    monkeypatch.setattr(config, "MEM_EOT", False)
    assert Session._mem_thr(f, "EagerEndOfTurn", 0.6) == 0.6


# ----------------------------------------------------------------------------- pre-warm
def test_prefetch_fills_per_property_cache(monkeypatch):
    knowledge._cache.clear()
    calls = []

    async def get_json(url, params, **kw):
        calls.append(url)
        if "w/api.php" in url:
            return {"search": [{"id": "Q182477", "label": "Nvidia", "description": "company"}]}
        b = lambda pid, v: {"item": {"value": "http://www.wikidata.org/entity/Q182477"}, "pid": {"value": pid},
                            "rank": {"value": "http://wikiba.se/ontology#NormalRank"}, "vLabel": {"value": v},
                            "v": {"value": "x"}}
        return {"results": {"bindings": [b("P112", "Jensen Huang"), b("P169", "Jensen Huang")]}}
    monkeypatch.setattr(knowledge, "get_json", get_json)

    async def go():
        assert await knowledge.prefetch("nvidia")
        n = len(calls)
        d = await knowledge.wikidata(knowledge.parse_fact("who founded nvidia"))
        return n, d
    n, d = asyncio.run(go())
    assert n == 2 and len(calls) == 2                   # the follow-up's lookup made no network call
    assert d and "Jensen Huang" in d[0].text
    knowledge._cache.clear()


def test_fact_value_is_the_asked_entity_and_demonyms_are_not_entities():
    from server.lanes import Doc
    words = "The iPhone is made by the American company Apple Inc. in Cupertino".split()
    sp = [{"type": "entity.mention.organization", "start": 6, "end": 7, "text": "American", "family": "entity"},
          {"type": "entity.mention.organization", "start": 8, "end": 9, "text": "Apple", "family": "entity"}]
    ents = memory.answer_entities(words, sp, "which company makes the iphone".split())
    assert ents == [("organization", "Apple Inc")]
    d = Doc(id="wikidata:Q2766:P176", title="iPhone · Wikidata", text="iPhone (smartphone): made by: Apple Inc.",
            meta={"tier": "facts", "fact": {"prop": "manufacturer", "pid": "P176", "item": "iPhone",
                                            "values": ["Apple Inc."]}})
    c = Conv()
    # the answer named nothing the tagger caught: the Wikidata value is what was asked for
    c.ask("which company makes the iphone", [(W, 4, 5)], "It is made in California.", [], [d])
    f = c.ask("who is its ceo")
    assert f.carried == "apple inc" and f.inline == "who is the ceo of apple inc"


def test_where_questions_keep_it_on_the_subject():
    c = Conv()
    c.ask("where is the eiffel tower", [(PL, 3, 5)], "The Eiffel Tower is on the Champ de Mars in Paris.",
          [("place", "Champ de Mars"), ("place", "Paris")])
    c2 = Conv()
    c2.m = memory.Memory.from_json(json.loads(json.dumps(c.m.to_json())))
    assert c.ask("how tall is it").carried == "eiffel tower"
    assert c2.ask("what is the weather there").carried in ("champ de mars", "paris")   # "there": the answer's place


def test_untagged_subject_and_incidental_people():
    c = Conv()
    c.ask("who painted the starry night", (), "Vincent van Gogh painted it in 1889.", [("person", "Vincent van Gogh")])
    f = c.ask("where is it displayed", (), "At MoMA in New York, near works by Edvard Munch.",
              [("organization", "MoMA"), ("place", "New York"), ("person", "Edvard Munch")])
    assert f.carried == "starry night"
    assert c.ask("when did he die").carried == "vincent van gogh"    # not Munch, only mentioned in passing


def test_superlative_question_it_is_the_answer():
    c = Conv()
    c.ask("what is the tallest mountain in the world", [(PL, 7, 8)], "Mount Everest is the tallest mountain.",
          [("place", "Mount Everest")])
    f = c.ask("how tall is it")
    assert f.carried == "mount everest" and router.plan(f).vertical == "wikidata"
