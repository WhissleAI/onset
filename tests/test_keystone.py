"""Keystone (server/keystone/): Arc-keyed relation keys, memory cells, the answer graph + activation, the VOI gate, and
the live hooks (router local tier, speculator ingest). No network, no embedding model: vectors are synthetic."""
import asyncio
import time

import numpy as np
import pytest

from server import config, lanes, router
from server.frame import Frame
from server.keystone import hook, voi
from server.keystone.cells import Cell, CellStore, key_sentences
from server.keystone.core import Flags, Keystone, Q
from server.keystone.graph import Graph
from server.keystone.relkey import key_of, key_of_frame


def vec(seed: int, d: int = 768) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def near(v: np.ndarray, eps: float, seed: int) -> np.ndarray:
    w = v + eps * vec(seed)
    return (w / np.linalg.norm(w)).astype(np.float32)


def F(text, intent="discourse.intent.qa_factoid", slots=()):
    return Frame(words=text.split(), intent=intent, intent_p=0.9, slots=tuple(slots))


def Doc(i, title, text, source="wikipedia", meta=None):
    return lanes.Doc(id=i, title=title, text=text, url=f"https://x/{i}", source=source, meta=meta or {})


# ------------------------------------------------------------------------------------------------- relation keys
def test_paraphrases_collapse_to_one_typed_key():
    org = [("organization", "nvidia")]
    keys = {key_of(t.split(), org).text() for t in ("who founded the company nvidia", "who started nvidia",
                                                     "who is the founder of nvidia")}
    assert keys == {"founded_by(nvidia)"}
    assert key_of("when was nvidia founded".split(), org).text() == "inception(nvidia)"   # same words, other answer


def test_relation_known_before_subject_and_hops():
    k = key_of("who founded the".split())
    assert k is not None and k.relation == "founded_by" and not k.complete
    assert key_of("where was the author of pride and prejudice born".split()).text() == \
        "born_place(@author_of(pride and prejudice))"
    assert key_of("how tall is the eiffel tower and what year was it built".split()) is None    # multi-part
    assert key_of("which is taller mount everest or k2 versus".split()) is None                  # comparison
    assert key_of("how do i get a red wine stain out of a carpet".split()) is None              # open-ended


def test_live_relations_carry_their_argument_and_short_ttl():
    k = key_of("what's the weather in chicago tomorrow".split(), [("place", "chicago"), ("date", "tomorrow")],
               "discourse.intent.weather_query")
    assert k.text() == "weather_in(chicago, tomorrow)" and k.vclass == "live" and k.ttl_s <= 900
    assert key_of("who founded nvidia".split()).vclass == "stable"


def test_key_of_frame_uses_the_repaired_question():
    f = F("who founded nvidia", slots=[("organization", "nvidia")])
    assert key_of_frame(f).text() == "founded_by(nvidia)"


# ------------------------------------------------------------------------------------------------- cells
def test_cells_two_stage_search_ttl_and_eviction():
    cs = CellStore(cap=4, max_cells=50)
    now = time.time()
    base = vec(1)
    for i in range(40):
        v = near(base, 2.0, 100 + i) if i else base
        cs.add(Cell(id=-1, kind="passage", title=f"T{i}", payload=f"text {i}", t=now, ttl=3600, doc_id=f"d{i}"), v)
    assert len(cs) == 40 and cs.alive.sum() == 40                       # grew past cap 4
    top = cs.search(base, k=3, now=now)
    assert top[0][0].doc_id == "d0" and top[0][2]["dense"] > 0.99
    assert cs.search(base, k=3, now=now + 7200) == []                   # expired: never served
    cs.add(Cell(id=-1, kind="passage", title="dup", payload="x", t=now + 5, ttl=3600, doc_id="d0"), base)
    assert len(cs) == 40                                                # same doc: refreshed, not duplicated
    for i in range(40, 70):
        cs.add(Cell(id=-1, kind="passage", title=f"T{i}", payload="y", t=now, ttl=3600, doc_id=f"d{i}"), vec(500 + i))
    assert len(cs) <= 50 and cs.evicted > 0


def test_cells_persist_roundtrip(tmp_path):
    cs = CellStore()
    now = time.time()
    cs.add(Cell(id=-1, kind="value", title="founded_by(nvidia)", payload="Jensen Huang", key="founded_by(nvidia)",
                t=now, ttl=3600, doc_id="value:founded_by(nvidia)"), vec(2), vec(3))
    cs.save(tmp_path)
    cs2 = CellStore()
    assert cs2.load(tmp_path) == 1
    c = next(iter(cs2.cells.values()))
    assert c.key == "founded_by(nvidia)" and cs2.has_q[c.id]
    assert cs2.question_nn(vec(3), 1)[0][1] > 0.99


def test_key_sentences_pick_the_relevant_ones():
    t = "Nvidia is a chip company. It was founded in 1993 by Jensen Huang. It is in Santa Clara."
    assert "Jensen Huang" in key_sentences(t, ["founded", "1993"], n=1)
    assert key_sentences(t, ["nvidia", "founded"], n=2).count(".") == 2


# ------------------------------------------------------------------------------------------------- graph
def test_graph_aliases_typed_edges_hops_and_ttl():
    g = Graph()
    n = g.node("Nvidia Corporation", qid="Q182477")
    assert g.resolve("nvidia") == n and g.resolve("NVIDIA's") == n            # prefix + possessive
    jen = g.node("Jensen Huang", kind="person")
    g.add_edge(n, "founded_by", "Jensen Huang", t=100.0, ttl=50.0, value_node=jen)
    g.add_edge(jen, "born_place", "Tainan", t=100.0, ttl=0.0)
    assert g.lookup("nvidia", "founded_by", now=120.0)[1][0]["value"] == "Jensen Huang"
    assert g.lookup("nvidia", "founded_by", now=200.0)[1] == []                # past its TTL
    assert g.lookup("@founded_by(nvidia)", "born_place", now=120.0)[1][0]["value"] == "Tainan"
    assert jen in dict(g.neighbours(n))                                       # value edge = co-occurrence


def test_activation_decays_and_spreads():
    g = Graph(session_half_s=60)
    a, b = g.node("Mozart"), g.node("Salzburg")
    g.link(a, b, 3.0)
    g.activate("s:1", a, 1.0, now=0.0)
    act = g.activation(["s:1"], now=0.0)
    assert act[a] == pytest.approx(1.0) and 0 < act[b] < act[a]
    assert g.activation(["s:1"], now=60.0)[a] == pytest.approx(0.5, rel=1e-3)
    assert g.most_active(["s:1"], now=1.0) == a


# ------------------------------------------------------------------------------------------------- core
def _q(text, slots, qv, intent="discourse.intent.qa_factoid", live=False):
    from server.keystone.ahash import arc_of
    k = key_of(text.split(), slots, intent)
    return Q(text=text, words=text.split(), slots=tuple(slots), intent=intent, live=live, qvec=qv, key=k,
             arc=arc_of(text.split(), intent, slots, (), live, k), user="u", session="s")


def test_write_then_typed_key_hit_for_a_paraphrase():
    ks = Keystone(Flags(hash=False))
    org = [("organization", "nvidia")]
    fact = Doc("wikidata:Q182477:P112", "Nvidia · Wikidata", "Nvidia (company): founded by: Jensen Huang.", "Wikidata",
               {"tier": "facts", "fact": {"prop": "founder", "values": ["Jensen Huang"], "item": "Nvidia"}})
    ks.write(_q("who founded the company nvidia", org, vec(10)), [fact], {fact.id: vec(11)}, tier="facts")
    cands, feats = ks.lookup(_q("who started nvidia", org, None))
    best = ks.heuristic(cands, feats)
    assert best is not None and best.path == "key" and "Jensen Huang" in best.cells[0][0].payload
    cands, _ = ks.lookup(_q("when was nvidia founded", org, None))
    assert not any(c.path == "key" for c in cands)                        # other relation: no hit


def test_live_questions_never_served_from_old_passages():
    ks = Keystone(Flags(hash=False))
    d = Doc("w1", "Chicago weather", "Rain in Chicago today.")
    v = vec(20)
    ks.write(_q("weather in chicago", [("place", "chicago")], v), [d], {"w1": v}, tier="web")
    cands, _ = ks.lookup(_q("what's the weather in chicago today", [("place", "chicago")], v,
                            "discourse.intent.weather_query", live=True))
    assert not any(c.path == "dense" for c in cands)


def test_plausible_prefilter_skips_the_embedding_for_unrelated_questions():
    ks = Keystone(Flags(hash=False))
    d = Doc("w1", "Mount Everest", "Mount Everest is Earth's highest mountain above sea level.")
    ks.write(_q("how tall is mount everest", [("place", "mount everest")], vec(30)), [d], {"w1": vec(31)})
    assert ks.plausible(_q("everest height in meters", [("place", "mount everest")], None))
    assert not ks.plausible(_q("best ramen in seattle", [("place", "seattle")], None))


# ------------------------------------------------------------------------------------------------- VOI
def test_voi_fit_is_calibrated_on_separable_data_and_respects_theta():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < x).astype(float)            # p(correct) = x
    X = np.stack([x, np.ones_like(x)], 1)
    w = voi.fit(X, y)
    p = 1 / (1 + np.exp(-(X @ w)))
    for lo in (0.2, 0.5, 0.8):
        m = (p >= lo) & (p < lo + 0.1)
        assert abs(y[m].mean() - p[m].mean()) < 0.08


# ------------------------------------------------------------------------------------------------- live hooks
@pytest.fixture
def index_on(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "INDEX", True)
    monkeypatch.setattr(config, "INDEX_PERSIST", False)
    monkeypatch.setattr(config, "ROUTER", True)
    monkeypatch.setattr(config, "ROUTER_LEADS", False)
    monkeypatch.setattr(config, "KEYSTONE_DIR", tmp_path)
    hook.reset(Keystone(Flags(hash=False)))
    yield
    hook.reset()


def test_router_serves_the_local_tier_free_and_labelled(index_on, monkeypatch):
    f = F("who wrote one hundred years of solitude", slots=[("work_product", "one hundred years of solitude")])
    d = Doc("https://en.wikipedia.org/wiki/One_Hundred_Years_of_Solitude", "One Hundred Years of Solitude",
            "One Hundred Years of Solitude is a 1967 novel by Colombian author Gabriel García Márquez.")
    hook.ingest(f, "who wrote one hundred years of solitude", [d], {d.id: vec(40)}, {"tier": "web"})
    assert len(hook.store().cells) == 2                          # the passage + the question's value cell
    monkeypatch.setattr(router, "plan", lambda fr: router.Plan("web", "", 0.2, {}, "test"))
    called = []

    async def web():
        called.append(1)
        r = lanes.WebResult([Doc("w9", "Web", "web")])
        r.backend = "wikipedia"
        return r
    m: dict = {}
    f2 = F("who is the author of one hundred years of solitude",
           slots=[("work_product", "one hundred years of solitude")])
    res = asyncio.run(router.route("author one hundred years solitude", f2, final=True, meter=m, web_fn=web))
    assert not called and m["tier"] == "index" and m["paid"] == 0 and m["index_path"] == "key"
    assert res[0].meta["tier"] == "index" and res[0].meta["snapshot"].startswith("from your Onset index · as of")
    assert res[0].source.startswith("Onset index")


def test_router_pays_when_the_index_has_nothing(index_on, monkeypatch):
    monkeypatch.setattr(router, "plan", lambda fr: router.Plan("web", "", 0.2, {}, "test"))
    called = []

    async def web():
        called.append(1)
        r = lanes.WebResult([Doc("w9", "Web", "web")])
        r.backend = "wikipedia"
        return r
    m: dict = {}
    asyncio.run(router.route("q", F("who wrote hamlet", slots=[("work_product", "hamlet")]), final=True, meter=m,
                             web_fn=web))
    assert called and m["tier"] == "web" and m["paid"] == 1


def test_hooks_are_noops_when_off_and_never_raise(monkeypatch):
    monkeypatch.setattr(config, "INDEX", False)
    assert asyncio.run(hook.local_first(F("who founded nvidia"), "q")) is None
    hook.ingest(F("x"), "x", [Doc("a", "A", "a")], {})          # off: nothing
    monkeypatch.setattr(config, "INDEX", True)
    monkeypatch.setattr(config, "INDEX_PERSIST", False)
    hook.reset()
    assert asyncio.run(hook.local_first(F("who founded nvidia"), "q")) is None     # empty store
    hook.ingest(None, "x", [], {})
    hook.ingest(F("x"), "x", [object()], {})                    # garbage in: counted as an error, no exception
    assert hook.STATS["errors"] >= 1
    hook.reset()


def test_hook_persistence_roundtrip_remaps_cell_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "INDEX", True)
    monkeypatch.setattr(config, "INDEX_PERSIST", True)
    monkeypatch.setattr(config, "KEYSTONE_DIR", tmp_path)
    ks = hook.reset(Keystone(Flags(hash=False)))
    for i in range(3):                                       # some churn so ids are not 0..n in order
        ks.cells.add(Cell(id=-1, kind="passage", title="x", payload="x", t=time.time(), ttl=1.0, doc_id=f"x{i}"),
                      vec(70 + i))
    ks.cells.remove(0)
    f = F("who wrote hamlet", slots=[("work_product", "hamlet")])
    d = Doc("h", "Hamlet", "Hamlet is a tragedy written by William Shakespeare.")
    hook.ingest(f, "who wrote hamlet", [d], {"h": vec(80)}, {"tier": "web"})
    hook.save(force=True)
    hook._ks = None
    ks2 = hook.store()
    cands, feats = ks2.lookup(hook.q_of(F("who is the author of hamlet", slots=[("work_product", "hamlet")]), "q"))
    best = ks2.heuristic(cands, feats)
    assert best is not None and best.path == "key" and "Shakespeare" in " ".join(c.payload for c, _ in best.cells)
    hook.reset()


def test_memory_and_answers_fold_into_the_graph(index_on):
    from server.memory import Memory
    mem = Memory()
    f = F("who founded nvidia", slots=[("organization", "nvidia")])
    fact = Doc("wikidata:Q182477:P112", "Nvidia · Wikidata", "Nvidia (company): founded by: Jensen Huang.", "Wikidata",
               {"tier": "facts", "fact": {"prop": "founder", "values": ["Jensen Huang"], "item": "Nvidia"}})
    mem.observe(f, "Nvidia was founded by Jensen Huang.", [{"type": "person", "text": "Jensen Huang"}], [fact])
    hook.absorb_memory(mem)
    g = hook.store().g
    n = g.resolve("nvidia")
    assert n == "Q182477" and g.lookup("nvidia", "founded_by")[1][0]["value"] == "Jensen Huang"
    assert g.most_active([f"s:{hook.SCOPE['session']}"]) is not None
    hook.ingest(f, "who founded nvidia", [fact], {fact.id: vec(90)}, {"tier": "facts"})
    hook.note_answer(f, "who founded nvidia", "Jensen Huang co-founded Nvidia in 1993.")
    vals = g.lookup("nvidia", "founded_by")[1]
    assert any("1993" in v["value"] for v in vals)
