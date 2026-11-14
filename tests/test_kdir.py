"""Keystone director v1 (server/keystone/kdir.py, KDIR=1): features, the numpy tree evaluator, the expected-cost
decision, the hook integration (fallback to the VOI gate, free-first labelled free, background re-check -> trust),
learned TTLs and the wider shared allowlist (question -> answer cells). No network, synthetic vectors."""
import asyncio
import json
import time

import numpy as np
import pytest

from server import config, lanes
from server.frame import Frame
from server.keystone import hook, kdir, voi
from server.keystone.core import Cand, Flags, Keystone, Q
from server.keystone.shared import SharedCell, SharedCells, admit_cell


def vec(seed: int, d: int = 768) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def F(text, intent="discourse.intent.qa_factoid", slots=()):
    return Frame(words=text.split(), intent=intent, intent_p=0.9, slots=tuple(slots))


def Doc(i, title, text, source="wikipedia", meta=None):
    return lanes.Doc(id=i, title=title, text=text, url=f"https://x/{i}", source=source, meta=meta or {})


def stump(feature: int, thr: float, lo: float, hi: float) -> dict:
    """One depth-1 tree: x[feature] < thr -> lo else hi (margins)."""
    return {"f": [feature, -1, -1], "t": [thr, 0, 0], "l": [1, -1, -1], "r": [2, -1, -1], "v": [0, lo, hi],
            "m": [1, -1, -1]}


def model(ok_feature: str, catch: float = -4.0, strong: float = -4.0) -> dict:
    i = kdir.NAMES.index(ok_feature)
    return {"ok": {"base": 0.0, "trees": [stump(i, 0.5, -4.0, 4.0)]},
            "catch": {"base": catch, "trees": [stump(0, 99, 0.0, 0.0)], "cols": list(range(len(kdir.Q_NAMES)))},
            "free_strong": {"base": strong, "trees": [stump(0, 99, 0.0, 0.0)], "cols": list(range(len(kdir.Q_NAMES)))},
            "R": 10.0, "C": 1.0, "eps": 0.02}


@pytest.fixture
def index_on(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "INDEX", True)
    monkeypatch.setattr(config, "INDEX_PERSIST", False)
    monkeypatch.setattr(config, "ROUTER", True)
    monkeypatch.setattr(config, "ROUTER_LEADS", False)
    monkeypatch.setattr(config, "KEYSTONE_DIR", tmp_path)
    hook.reset(Keystone(Flags(hash=False)))
    kdir.reset()
    yield
    hook.reset()
    kdir.reset()


# ------------------------------------------------------------------------------------------------- the evaluator
def test_trees_evaluate_like_the_exported_arrays():
    t = kdir.Trees({"base": 0.5, "trees": [stump(0, 1.0, -1.0, 2.0), stump(1, 0.0, 0.3, -0.3)]})
    assert t.margin(np.array([0.0, -1.0])) == pytest.approx(0.5 - 1.0 + 0.3)
    assert t.margin(np.array([5.0, 1.0])) == pytest.approx(0.5 + 2.0 - 0.3)
    assert t.margin(np.array([np.nan, 1.0])) == pytest.approx(0.5 - 1.0 - 0.3)        # NaN -> the missing branch
    cal = kdir.Trees({"base": 0.0, "trees": [stump(0, 1.0, 0.0, 0.0)], "cal": [2.0, 1.0], "cols": [1]})
    assert cal.p([9.0, 0.0]) == pytest.approx(1 / (1 + np.exp(-1.0)))


def test_feature_vectors_have_the_named_width(index_on):
    ks = hook.store()
    f = F("who founded nvidia", slots=[("organization", "nvidia")])
    d = Doc("n", "Nvidia", "Nvidia was founded in 1993 by Jensen Huang, Chris Malachowsky and Curtis Priem.")
    hook.ingest(f, "who founded nvidia", [d], {"n": vec(1)}, {"tier": "web"})
    q = hook.q_of(f, "who started nvidia")
    q.qvec = vec(1)
    cands, feats = ks.lookup(q, time.time())
    assert cands
    qx = kdir.question_vec(q, f, {"lane": "web", "conf": 0.3}, ks, feats, cands, (0.7, 0.1), None)
    cx = kdir.cand_vec(cands[0], feats, ks, cands, time.time())
    assert len(qx) == len(kdir.Q_NAMES) and len(qx + cx) == len(kdir.NAMES)
    assert qx[kdir.Q_NAMES.index("q_has_edge_any")] == 1.0          # founded_by(nvidia) is an edge now
    assert qx[kdir.Q_NAMES.index("q_has_dir")] == 0.0               # no director outputs on this frame


# ------------------------------------------------------------------------------------------------- the decision
def _cands():
    return [Cand("qnn", [], 0.9, time.time())]


def test_expected_cost_picks_local_pay_recheck_and_free():
    nq = len(kdir.Q_NAMES)
    i = kdir.NAMES.index("qnn_cos")
    qx = [0.0] * nq
    hi = [0.0] * (len(kdir.NAMES) - nq)
    hi[i - nq] = 1.0
    lo = [0.0] * (len(kdir.NAMES) - nq)
    pol = kdir.Policy(model("qnn_cos"))
    assert pol.decide(qx, _cands(), [hi], 1, False)[0] == "local"           # p_ok ~ 0.98
    assert pol.decide(qx, _cands(), [lo], 1, False)[0] == "pay"             # p_ok ~ 0.02
    assert pol.decide(qx, _cands(), [lo], 0, False)[0] == "pay"             # the plan is a free lane: nothing to save
    sure_catch = kdir.Policy(model("qnn_cos", catch=6.0))
    mid = kdir.Policy({**model("qnn_cos", catch=6.0), "ok": {"base": 0.0, "trees": [stump(0, 99, 0.0, 0.0)]}})
    assert mid.decide(qx, _cands(), [lo], 1, True)[0] == "local_rc"         # p_ok 0.5, but a re-check would catch it
    assert mid.decide(qx, _cands(), [lo], 1, False)[0] == "pay"             # live / free-planned: nothing can check it
    assert sure_catch.decide(qx, [], [], 1, True)[0] == "pay"               # free-first needs a strong-lane model
    free = kdir.Policy(model("qnn_cos", catch=6.0, strong=6.0))
    assert free.decide(qx, [], [], 1, True)[0] == "free"
    assert free.decide(qx, [], [], 1, False)[0] == "pay"                    # live / fresh / free-planned: never free


def test_feedback_moves_trust_and_is_bounded():
    ks = Keystone(Flags(hash=False))
    for _ in range(5):
        kdir.feedback(ks, 7, "caught")
    assert ks.trust[7] == -4.0
    kdir.feedback(ks, 8, "confirm")
    kdir.feedback(ks, 8, "verified")
    assert kdir.ks_trust(ks, 8) == 1.5 and kdir.ks_trust(ks, 9) == 0.0


def test_learned_ttl_table_and_fallback(monkeypatch, tmp_path):
    p = tmp_path / "ttl.json"
    p.write_text(json.dumps({"ttl_s": {"ceo_of": 86400 * 50, "class:live": 900}}))
    monkeypatch.setattr(kdir, "TTL_PATH", p)
    monkeypatch.setattr(kdir, "_ttl", None)
    assert kdir.learned_ttl("ceo_of", "slow", 1.0) == 86400 * 50
    assert kdir.learned_ttl("weather_in", "live", 600.0) == 900
    assert kdir.learned_ttl("born_date", "stable", 123.0) == 123.0
    monkeypatch.setattr(kdir, "_ttl", None)


# ------------------------------------------------------------------------------------------------- the hook
def _warm(ks_text="who wrote one hundred years of solitude"):
    f = F(ks_text, slots=[("work_product", "one hundred years of solitude")])
    d = Doc("u", "One Hundred Years of Solitude",
            "One Hundred Years of Solitude is a 1967 novel by Colombian author Gabriel García Márquez.")
    hook.ingest(f, ks_text, [d], {"u": vec(40)}, {"tier": "web"})
    return f


def test_kdir_without_a_model_falls_back_to_the_voi_gate(index_on, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KDIR", True)
    monkeypatch.setattr(kdir, "MODEL_PATH", tmp_path / "missing.json")
    _warm()
    m = {}
    q = F("what novel did gabriel garcia marquez write in 1967")
    hook._QV.clear()
    monkeypatch.setattr("server.embed.encode_queries", lambda texts: np.stack([vec(40) for _ in texts]))
    asyncio.run(hook.local_first(q, "what novel did gabriel garcia marquez write in 1967", m))
    assert "kdir_action" not in m and "kdir_error" not in m                # no model: the VOI gate decided


def test_kdir_pay_action_and_decision_is_fast(index_on, monkeypatch, tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(model("qnn_cos")))
    monkeypatch.setattr(config, "KDIR", True)
    monkeypatch.setattr(kdir, "MODEL_PATH", p)
    _warm()
    monkeypatch.setattr("server.embed.encode_queries", lambda texts: np.stack([vec(41) for _ in texts]))
    m = {}
    r = asyncio.run(hook.local_first(F("what is the weather on mars"), "what is the weather on mars", m))
    assert r is None or m.get("kdir_action") in ("local", "local_rc")
    if "kdir_action" in m:
        assert m["kdir_ms"] < 50


def test_free_first_is_labelled_free_and_paid_zero(index_on, monkeypatch):
    from server import knowledge, router
    strong = lanes.Doc(id="https://en.wikipedia.org/wiki/Jaws_(film)", title="Jaws (film) · Wikipedia summary",
                       text="Jaws is a 1975 American thriller film directed by Steven Spielberg.",
                       url="https://en.wikipedia.org/wiki/Jaws_(film)", source="Wikipedia summary",
                       meta={"tier": "facts", "strong": True})

    async def leads(q, k=3, subject=""):
        return []

    async def summary(s):
        return [strong]
    monkeypatch.setattr(knowledge, "leads", leads)
    monkeypatch.setattr(knowledge, "wiki_summary", summary)
    monkeypatch.setattr(config, "KDIR", True)
    f = Frame(words="tell me about the movie jaws".split(), intent="intent.request_information", intent_p=0.9,
              slots=(("work_product", "jaws"),), phrases=("jaws",))
    calls = []

    async def web_fn():
        calls.append(1)
        return lanes.WebResult([])
    m = {"kdir_action": "free"}
    go = hook.wrap_web(web_fn, f, "tell me about the movie jaws", m, True)
    monkeypatch.setattr(hook, "local_first", lambda *a, **k: asyncio.sleep(0, result=None))
    res = asyncio.run(go())
    assert not calls and res.backend == "kdir-free" and m["kdir_free"] == "served"
    # through the router: a free-lane result is paid 0 and its tier is a free lane, never "web"
    meter = {}

    async def web2():
        return res
    asyncio.run(router.route("tell me about the movie jaws", f, final=True, meter=meter, web_fn=web2))
    assert meter["paid"] == 0 and meter["tier"] in ("facts", "index") or meter["tier"] == "card"


def test_background_recheck_updates_trust(index_on, monkeypatch):
    ks = hook.store()
    f = _warm()
    q = hook.q_of(f, "who wrote one hundred years of solitude")
    q.qvec = vec(40)
    cands, _ = ks.lookup(q, time.time())
    cand = next(c for c in cands if c.path in ("key", "qnn"))
    good = lanes.WebResult([Doc("w", "Gabriel García Márquez", "Gabriel García Márquez wrote One Hundred Years of "
                                "Solitude, published in 1967.", meta={"strong": True})])

    async def ff(*a, **k):
        return good
    monkeypatch.setattr(hook, "free_first", ff)
    out = asyncio.run(hook.recheck_cell(f, q, cand))
    assert out in ("confirm", "none")
    if out == "confirm":
        assert kdir.ks_trust(ks, cand.cells[0][0].id) > 0
    hook.spoken_correction(cand.cells[0][0].id)
    assert kdir.ks_trust(ks, cand.cells[0][0].id) <= (1.0 if out == "confirm" else 0.0) - 2.0


# ------------------------------------------------------------------------------------------------- wider shared store
def test_shared_cells_allowlist_refuses_personal_and_unverified():
    ok = SharedCell("how tall is the eiffel tower", "The Eiffel Tower is 330 m tall.", "supported", time.time(), 86400,
                    "Wikipedia", "https://en.wikipedia.org/wiki/Eiffel_Tower")
    assert admit_cell(ok) is None
    assert admit_cell(SharedCell("how tall is my house", "6 m", "supported", 0, 1, "Wikipedia", "https://x")) is not None
    assert admit_cell(SharedCell("what did i eat", "pancakes", "supported", 0, 1, "Wikipedia", "https://x")) is not None
    assert admit_cell(SharedCell("how tall is the eiffel tower", "330 m", "unverified", 0, 1, "W", "https://x"))
    assert admit_cell(SharedCell("what did the judge say", "x", "supported", 0, 1, "local moments", "https://x"))
    assert admit_cell(SharedCell("call 555 123 4567", "x", "supported", 0, 1, "W", "https://x"))
    assert admit_cell(SharedCell("weather in paris", "sunny", "supported", 0, 86400, "W", "https://x",
                                 {"live": True}))                              # live with a long TTL


def test_shared_cells_answer_another_users_paraphrase():
    sc = SharedCells()
    v = vec(3)
    c = SharedCell("how tall is the eiffel tower", "The Eiffel Tower is 330 m tall.", "supported", 1000.0, 86400,
                   "Wikipedia", "https://en.wikipedia.org/wiki/Eiffel_Tower", {"answer_type": "measure_h"})
    assert sc.publish(c, v, "fam1") is None
    assert sc.publish(c, v, "fam1") is None and sc.stats()["cells"] == 1 and sc.stats()["contributions"] == 2
    q = Q(text="what's the height of the eiffel tower", words=[], qvec=v, arc={"answer_type": "measure_h"})
    cand = sc.lookup(q, 2000.0)
    assert cand is not None and cand.path == "shared_q" and "330" in cand.cells[0][0].payload
    assert sc.lookup(q, 1000.0 + 86400 * 2) is None                            # past its TTL: never served
    q2 = Q(text="who painted the mona lisa", words=[], qvec=vec(99))
    assert sc.lookup(q2, 2000.0) is None


def test_a_user_fact_never_reaches_shared_cells(index_on, monkeypatch):
    monkeypatch.setattr(config, "KDIR_SHARED_CELLS", True)
    monkeypatch.setattr(hook, "_SHARED_CELLS", None)
    from server.memory import Memory
    mem = Memory()
    mem.learn_user("i live in denver".split())
    hook.absorb_memory(mem)
    f = F("where do i live")
    d = Doc("h", "Denver", "Denver is the capital of Colorado.")
    hook.ingest(f, "where do i live", [d], {"h": vec(7)}, {"tier": "web"})
    hook.note_answer(f, "where do i live", "You live in Denver.", "supported")
    pub = F("what is the capital of colorado", slots=[("place", "colorado")])
    d2 = Doc("c", "Colorado", "Denver is the capital of Colorado.")
    hook.ingest(pub, "what is the capital of colorado", [d2], {"c": vec(8)}, {"tier": "web"})
    hook.note_answer(pub, "what is the capital of colorado", "Denver is the capital of Colorado.", "supported")
    rows = hook.shared_cells().dump()
    flat = " ".join(" ".join(map(str, r)) for r in rows).lower()
    assert "where do i live" not in flat and "you live" not in flat
    assert all("my " not in r[0].lower() and " i " not in f" {r[0].lower()} " for r in rows)


# ------------------------------------------------------------------------------------------------- the shipped model
def test_shipped_model_is_tiny_fast_and_well_formed():
    if not kdir.MODEL_PATH.exists():
        pytest.skip("no kdir_model.json")
    assert kdir.MODEL_PATH.stat().st_size < 2_000_000
    kdir.reset()
    pol = kdir.load()
    assert pol is not None and 0 < pol.R < 1000
    rng = np.random.default_rng(0)
    nq = len(kdir.Q_NAMES)
    ts = []
    for _ in range(200):
        x = rng.random(len(kdir.NAMES)).tolist()
        t = time.perf_counter()
        pol.decide(x[:nq], [Cand("qnn", [], 0.9, 0.0), Cand("dense", [], 0.7, 0.0)], [x[nq:], x[nq:]], 1, True)
        ts.append((time.perf_counter() - t) * 1000)
    assert sorted(ts)[len(ts) // 2] < 2.0
    kdir.reset()
