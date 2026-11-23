"""The on-device answer backend (server/answer.py backend "local", server/answer_lm.py).

A tiny fake engine stands in for the 4-bit Qwen2.5-0.5B-Instruct ONNX graph, so the real worker thread, incremental
detokenisation, cancel, fallbacks and usage accounting run without the model. The real model is exercised by the
opt-in smoke test at the end (ANSWER_LM_SMOKE=1 with the model installed in MODELS_DIR/answer-lm).

Run: .venv/bin/python -m pytest -q tests/test_answer_local.py
"""
import asyncio
import os
import threading
import time

import pytest

from server import answer, answer_lm, config, lanes, meter, notices, router


DOCS = [lanes.Doc(id="1", title="Australia", text="Australia is a country in Oceania. Its capital is Canberra."),
        lanes.Doc(id="2", title="Canberra", text="Canberra is the capital city of Australia, founded in 1913.")]


class FakeEngine:
    """Stands in for answer_lm.OnnxEngine: word-level ids (WORDS[i]), generate() streams `script` (first token after
    `ttft`, then `gap` seconds apart), checking stop() before every token like the real loop. Records how many tokens
    it produced and on which thread."""

    WORDS = ["<eos>", "Canberra", "is", "the", "capital", "of", "Australia.", "[2]", "x"]

    def __init__(self, script, ttft=0.0, gap=0.0, fail=None):
        self.script, self.ttft, self.gap, self.fail = script, ttft, gap, fail
        self.produced = 0
        self.threads = set()
        self.calls = 0

    def prompt_ids(self, system, user):
        return [1] * len((system + " " + user).split())

    def decode(self, ids):
        return " ".join(self.WORDS[i] for i in ids if i)

    def generate(self, ids, max_new, on_token, stop):
        self.calls += 1
        self.threads.add(threading.current_thread().name)
        if self.fail:
            raise RuntimeError(self.fail)
        time.sleep(self.ttft)
        for tid in self.script[:max_new]:
            if stop():
                return
            if tid == 0:
                return
            on_token(tid)
            self.produced += 1
            time.sleep(self.gap)


@pytest.fixture
def fake_lm(monkeypatch):
    """Install a fake loaded engine; returns the installer."""
    monkeypatch.setattr(router, "pick_card", lambda q, d: None)

    def install(engine):
        monkeypatch.setattr(answer_lm, "_engine", engine)
        return engine
    yield install
    answer_lm._engine = None


async def _drain(st):
    return [t async for t in st.follow()]


# ----------------------------------------------------------------------------------------------- backend choice
def test_backend_choice(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for b, want in (("local", "local"), ("extractive", "extractive"), ("nonsense", "local"), ("claude", "local")):
        monkeypatch.setattr(config, "ANSWER_BACKEND", b)
        assert answer.backend() == want, b
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")          # claude only when chosen AND a key is set
    assert answer.backend() == "claude" and answer.model_name() == config.ANSWER_MODEL
    monkeypatch.setattr(config, "ANSWER_BACKEND", "local")
    assert answer.backend() == "local" and answer.model_name().startswith("local:")


def test_default_backend_needs_no_anthropic_package():
    import sys
    # importing the answer module must not import anthropic (the app can ship without it)
    assert "anthropic" not in sys.modules or answer.backend() != "claude"
    src = open(answer.__file__).read()
    assert "\nfrom anthropic" not in src and "\nimport anthropic" not in src


# ----------------------------------------------------------------------------------------------- streaming
def test_local_stream_answers_off_the_event_loop_and_cites(fake_lm):
    m = fake_lm(FakeEngine([1, 2, 3, 4, 5, 6, 0]))          # "Canberra is the capital of Australia." (no citation)

    async def go():
        st = answer.Stream("what is the capital of australia", DOCS)
        toks = await asyncio.wait_for(_drain(st), 5)
        return st, toks
    st, toks = asyncio.run(go())
    assert st.error is None and st.fallback is None and st.winner == 0 and st.t_first is not None
    assert st.text().startswith("Canberra is the capital of Australia.")
    assert st.text().rstrip().endswith("[1]") or st.text().rstrip().endswith("[2]")   # citation added
    assert "".join(toks) == st.text()
    assert m.threads == {"answer-lm"}                       # generated on the worker thread, not the event loop
    u = st.usage()
    assert u["requests"] == 1 and u["in"] > 0 and u["out"] == 6 and u["win_out"] == 6 and not u["est"]
    assert answer.health()["status"] == "live" and answer.health()["backend"] == "local"


def test_model_citation_is_kept(fake_lm):
    fake_lm(FakeEngine([1, 2, 3, 4, 5, 6, 7, 0]))            # "... Australia. [2]"

    async def go():
        st = answer.Stream("capital of australia", DOCS)
        await asyncio.wait_for(_drain(st), 5)
        return st
    st = asyncio.run(go())
    assert st.text().count("[") == 1 and st.text().rstrip().endswith("[2]")


def test_cancel_stops_generation(fake_lm):
    m = fake_lm(FakeEngine([8] * 200, gap=0.01))             # would take 2 s

    async def go():
        st = answer.Stream("q", DOCS)
        while not st.tokens:
            await asyncio.sleep(0.005)
        st.cancel()
        t = time.perf_counter()
        await asyncio.wait_for(_drain(st), 1)
        await asyncio.sleep(0.1)                            # let the worker see the stop flag
        return st, time.perf_counter() - t
    st, took = asyncio.run(go())
    assert st.done and st.error == "cancelled"
    assert m.produced < 40, m.produced                      # stopped within a token or two, not 200
    assert not answer_lm._lock.locked()                     # the model is free for the next answer


def test_task_cancel_also_stops_generation(fake_lm):
    m = fake_lm(FakeEngine([8] * 200, gap=0.01))

    async def go():
        st = answer.Stream("q", DOCS)
        while not st.tokens:
            await asyncio.sleep(0.005)
        st.task.cancel()
        await asyncio.sleep(0.15)
        return st
    st = asyncio.run(go())
    assert st.done and st.error == "cancelled" and m.produced < 40


def test_second_stream_waits_for_the_model_then_answers(fake_lm):
    fake_lm(FakeEngine([1, 2, 3, 4, 5, 6, 0], gap=0.01))

    async def go():
        a = answer.Stream("capital of australia", DOCS, tag="draft")
        b = answer.Stream("capital of australia", DOCS, tag="final")
        await asyncio.wait_for(asyncio.gather(_drain(a), _drain(b)), 5)
        return a, b
    a, b = asyncio.run(go())
    assert a.text() and a.text() == b.text() and not a.error and not b.error


# ----------------------------------------------------------------------------------------------- fallbacks
def test_missing_model_falls_back_to_extractive_and_opens_breaker(monkeypatch):
    monkeypatch.setattr(router, "pick_card", lambda q, d: None)
    assert not answer_lm.installed()                        # conftest points ANSWER_LM_DIR at an empty tmp folder

    async def go():
        st = answer.Stream("what is the capital of australia", DOCS)
        await asyncio.wait_for(_drain(st), 5)
        return st
    st = asyncio.run(go())
    assert st.fallback == "extractive" and "not installed" in st.fallback_reason and "Canberra" in st.text()
    h = answer.health()
    assert h["status"] == "down" and h["backend"] == "local"
    n = notices.build({"status": "live", "cache_entries": 0}, h)
    assert n[0]["id"] == "answer-model" and "fetch answer-lm" in n[0]["hint"]
    st2 = asyncio.run(go())                                 # breaker open: straight to extractive
    assert st2.fallback == "extractive" and "answer model down" in st2.fallback_reason
    assert st.usage()["requests"] == 1 and st.usage()["in"] == 0      # nothing was generated: nothing counted


def test_generation_error_falls_back(fake_lm):
    fake_lm(FakeEngine([], fail="MPS backend out of memory"))

    async def go():
        st = answer.Stream("what is the capital of australia", DOCS)
        await asyncio.wait_for(_drain(st), 5)
        return st
    st = asyncio.run(go())
    assert st.fallback == "extractive" and "out of memory" in st.fallback_reason and st.text()


def test_no_first_token_by_deadline_falls_back_without_breaker(fake_lm, monkeypatch):
    m = fake_lm(FakeEngine([1, 2, 0], ttft=0.6))
    monkeypatch.setattr(config, "ANSWER_DEADLINE_MS", 150)

    async def go():
        st = answer.Stream("what is the capital of australia", DOCS)
        await asyncio.wait_for(_drain(st), 5)
        await asyncio.sleep(0.6)
        return st
    st = asyncio.run(go())
    assert st.fallback == "extractive" and "no token" in st.fallback_reason
    assert answer.health()["status"] != "down"              # slow is not broken
    assert m.produced <= 1                                   # the late generation was stopped


def test_extractive_backend_makes_no_model_call(fake_lm, monkeypatch):
    m = fake_lm(FakeEngine([1, 0]))
    monkeypatch.setattr(config, "ANSWER_BACKEND", "extractive")

    async def go():
        st = answer.Stream("what is the capital of australia", DOCS)
        await asyncio.wait_for(_drain(st), 5)
        return st
    st = asyncio.run(go())
    assert m.calls == 0 and st.fallback == "extractive" and "Canberra" in st.text()


# ----------------------------------------------------------------------------------------------- warm / probe / meter
def test_warm_probes_the_local_model(fake_lm):
    m = fake_lm(FakeEngine([1, 0]))
    h = asyncio.run(answer.probe())
    assert m.calls == 1 and h["status"] == "live" and h["backend"] == "local" and h["probe_ms"] >= 0


def test_local_answers_cost_nothing():
    p = meter.prices()
    assert p["model"].startswith("local:") and p["in_usd_mtok"] == 0 and p["verified"]
    c = meter.cost(0, {"ans_in": 600, "ans_out": 20, "waste_in": 600, "waste_out": 5})
    assert c["usd_llm"] == 0 and c["usd"] == 0


def test_local_prompt_has_no_example_to_parrot():
    # a 0.5B model copied SYSTEM's recording example into a third of its answers; SYSTEM_LOCAL has none
    assert "Hunter" not in answer.SYSTEM_LOCAL and '"' not in answer.SYSTEM_LOCAL.replace('("X, sorry, I mean Y")', "")
    assert len(answer.SYSTEM_LOCAL.split()) < 140


def test_cite_helper():
    assert answer._cite("Canberra is the capital.", DOCS) in (" [1]", " [2]")
    assert answer._cite("Canberra [2].", DOCS) == ""
    assert answer._cite("", DOCS) == "" and answer._cite("Zzz qqq.", DOCS) == ""


# ----------------------------------------------------------------------------------------------- the real model
@pytest.mark.skipif(os.environ.get("ANSWER_LM_SMOKE") != "1", reason="opt-in: ANSWER_LM_SMOKE=1 (loads the real model)")
def test_real_model_smoke(monkeypatch):
    from server import modelpaths
    d = modelpaths.resolve("answer-lm", "ANSWER_LM_DIR")
    monkeypatch.setattr(config, "ANSWER_LM_DIR", d)
    if not answer_lm.installed():
        pytest.skip(f"answer-lm not installed in {d}")
    monkeypatch.setattr(router, "pick_card", lambda q, d: None)
    answer_lm.unload()

    async def go():
        h = await answer.probe(timeout=120)
        st = answer.Stream("what is the capital of australia", DOCS)
        await asyncio.wait_for(_drain(st), 30)
        return h, st
    try:
        h, st = asyncio.run(go())
    finally:
        answer_lm.unload()
    assert h["status"] == "live", h
    assert st.error is None and st.fallback is None and "canberra" in st.text().lower(), st.text()
    assert "[" in st.text() and len(st.text().split()) < 40
    assert (st.t_first - st.t_start) < 3.0


@pytest.mark.skipif(os.environ.get("ANSWER_LM_SMOKE") != "1", reason="opt-in: ANSWER_LM_SMOKE=1 (needs the model + transformers)")
def test_chat_template_matches_transformers(monkeypatch):
    from server import modelpaths
    d = modelpaths.resolve("answer-lm", "ANSWER_LM_DIR")
    monkeypatch.setattr(config, "ANSWER_LM_DIR", d)
    if not answer_lm.installed():
        pytest.skip(f"answer-lm not installed in {d}")
    tf = pytest.importorskip("transformers")
    tok = tf.AutoTokenizer.from_pretrained(str(d))
    user = answer.build_prompt("who is Ünïcode 日本?", DOCS)
    want = tok.apply_chat_template([{"role": "system", "content": answer.SYSTEM_LOCAL}, {"role": "user", "content": user}],
                                   add_generation_prompt=True, return_dict=True)["input_ids"]
    answer_lm.unload()
    try:
        assert answer_lm.load().prompt_ids(answer.SYSTEM_LOCAL, user) == list(want)
    finally:
        answer_lm.unload()
