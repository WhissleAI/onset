"""Cost meter: answer-model token capture (answer.py), waste accounting, $ arithmetic, metric fields.

Run: .venv/bin/python -m pytest -q tests/test_meter.py
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import answer, config, lanes, meter, router  # noqa: E402
from test_speculation import FakeStream, fakes, speak  # noqa: E402,F401  (fakes: autouse fake lanes + answer model)

RealStream = answer.Stream          # test_speculation's autouse fixture swaps answer.Stream for FakeStream


# ----------------------------------------------------------------------------- fake Anthropic stream with usage
class UsageMsgStream:
    """Streams n text chunks; message_start usage (input) arrives at open, message_delta (final output) at the end."""

    def __init__(self, spec):
        self.spec = spec
        self.snap = None

    async def __aenter__(self):
        await asyncio.sleep(self.spec.get("open", 0.0))
        self.snap = SimpleNamespace(stop_reason=None, usage=SimpleNamespace(
            input_tokens=self.spec["in"], output_tokens=1, cache_creation_input_tokens=None,
            cache_read_input_tokens=None))
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def current_message_snapshot(self):
        assert self.snap is not None
        return self.snap

    @property
    def text_stream(self):
        spec, me = self.spec, self

        async def gen():
            await asyncio.sleep(spec["ttft"])
            for i in range(spec["n"]):
                yield "word " * 2                     # 10 characters per chunk
                await asyncio.sleep(spec.get("gap", 0.0))
            me.snap.usage.output_tokens = spec["out"]
            me.snap.stop_reason = "end_turn"
        return gen()


class UsageClient:
    def __init__(self, specs):
        self.specs, self.calls = list(specs), 0

    @property
    def messages(self):
        return self

    def stream(self, **kw):
        s = self.specs[min(self.calls, len(self.specs) - 1)]
        self.calls += 1
        return UsageMsgStream(s)


DOCS = [lanes.Doc(id="1", title="Nvidia", text="Nvidia was founded by Jensen Huang, Chris Malachowsky and Curtis Priem.")]


@pytest.fixture
def no_card(monkeypatch, claude_backend):
    monkeypatch.setattr(config, "ANSWER_FALLBACK", False)
    monkeypatch.setattr(router, "pick_card", lambda q, d: None)
    answer.reset_status()


def test_stream_usage_exact_from_api(monkeypatch, no_card):
    monkeypatch.setattr(answer, "_client", UsageClient([{"in": 812, "out": 37, "ttft": 0.01, "n": 3}]))

    async def go():
        st = RealStream("who founded nvidia", DOCS)
        await st.task
        return st
    st = asyncio.run(go())
    u = st.usage()
    assert u == {"in": 812, "out": 37, "win_in": 812, "win_out": 37, "requests": 1, "est": False}
    t = meter.tokens([st], st)
    assert (t["ans_in"], t["ans_out"], t["waste_in"], t["waste_out"]) == (812, 37, 0, 0)


def test_cancelled_draft_tokens_are_waste_and_estimated(monkeypatch, no_card):
    # a draft cancelled mid-stream: its input is exact (message_start came), its output is estimated from what streamed
    monkeypatch.setattr(answer, "_client", UsageClient([{"in": 700, "out": 999, "ttft": 0.01, "n": 50, "gap": 0.02}]))

    async def go():
        st = RealStream("who founded nvidia", DOCS, tag="draft")
        await asyncio.sleep(0.1)
        st.cancel()
        await asyncio.sleep(0.02)
        return st
    st = asyncio.run(go())
    u = st.usage()
    assert u["in"] == 700 and u["est"] and 0 < u["out"] < 999
    final = SimpleNamespace(usage=lambda: {"in": 900, "out": 40, "win_in": 900, "win_out": 40, "requests": 1,
                                           "est": False}, by=None)
    t = meter.tokens([st, final], final)
    assert t["ans_in"] == 900 and t["waste_in"] == 700 and t["waste_out"] == u["out"]
    assert t["waste_streams"] == 1 and t["tokens_est"] and t["llm_requests"] == 2


def test_hedge_loser_counts_as_waste(monkeypatch, no_card):
    # request 0 is slow (no token by HEDGE_MS): the duplicate wins; the loser's prompt was still billed
    monkeypatch.setattr(config, "HEDGE_MS", 50)
    monkeypatch.setattr(answer, "_client", UsageClient([{"in": 500, "out": 30, "ttft": 0.5, "n": 2},
                                                        {"in": 500, "out": 30, "ttft": 0.01, "n": 2}]))

    async def go():
        st = RealStream("who founded nvidia", DOCS)
        await st.task
        return st
    st = asyncio.run(go())
    assert st.hedged and st.winner == 1
    t = meter.tokens([st], st)
    assert t["ans_in"] == 500 and t["ans_out"] == 30
    assert t["waste_in"] == 500 and t["llm_requests"] == 2


def test_card_answer_uses_zero_tokens(monkeypatch, claude_backend):
    card = {"spoken": "It is 21 degrees in Paris.", "kind": "weather"}
    monkeypatch.setattr(router, "pick_card", lambda q, d: (0, card))
    monkeypatch.setattr(config, "ROUTER", True)
    monkeypatch.setattr(config, "ROUTER_CARD_ANSWER", True)

    class NoCalls:
        @property
        def messages(self):
            raise AssertionError("a card answer makes no model request")
    monkeypatch.setattr(answer, "_client", NoCalls())

    async def go():
        st = RealStream("weather in paris", DOCS)
        await st.task
        return st
    st = asyncio.run(go())
    assert st.by == "card"
    assert st.usage() == {"in": 0, "out": 0, "win_in": 0, "win_out": 0, "requests": 0, "est": False}
    f = meter.fields([st], st, {"paid_calls": 0, "spec_fires": 1, "spec_used": 1})
    assert f["ans_in"] == f["ans_out"] == f["waste_in"] == 0 and f["usd"] == 0 and f["usd_llm"] == 0


def test_connection_failure_is_not_billed(monkeypatch, no_card):
    class Down:
        @property
        def messages(self):
            return self

        def stream(self, **kw):
            class S:
                async def __aenter__(self):
                    raise ConnectionError("down")

                async def __aexit__(self, *a):
                    return False
            return S()
    monkeypatch.setattr(answer, "_client", Down())

    async def go():
        st = RealStream("q", DOCS)
        await st.task
        return st
    st = asyncio.run(go())
    assert st.error and st.usage()["in"] == 0 and st.usage()["out"] == 0


# ----------------------------------------------------------------------------- $ arithmetic
def test_cost_arithmetic(monkeypatch, claude_backend):
    monkeypatch.setattr(config, "ANSWER_MODEL", "claude-haiku-5-5")
    p = meter.prices()
    assert p["verified"] and (p["in_usd_mtok"], p["out_usd_mtok"]) == (0.10, 0.50) and p["credit_usd"] == 0.008
    tok = {"ans_in": 1000, "ans_out": 100, "waste_in": 2000, "waste_out": 60}
    c = meter.cost(2, tok)
    assert c["usd_web"] == pytest.approx(2 * 0.008)
    assert c["usd_llm"] == pytest.approx((3000 * 0.10 + 160 * 0.50) / 1e6)
    assert c["usd_llm_waste"] == pytest.approx((2000 * 0.10 + 60 * 0.50) / 1e6)
    assert c["usd"] == pytest.approx(c["usd_web"] + c["usd_llm"], abs=1e-6) and c["usd_est"]
    # actual Tavily credits (an advanced search = 2) beat paid calls x 1
    assert meter.cost(1, tok, credits=2.0)["usd_web"] == pytest.approx(0.016)
    assert meter.cost(0, {})["usd"] == 0


def test_unlisted_model_price_is_unverified(monkeypatch, claude_backend):
    monkeypatch.setattr(config, "ANSWER_MODEL", "claude-unknown-9")
    c = meter.cost(1, {"ans_in": 1000, "ans_out": 10})
    assert c["usd_llm"] is None and c["usd"] == pytest.approx(0.008) and not c["price_verified"]
    monkeypatch.setattr(config, "METER_ANSWER_IN_USD_MTOK", "1.0")
    monkeypatch.setattr(config, "METER_ANSWER_OUT_USD_MTOK", "5.0")
    c = meter.cost(0, {"ans_in": 1000, "ans_out": 10})
    assert c["usd_llm"] == pytest.approx((1000 * 1.0 + 10 * 5.0) / 1e6) and not c["price_verified"]


# ----------------------------------------------------------------------------- metric fields (speculator + session)
class MeteredStream(FakeStream):
    """FakeStream that reports usage like answer.Stream: 600 in, 20 out per finished stream; a cancelled one 600 / 5."""

    def usage(self):
        if self.tokens:
            return {"in": 600, "out": 20, "win_in": 600, "win_out": 20, "requests": 1, "est": False}
        return {"in": 600, "out": 5, "win_in": 0, "win_out": 0, "requests": 1, "est": True}


METER_KEYS = ("ans_in", "ans_out", "waste_in", "waste_out", "llm_requests", "tokens_est", "usd", "usd_web", "usd_llm",
              "usd_llm_waste", "usd_est", "wasted_fires", "paid_calls", "free_calls")


@pytest.fixture
def metered(monkeypatch, claude_backend):
    from server import speculator
    monkeypatch.setattr(answer, "Stream", MeteredStream)
    monkeypatch.setattr(speculator.answer, "Stream", MeteredStream)
    monkeypatch.setattr(config, "ANSWER_MODEL", "claude-haiku-5-5")


@pytest.mark.parametrize("mode", ["headstart", "baseline"])
def test_metric_has_meter_fields(metered, mode):
    m, ev = asyncio.run(speak("who founded the company Nvidia", mode))
    metric = [e for e in ev if e["t"] == "metric"][-1]
    for k in METER_KEYS:
        assert k in metric, k
    assert metric["ans_in"] == 600 and metric["ans_out"] == 20
    n_streams = len(MeteredStream.instances)
    assert metric["llm_requests"] == n_streams
    # every stream that did not answer is waste (Head Start drafts; the baseline has none)
    assert metric["waste_in"] == 600 * (n_streams - 1)
    if mode == "baseline":
        assert metric["waste_in"] == 0 and metric["wasted_fires"] == 0
    exp = metric["paid_calls"] * 0.008 + ((600 * n_streams) * 0.10
                                          + (20 + metric["waste_out"]) * 0.50) / 1e6
    assert metric["usd"] == pytest.approx(exp, abs=2e-6)
    # the final tally at close: one usage event, same question, answered
    us = [e for e in ev if e["t"] == "usage"]
    assert len(us) == 1 and us[0]["answered"] and us[0]["q"] == metric["q"]
    assert us[0]["ans_in"] == 600 and us[0]["llm_requests"] == n_streams


def test_unused_drafts_are_counted_as_waste(metered):
    """Speculator close: drafts that did not answer (one finished, one cancelled mid-stream) are waste, the reused
    one is the answer; the final usage event carries the tally even after the metric went out."""
    from server.speculator import Speculator

    class S:
        def __init__(self, u, by=None):
            self._u, self.by = u, by

        def usage(self):
            return self._u

        def cancel(self):
            pass
    used = S({"in": 600, "out": 20, "win_in": 600, "win_out": 20, "requests": 1, "est": False})
    done_unused = S({"in": 550, "out": 18, "win_in": 550, "win_out": 18, "requests": 1, "est": False})
    cancelled = S({"in": 500, "out": 4, "win_in": 0, "win_out": 0, "requests": 1, "est": True})
    ev = []

    async def go():
        sp = Speculator(ev.append, mode="headstart")
        sp.streams = [cancelled, done_unused, used]
        sp.meter_used = used
        sp.meter_metric = {"paid_calls": 1, "spec_fires": 3, "spec_used": 1}
        sp.close()
        sp.close()                                    # once
    asyncio.run(go())
    (u,) = [e for e in ev if e["t"] == "usage"]
    assert (u["ans_in"], u["ans_out"]) == (600, 20)
    assert (u["waste_in"], u["waste_out"], u["waste_streams"]) == (1050, 22, 2) and u["tokens_est"]
    assert u["wasted_fires"] == 2 and u["answered"]
    assert u["usd_llm_waste"] == pytest.approx((1050 * 0.10 + 22 * 0.50) / 1e6)
    assert u["usd"] == pytest.approx(0.008 + ((1650 * 0.10) + 42 * 0.50) / 1e6, abs=1e-6)


def test_superseded_question_still_reports_tokens(metered):
    from server.speculator import Speculator
    ev = []

    class S:
        by = None

        def usage(self):
            return {"in": 400, "out": 10, "win_in": 400, "win_out": 10, "requests": 1, "est": False}

        def cancel(self):
            pass

    async def go():
        sp = Speculator(ev.append, mode="headstart")
        sp.streams = [S()]
        sp.close()
    asyncio.run(go())
    (u,) = [e for e in ev if e["t"] == "usage"]
    assert not u["answered"] and u["waste_in"] == 400 and u["ans_in"] == 0
