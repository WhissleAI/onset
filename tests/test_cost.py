"""Cost parity (server/cost.py, COST_MODE): one paid web call per question sent when the meaning is final, drafts only
on a final meaning, no draft for a meaning already being answered, prompt caching of the answer model's system prompt
and its pricing in the meter.

Run: .venv/bin/python -m pytest -q tests/test_cost.py
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_speculation as ts  # noqa: E402  (fake lanes, fake answer model, speak())
from server import answer, config, cost, frame, meter, searchcache, speculator  # noqa: E402

fakes = ts.fakes          # the same autouse fixture (fake web / embeddings / answer stream)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(config, "CACHE_SEM", False)        # every question pays for its own searches
    monkeypatch.setattr(config, "SPEC_EFF", True)
    monkeypatch.setattr(config, "ROUTER", False)            # the web lane is the fake's: no router plans in the way
    searchcache.reset()
    yield
    searchcache.reset()


def F(words, slots=(), phrases=()):
    ws = words.split()
    c = tuple(w for w in (frame.norm(x) for x in ws) if w and w not in frame.STOP)
    return frame.Frame(words=ws, intent="discourse.intent.qa_factoid", intent_p=0.9, slots=tuple(sorted(slots)),
                       content=c, phrases=tuple(phrases))


# ----------------------------------------------------------------------------------------------- presets
def test_profiles(monkeypatch):
    sp = cost.profile("speed")
    assert (sp.paid_max, sp.final_max, sp.draft_budget, sp.final_eot) == (
        config.SPEC_PAID_MAX, config.SPEC_PAID_MAX, config.DRAFT_BUDGET, None)
    pa = cost.profile("parity")
    assert pa.paid_max == 1 and pa.final_max == 1 + config.COST_PARITY_FINAL_EXTRA
    assert pa.draft_budget == config.COST_PARITY_DRAFTS and pa.final_eot == config.COST_PARITY_EOT
    monkeypatch.setattr(config, "COST_MODE", "parity")
    assert cost.profile().name == "parity" and cost.profile(None).name == "parity"
    assert cost.valid("Parity") == "parity" and cost.valid("cheap") is None and cost.valid(None) is None


# ----------------------------------------------------------------------------------------------- parity fire policy
def test_parity_paid_call_waits_for_the_final_meaning():
    async def go():
        sp = speculator.Speculator(lambda e: None, "headstart", cost_mode="parity")
        sp.eot_seen = True                       # a Flux-like ear: end-of-turn scores arrive
        f = F("who founded the company Nvidia", slots=[("organization", "nvidia")], phrases=["nvidia"])
        sp.on_frame(f)
        await asyncio.sleep(0.4)                 # past SPEC_SETTLE_MS: speed would have fired by now
        sp.on_frame(f)
        assert not [x for x in sp.fires if x.lane == "web"]
        sp.on_eot(0.1)
        assert not [x for x in sp.fires if x.lane == "web"]
        sp.on_eot(0.95)                          # the turn is over: these are the final words
        assert len([x for x in sp.fires if x.lane == "web"]) == 1
        sp.on_eot(0.97)
        assert len([x for x in sp.fires if x.lane == "web"]) == 1
        sp.close()
    asyncio.run(go())


def test_speed_fires_on_settle_without_turn_score():
    async def go():
        sp = speculator.Speculator(lambda e: None, "headstart", cost_mode="speed")
        f = F("who founded the company Nvidia", slots=[("organization", "nvidia")], phrases=["nvidia"])
        sp.on_frame(f)
        await asyncio.sleep(config.SPEC_SETTLE_MS / 1000 + 0.05)
        sp.on_frame(f)
        assert len([x for x in sp.fires if x.lane == "web"]) == 1
        sp.close()
    asyncio.run(go())


@pytest.mark.parametrize("text", ["how many bones are there in an adult human body"])
def test_parity_end_to_end_costs_what_baseline_costs(monkeypatch, text):
    monkeypatch.setattr(config, "COST_MODE", "parity")
    hs, _ = asyncio.run(ts.speak(text, "headstart"))
    n_hs = len(ts.WEB_CALLS)
    ts.WEB_CALLS.clear()
    bl, _ = asyncio.run(ts.speak(text, "baseline"))
    assert hs["cost_mode"] == "parity" and hs.get("answer")
    assert hs["paid_calls"] <= bl["paid_calls"] == 1
    assert n_hs == hs["paid_calls"]
    assert hs["drafts"] <= config.COST_PARITY_DRAFTS


def test_parity_meaning_change_gets_one_more_call(monkeypatch):
    """The tagger gate (this harness's stand-in for an end-of-turn score) opens on "who founded the company": the one
    paid call goes there; "... Nvidia" names a new subject, its coverage check fails, the endpoint searches it (the
    one extra call parity allows), never more."""
    monkeypatch.setattr(config, "COST_MODE", "parity")
    m, _ = asyncio.run(ts.speak("who founded the company Nvidia", "headstart"))
    assert m.get("answer") and "nvidia" in m["query"]
    assert m["paid_calls"] <= cost.profile("parity").final_max


def test_parity_multi_part_stays_within_the_plan_cap(monkeypatch):
    """A two-part question (server/planner.py) searches each part; parity holds the parts to the final meaning. (The
    Arc-style tagger gate in this harness opens on "who founded the company", one premature call the cap absorbs.)"""
    monkeypatch.setattr(config, "COST_MODE", "parity")
    m, _ = asyncio.run(ts.speak("who founded the company Nvidia and when did it go public", "headstart"))
    p = cost.profile("parity")
    assert m.get("answer") and m["paid_calls"] <= p.final_max + config.PLAN_EXTRA_PAID


def test_parity_correction_may_search_the_replacement(monkeypatch):
    monkeypatch.setattr(config, "COST_MODE", "parity")
    m, _ = asyncio.run(ts.speak("what's the capital of Canada actually make that Turkey", "headstart", gap=0.3))
    assert "turkey" in m["query"]
    assert any("turkey" in q for q in ts.WEB_CALLS)
    assert m["paid_calls"] <= cost.profile("parity").final_max


# ----------------------------------------------------------------------------------------------- drafts
def test_no_draft_for_a_meaning_being_answered():
    async def go():
        sp = speculator.Speculator(lambda e: None, "headstart", cost_mode="speed")
        f = F("who founded the company Nvidia", slots=[("organization", "nvidia")], phrases=["nvidia"])
        fire = sp._fire("web", f.query(), f, lambda: ts.fake_web(f.query()))
        await fire.task
        sp.answering.add(f.meaning_sig)          # a soft / final answer for exactly this meaning is streaming
        sp._maybe_draft(f)
        await asyncio.sleep(0.02)
        assert not sp.drafts and not sp.draft_busy
        sp.answering.clear()
        sp.last_sig = f.sig
        sp._maybe_draft(f)                       # (control: without it, the same frame drafts)
        await asyncio.sleep(0.05)
        assert len(sp.drafts) == 1
        sp.close()
    asyncio.run(go())


def test_parity_drafts_only_a_final_meaning():
    async def go():
        sp = speculator.Speculator(lambda e: None, "headstart", cost_mode="parity")
        sp.eot_seen = True
        f = F("who founded the company Nvidia", slots=[("organization", "nvidia")], phrases=["nvidia"])
        sp.last_sig, sp.last_frame = f.sig, f
        fire = sp._fire("web", f.query(), f, lambda: ts.fake_web(f.query()))
        await fire.task
        sp.eot_p = 0.1
        sp._maybe_draft(f)
        await asyncio.sleep(0.02)
        assert not sp.drafts
        sp.on_eot(0.95)                          # final meaning: the draft starts (and will be the answer)
        await asyncio.sleep(0.05)
        assert len(sp.drafts) == 1
        sp.close()
    asyncio.run(go())


# ----------------------------------------------------------------------------------------------- prompt caching
def test_system_prompt_is_cache_marked(monkeypatch):
    sp = answer.system_param()
    assert sp[0]["text"] == answer.SYSTEM and sp[0]["cache_control"] == {"type": "ephemeral"}
    monkeypatch.setattr(config, "COST_CACHE", False)
    assert answer.system_param() == answer.SYSTEM


def test_meter_prices_cache_hits_and_writes():
    p = {"in_usd_mtok": 0.10, "out_usd_mtok": 0.50, "credit_usd": 0.008, "credits_per_call": 1, "verified": True,
         "cache_read_mult": 0.1, "cache_write_mult": 1.25}
    plain = meter.cost(1, {"ans_in": 1000, "ans_out": 100}, p=p)
    hit = meter.cost(1, {"ans_in": 1000, "ans_out": 100, "ans_cache_read": 700}, p=p)
    wr = meter.cost(1, {"ans_in": 1000, "ans_out": 100, "ans_cache_write": 700}, p=p)
    assert plain["usd_llm"] == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)
    assert hit["usd_llm"] == pytest.approx((300 * 0.10 + 700 * 0.01 + 100 * 0.50) / 1e6)
    assert wr["usd_llm"] == pytest.approx((300 * 0.10 + 700 * 0.125 + 100 * 0.50) / 1e6)
    w = meter.cost(0, {"ans_in": 0, "waste_in": 1000, "waste_cache_read": 700}, p=p)
    assert w["usd_llm_waste"] == pytest.approx((300 * 0.10 + 700 * 0.01) / 1e6) and w["usd_web"] == 0


def test_meter_splits_cache_tokens_by_answer_and_waste():
    class St:
        def __init__(self, u):
            self.u = u

        def usage(self):
            return self.u
    used = St({"in": 1600, "out": 60, "win_in": 800, "win_out": 30, "requests": 2, "est": False,
               "cache_read": 1400, "cache_write": 0, "win_cache_read": 700, "win_cache_write": 0})
    draft = St({"in": 900, "out": 10, "win_in": 900, "win_out": 10, "requests": 1, "est": False,
                "cache_read": 0, "cache_write": 700})
    t = meter.tokens([used, draft], used)
    assert (t["ans_in"], t["ans_cache_read"], t["waste_in"]) == (800, 700, 1700)
    assert (t["waste_cache_read"], t["waste_cache_write"]) == (700, 700)
