"""Shared isolation: the degraded-mode breakers and the web replay cache are process-wide state."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import answer, config, lanes, searchcache  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_fallback_state(monkeypatch, tmp_path):
    # never read the real runs/web_cache.json from a test: a test opts in by installing its own cache
    monkeypatch.setattr(config, "WEB_CACHE_PATH", tmp_path / "no_web_cache.json")
    monkeypatch.setattr(config, "WEB_CACHE_RECORD", False)
    # the cross-question search cache is process-wide: off unless a test opts in (tests/test_spec_eff.py), so one
    # test's (or one mode's) searches never answer another's
    monkeypatch.setattr(config, "CACHE_SEM", False)
    monkeypatch.setattr(config, "CACHE_PERSIST", False)
    searchcache.reset()
    # the tiered router (server/router.py) calls free APIs (Wikidata, Open-Meteo, ...): off unless a test opts in
    # (tests/test_router.py does, with a mock transport)
    monkeypatch.setattr(config, "ROUTER", False)
    # the after-answer re-check (server/recheck.py) calls Wikidata: off unless a test opts in (tests/test_recheck.py)
    monkeypatch.setattr(config, "VERIFY_RESEARCH", False)
    # Keystone (server/keystone/) is process-wide and persists to the data dir: off and in-memory unless a test opts
    # in (tests/test_keystone.py)
    monkeypatch.setattr(config, "INDEX", False)
    monkeypatch.setattr(config, "INDEX_PERSIST", False)
    monkeypatch.setattr(config, "INDEX_SHARED", False)
    # the speculation tests were written against the "speed" budget (SPEC_PAID_MAX, DRAFT_BUDGET, settle-time fires);
    # tests/test_cost.py covers COST_MODE=parity (the default) explicitly
    monkeypatch.setattr(config, "COST_MODE", "speed")
    # the on-device answer model (server/answer_lm.py) is never loaded by accident: a folder that does not exist, so a
    # real answer.Stream falls back to the extractive sentence (as it did without an Anthropic key). Tests of the local
    # backend install a fake generator (tests/test_answer_local.py); the real-model smoke test opts in (ANSWER_LM_SMOKE=1)
    monkeypatch.setattr(config, "ANSWER_BACKEND", "local")
    monkeypatch.setattr(config, "ANSWER_LM_DIR", tmp_path / "no-answer-lm")
    lanes.set_web_cache(None)
    lanes.reset_web_status()
    answer.reset_status()
    yield
    lanes.set_web_cache(None)
    lanes.reset_web_status()
    answer.reset_status()


@pytest.fixture
def claude_backend(monkeypatch):
    """ANSWER_BACKEND=claude with a (fake) key: for tests that drive the Claude path with a fake client (answer._client)."""
    monkeypatch.setattr(config, "ANSWER_BACKEND", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
