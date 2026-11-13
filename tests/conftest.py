"""Shared isolation: the degraded-mode breakers and the web replay cache are process-wide state."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import answer, config, lanes  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_fallback_state(monkeypatch, tmp_path):
    # never read the real runs/web_cache.json from a test: a test opts in by installing its own cache
    monkeypatch.setattr(config, "WEB_CACHE_PATH", tmp_path / "no_web_cache.json")
    monkeypatch.setattr(config, "WEB_CACHE_RECORD", False)
    # the tiered router (server/router.py) calls free APIs (Wikidata, Open-Meteo, ...): off unless a test opts in
    # (tests/test_router.py does, with a mock transport)
    monkeypatch.setattr(config, "ROUTER", False)
    lanes.set_web_cache(None)
    lanes.reset_web_status()
    answer.reset_status()
    yield
    lanes.set_web_cache(None)
    lanes.reset_web_status()
    answer.reset_status()
