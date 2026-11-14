"""The macOS app reuses earlier searches across launches (CACHE_PERSIST=1, CACHE_PATH in the app data dir):
entries survive a restart, the file stays bounded and valid JSON, and POST /api/cache/clear forgets them."""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from server import config, searchcache


def _key(i: int) -> searchcache.QKey:
    return searchcache.QKey(topic="general", subject=frozenset({f"thing{i}"}), phrases=(f"thing{i}",), terms=frozenset({"tall", f"thing{i}"}),
                            nums=frozenset(), family="")


def _docs(i: int):
    return [SimpleNamespace(id=f"d{i}", title=f"Thing {i}", text="text", url=f"https://example.org/{i}", source="web")]


def test_cache_persists_across_restart_bounded_and_atomic(tmp_path, monkeypatch):
    path = tmp_path / "data" / "search_cache.json"
    monkeypatch.setattr(config, "CACHE_PERSIST", True)
    monkeypatch.setattr(config, "CACHE_SEM", True)
    monkeypatch.setattr(config, "CACHE_MAX", 3)
    c = searchcache.SearchCache(path)
    for i in range(5):
        c.put(_key(i), f"thing {i}", _docs(i), "tavily")
    data = json.loads(path.read_text())                  # always a whole JSON document (tmp file + rename)
    assert len(data["entries"]) == 3                      # bounded by CACHE_MAX (oldest dropped)
    assert not path.with_suffix(".tmp").exists()
    c2 = searchcache.SearchCache(path)                    # "next launch"
    hit = c2.get(_key(4))
    assert hit is not None and hit.entry.docs[0]["url"].endswith("/4")
    assert hit.as_of and hit.age_s >= 0                   # what the UI labels "cached · as of …"
    assert c2.get(_key(0)) is None


def test_clear_cache_endpoint(tmp_path, monkeypatch):
    from server import app as appmod
    path = tmp_path / "search_cache.json"
    monkeypatch.setattr(config, "CACHE_PERSIST", True)
    monkeypatch.setattr(config, "CACHE_SEM", True)
    monkeypatch.setattr(config, "CACHE_PATH", path)
    searchcache.reset(path)
    searchcache.cache().put(_key(1), "thing 1", _docs(1), "tavily")
    assert path.exists() and len(searchcache.cache()) == 1
    r = TestClient(appmod.app).post("/api/cache/clear")
    assert r.status_code == 200 and r.json()["cleared"] == 1
    assert len(searchcache.cache()) == 0 and not path.exists()
    searchcache.reset()
