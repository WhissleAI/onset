"""Mic runs over one websocket (the Mac app bug: 0.1-0.6 s of audio per push-to-talk press).

The page tags every run with its own id (rid); the server echoes it on that run's events. A press that starts while
the previous run is still flushing gets the previous run's run_done AFTER its own start: without the id the page took
it for its own and turned the new mic off (no audio, no stop sent), and every later press repeated it.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import app as appmod
from server import ear as earmod
from server import lanes


class FakeEar:
    def __init__(self, sinks, domain="assistant", on_event=None):
        self.fed = 0

    async def feed(self, pcm):
        self.fed += len(pcm)

    async def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod.app.router, "on_startup", [])        # no model warm-up in a unit test
    monkeypatch.setattr(earmod, "make_ear", lambda sinks, domain="assistant", on_event=None: FakeEar(sinks))
    monkeypatch.setattr(lanes, "_index", lanes.LocalIndex.__new__(lanes.LocalIndex))
    lanes._index.docs = []
    logged = []
    monkeypatch.setattr(appmod, "rlog", lambda s: logged.append(s))
    with TestClient(appmod.app) as c:
        yield c, logged


def _until(ws, t):
    for _ in range(200):
        e = ws.receive_json()
        if e.get("t") == t:
            return e
    raise AssertionError(f"no {t}")


def test_run_done_carries_the_page_run_id_and_stop_logs_capture_counters(client):
    c, logged = client
    with c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["t"] == "ready"
        ws.send_json({"t": "start", "modes": ["headstart"], "rid": 7})
        ws.send_bytes(np.zeros(1600, dtype="<f4").tobytes())
        ws.send_json({"t": "stop", "dbg": {"why": "release", "held_s": 3.21, "frames": 1203, "ctx": "running",
                                           "nested": {"x": 1}, "muted": False}})
        done = _until(ws, "run_done")
        assert done["rid"] == 7
        # the next press: its start makes the server end nothing (already stopped); its own run_done has its own id
        ws.send_json({"t": "start", "modes": ["headstart"], "rid": 8})
        ws.send_json({"t": "start", "modes": ["headstart"], "rid": 9})     # a press while run 8 is open: ends run 8
        late = _until(ws, "run_done")
        assert late["rid"] == 8                                             # the page (on run 9) ignores it
    stop = next(s for s in logged if "stop (client)" in s)
    assert "why=release" in stop and "held_s=3.21" in stop and "frames=1203" in stop and "ctx=running" in stop
    assert "nested" not in stop and "muted" not in stop                    # only flat scalars, no bools
    assert any("stopping: 0.1s of audio" in s for s in logged)


def test_runs_without_an_id_keep_the_old_events(client):
    c, _ = client
    with c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"t": "start", "modes": ["headstart"], "rid": True})   # not an int: ignored
        ws.send_json({"t": "stop"})
        assert "rid" not in _until(ws, "run_done")


def test_page_ignores_events_of_other_runs_and_keeps_one_capture():
    js = (appmod.WEB / "app.js").read_text()
    assert "if (e.rid != null && (!run || e.rid !== run.rid)) return;" in js
    assert 'obj = { ...obj, rid: ++ridSeq }' in js and "run = { rid: ridSeq," in js
    # one capture per session, gated per press; released after a quiet minute
    assert "async function openCapture()" in js and "setTimeout(releaseCapture, MIC_KEEP_MS)" in js
    assert 'send({ t: "stop", dbg })' in js
