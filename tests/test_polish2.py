"""polish2: the typed box under stress (rapid typing, Enter spam, long text, a new question while one is open), the one
trust line under answers (web/confidence.js), the answer card markup, and the answer prompt's no-preamble rule."""
import json
import shutil
import subprocess
import time

import pytest

from server import answer, config, speculator
from server import app as appmod

ROOT = config.ROOT
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


def _stub(monkeypatch, finalized):
    async def finalize(self, f, t_end, **kw):
        finalized.append(f.text)
        self.emit({"t": "answer", "delta": "An answer [1].", "ms": 1})
        self.emit({"t": "metric", "ms": 3, "ready_ms": 5})
        return {"mode": self.mode, "question": f.text, "path": "stub", "answer": "An answer [1]."}
    monkeypatch.setattr(speculator.Speculator, "finalize", finalize)
    monkeypatch.setattr(speculator.Speculator, "on_frame", lambda self, f, now=None: None)


def _client(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(appmod.app.router, "on_startup", [])
    monkeypatch.setattr(appmod.ear, "make_ear", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no ear")))
    return TestClient(appmod.app)


def _until(ws, pred, limit=400):
    got = []
    for _ in range(limit):
        e = ws.receive_json()
        got.append(e)
        if pred(e):
            return got
    raise AssertionError(f"no matching event in {limit}: {[x.get('t') for x in got][-20:]}")


def _keystrokes(ws, rid, text, final=True):
    for k in range(1, len(text) + 1):
        ws.send_json({"t": "type", "rid": rid, "text": text[:k], "final": False})
    if final:
        ws.send_json({"t": "type", "rid": rid, "text": text, "final": True})


QUESTIONS = ["what can i do in san francisco next week",
             "i'm going to see it next week what should i do there",
             "who founded nvidia"]


@pytest.mark.parametrize("text", QUESTIONS)
def test_typed_question_always_answers(monkeypatch, text):
    """Every typed question that reaches Enter gets an answer and a run_done (a user statement included)."""
    finalized = []
    _stub(monkeypatch, finalized)
    with _client(monkeypatch) as c, c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["t"] == "ready"
        ws.send_json({"t": "start", "typed": True, "modes": ["headstart", "baseline"], "rid": 1})
        _keystrokes(ws, 1, text)
        got = _until(ws, lambda e: e["t"] == "run_done")
    assert {e["mode"] for e in got if e["t"] == "metric"} == {"headstart", "baseline"}, [e["t"] for e in got]


def test_typed_enter_spam_and_next_question(monkeypatch):
    """Enter pressed many times, then a second question typed straight away (the page starts a new run): both answered,
    the server never crashes, the connection stays open."""
    finalized = []
    _stub(monkeypatch, finalized)
    with _client(monkeypatch) as c, c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["t"] == "ready"
        ws.send_json({"t": "start", "typed": True, "modes": ["headstart", "baseline"], "rid": 1})
        _keystrokes(ws, 1, "what is the capital of australia")
        for _ in range(8):                                   # Enter spam on the same (closed) question
            ws.send_json({"t": "type", "rid": 1, "text": "what is the capital of australia", "final": True})
        ws.send_json({"t": "start", "typed": True, "modes": ["headstart", "baseline"], "rid": 2})
        _keystrokes(ws, 2, "how tall is the eiffel tower " * 20)    # long text
        got = _until(ws, lambda e: e["t"] == "run_done" and e.get("rid") == 2, limit=2000)
    assert any(e["t"] == "metric" and e.get("rid") == 2 for e in got)
    assert not [e for e in got if e["t"] == "error"], [e for e in got if e["t"] == "error"]


# ----------------------------------------------------------------------------- the one trust line (node)
def _merge(state: dict) -> dict:
    js = f"""
    global.window = {{}}; global.document = {{ getElementById: () => null }};
    require({json.dumps(str(ROOT / 'web' / 'confidence.js'))});
    process.stdout.write(JSON.stringify(window.HSConf._merge({json.dumps(state)})));
    """
    out = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@needs_node
def test_trust_line_merges_confidence_and_check():
    agree = {"status": "agree", "label": "✓ 2 sources agree"}
    one = {"status": "supported", "label": "✓ matches 1 source"}
    m = _merge({"conf": {"conf": 0.93, "band": "high", "reason": "matches a source"}, "verify": agree})
    assert m == {"band": "high", "word": "High confidence", "detail": "2 sources agree"}
    # the screenshot's contradiction: "Low · sources don't say" + "✓ matches 1 source" -> one plain "Unverified" line
    m = _merge({"conf": {"conf": 0.2, "band": "low", "reason": "sources don't say"}, "verify": one})
    assert m["word"] == "Unverified" and "matches" not in m["detail"] and m["band"] == "low"
    m = _merge({"conf": {"conf": 0.5, "band": "mid", "reason": "weak match in sources"},
                "verify": {"status": "disagree", "label": "⚠ sources disagree: 1969 vs 1970"}})
    assert m["detail"] == "sources disagree: 1969 vs 1970"
    # no calibration model: the check alone, still plain words
    assert _merge({"verify": one}) == {"band": "high", "word": "Checked", "detail": "matches a source"}
    assert _merge({}) is None


@needs_node
def test_trust_line_recheck():
    s = {"conf": {"conf": 0.9, "band": "high", "reason": "from Wikidata"},
         "recheck": {"status": "corrected", "label": "corrected"}}
    assert _merge(s)["word"] == "Corrected"
    s["recheck"] = {"status": "rechecked", "label": "re-checked ✓"}
    assert _merge(s)["detail"] == "from Wikidata, re-checked"


def test_answer_card_markup():
    js = (ROOT / "web" / "app.js").read_text()
    css = (ROOT / "web" / "style.css").read_text()
    assert '<sup class="cites">' in js and "fitAnswer(" in js
    assert "lane-chip lc-" not in js                       # the "web · tavily" chip is gone (route chip tooltip)
    assert "max-height: calc(var(--alh) * var(--aln))" in css
    plan = (ROOT / "web" / "plan.js").read_text()
    assert "HSConf.verify" in plan and "HSConf.recheck" in plan


def test_typed_box_enter_handling():
    js = (ROOT / "web" / "app.js").read_text()
    html = (ROOT / "web" / "lab.html").read_text()
    # Enter submits from the box's own keydown (not only the form's implicit submission); an IME commit doesn't
    assert '$("#typeBox").addEventListener("keydown"' in js and "e.isComposing" in js
    # Enter again on the same unedited question while it is answered does not re-ask it (a new run per press)
    assert "typed.asked === text && run && run.rid === typed.askedRid && run.running" in js
    # a pause in typing says Enter answers; no inline predictions / autocorrect in the box (WKWebView)
    assert "press Enter for the answer" in js
    assert 'writingsuggestions="false"' in html and 'autocorrect="off"' in html


@needs_node
def test_enter_spam_guard_logic():
    """typeSend's guard, extracted: a repeated final on the same text and the same still-running run is dropped."""
    js = """
    const typed = { open: false, asked: "who founded nvidia", askedRid: 3 };
    const run = { rid: 3, running: true };
    const drop = (text, r) => typed.asked === text && r && r.rid === typed.askedRid && r.running;
    process.stdout.write(JSON.stringify([drop("who founded nvidia", run), drop("who founded nvidia", { ...run, running: false }),
      drop("who founded nvidia?", run), drop("who founded nvidia", { rid: 4, running: true })]));
    """
    out = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=60)
    assert json.loads(out.stdout) == [True, False, False, False]


def test_app_shell_recovers_from_page_process_end():
    main = (ROOT / "macapp" / "Sources" / "main.swift").read_text()
    assert "func webViewWebContentProcessDidTerminate" in main
    i = main.index("func webViewWebContentProcessDidTerminate")
    body = main[i:i + 1500]
    assert "appLog(" in body and "reloadAfterEnd()" in body and "pageEnds.count > 3" in body
    assert "window.__hsNotice" in main and "showPendingNotice()" in main
    js = (ROOT / "web" / "app.js").read_text()
    assert "window.__hsNotice = (msg)" in js and 'postMessage({ t: "log"' in js


def test_answer_prompt_no_preamble():
    s = answer.SYSTEM
    assert "Never restate or interpret the question" in s
    assert "at most six words" in s
