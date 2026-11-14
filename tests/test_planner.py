"""Planner (server/planner.py) + verify (server/verify.py) + the speculator's plan path.

- decompose(): comparisons, conjunctions, multi-hop split; single-part questions (every prefix of the 58 / showcase
  sets is checked by eval/multipart/plan_textcheck.py) return None.
- Head Start fires the parts while the question is spoken, in parallel; the answer gets the sources grouped by part.
- the baseline searches the same parts after the endpoint.
- a correction re-plans (the corrected-away part's search goes stale).
- a multi-hop part waits for its hop's entity (a Wikidata fact value) and then fires.
- verify: agreement, disagreement, tolerance, unverified; runs after the answer.

Run: .venv/bin/python -m pytest -q tests/test_planner.py
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import test_speculation as ts  # noqa: E402
from server import config, frame, lanes, planner, verify  # noqa: E402
from server.tagger import Tagger  # noqa: E402

fakes = ts.fakes


@pytest.fixture(scope="module")
def tagger():
    t = Tagger(domain="assistant")
    yield t
    t.close()


def _frame(tagger, text):
    r = tagger.tag_text(text)
    return frame.build(text.split(), r.get("spans", []), r.get("current"))


def _parts(tagger, text):
    mp = planner.decompose(_frame(tagger, text))
    return [p.text for p in mp.parts] if mp else None


def test_decompose_shapes(tagger):
    assert _parts(tagger, "which is taller, Mount Everest or K2") == ["how tall is mount everest", "how tall is k2"]
    assert _parts(tagger, "who was born first, Mozart or Beethoven") == ["when was mozart born",
                                                                          "when was beethoven born"]
    assert _parts(tagger, "flights to Denver and the weather there") == ["flights to denver",
                                                                         "what's the weather in denver"]
    assert _parts(tagger, "who founded Nvidia and when was it founded") == ["who founded nvidia",
                                                                            "when was nvidia founded"]
    assert _parts(tagger, "compare Nvidia and AMD revenue") == ["nvidia revenue", "amd revenue"]
    assert _parts(tagger, "how old was the founder of Amazon when it was founded") == [
        "who founded amazon", "when was {B} born", "when was amazon founded"]


@pytest.mark.parametrize("q", ["who founded the company Nvidia", "who wrote pride and prejudice",
                               "who was the first woman to win a Nobel Prize", "who is the founder of Tesla",
                               "what's the difference between a crocodile and an alligator",
                               "so, um, what exactly is a Roth IRA and how does it work",
                               "how long do I cook a two and a half pound meatloaf",
                               "how do I open the terminal in VS Code", "which is the tallest mountain"])
def test_single_part_not_split(tagger, q):
    assert planner.decompose(_frame(tagger, q)) is None


def test_plan_off(tagger, monkeypatch):
    monkeypatch.setattr(config, "PLAN", False)
    assert planner.decompose(_frame(tagger, "which is taller, Mount Everest or K2")) is None


def test_bridge_value():
    d = lanes.Doc(id="wd", title="Amazon · Wikidata", text="Amazon: founded by: Jeff Bezos.",
                  meta={"fact": {"values": ["Jeff Bezos"]}})
    assert planner.bridge_of([d]) == "Jeff Bezos"
    assert planner.bridge_of([lanes.Doc(id="x", title="t", text="t")]) is None


def _plans(events):
    return [e for e in events if e["t"] == "plan"]


def test_headstart_fires_parts_mid_sentence():
    m, ev = asyncio.run(ts.speak("which is taller Everest or K2", "headstart"))
    fired = ts.specs(ev, state="fired")
    parts = [e for e in fired if e.get("part")]
    assert {e["part"] for e in parts} >= {"how tall is everest", "how tall is k2"}
    ep = next(e["ms"] for e in ev if e["t"] == "endpoint")
    assert any(e["ms"] < ep for e in parts if e["part"] == "how tall is everest")     # before the speaker stopped
    assert not any(e["query"] == "taller everest k2" for e in fired)                 # not the whole sentence
    assert m["path"] == "plan" and [p["text"] for p in m["plan"]["parts"]] == ["how tall is everest",
                                                                                "how tall is k2"]
    st = ts.FakeStream.instances[-1]
    assert "2 parts" in st.hint and "(1) how tall is everest" in st.hint
    assert _plans(ev) and _plans(ev)[-1]["final"]
    assert "verify" in m


def test_baseline_searches_parts_after_endpoint():
    m, ev = asyncio.run(ts.speak("which is taller Everest or K2", "baseline"))
    fired = ts.specs(ev, state="fired")
    ep = next(e["ms"] for e in ev if e["t"] == "endpoint")
    assert {e.get("part") for e in fired} == {"how tall is everest", "how tall is k2"}
    assert all(e["lane"] == "final-web" and e["ms"] >= ep for e in fired)
    assert m["path"] == "plan"


def test_single_part_unchanged():
    m, ev = asyncio.run(ts.speak("who founded the company Nvidia", "headstart"))
    assert m["path"] != "plan" and "plan" not in m
    assert not _plans(ev) and not any(e.get("part") for e in ts.specs(ev))


def test_correction_replans():
    m, ev = asyncio.run(ts.speak("who was born first Mozart or Beethoven actually make that Bach", "headstart",
                                 gap=0.5, endpoint_delay=0.8))      # a pause after "Beethoven": its part fires
    parts = [p["text"] for p in m["plan"]["parts"]]
    assert parts == ["when was mozart born", "when was bach born"]
    bee = [e for e in ts.specs(ev) if e.get("part") == "when was beethoven born"]
    assert bee and any(e["state"] == "fired" for e in bee)          # it was searched mid-sentence ...
    assert any(e["state"] == "stale" for e in ts.specs(ev, id=bee[0]["id"]))   # ... and went stale on the correction
    assert any(e.get("part") == "when was bach born" for e in ts.specs(ev, state="fired"))


def test_multihop_bridge(monkeypatch):
    async def web(query, topic="general", final=False, frame=None, meter=None, **kw):
        ts.WEB_CALLS.append(query)
        await asyncio.sleep(0.05)
        if "founded amazon" in query:
            return [lanes.Doc(id="wd:amazon", title="Amazon · Wikidata", text="Amazon: founded by: Jeff Bezos.",
                              meta={"tier": "facts", "strong": True, "fact": {"values": ["Jeff Bezos"]}})]
        return [lanes.Doc(id=f"web:{query}", title=query.title(), text=f"About {query}: 1964.", source="fake")]
    monkeypatch.setattr(lanes, "web", web)
    m, ev = asyncio.run(ts.speak("how old was the founder of Amazon when it was founded", "headstart"))
    assert m["path"] == "plan"
    texts = [p["text"] for p in m["plan"]["parts"]]
    assert "when was Jeff Bezos born" in texts
    assert any("jeff bezos" in q for q in ts.WEB_CALLS)


def test_verify_agree_disagree_unverified():
    D = lambda i, t, u: lanes.Doc(id=i, title=i, text=t, url=u)
    docs = [D("a", "Nvidia was founded in 1993 by Jensen Huang.", "u1"),
            D("b", "Founded in April 1993, Nvidia makes GPUs; Jensen Huang is CEO.", "u2")]
    v = verify.check("Nvidia was founded in 1993 by Jensen Huang [1][2].", docs, "when was nvidia founded")
    assert v["status"] == "agree" and v["label"] == "✓ 2 sources agree"
    T = lambda t, x, u: lanes.Doc(id=u, title=t, text=x, url=u)
    docs2 = [T("Jean Dupont", "Jean Dupont was born in 1993 in Lyon.", "u1"),
             T("Jean Dupont (painter)", "Some say Dupont was born in 1991 in Lyon.", "u2"),
             T("Lyon Football", "The club was founded in 1950; Dupont was born nearby.", "u3")]
    v = verify.check("Jean Dupont was born in 1993 [1].", docs2, "when was jean dupont born")
    assert v["status"] == "disagree" and "1993 vs 1991" in v["label"]
    v = verify.check("Jean Dupont was born in 1993 [1].", [docs2[0], docs2[2]], "when was jean dupont born")
    assert v["status"] == "supported"          # another page's year (a club, not Dupont) is not a disagreement
    v = verify.check("Revenue was $60.9 billion [1].", [D("a", "Revenue: US$60,922 million (FY2024)", "u")], "q")
    assert v["status"] == "supported"
    v = verify.check("It is about 4,200 km long [1].", [D("a", "A river in Africa.", "u")], "q")
    assert v["status"] == "unverified"
    assert verify.check("I don't know.", docs, "q")["status"] == "unverified"
