"""Memory graph, small panel + full-size view (web/graphlayout.js, web/graph.js, web/graphbig.js).
- layout helpers (run under node): labels never overlap and stay in bounds, the focus and the current turn are labelled
  first, fit-to-bounds, one cell per turn (including a turn that only went back to earlier entities);
- wiring: the full-size view is loaded, opens from the small graph and an expand button, and follows the Results
  overlay rules (Esc, focus trap, focus returns, keys stopped before push-to-talk).
Run: .venv/bin/python -m pytest -q tests/test_graph_layout.py
"""
import json
import shutil
import subprocess

import pytest

from server import config

ROOT = config.ROOT
NODE = shutil.which("node")


def _node(script: str):
    out = subprocess.run([NODE, "-e", f"const L = require({json.dumps(str(ROOT / 'web' / 'graphlayout.js'))});\n{script}"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def _busy_graph(turns=8, per=3):
    nodes, edges = [], []
    for t in range(1, turns + 1):
        for i in range(per):
            nodes.append({"id": f"n{t}_{i}", "label": f"Rose Garden Inn number {t}-{i}", "kind": "place", "turns": [t]})
            if i:
                edges.append({"a": f"n{t}_0", "b": f"n{t}_{i}", "rel": "near", "kind": "said", "turn": t})
        if t > 1:
            edges.append({"a": f"n{t}_0", "b": f"n{t-1}_0", "rel": "located in", "kind": "wikidata", "turn": t})
    nodes[0]["turns"] = [1, turns]
    return {"nodes": nodes, "edges": edges, "user": [{"key": "home", "label": "you live in San Jose", "v": "san jose"}],
            "turns": [{"t": t, "q": f"question {t}"} for t in range(1, turns + 1)] + [{"t": turns + 1, "q": "and when was it built"}]}


needs_node = pytest.mark.skipif(NODE is None, reason="node not installed")


@needs_node
def test_place_labels_no_overlap_in_bounds_priority_first():
    r = _node("""
      const items = [];
      for (let i = 0; i < 30; i++) items.push({id: 'n' + i, x: 20 + (i % 6) * 18, y: 20 + Math.floor(i / 6) * 14, r: 5, w: 60, h: 12});
      const out = L.placeLabels(items, {bounds: {x: 0, y: 0, w: 200, h: 120}, max: 8});
      console.log(JSON.stringify({ids: Object.keys(out), boxes: Object.values(out)}));
    """)
    boxes = r["boxes"]
    assert 0 < len(boxes) <= 8 and r["ids"][0] == "n0"          # the highest-priority label is always placed first
    for b in boxes:
        assert b["x"] >= 0 and b["y"] >= 0 and b["x"] + b["w"] <= 200 and b["y"] + b["h"] <= 120
    for i, a in enumerate(boxes):
        for c in boxes[i + 1:]:
            assert not (a["x"] < c["x"] + c["w"] and c["x"] < a["x"] + a["w"] and a["y"] < c["y"] + c["h"] and c["y"] < a["y"] + a["h"])


@needs_node
def test_rank_focus_then_current_turn():
    g = _busy_graph()
    r = _node(f"""
      const a = L.augment({json.dumps(g)});
      const order = L.rank(a.nodes, a.edges, 'n3_1', a.now).map(n => n.id);
      console.log(JSON.stringify({{order, now: a.now, you: a.nodes.some(n => n.id === '__you')}}));
    """)
    assert r["order"][0] == "n3_1" and r["now"] == 9 and r["you"]
    assert r["order"][1] == "n1_0"            # mentioned again in turn 8 (the last turn with entities) ranks above older ones


@needs_node
def test_fit_keeps_points_in_box_and_ellipsis():
    r = _node("""
      const f = L.fit([{x: -500, y: 3}, {x: 900, y: 40}, {x: 10, y: -300}], 300, 160, {l: 14, r: 14, t: 12, b: 22}, 1.4);
      const pts = [{x: -500, y: 3}, {x: 900, y: 40}, {x: 10, y: -300}].map(f.map);
      console.log(JSON.stringify({pts, e: L.ellipsize('San Jose Mineta International Airport', 18)}));
    """)
    for p in r["pts"]:
        assert 13.9 <= p["x"] <= 286.1 and 11.9 <= p["y"] <= 138.1
    assert r["e"].endswith("…") and len(r["e"]) == 18


@needs_node
def test_group_layout_one_cell_per_turn_labels_apart():
    g = _busy_graph()
    r = _node(f"""
      const out = L.groupLayout({json.dumps(g)}, {{maxW: 900, measure: s => s.length * 7, rnd: (() => {{ let s = 1; return () => (s = (s * 16807) % 2147483647) / 2147483647; }})()}});
      console.log(JSON.stringify({{cells: out.cells.map(c => ({{key: c.key, x: c.x, y: c.y, w: c.w, h: c.h, n: c.nodes.length, refs: c.refs, now: c.now}})), pos: out.pos, width: out.width}}));
    """)
    keys = [c["key"] for c in r["cells"]]
    assert keys == ["you"] + list(range(1, 10))                # a turn with no new entity still gets its cell
    assert r["cells"][-1]["n"] == 0 and r["cells"][-1]["now"]
    assert r["width"] <= 900 + 1
    for c in r["cells"]:                                          # cells don't overlap each other
        for d in r["cells"]:
            if c is not d:
                assert not (c["x"] < d["x"] + d["w"] and d["x"] < c["x"] + c["w"] and c["y"] < d["y"] + d["h"] and d["y"] < c["y"] + c["h"])
    cell = {c["key"]: c for c in r["cells"]}
    for t in range(1, 9):                                         # every node sits in its first turn's cell
        for i in range(3):
            p, c = r["pos"][f"n{t}_{i}"], cell[t]
            assert c["x"] <= p["x"] <= c["x"] + c["w"] and c["y"] <= p["y"] <= c["y"] + c["h"]
        # labels (width 7 px/char, under the node) of one cell don't overlap
        boxes = [(r["pos"][f"n{t}_{i}"]["x"] - 7 * len(f"Rose Garden Inn number {t}-{i}") / 2, r["pos"][f"n{t}_{i}"]["y"]) for i in range(3)]
        for a in range(3):
            for b in range(a + 1, 3):
                w = 7 * len("Rose Garden Inn number 1-1")
                assert abs(boxes[a][0] - boxes[b][0]) >= w - 1 or abs(boxes[a][1] - boxes[b][1]) >= 16


def test_web_wiring_full_size_view():
    html = (ROOT / "web" / "lab.html").read_text()
    i_l, i_g, i_b = html.index("graphlayout.js"), html.index('src="graph.js'), html.index("graphbig.js")
    assert i_l < i_g < i_b and "graphbig.css" in html
    g = (ROOT / "web" / "graph.js").read_text()
    assert 'id="memExpand"' in g and 'id="memStage"' in g and "HSGraphBig" in g and "placeLabels" in g
    assert '<div class="pers-sub" id="persSub">' in g                     # "For you" sub line has its own row
    b = (ROOT / "web" / "graphbig.js").read_text()
    for s in ('"Escape"', '"Tab"', "stopPropagation", 'addEventListener("keydown", onKey, true)', "gesturechange", "wheel",
              'aria-modal', "mgSearch", "Legend", 'data-mg="fit"', 'data-mg="in"', 'data-mg="out"'):
        assert s in b, s
    assert "/Users/" not in b + g
