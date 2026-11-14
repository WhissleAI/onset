"""Build the GitHub Pages site into _site/ (stdlib only; the Pages workflow runs it).

  python3 scripts/pages/build.py [--out _site]
  python3 -m http.server 8812 -d _site          # local preview: http://localhost:8812/

_site/index.html        landing page (site/index.html; {{tokens}} filled from server/scorecard.json)
_site/demo/             the real web UI (web/, copied as is) + site/demo/demo.{js,css} injected: a recorded-run transport
_site/demo/data/        recorded runs + clips (site/demo-data, from scripts/pages/record.py), index.json, api/*.json
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
DOWNLOAD = "https://github.com/WhissleAI/headstart/releases/latest"
REPO = "https://github.com/WhissleAI/headstart"


def demo_index(data: Path) -> dict:
    spec = json.loads((ROOT / "scripts" / "pages" / "scenarios.json").read_text())
    scen, questions, convs, audio = [], [], [], []
    first = None
    for sc in spec["scenarios"]:
        p = data / "runs" / f"{sc['id']}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        first = first or rec
        row = {k: sc[k] for k in ("id", "title", "tag", "blurb", "file", "files", "turns", "text", "typed", "char_ms",
                                  "enter_ms", "open") if k in sc}
        scen.append(row)
        if sc.get("file"):
            questions.append({"file": sc["file"], "text": sc["text"], "kind": sc["tag"].lower()})
            audio.append(sc["file"])
        elif sc.get("files"):
            convs.append({"id": sc["id"], "turns": sc["turns"], "files": sc["files"], "kind": "conversation"})
            audio += sc["files"]
    pre = {e.get("t"): e for e in (first or {}).get("pre", [])}
    health = json.loads((data / "api" / "health.json").read_text())
    return {"download": DOWNLOAD, "repo": REPO, "scenarios": scen, "questions": questions, "conversations": convs,
            "audio": audio, "ready": pre.get("ready") or {"t": "ready", "mode": "sys", "warm": True},
            "hello": [e for k, e in pre.items() if k in ("memory", "persona")],
            "emptyGraph": {"nodes": [], "edges": [], "user": [], "focus": None, "turns": []},
            "health": health}


def build_demo(out: Path):
    demo = out / "demo"
    shutil.copytree(ROOT / "web", demo)
    for f in ("demo.js", "demo.css"):
        shutil.copy2(SITE / "demo" / f, demo / f)
    data = demo / "data"
    shutil.copytree(SITE / "demo-data", data)
    (data / "index.json").write_text(json.dumps(demo_index(data), separators=(",", ":")))
    sc = json.loads((ROOT / "server" / "scorecard.json").read_text())
    meter = json.loads((data / "api" / "meter.json").read_text())
    health = json.loads((data / "api" / "health.json").read_text())
    sc["now"] = {"cost_mode": meter.get("cost_mode"), "web_backend": health.get("web_backend"), "keystone": True}
    (data / "api" / "scorecard.json").write_text(json.dumps(sc))
    page = (demo / "index.html").read_text()
    page = page.replace("<title>Onset</title>", "<title>Onset demo</title>", 1)
    page = page.replace("</head>", '<link rel="stylesheet" href="demo.css?v=1">\n</head>', 1)
    i = page.index("<script src=")
    page = page[:i] + '<script src="demo.js?v=1"></script>\n' + page[i:]
    (demo / "index.html").write_text(page)


def build_landing(out: Path):
    sc = json.loads((ROOT / "server" / "scorecard.json").read_text())
    tok = {"DOWNLOAD": DOWNLOAD, "REPO": REPO, "HEADLINE_LABEL": html.escape(sc.get("headline_label", ""))}
    cards = []
    for h in sc.get("headline", []):
        s = h.get("source") or {}
        cards.append(f'<div class="stat"><b>{html.escape(h["big"])}</b><span>{html.escape(h["label"])}</span>'
                     f'<small>{html.escape(h.get("detail", ""))}</small>'
                     f'<a class="src" href="{REPO}/blob/main/{html.escape(s.get("doc", ""))}">'
                     f'{html.escape(s.get("doc", ""))} · {html.escape(s.get("section", ""))}</a></div>')
    tok["HEADLINE"] = "\n".join(cards)
    rows = []
    for p in sc.get("problems", []):
        s = p.get("source") or {}
        rows.append(f'<li><span class="pn">{html.escape(p["n"])}</span><div><h3>{html.escape(p["problem"])}</h3>'
                    f'<p>{html.escape(p["answer"])}</p></div><div class="pv"><b>{html.escape(p["big"])}</b>'
                    f'<span>{html.escape(p["unit"])}</span><small>{html.escape(p.get("vs", ""))}</small>'
                    f'<a class="src" href="{REPO}/blob/main/{html.escape(s.get("doc", ""))}">'
                    f'{html.escape(s.get("doc", ""))}{" · " + html.escape(s["section"]) if s.get("section") else ""}'
                    f'{" · " + html.escape(p["lane"]) if p.get("lane") else ""}</a></div></li>')
    tok["PROBLEMS"] = "\n".join(rows)
    page = (SITE / "index.html").read_text()
    page = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: tok[m.group(1)], page)
    (out / "index.html").write_text(page)
    for f in SITE.iterdir():
        if f.is_file() and f.name != "index.html" and f.suffix in (".css", ".js", ".svg", ".png", ".ico"):
            shutil.copy2(f, out / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "_site"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    build_landing(out)
    build_demo(out)
    (out / ".nojekyll").write_text("")
    n = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"built {out} ({n / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
