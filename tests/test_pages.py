"""GitHub Pages build (scripts/pages/build.py): recorded runs are complete, clean, and wired into the real UI."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build(tmp_path):
    out = tmp_path / "_site"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "pages" / "build.py"), "--out", str(out)], check=True,
                   capture_output=True)
    return out


def test_build_wires_demo_transport_before_app(tmp_path):
    out = _build(tmp_path)
    page = (out / "demo" / "index.html").read_text()
    assert page.index("demo.js") < page.index("app.js")
    assert "demo.css" in page
    landing = (out / "index.html").read_text()
    assert "{{" not in landing
    assert "releases/latest" in landing


def test_every_scenario_has_a_run_and_its_audio(tmp_path):
    out = _build(tmp_path)
    data = out / "demo" / "data"
    idx = json.loads((data / "index.json").read_text())
    spec = json.loads((ROOT / "scripts" / "pages" / "scenarios.json").read_text())
    assert [s["id"] for s in idx["scenarios"]] == [s["id"] for s in spec["scenarios"]]
    for name in idx["audio"]:
        assert (data / "audio" / (Path(name).stem + ".mp3")).exists(), name
    for sc in idx["scenarios"]:
        rec = json.loads((data / "runs" / f"{sc['id']}.json").read_text())
        ts = [x["e"].get("t") for x in rec["events"]]
        assert "run_done" in ts, sc["id"]
        assert "answer" in ts, sc["id"]
        ats = [x["at"] for x in rec["events"]]
        assert ats == sorted(ats), sc["id"]


def test_published_files_are_clean():
    sys.path.insert(0, str(ROOT / "scripts" / "pages"))
    from sanitize import check
    for p in (ROOT / "site").rglob("*"):
        if p.is_file() and p.suffix in (".json", ".html", ".js", ".css"):
            txt = p.read_text()
            check(txt)
            assert not re.search(r'"rid"\s*:', txt) or p.suffix != ".json", p
            assert '"client"' not in txt, p
