"""The consumer page (web/index.html + onset.js): only the product, and the automation contract the release smoke uses."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text()
JS = (ROOT / "web" / "onset.js").read_text()


def test_consumer_page_has_no_engine_internals():
    low = HTML.lower()
    for word in ("baseline", "race", "speculat", "timeline", "query frame", "replay a question", "arc tagger",
                 "a/b", "fonts.googleapis", "eval/audio"):
        assert word not in low, word


def test_release_smoke_contract():
    # macapp/Sources/Automation.swift drives these
    for needle in ('id="typeBox"', 'id="typeForm"'):
        assert needle in HTML
    for needle in ("window.__hs", "get ready()", "replay", "mic: {", "stats:", "__hsNotice"):
        assert needle in JS, needle


def test_runs_onset_only_and_opens_files_through_the_app():
    assert 'MODES = [MODE]' in JS and 'MODE = "headstart"' in JS
    assert 't: "open", path' in JS
