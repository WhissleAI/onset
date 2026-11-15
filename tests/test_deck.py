"""Pitch deck (scripts/deck/deck.html -> docs/Onset_deck.pdf, docs/img/deck/).
- every number on a slide sits in an element with data-src="<doc>" and appears verbatim in that doc
  (two derived numbers, 0.49 s and 27%, are recomputed from the doc instead);
- no digits outside data-src, except names, source lines and page numbers;
- 5 slides, the PDF is built (5 pages, < 15 MB), linked from the README and the landing page; no paths or keys.
Run: .venv/bin/python -m pytest -q tests/test_deck.py
"""
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "scripts" / "deck" / "deck.html"
EXEMPT = {"src", "pg", "nm"}                      # source lines, page numbers, product names with digits
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?(?:/\d+)?%?")


class _Spans(HTMLParser):
    """(text, data-src or None, exempt?) for every text node inside a <section class="slide">."""
    VOID = {"img", "br", "meta", "link", "path", "rect", "input"}

    def __init__(self):
        super().__init__()
        self.stack, self.out, self.slides = [], [], 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "section" and "slide" in (a.get("class") or "").split():
            self.slides += 1
        if tag in self.VOID:
            return
        self.stack.append((tag, a.get("data-src"), set((a.get("class") or "").split())))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        while self.stack:
            t, _, _ = self.stack.pop()
            if t == tag:
                break

    def handle_data(self, data):
        if not any(t == "section" for t, _, _ in self.stack) or any(t in ("style", "script", "title") for t, _, _ in self.stack):
            return
        if not data.strip():
            return
        src = next((s for _, s, _ in reversed(self.stack) if s), None)
        exempt = any(c & EXEMPT for _, _, c in self.stack)
        self.out.append((data.strip(), src, exempt))


def _spans():
    p = _Spans()
    p.feed(DECK.read_text())
    return p


def _doc(path: str) -> str:
    return (ROOT / path).read_text().replace("**", "")


def _derived_ok(n: str, doc: str) -> bool:
    """Numbers that are rounded or computed from the doc, not quoted."""
    if n == "0.49":           # 486 ms paired median over the 100 questions, in seconds
        return "**486 ms** (76/100)" in (ROOT / "docs/RETRIEVAL.md").read_text() and round(486 / 1000, 2) == 0.49
    if n == "27%":            # 1 - 46/63 credits
        return "Total: 46 credits for Onset" in doc and "63 for plain search" in doc and round((1 - 46 / 63) * 100) == 27
    return False


def test_five_slides():
    assert _spans().slides == 5


def _numbered():
    return [(t, s) for t, s, ex in _spans().out if s and NUM.search(t)]


@pytest.mark.parametrize("text,src", _numbered(), ids=lambda x: x[:40] if isinstance(x, str) else None)
def test_every_number_is_in_its_cited_doc(text, src):
    doc = _doc(src)
    for n in NUM.findall(text):
        if _derived_ok(n, doc):
            continue
        assert n in doc, f"{n!r} (in {text!r}) is not in {src}"


def test_no_unsourced_digits():
    bad = [t for t, s, ex in _spans().out if not s and not ex and re.search(r"\d", t)]
    assert not bad, f"digits outside a data-src element: {bad}"


def test_headline_numbers_present():
    txt = " ".join(t for t, _, _ in _spans().out)
    for n in ("0.49 s", "76/100", "27%", "0.46 vs 0.63", "80/82", "79/82", "0.64 → 0.36", "58/58", "0.369 vs 0.464",
              "ECE 0.034", "AUC 0.72", "65/69 → 6/69", "0.40 → 0.84", "0.77 → 0.84", "32/34", "2/34", "22/26", "13/26",
              "16/16", "5/16", "86.8%", "49.1%"):
        assert n in txt, n


def test_copy_rules():
    raw = DECK.read_text()
    vis = " ".join(t for t, _, _ in _spans().out)
    assert "LLM" not in vis and "Outpost" not in raw
    assert "signed and notarized" in vis                    # MACAPP.md "Release": DMG notarized and stapled
    assert vis.count("Recorded run of the real app") == 3
    assert "/Users/" not in raw and "sk-" not in raw and "tvly-" not in raw


def test_assets_exist_and_are_local():
    raw = DECK.read_text()
    refs = re.findall(r'src="(frames/[^"]+)"', raw) + re.findall(r"url\((fonts/[^)]+)\)", raw)
    assert len(refs) >= 8
    for ref in refs:
        assert (DECK.parent / ref).exists(), ref
    assert "http://" not in raw.split("</style>")[0] and "https://" not in raw.split("</style>")[0]
    assert (DECK.parent / "fonts" / "OFL-NOTICE.txt").exists()


def test_pdf_and_links():
    pdf = ROOT / "docs" / "Onset_deck.pdf"
    assert pdf.exists() and pdf.stat().st_size < 15e6
    data = pdf.read_bytes()
    assert data[:5] == b"%PDF-" and len(re.findall(rb"/Type\s*/Page(?![s\w])", data)) == 5
    assert sorted(p.name for p in (ROOT / "docs" / "img" / "deck").glob("slide-*.png")) == [f"slide-{i}.png" for i in range(1, 6)]
    assert "docs/Onset_deck.pdf" in (ROOT / "README.md").read_text()
    site = (ROOT / "site" / "index.html").read_text()
    assert site.count("Slides (PDF)") == 2 and "/blob/main/docs/Onset_deck.pdf" in site
