"""Build the Onset pitch deck: scripts/deck/deck.html -> docs/Onset_deck.pdf + docs/img/deck/slide-N.png.

  python3 scripts/deck/build.py                 # needs the Python playwright package + a Chromium
  python3 scripts/deck/build.py --frames        # first re-capture the demo frames (scripts/deck/frames/*.jpg)
                                                #   from the Pages demo (runs scripts/pages/build.py into a temp dir)

Chromium: Playwright's own, or set DECK_CHROMIUM to an executable; otherwise the newest headless shell in the
Playwright cache is used. Slides are 1920x1080 CSS px; the PDF is one slide per page with selectable text and the
Sora / Atkinson Hyperlegible fonts (OFL, scripts/deck/fonts) embedded. Every number on a slide is checked against its
cited doc by tests/test_deck.py.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT / "scripts" / "deck"
PDF = ROOT / "docs" / "Onset_deck.pdf"
PNG_DIR = ROOT / "docs" / "img" / "deck"
W, H = 1920, 1080


def _launch(p):
    exe = os.environ.get("DECK_CHROMIUM")
    if exe:
        return p.chromium.launch(executable_path=exe)
    try:
        return p.chromium.launch()
    except Exception:
        cache = Path.home() / "Library" / "Caches" / "ms-playwright"
        if not cache.exists():
            cache = Path.home() / ".cache" / "ms-playwright"
        found = sorted(glob.glob(str(cache / "chromium*" / "*" / "chrome-headless-shell")) +
                       glob.glob(str(cache / "chromium*" / "*" / "Chromium.app" / "Contents" / "MacOS" / "Chromium")))
        if not found:
            raise
        return p.chromium.launch(executable_path=found[-1])


HIDE_CHROME = """() => {
  const bar = document.getElementById('demoBar'); if (bar) bar.style.display = 'none';
  for (const el of document.querySelectorAll('body *')) if (getComputedStyle(el).position === 'fixed') el.style.visibility = 'hidden';
  const t = document.getElementById('transcript').closest('.card').getBoundingClientRect();
  const a = document.getElementById('answers').getBoundingClientRect();
  return {x: t.x - 12, y: t.y - 12 + scrollY, width: t.width + 24, height: (a.y - t.y) + 330};
}"""


def capture_frames(p, port: int = 8841):
    """Race / follow-up / open-mic frames from the Pages demo (recorded runs of the real app), cropped to the
    'What Arc hears' card and the top of the race. The demo bar, memory dock and meter (fixed overlays) are hidden."""
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp) / "site"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "pages" / "build.py"), "--out", str(site)], check=True)
        srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "-d", str(site)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        try:
            b = _launch(p)
            for name, wait_s in (("race", 10), ("followup", 12), ("openmic", 12)):
                pg = b.new_page(viewport={"width": 1280, "height": 1100}, device_scale_factor=1.5)
                pg.goto(f"http://localhost:{port}/demo/?play={name}")
                pg.wait_for_timeout(wait_s * 1000)
                clip = pg.evaluate(HIDE_CHROME)
                pg.wait_for_timeout(300)
                png = Path(tmp) / f"{name}.png"
                pg.screenshot(path=str(png), clip=clip)
                pg.close()
                out = DECK / "frames" / f"{name}.jpg"
                if sys.platform == "darwin":
                    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "85", str(png), "--out", str(out)],
                                   check=True, stdout=subprocess.DEVNULL)
                else:
                    from PIL import Image
                    Image.open(png).convert("RGB").save(out, quality=85)
                print("frame", out.relative_to(ROOT))
            b.close()
        finally:
            srv.terminate()


def build(p):
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    b = _launch(p)
    pg = b.new_page(viewport={"width": W, "height": H})
    pg.goto((DECK / "deck.html").as_uri())
    pg.evaluate("document.fonts.ready.then(() => true)")
    pg.wait_for_timeout(300)
    fonts = pg.evaluate("[...document.fonts].filter(f => f.status === 'loaded').map(f => f.family + ' ' + f.weight)")
    assert any("Sora" in f for f in fonts) and any("Atkinson" in f for f in fonts), fonts
    over = pg.evaluate("""() => [...document.querySelectorAll('.slide')].flatMap((s, i) => {
        const r = s.getBoundingClientRect();
        return [...s.querySelectorAll('*')].filter(e => { if (e.closest('.img')) return false; const q = e.getBoundingClientRect();
          return q.width && (q.right > r.right + 1 || q.bottom > r.bottom + 1 || q.left < r.left - 1); })
          .map(e => `slide ${i + 1}: ${e.tagName}.${e.className}`); })""")
    if over:
        print("WARNING overflow:", *over[:10], sep="\n  ")
    slides = pg.query_selector_all(".slide")
    for old in PNG_DIR.glob("slide-*.png"):
        old.unlink()
    for i, s in enumerate(slides, 1):
        s.screenshot(path=str(PNG_DIR / f"slide-{i}.png"))
    pg.emulate_media(media="print")
    pg.pdf(path=str(PDF), width=f"{W}px", height=f"{H}px", print_background=True,
           margin={"top": "0", "right": "0", "bottom": "0", "left": "0"}, prefer_css_page_size=True)
    b.close()
    print(f"{PDF.relative_to(ROOT)}: {len(slides)} slides, {PDF.stat().st_size / 1e6:.2f} MB; PNGs in {PNG_DIR.relative_to(ROOT)}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", action="store_true", help="re-capture the demo frames first")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        if a.frames:
            capture_frames(p)
        build(p)


if __name__ == "__main__":
    main()
