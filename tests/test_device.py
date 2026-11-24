"""Device lane (server/device.py): parse is conservative, actions wait for the endpoint and run once, file results are
cards with clickable rows. Every subprocess is mocked: no Music, no apps, no Spotlight."""
import asyncio
import time

import pytest

from server import answer, config, device, lanes, router
from server.frame import Frame


@pytest.fixture
def dev(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DEVICE", True)
    monkeypatch.setattr(config, "ROUTER", True)
    monkeypatch.setattr(device.sys, "platform", "darwin")
    monkeypatch.setattr(device, "_apps_cache", (time.monotonic() + 60, {
        "safari": "/Applications/Safari.app", "notes": "/System/Applications/Notes.app",
        "google chrome": "/Applications/Google Chrome.app", "spotify": "/Applications/Spotify.app"}))
    monkeypatch.setenv("DEVICE_HOME", str(tmp_path))
    device._done.clear()
    calls = []

    async def fake_run(*argv, timeout=1.5):
        calls.append(argv)
        if argv[0] == "mdfind":
            return 0, ""
        if argv[0] == "osascript" and "search library" in argv[2]:
            return 0, "Kind of Blue by Miles Davis\n" if "blue" in argv[3] else "none\n"
        return 0, ""
    monkeypatch.setattr(device, "_run", fake_run)
    return calls, tmp_path


@pytest.mark.parametrize("text,kind,term", [
    ("find my tax pdf", "files", "tax"),
    ("where is the resume I edited last week", "files", "resume"),
    ("open my notes about the trip", "files", "notes trip"),
    ("show my recent downloads", "recent", ""),
    ("play some jazz", "play", "jazz"),
    ("Play Taylor Swift.", "play", "taylor swift"),
    ("pause the music", "pause", ""),
    ("next song", "next", ""),
    ("skip this song", "next", ""),
    ("open Safari", "open_app", "Safari"),
    ("launch Notes", "open_app", "Notes"),
    ("open notes", "open_app", "Notes"),
    ("can you open chrome please", "open_app", "Google Chrome"),
])
def test_parse_positive(dev, text, kind, term):
    q = device.parse(text)
    assert q is not None and q.kind == kind and q.term == term


@pytest.mark.parametrize("text", [
    "who played Gandalf", "who played in the super bowl", "where is the Eiffel tower", "open source licenses",
    "what is the capital of France", "find a good restaurant near me", "play chess", "where is my car",
    "how do I open a pdf", "open the door", "what are five good books to read", "find the best pizza in town",
])
def test_parse_negative(dev, text):
    assert device.parse(text) is None


def test_off_outside_macos(dev, monkeypatch):
    monkeypatch.setattr(device.sys, "platform", "linux")
    assert device.parse("play some jazz") is None
    monkeypatch.setattr(device.sys, "platform", "darwin")
    monkeypatch.setattr(config, "DEVICE", False)
    assert device.parse("play some jazz") is None


def test_spotlight_query_is_escaped(dev):
    q = device.DeviceQ("files", 'tax" || kMDItemFSName == "*', "pdf", 7)
    s = device.spotlight_query(q)
    assert '\\' not in s and s.count('"') % 2 == 0 and "com.adobe.pdf" in s and "$time.today(-7)" in s


def _F(text):
    return Frame(words=text.split())


def test_router_plans_device_first(dev):
    p = router.plan(_F("play some jazz"))
    assert p.lane == "card" and p.vertical == "device" and p.skip_web


def test_action_never_runs_while_speculating(dev):
    calls, _ = dev
    web_called = []

    async def web():
        web_called.append(1)
        return lanes.WebResult([])
    with pytest.raises(lanes.Skip):
        asyncio.run(router.route("", _F("play some jazz"), final=False, meter={}, web_fn=web))
    assert not calls and not web_called


def test_action_runs_once_at_endpoint(dev):
    calls, _ = dev

    async def web():
        raise AssertionError("no web for the device lane")

    async def both():
        a = await router.route("", _F("play kind of blue"), final=True, meter={}, web_fn=web)
        b = await router.route("", _F("play kind of blue"), final=True, meter={}, web_fn=web)   # the other race mode
        return a, b
    a, b = asyncio.run(both())
    assert len([c for c in calls if c[0] == "osascript"]) == 1
    card = a[0].meta["card"]
    assert card["spoken"] == "Playing Kind of Blue by Miles Davis in Music." and a[0].meta["device"]["kind"] == "played"
    assert b[0].meta["card"]["spoken"] == card["spoken"]


def test_play_not_in_library_opens_search(dev):
    calls, _ = dev
    q = device.parse("play some jazz")
    d = asyncio.run(device.docs(q))
    assert any(c[0] == "open" and c[1].startswith("music://music.apple.com/search?term=jazz") for c in calls)
    assert "Apple Music search" in d[0].meta["card"]["spoken"]


def test_open_app_uses_installed_path(dev):
    calls, _ = dev
    d = asyncio.run(device.docs(device.parse("open safari")))
    assert ("open", "-a", "/Applications/Safari.app") in calls and d[0].meta["card"]["spoken"] == "Opened Safari."


def test_files_walk_fallback_and_rows(dev):
    calls, home = dev
    (home / "Documents" / "taxes").mkdir(parents=True)
    (home / "Documents" / "taxes" / "2025_tax_return.pdf").write_text("x")
    (home / "Documents" / "tax_notes.txt").write_text("x")
    (home / "Desktop").mkdir()
    (home / "Desktop" / "tax-script.py").write_text("x")
    d = asyncio.run(device.docs(device.parse("find my tax pdf")))
    assert calls[0][0] == "mdfind"                        # Spotlight first; its empty answer falls back to the walk
    assert d[0].meta["card"]["spoken"].startswith("Found 2025_tax_return.pdf in Documents/taxes")
    rows = d[1:]
    assert [r.title for r in rows] == ["2025_tax_return.pdf"]
    assert rows[0].meta["device"] == {"kind": "file", "path": str(home / "Documents" / "taxes" / "2025_tax_return.pdf")}
    assert rows[0].url.startswith("file://") and rows[0].brief()["device"]["kind"] == "file"


def test_no_files_says_so(dev):
    d = asyncio.run(device.docs(device.parse("find my passport")))
    assert d[0].meta["card"]["spoken"] == "I couldn't find any files matching “passport” on this Mac."
    assert len(d) == 1


def test_card_answers_without_model(dev):
    d = asyncio.run(device.docs(device.parse("pause the music")))
    assert router.pick_card("pause the music", d) is not None
    assert router.pick_card("play some jazz", d) is None

    async def go():
        s = answer.Stream("pause the music", d)
        for _ in range(50):
            if s.done:
                break
            await asyncio.sleep(0.01)
        return s
    s = asyncio.run(go())
    assert s.by == "card" and s.text().startswith("Paused the music.")


def test_slow_spotlight_falls_back_to_the_folder_walk(monkeypatch, tmp_path):
    """Spotlight timing out under load used to return "no files" (installed 0.2.3: "find my resume" found nothing)."""
    import asyncio as _a
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Downloads" / "Jane Resume.pdf").write_text("x")
    (tmp_path / "Desktop" / "work" / "repo" / "recipes" / "x").mkdir(parents=True)
    (tmp_path / "Desktop" / "work" / "repo" / "recipes" / "x" / "RESUME_PLAN.md").write_text("x")
    (tmp_path / "Desktop" / "work" / "repo" / "recipes" / "x" / "copy").mkdir()
    monkeypatch.setenv("DEVICE_HOME", str(tmp_path))

    async def slow(*a, **k):
        raise _a.TimeoutError
    monkeypatch.setattr(device, "_run", slow)
    rows = _a.run(device.find_files(device.DeviceQ("files", "resume", "", 0, "", "", "")))
    assert rows and rows[0]["name"] == "Jane Resume.pdf"          # the document first, the project file after
    assert [r["name"] for r in rows].count("RESUME_PLAN.md") <= 1
