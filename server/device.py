"""Device lane: questions about, and requests for, this Mac. Router vertical "device" (server/router.py).

  files : "find my tax pdf", "where is the resume I edited last week", "open my notes about the trip",
          "show my recent downloads"  -> a Spotlight search (mdfind, -onlyin $HOME), newest first, top 5.
  music : "play some jazz", "play Taylor Swift", "pause the music", "next song"  -> Music.app (AppleScript); a term
          with no match in the library opens the Apple Music search for it.
  apps  : "open Safari", "launch Notes"  -> `open -a` for an app installed in /Applications (or the system's).

Conservative on purpose: only clearly device-directed phrasing matches ("who played Gandalf", "where is the Eiffel
tower", "open source licenses" do not). Every result is a card (router card path): its one-line `spoken` IS the
answer, no model and no web. Side effects (music, apps) run only on the FINAL endpoint, never while the speaker is
still talking (router.route raises Skip for them while speculating); a file search has none, so it may speculate.

Nothing leaves the Mac: subprocesses only (argv, never a shell), each with a short timeout. DEVICE=0 turns the lane
off; it is always off off macOS.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from . import config

ACTIONS = ("play", "pause", "resume", "next", "previous", "open_app")   # kinds with side effects


@dataclass(frozen=True)
class DeviceQ:
    kind: str                     # files | recent | play | pause | resume | next | previous | open_app
    term: str = ""                # file words / music search / app name
    ftype: str = ""               # pdf | image | spreadsheet | presentation | screenshot | folder | document | ""
    since_days: int = 0           # files changed in the last N days (0 = any time)
    folder: str = ""              # recent: Downloads | Desktop | Documents
    app: str = ""                 # open_app: the app bundle's path
    service: str = ""             # play: "spotify" when asked for

    @property
    def action(self) -> bool:
        return self.kind in ACTIONS

    def key(self) -> dict:
        return {"kind": self.kind, "term": self.term, "ftype": self.ftype, "since_days": self.since_days,
                "folder": self.folder, "service": self.service}


def enabled() -> bool:
    return config.DEVICE and sys.platform == "darwin"


# ------------------------------------------------------------------------------------------------- parse
_LEAD = re.compile(r"^(?:(?:hey|ok|okay) onset\s+|onset\s+)?(?:(?:can|could|would|will) you\s+|please\s+|"
                   r"i (?:want|need) (?:you )?to\s+|(?:let'?s|lets)\s+|help me\s+|go (?:and )?)*")
_TAIL = re.compile(r"\s+(?:please|for me|now|right now|on (?:my|this) (?:mac|computer|laptop))+$")
_FILE_WORDS = {
    "pdf": "pdf", "pdfs": "pdf",
    "photo": "image", "photos": "image", "picture": "image", "pictures": "image", "image": "image", "images": "image",
    "pic": "image", "pics": "image",
    "spreadsheet": "spreadsheet", "spreadsheets": "spreadsheet", "excel": "spreadsheet",
    "presentation": "presentation", "presentations": "presentation", "slides": "presentation", "deck": "presentation",
    "keynote": "presentation", "powerpoint": "presentation",
    "screenshot": "screenshot", "screenshots": "screenshot",
    "folder": "folder", "folders": "folder",
    "file": "", "files": "", "document": "document", "documents": "document", "doc": "document", "docs": "document",
}
_NAMED = {"resume", "cv", "notes", "note", "receipt", "receipts", "invoice", "invoices", "contract", "lease",
          "passport", "statement", "statements", "report", "essay", "thesis", "taxes", "tax", "w2", "1099",
          "recording", "recordings", "video", "videos", "download", "downloads"}
_DROP = {"my", "the", "a", "an", "all", "any", "some", "me", "i", "that", "which", "file", "files", "called", "named",
         "with", "about", "on", "of", "from", "in", "for", "this", "mac", "computer", "laptop", "edited", "made",
         "wrote", "written", "saved", "created", "changed", "modified", "downloaded", "worked", "opened", "was",
         "were", "last", "week", "month", "yesterday", "today", "recent", "recently", "latest", "new", "newest",
         "is", "are", "where", "did", "put", "it", "and", "or", "to", "document", "documents", "doc", "docs",
         "folder", "folders", "find", "show", "open", "get", "pull", "up", "search", "look", "locate"}
_SINCE = [(re.compile(r"\btoday\b"), 1), (re.compile(r"\byesterday\b"), 2),
          (re.compile(r"\b(?:last|this|past) week\b"), 7), (re.compile(r"\b(?:last|this|past) month\b"), 31),
          (re.compile(r"\b(?:recent|recently|latest|newest)\b"), 14)]
_RECENT = re.compile(r"^(?:show|open|find|list|what are|what's in|whats in)?\s*(?:me\s+)?(?:my|the)\s+"
                     r"(?:recent|latest|last|newest)\s+(downloads|desktop files|documents)$")
_FILE_VERB = re.compile(r"^(?:find|search(?: for)?|look for|look up|locate|show(?: me)?|open|pull up|get)\s+(.+)$")
_WHERE = re.compile(r"^(?:where(?:'s| is| are| did i (?:put|save))|where's)\s+(.+)$")

_CTRL = [
    (re.compile(r"^(?:pause|stop)(?: the)?(?: music| song| playback| it| playing)?$"), "pause"),
    (re.compile(r"^(?:resume|unpause|continue)(?: the)?(?: music| song| playback| playing)?$"), "resume"),
    (re.compile(r"^play$|^play (?:the )?music$|^keep playing$"), "resume"),
    (re.compile(r"^(?:next|skip)(?: (?:this|the))?(?: song| track)?$|^(?:play )?(?:the )?next (?:song|track)$|"
                r"^skip (?:this|it)$"), "next"),
    (re.compile(r"^(?:previous|go back)(?: (?:song|track))?$|^(?:play )?(?:the )?(?:previous|last) (?:song|track)$"),
     "previous"),
]
_PLAY = re.compile(r"^(?:play|put on|shuffle|listen to)\s+(?:me\s+)?(?:some\s+|a little\s+|a bit of\s+|the\s+)?(.+)$")
_NOT_PLAY = re.compile(r"\b(?:game|games|chess|role|part|roles|parts|video of|movie|trailer)\b")
_OPEN_APP = re.compile(r"^(?:open|launch|start|run|bring up|switch to)\s+(?:the\s+|my\s+)?(.+?)(?:\s+app)?$")

_APP_DIRS = ["/Applications", "/System/Applications", "/System/Applications/Utilities", "/Applications/Utilities",
             str(Path.home() / "Applications")]
_apps_cache: tuple[float, dict] | None = None
_done: dict = {}                     # action key -> (expires, future of its result): see docs()


def installed_apps() -> dict:
    """lowercased app name -> .app path (cached 60 s)."""
    global _apps_cache
    if _apps_cache and _apps_cache[0] > time.monotonic():
        return _apps_cache[1]
    out: dict = {}
    for d in _APP_DIRS:
        try:
            for p in Path(d).iterdir():
                if p.suffix == ".app":
                    out.setdefault(p.stem.lower(), str(p))
        except OSError:
            continue
    _apps_cache = (time.monotonic() + 60, out)
    return out


def find_app(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip().lower())
    if not name or len(name) < 2:
        return ""
    apps = installed_apps()
    if name in apps:
        return apps[name]
    alias = {"chrome": "google chrome", "settings": "system settings", "preferences": "system settings",
             "vs code": "visual studio code", "vscode": "visual studio code", "word": "microsoft word",
             "excel": "microsoft excel", "powerpoint": "microsoft powerpoint", "outlook": "microsoft outlook",
             "teams": "microsoft teams", "app store": "app store", "facetime": "facetime"}.get(name)
    if alias and alias in apps:
        return apps[alias]
    hits = [k for k in apps if k.startswith(name + " ") or k == name]
    return apps[hits[0]] if len(hits) == 1 else ""


def _norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[?!.,;:]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _LEAD.sub("", t).strip()
    t = _TAIL.sub("", t).strip()
    return t


def _file_q(rest: str, my: bool, where: bool) -> DeviceQ | None:
    words = rest.split()
    ftype = ""
    has_file_word = False
    for w in words:
        if w in _FILE_WORDS:
            has_file_word = True
            ftype = ftype or _FILE_WORDS[w]
        elif w in _NAMED:
            has_file_word = True
    if not has_file_word and not (my and not where):
        return None
    since = 0
    for rx, days in _SINCE:
        if rx.search(rest):
            since = days
            break
    terms = [w for w in words if w not in _DROP and w not in _FILE_WORDS]
    if not terms and not ftype:
        return None
    return DeviceQ("files", " ".join(terms)[:60], ftype, since)


def parse(text: str) -> DeviceQ | None:
    if not enabled() or not text:
        return None
    t = _norm(text)
    if not t or len(t.split()) > 14:
        return None
    for rx, kind in _CTRL:
        if rx.match(t):
            return DeviceQ(kind)
    m = _RECENT.match(t)
    if m:
        folder = {"downloads": "Downloads", "desktop files": "Desktop", "documents": "Documents"}[m.group(1)]
        return DeviceQ("recent", folder=folder, since_days=14)
    m = _PLAY.match(t)
    if m and not _NOT_PLAY.search(t):
        term = m.group(1)
        service = ""
        sm = re.search(r"\s+(?:on|in|from|with) (spotify|apple music|music|itunes)$", term)
        if sm:
            service = "spotify" if sm.group(1) == "spotify" else ""
            term = term[: sm.start()].strip()
        term = re.sub(r"\s+(?:music|songs?|playlist)$", lambda x: x.group(0) if term.count(" ") == 0 else "", term)
        if term:
            return DeviceQ("play", term[:60], service=service)
    m = _WHERE.match(t)
    if m:
        rest = m.group(1)
        return _file_q(rest, " my " in f" {rest} ", True)
    m = _OPEN_APP.match(t)
    if m and not re.search(r"\bmy\b", t):
        app = find_app(m.group(1))
        if app:                                        # "open notes" is the Notes app; "open my notes" is files
            return DeviceQ("open_app", Path(app).stem, app=app)
    m = _FILE_VERB.match(t)
    if m:
        rest = m.group(1)
        my = " my " in f" {rest} "
        q = _file_q(rest, my, False)
        if q:
            return q
    return None


# ------------------------------------------------------------------------------------------------- run
async def _run(*argv: str, timeout: float = 1.5) -> tuple[int, str]:
    """Run argv (no shell); (returncode, stdout). Tests replace this."""
    p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
    except asyncio.TimeoutError:
        try:
            p.kill()
        except ProcessLookupError:
            pass
        raise
    return p.returncode or 0, out.decode("utf-8", "replace")


def _esc(s: str) -> str:
    """A value inside a Spotlight query string literal."""
    return re.sub(r'["\\*]', "", s).strip()


_TYPES = {"pdf": 'kMDItemContentType == "com.adobe.pdf"',
          "image": 'kMDItemContentTypeTree == "public.image"',
          "spreadsheet": 'kMDItemContentTypeTree == "public.spreadsheet"',
          "presentation": 'kMDItemContentTypeTree == "public.presentation"',
          "screenshot": "kMDItemIsScreenCapture == 1",
          "folder": 'kMDItemContentType == "public.folder"',
          "document": '(kMDItemContentTypeTree == "public.content" && kMDItemContentType != "public.folder")'}


def spotlight_query(q: DeviceQ) -> str:
    parts = []
    for w in q.term.split()[:5]:
        w = _esc(w)
        if w:
            parts.append(f'(kMDItemDisplayName == "*{w}*"cd || kMDItemTextContent == "{w}"cdw)')
    if q.ftype in _TYPES:
        parts.append(_TYPES[q.ftype])
    if q.since_days:
        parts.append(f"kMDItemFSContentChangeDate >= $time.today(-{q.since_days})")
    return " && ".join(parts) or 'kMDItemDisplayName == "*"'


_SKIP = re.compile(r"/(?:Library|\.[^/]+|node_modules|__pycache__|site-packages|\.Trash)/|\.app/|\.photoslibrary/|"
                   r"\.musiclibrary/|/Containers/")


def _home() -> Path:
    return Path(os.environ.get("DEVICE_HOME") or Path.home())


def _where(p: Path) -> str:
    home = _home()
    try:
        rel = p.parent.relative_to(home)
        return str(rel) if str(rel) != "." else "your home folder"
    except ValueError:
        return str(p.parent)


def _ago(ts: float) -> str:
    d = time.time() - ts
    if d < 3600:
        return "just now" if d < 120 else f"{int(d // 60)} minutes ago"
    if d < 86400:
        return f"{int(d // 3600)} hours ago" if d >= 7200 else "an hour ago"
    if d < 2 * 86400:
        return "yesterday"
    if d < 30 * 86400:
        return f"{int(d // 86400)} days ago"
    return time.strftime("%b %-d, %Y", time.localtime(ts))


async def find_files(q: DeviceQ, n: int = 5) -> list[dict]:
    home = _home()
    if q.kind == "recent":
        folder = home / q.folder
        try:
            items = [p for p in folder.iterdir() if not p.name.startswith(".")]
        except OSError:
            return []
        paths = items
    else:
        try:
            _, out = await _run("mdfind", "-onlyin", str(home), spotlight_query(q), timeout=config.DEVICE_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError):
            return []
        paths = [Path(x) for x in out.splitlines()[:400] if x.strip()]
        if not paths:
            # Spotlight off, still indexing, or its index lost (a full disk drops it): walk the usual folders by name
            paths = await asyncio.to_thread(_walk, q, config.DEVICE_TIMEOUT_S)
    rows = []
    terms = q.term.split()
    for p in paths:
        s = str(p)
        if _SKIP.search(s[len(str(home)):] + "/") and q.kind != "recent":
            continue
        if p.suffix.lower() in _CODE and q.kind != "recent":
            continue                                   # source files match words too ("resume-sweep.test.ts")
        try:
            st = p.stat()
        except OSError:
            continue
        name_hit = sum(t in p.name.lower() for t in terms)
        rows.append((name_hit, st.st_mtime, p))
    rows.sort(key=lambda r: (-r[0], -r[1]))
    return [{"path": str(p), "name": p.name, "where": _where(p), "modified": _ago(m), "mtime": m}
            for _, m, p in rows[:n]]


_EXT = {"pdf": {".pdf"}, "image": {".jpg", ".jpeg", ".png", ".heic", ".gif", ".webp", ".tiff"},
        "spreadsheet": {".xlsx", ".xls", ".csv", ".numbers"}, "presentation": {".pptx", ".ppt", ".key"},
        "document": {".pdf", ".doc", ".docx", ".pages", ".txt", ".md", ".rtf", ".odt"}}
_CODE = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".py", ".pyc", ".swift", ".json", ".jsonl", ".go", ".rs", ".c", ".h",
         ".cc", ".cpp", ".java", ".kt", ".rb", ".sh", ".yml", ".yaml", ".toml", ".lock", ".log", ".map", ".css",
         ".scss", ".html", ".vue", ".o", ".so", ".dylib", ".pyi", ".ipynb"}
_WALK_DIRS = ("Desktop", "Documents", "Downloads", "Pictures", "Movies", "Music")


def _walk(q: DeviceQ, budget_s: float) -> list[Path]:
    """Filename match over the user's usual folders (breadth-first, time-boxed); Spotlight's fallback."""
    home, t_end = _home(), time.monotonic() + budget_s
    terms, cutoff = q.term.split(), (time.time() - q.since_days * 86400) if q.since_days else 0
    exts = _EXT.get(q.ftype)
    queue, out = [(home / d, 0) for d in _WALK_DIRS], []
    while queue and time.monotonic() < t_end and len(out) < 200:
        d, depth = queue.pop(0)
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            n = e.name
            if n.startswith(".") or n in ("node_modules", "Library", "__pycache__") or n.endswith((".app", "library")):
                continue
            low = n.lower()
            try:
                is_dir = e.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir and depth < 5:
                queue.append((Path(e.path), depth + 1))
            if terms and not all(t in low for t in terms):
                continue
            if q.ftype == "folder" and not is_dir:
                continue
            if q.ftype == "screenshot" and not low.startswith(("screenshot", "screen shot")):
                continue
            if exts and (is_dir or Path(low).suffix not in exts):
                continue
            if cutoff:
                try:
                    if e.stat().st_mtime < cutoff:
                        continue
                except OSError:
                    continue
            out.append(Path(e.path))
    return out


_MUSIC_PLAY = """on run argv
  set q to item 1 of argv
  tell application "Music"
    set hits to (search library playlist 1 for q)
    if (count of hits) is 0 then return "none"
    play item 1 of hits
    set t to item 1 of hits
    return (name of t) & " by " & (artist of t)
  end tell
end run"""
_MUSIC_CTRL = {"pause": "pause", "resume": "play", "next": "next track", "previous": "previous track"}


async def _music(kind: str, term: str = "", service: str = "") -> tuple[bool, str, str]:
    """(ok, spoken, what) for a music request."""
    if kind == "play":
        if service == "spotify" and find_app("spotify"):
            await _run("open", f"spotify:search:{quote(term)}", timeout=3.0)
            return True, f"Opened Spotify search for {term}.", "searched"
        try:
            rc, out = await _run("osascript", "-e", _MUSIC_PLAY, term, timeout=config.DEVICE_ACTION_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError):
            rc, out = 1, ""
        out = out.strip()
        if rc == 0 and out and out != "none":
            return True, f"Playing {out} in Music.", "played"
        await _run("open", f"music://music.apple.com/search?term={quote(term)}", timeout=3.0)
        return True, f"{term[:1].upper() + term[1:]} isn't in your Music library, so I opened Apple Music search " \
                     f"for it.", "searched"
    cmd = _MUSIC_CTRL[kind]
    script = f'if application "Music" is running then tell application "Music" to {cmd}'
    try:
        rc, _ = await _run("osascript", "-e", script, timeout=config.DEVICE_ACTION_TIMEOUT_S)
    except (asyncio.TimeoutError, OSError):
        rc = 1
    said = {"pause": "Paused the music.", "resume": "Playing.", "next": "Skipped to the next song.",
            "previous": "Back to the previous song."}[kind]
    return rc == 0, said if rc == 0 else "I couldn't reach the Music app.", kind


async def run(q: DeviceQ):
    """The device card for q (a verticals.Card)."""
    from .verticals import Card
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if q.kind in ("files", "recent"):
        files = await find_files(q)
        what = q.term or {"pdf": "PDFs", "image": "photos", "spreadsheet": "spreadsheets",
                          "presentation": "presentations", "screenshot": "screenshots", "folder": "folders",
                          "document": "documents"}.get(q.ftype, "files")
        if q.kind == "recent":
            one = q.folder.lower().rstrip("s") if q.folder == "Downloads" else f"file in {q.folder}"
            spoken = (f"Your latest {one} is {files[0]['name']}, from {files[0]['modified']}." if files
                      else f"Your {q.folder} folder is empty.")
        elif not files:
            spoken = f"I couldn't find any files matching “{what}” on this Mac."
        elif len(files) == 1:
            spoken = f"Found {files[0]['name']} in {files[0]['where']}, changed {files[0]['modified']}."
        else:
            spoken = f"Found {len(files)} files. The best match is {files[0]['name']} in {files[0]['where']}, " \
                     f"changed {files[0]['modified']}."
        detail = "\n".join(f"{f['name']} · {f['where']} · {f['modified']}" for f in files)
        return Card(vertical="device", title="On this Mac", spoken=spoken, detail=detail, source="this Mac",
                    as_of=stamp, args=q.key(), live=False,
                    data={"kind": "files", "files": [{k: f[k] for k in ("path", "name", "where", "modified")}
                                                     for f in files]}), files
    if q.kind == "open_app":
        try:
            rc, _ = await _run("open", "-a", q.app, timeout=config.DEVICE_ACTION_TIMEOUT_S)
        except (asyncio.TimeoutError, OSError):
            rc = 1
        spoken = f"Opened {q.term}." if rc == 0 else f"I couldn't open {q.term}."
        return Card(vertical="device", title=q.term, spoken=spoken, source="this Mac", as_of=stamp, args=q.key(),
                    live=False, data={"kind": "opened", "ok": rc == 0, "app": q.app}), []
    ok, spoken, what = await _music(q.kind, q.term, q.service)
    return Card(vertical="device", title="Music", spoken=spoken, source="this Mac", as_of=stamp, args=q.key(),
                live=False, data={"kind": what, "ok": ok}), []


async def docs(q: DeviceQ) -> list:
    """Docs for the router: the card first, then one doc per file found (clickable rows in the UI)."""
    from .lanes import Doc
    if q.action:
        # one request, one action: both race modes (and a re-asked endpoint) route the same final question; the
        # second within a few seconds gets the first one's result instead of playing / opening again
        k = repr(sorted(q.key().items()))
        hit = _done.get(k)
        if hit and hit[0] > time.monotonic():
            card, files = await hit[1]
        else:
            fut = asyncio.ensure_future(run(q))
            _done[k] = (time.monotonic() + 8.0, fut)
            card, files = await fut
    else:
        card, files = await run(q)
    d = card.doc()
    d.lane = "device"
    d.meta["device"] = {"kind": card.data.get("kind"), **({"ok": card.data["ok"]} if "ok" in card.data else {})}
    out = [d]
    for f in files:
        out.append(Doc(id=f"device:file:{f['path']}", title=f["name"], text=f"{f['where']} · changed {f['modified']}",
                       url="file://" + quote(f["path"]), lane="device", source="this Mac", score=0.9,
                       meta={"tier": "card", "device": {"kind": "file", "path": f["path"]}}))
    return out


def fits(args: dict, question: str) -> bool:
    q = parse(question)
    return q is not None and q.key() == args
