"""Onset for Windows: the app shell (the Windows twin of macapp/Sources/*.swift).

One window (Edge WebView2 through pywebview) over the same local engine the Mac app runs:
- checks the models with scripts/fetch_models.py (status / fetch, the public release manifest), shows the first-run
  download, then starts macapp/server_main.py on a free localhost port and waits for /api/health warm;
- restarts the server if it crashes (at most 3 times in 2 minutes), stops it when the window closes;
- loads the consumer page (web/index.html) and answers its bridge (window.pywebview.api.message: settings, log, mic,
  open a found file);
- Settings: "Help improve Onset" + Advanced (your own service keys, kept in Windows Credential Manager; speech);
- checks GitHub releases (WhissleAI/onset) once a day for a newer Onset-<v>-win-x64-setup.exe.

Run from the install folder:  python\\pythonw.exe -I -B winapp\\onset_win.py   (the Start-menu shortcut does this)
Dev run from a checkout:      python winapp/onset_win.py --dev    (system Python with winapp/requirements-win.txt)
Data: %LOCALAPPDATA%\\Onset\\{models,data,logs,settings.json}. Build: docs/BUILD_WINDOWS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the app folder (install dir, or a checkout)


def _conf() -> dict:
    out = {}
    for line in (ROOT / "macapp" / "app.conf").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


CONF = _conf()
APP_NAME = CONF.get("APP_NAME", "Onset")
VERSION = CONF.get("APP_VERSION", "0.0.0")
APP_ID = CONF.get("BUNDLE_ID", "ai.whissle.onset")
REPO = "WhissleAI/onset"
# ONSET_HOME overrides the data folder (tests, a second profile)
BASE = Path(os.environ["ONSET_HOME"]) if os.environ.get("ONSET_HOME") else \
    Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / APP_NAME
MODELS, DATA, LOGS = BASE / "models", BASE / "data", BASE / "logs"
SETTINGS_FILE = BASE / "settings.json"
KEYS = {"ANTHROPIC_API_KEY": "Anthropic (longer answers written by Claude)",
        "TAVILY_API_KEY": "Tavily (web search)",
        "TELEM_API_KEY": "Telem (web search router)",
        "DEEPGRAM_API_KEY": "Deepgram (cloud speech recognition)"}
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0       # CREATE_NO_WINDOW: no console flashes


def log(msg: str):
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(LOGS / "app.log", "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------ settings + keys
def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(s: dict):
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    os.replace(tmp, SETTINGS_FILE)


def get_key(k: str) -> str | None:
    try:
        import keyring
        return keyring.get_password(APP_NAME, k) or None
    except Exception as e:                               # no backend: keys simply unavailable
        if not getattr(get_key, "warned", False):
            get_key.warned = True
            log(f"keyring unavailable: {type(e).__name__}")
        return None


def set_key(k: str, v: str | None):
    import keyring
    if v:
        keyring.set_password(APP_NAME, k, v)
    else:
        try:
            keyring.delete_password(APP_NAME, k)
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------ the engine
def python_exe() -> str:
    """The console-less server needs python.exe (pythonw has no stdio to redirect on some builds)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def manifest() -> Path | None:
    rel = ROOT / "models" / "manifest.release.json"
    return rel if rel.exists() else None


def free_port(preferred: int | None) -> int:
    for p in ([preferred] if preferred else []) + [0]:
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", p))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("no free port")


def server_env(settings: dict) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONSTARTUP", *KEYS)}
    for k in KEYS:
        v = get_key(k)
        if v:
            env[k] = v
    env.update({
        "APP_NAME": APP_NAME, "MODELS_DIR": str(MODELS), "HEADSTART_DATA_DIR": str(DATA),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
        "TOKENIZERS_PARALLELISM": "false", "ONNX_RUNTIME": "1",
        "CACHE_PERSIST": env.get("CACHE_PERSIST", "1"), "CACHE_PATH": str(DATA / "search_cache.json"),
        "MEM_DIR": str(DATA / "memory"), "KNOWLEDGE_DIR": str(MODELS / "index-data" / "knowledge"),
        "TRAIN_COLLECT": "1" if settings.get("trainCollect", True) else "0",
    })
    # web: Tavily with its key, else Telem with its key, else DuckDuckGo (free, no key)
    if "WEB_BACKEND" not in os.environ:
        env["WEB_BACKEND"] = "tavily" if "TAVILY_API_KEY" in env else "telem" if "TELEM_API_KEY" in env else "duckduckgo"
    if env["WEB_BACKEND"] == "tavily" and "TELEM_API_KEY" in env:
        env.setdefault("WEB_FAILOVER", "telem")
    ear = settings.get("ear", "auto")
    env["EAR"] = "flux" if ear == "flux" or (ear == "auto" and "DEEPGRAM_API_KEY" in env) else "arc"
    return env


class Engine:
    """Model check / download, then the server: the same state machine as macapp/Sources/Server.swift."""

    def __init__(self, on_state):
        self.on_state = on_state
        self.state: dict = {"s": "idle"}
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.stopping = False
        self.crashes: list[float] = []
        self.lock = threading.Lock()

    def set(self, **st):
        self.state = st
        try:
            self.on_state(st)
        except Exception as e:
            log(f"state callback: {e}")

    def _fetch(self, args: list[str], on_line=None) -> tuple[int, str]:
        m = manifest()
        cmd = [python_exe(), "-I", "-B", str(ROOT / "scripts" / "fetch_models.py"), "--dest", str(MODELS)] + \
              (["--manifest", str(m)] if m else []) + args
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=server_env(load_settings()), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, creationflags=NO_WINDOW, text=True, encoding="utf-8",
                             errors="replace")
        out = []
        for line in p.stdout:
            out.append(line)
            if on_line:
                on_line(line.rstrip("\n"))
        return p.wait(), "".join(out)

    def status(self) -> list[dict]:
        rc, out = self._fetch(["status", "--json"])
        for line in reversed(out.splitlines()):
            if line.startswith("{"):
                try:
                    return json.loads(line).get("models", [])
                except ValueError:
                    break
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "models.log").write_text(out, encoding="utf-8")
        return []

    def start(self):
        threading.Thread(target=self._start, daemon=True, name="engine-start").start()

    def _start(self):
        self.set(s="checking")
        MODELS.mkdir(parents=True, exist_ok=True)
        rows = self.status()
        missing = [r for r in rows if r.get("required") and not r.get("ok")]
        if not rows:
            self.set(s="failed", msg="Onset could not check its files. Open the logs for details.")
        elif missing:
            self.set(s="needs", bytes=sum(int(r.get("bytes") or 0) for r in missing),
                     downloadable=all(r.get("downloadable") for r in missing))
        else:
            self.launch()

    def download(self):
        def run():
            self.set(s="downloading", done=0, total=0)

            def line(l):
                try:
                    o = json.loads(l)
                except ValueError:
                    return
                if o.get("event") == "progress":
                    self.set(s="downloading", done=int(o.get("done") or 0), total=int(o.get("total") or 0))
                elif o.get("event") == "unpack":
                    st = dict(self.state)
                    st["checking"] = True
                    self.set(**st)
            self._fetch(["--json-progress", "fetch"], line)
            self._start()
        threading.Thread(target=run, daemon=True, name="download").start()

    def launch(self):
        with self.lock:
            self.stopping = False
            LOGS.mkdir(parents=True, exist_ok=True)
            DATA.mkdir(parents=True, exist_ok=True)
            st = load_settings()
            self.port = free_port(st.get("port"))
            st["port"] = self.port
            save_settings(st)
            logf = LOGS / "server.log"
            for i in range(4, -1, -1):                     # keep 5 old logs
                a, b = (LOGS / (f"server.log.{i}" if i else "server.log")), LOGS / f"server.log.{i + 1}"
                if a.exists():
                    try:
                        os.replace(a, b)
                    except OSError:
                        pass
            out = open(logf, "w", encoding="utf-8")
            cmd = [python_exe(), "-I", "-B", str(ROOT / "macapp" / "server_main.py"), "--port", str(self.port),
                   "--parent-pid", str(os.getpid())]
            try:
                self.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=server_env(st), stdout=out,
                                             stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            except OSError as e:
                self.set(s="failed", msg=f"Onset could not start its engine ({e}).")
                return
            self.started = time.time()
        self.set(s="starting", secs=0)
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True, name="server-watch").start()

    def _watch(self, p: subprocess.Popen):
        ready = False
        while p.poll() is None:
            secs = int(time.time() - self.started)
            if not ready:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/health", timeout=2) as r:
                        if json.loads(r.read()).get("warm"):
                            ready = True
                            self.set(s="ready", port=self.port)
                except Exception:
                    pass
                if not ready:
                    if secs > 300:
                        self.set(s="failed", msg="Onset took too long to start. Open the logs for details.")
                        return
                    self.set(s="starting", secs=secs)
            time.sleep(0.5 if not ready else 1.0)
        if self.stopping or p is not self.proc:
            return
        now = time.time()
        self.crashes = [t for t in self.crashes if now - t < 120] + [now]
        log(f"server exited {p.returncode}")
        if len(self.crashes) > 3:
            self.set(s="failed", msg=f"Onset's engine stopped {len(self.crashes)} times in 2 minutes. Open the logs for details.")
            return
        self.set(s="starting", secs=0)
        time.sleep(1.0)
        if not self.stopping:
            self.launch()

    def stop(self):
        self.stopping = True
        p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(4)
            except subprocess.TimeoutExpired:
                p.kill()

    def restart(self):
        self.stop()
        self.crashes = []
        self.launch()


# ------------------------------------------------------------------------------------------------ pages
SPLASH = """<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#fbfbfd;--ink:#1d1d1f;--ink2:#6e6e73;--line:#e5e5ea;--a:#4f46e5;--b:#14b8a6}
@media (prefers-color-scheme:dark){:root{--bg:#111113;--ink:#f5f5f7;--ink2:#a1a1a6;--line:#2c2c30;--a:#818cf8}}
body{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);color:var(--ink);
font:15px/1.5 "Segoe UI Variable","Segoe UI",system-ui,sans-serif}
.c{text-align:center;max-width:440px;padding:24px}.m{width:72px;height:72px;border-radius:20px;margin:0 auto 18px;display:flex;
align-items:center;justify-content:center;gap:6px;background:linear-gradient(135deg,var(--a),var(--b))}.m i{width:6px;border-radius:3px;background:#fff}
.m i:nth-child(1){height:18px}.m i:nth-child(2){height:34px}.m i:nth-child(3){height:24px}h1{margin:0 0 4px;font-size:28px}
p{color:var(--ink2);margin:6px 0}.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:18px 0 6px}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--a),var(--b));transition:width .3s}
button{font:inherit;border:0;border-radius:999px;padding:9px 20px;margin:14px 4px 0;cursor:pointer;background:var(--line);color:var(--ink)}
button.p{background:var(--a);color:#fff}[hidden]{display:none!important}</style></head><body><div class="c">
<div class="m"><i></i><i></i><i></i></div><h1>Onset</h1><p>Ask out loud. The answer is ready when you stop.</p>
<div id="body"><p>Starting…</p></div></div><script>
const B=document.getElementById('body'),api=()=>window.pywebview&&window.pywebview.api;
const mb=n=>n>=1e9?(n/1e9).toFixed(1)+' GB':Math.round(n/1e6)+' MB';
window.onState=function(st){let h='';
if(st.s==='checking'||st.s==='idle')h='<p>Starting…</p>';
else if(st.s==='needs')h=st.downloadable?`<p><b>One-time setup</b></p><p>Onset listens and answers on this PC, so your voice never leaves it. It needs a one-time download (${mb(st.bytes)}).</p><button class="p" onclick="api().download()">Download</button>`
 :'<p>Some files are missing and cannot be downloaded. Reinstall Onset.</p><button onclick="api().open_logs()">Open logs</button>';
else if(st.s==='downloading'){const f=st.total?st.done/st.total:0;h=`<div class="bar"><i style="width:${(f*100).toFixed(1)}%"></i></div><p>${st.checking?'Checking files…':st.total?'Downloading: '+mb(st.done)+' of '+mb(st.total):'Preparing download…'}</p>`}
else if(st.s==='starting')h=`<div class="bar"><i style="width:${Math.min(95,st.secs*1.6)}%"></i></div><p>Getting ready… ${st.secs} s</p><p style="font-size:12px">The first start takes about a minute.</p>`;
else if(st.s==='ready')h='<p>Loading…</p>';
else if(st.s==='failed')h=`<p>${st.msg}</p><button class="p" onclick="api().retry()">Try again</button><button onclick="api().open_logs()">Open logs</button>`;
B.innerHTML=h};
window.addEventListener('pywebviewready',()=>api().hello());</script></body></html>"""


def settings_html() -> str:
    st = load_settings()
    rows = "".join(
        f'<label class="k"><span>{label}</span><input type="password" name="{k}" autocomplete="off" '
        f'placeholder="{"Saved" if get_key(k) else "Not set"}"></label>' for k, label in KEYS.items())
    ear = st.get("ear", "auto")
    radio = "".join(f'<label><input type="radio" name="ear" value="{v}"{" checked" if ear == v else ""}> {t}</label>'
                    for v, t in (("auto", "On this PC (recommended)"), ("flux", "Deepgram (cloud, needs a key)")))
    return """<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#fbfbfd;--ink:#1d1d1f;--ink2:#6e6e73;--line:#e5e5ea;--a:#4f46e5}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1f;--ink:#f5f5f7;--ink2:#a1a1a6;--line:#2c2c30;--a:#818cf8}}
body{margin:0;padding:22px;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI Variable","Segoe UI",system-ui,sans-serif}
h2{font-size:15px;margin:0 0 6px}small{color:var(--ink2);display:block;margin:2px 0 14px}label{display:block;margin:6px 0}
.k span{display:block;font-size:13px;color:var(--ink2)}.k input{width:100%;box-sizing:border-box;padding:6px 8px;border:1px solid var(--line);
border-radius:6px;background:transparent;color:var(--ink);font:inherit}details{margin-top:8px}summary{cursor:pointer;font-weight:600}
.row{display:flex;justify-content:space-between;align-items:center;margin-top:18px}button{font:inherit;border:0;border-radius:999px;
padding:7px 18px;background:var(--a);color:#fff;cursor:pointer}#note{color:var(--ink2);font-size:12px}</style></head><body>
<h2>Privacy</h2><label><input type="checkbox" id="collect" CHECKED> Help improve Onset</label>
<small>Keeps anonymous timings of your questions on this PC to tune Onset. No audio. Nothing is uploaded.</small>
<details><summary>Advanced</summary><small>Onset works without any keys: answers are written on this PC and web results come
from DuckDuckGo. Add your own keys for longer answers or other services. Keys stay in Windows Credential Manager.</small>
ROWS<h2 style="margin-top:14px">Speech recognition</h2>RADIO
<label style="margin-top:10px"><input type="checkbox" id="forget"> Remove saved keys</label></details>
<div class="row"><span id="note"></span><button onclick="save()">Save</button></div><script>
function save(){const keys={};document.querySelectorAll('.k input').forEach(i=>{if(i.value.trim())keys[i.name]=i.value.trim()});
const ear=(document.querySelector('input[name=ear]:checked')||{}).value||'auto';
window.pywebview.api.save_settings({keys,ear,collect:document.getElementById('collect').checked,forget:document.getElementById('forget').checked})
.then(()=>{document.getElementById('note').textContent='Saved.';document.querySelectorAll('.k input').forEach(i=>{i.value=''})})}
</script></body></html>""".replace("ROWS", rows).replace("RADIO", radio).replace(
        "CHECKED", "checked" if st.get("trainCollect", True) else "")


# ------------------------------------------------------------------------------------------------ the bridge
RISKY = {".exe", ".bat", ".cmd", ".com", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh", ".msi", ".msp", ".scr",
         ".lnk", ".url", ".hta", ".cpl", ".jar", ".py", ".pyw", ".reg", ".appref-ms", ".application", ".msc", ".pif"}


class Api:
    """window.pywebview.api for both pages: the splash (download / retry / logs) and the app page (message)."""

    def __init__(self, app: "App"):
        self._app = app

    def hello(self):
        self._app.push_state()

    def download(self):
        self._app.engine.download()

    def retry(self):
        self._app.engine.start()

    def open_logs(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        os.startfile(str(LOGS)) if sys.platform == "win32" else None

    def save_settings(self, s: dict):
        st = load_settings()
        st["trainCollect"] = bool(s.get("collect", True))
        st["ear"] = s.get("ear") if s.get("ear") in ("auto", "flux") else "auto"
        save_settings(st)
        if s.get("forget"):
            for k in KEYS:
                set_key(k, None)
        for k, v in (s.get("keys") or {}).items():
            if k in KEYS and v:
                set_key(k, v)
        threading.Thread(target=self._app.engine.restart, daemon=True).start()
        return True

    def message(self, m: dict):
        """The app page's bridge (web/onset.js post()): the Mac's window.webkit.messageHandlers.onset."""
        t = (m or {}).get("t")
        if t == "settings":
            self._app.open_settings()
        elif t == "log":
            log("page: " + str(m.get("msg", ""))[:400])
        elif t == "open":
            open_found_file(str(m.get("path", "")))
        return True


def open_found_file(path: str):
    """A file from a "find my …" answer: opened with its default app; anything that runs code is only shown in
    Explorer. Only existing files under the user's profile."""
    if sys.platform != "win32" or not path:
        return
    p = Path(path)
    try:
        p = p.resolve(strict=True)
    except OSError:
        return
    home = Path.home().resolve()
    if home not in p.parents and p != home:
        return
    if p.is_file() and p.suffix.lower() in RISKY:
        subprocess.Popen(["explorer", "/select,", str(p)], creationflags=NO_WINDOW)
    else:
        os.startfile(str(p))


# ------------------------------------------------------------------------------------------------ the app
class App:
    def __init__(self, dev: bool):
        import webview
        self.webview = webview
        self.dev = dev
        self.engine = Engine(self.on_state)
        self.api = Api(self)
        self.window = webview.create_window(APP_NAME, html=SPLASH, js_api=self.api, width=960, height=780,
                                            min_size=(420, 520), background_color="#FBFBFD")
        self.settings_win = None
        self.loaded_port = 0
        self.mic_hooked = False
        self.window.events.closed += self.on_closed
        self.window.events.loaded += self.on_loaded

    def push_state(self):
        self.on_state(self.engine.state)

    def on_state(self, st: dict):
        if st.get("s") == "ready":
            if st["port"] != self.loaded_port:
                self.loaded_port = st["port"]
                page = "lab.html" if self.dev and os.environ.get("ONSET_LAB") == "1" else ""
                self.window.load_url(f"http://127.0.0.1:{st['port']}/{page}")
            return
        if self.loaded_port and st.get("s") == "starting":
            # the engine restarted (Settings saved, or a crash): back to the splash until it is warm again
            self.loaded_port = 0
            self.window.load_html(SPLASH)
            return
        try:
            self.window.evaluate_js(f"window.onState && window.onState({json.dumps(st)})")
        except Exception:
            pass

    def on_loaded(self):
        self.allow_microphone()

    def allow_microphone(self):
        """WebView2 asks for the microphone per page with its own prompt; Onset's own local page gets it outright
        (Windows still applies Settings > Privacy > Microphone)."""
        if self.mic_hooked or sys.platform != "win32":
            return
        try:
            from Microsoft.Web.WebView2.Core import CoreWebView2PermissionKind, CoreWebView2PermissionState
            core = self.window.native.browser.webview.CoreWebView2

            def on_perm(sender, args):
                uri = str(args.Uri)
                if args.PermissionKind == CoreWebView2PermissionKind.Microphone and \
                        uri.startswith(f"http://127.0.0.1:{self.loaded_port}/"):
                    args.State = CoreWebView2PermissionState.Allow
            core.PermissionRequested += on_perm
            self.mic_hooked = True
        except Exception as e:                            # the WebView2 prompt still works without this
            log(f"microphone hook: {type(e).__name__}: {e}")

    def open_settings(self):
        if self.settings_win is not None:
            try:
                self.settings_win.show()
                return
            except Exception:
                self.settings_win = None
        self.settings_win = self.webview.create_window(f"{APP_NAME} Settings", html=settings_html(), js_api=self.api,
                                                       width=520, height=640, resizable=False)
        self.settings_win.events.closed += lambda: setattr(self, "settings_win", None)

    def on_closed(self):
        self.engine.stop()

    def check_update(self):
        st = load_settings()
        if time.time() - float(st.get("updateChecked", 0)) < 86400 or st.get("autoUpdate") is False:
            return
        st["updateChecked"] = time.time()
        save_settings(st)
        try:
            req = urllib.request.Request(f"https://api.github.com/repos/{REPO}/releases/latest",
                                         headers={"Accept": "application/vnd.github+json", "User-Agent": APP_NAME})
            with urllib.request.urlopen(req, timeout=8) as r:
                rel = json.loads(r.read())
        except Exception as e:
            log(f"update check: {type(e).__name__}")
            return
        tag = str(rel.get("tag_name", "")).lstrip("v")
        newer = _ver(tag) > _ver(VERSION)
        asset = next((a for a in rel.get("assets", []) if str(a.get("name", "")).endswith("-win-x64-setup.exe")), None)
        if newer and asset:
            time.sleep(5)
            if self.window.create_confirmation_dialog("Update available",
                                                      f"Onset {tag} is available (you have {VERSION}). Download it now?"):
                import webbrowser
                webbrowser.open(asset["browser_download_url"])

    def run(self):
        def started():
            self.engine.start()
            threading.Thread(target=self.check_update, daemon=True, name="update").start()
        BASE.mkdir(parents=True, exist_ok=True)
        self.webview.start(started, private_mode=False, storage_path=str(BASE / "webview"), debug=self.dev)


def _ver(v: str) -> tuple:
    out = []
    for part in v.split("."):
        try:
            out.append(int(part))
        except ValueError:
            out.append(0)
    return tuple(out)


def single_instance() -> bool:
    if sys.platform != "win32":
        return True
    import ctypes
    ctypes.windll.kernel32.CreateMutexW(None, False, f"Local\\{APP_ID}")
    return ctypes.windll.kernel32.GetLastError() != 183          # ERROR_ALREADY_EXISTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="devtools on; ONSET_LAB=1 opens the developer view")
    a = ap.parse_args()
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)   # taskbar: Onset, not Python
    if not single_instance():
        return 0
    log(f"{APP_NAME} {VERSION} starting from {ROOT}")
    App(a.dev).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
