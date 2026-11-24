"""Head Start server: FastAPI + one WebSocket per client.

Client -> server
  binary frame                        Float32Array (little-endian) mono 16 kHz PCM, any length (100 ms recommended)
  {"t":"start","modes":["headstart","baseline"], "domain":"assistant"}   start a run (both modes race on one ear)
                                    "open": true = open mic (clicked, keeps listening): a statement that is not a
                                    request gets a "heard" event instead of a search + answer; push-to-talk omits it
  {"t":"stop"}                        flush the ear (closes the open utterance) and end the run
  {"t":"replay","file":"q01.wav","modes":[...]}   server streams eval/audio/<file> through the ear in real time
  {"t":"hello","client":"<random id>"}  this browser's id (localStorage): its memory is kept in runs/memory/ across
                                    reconnects (server/memory.py Store; TTLs MEM_CONV_TTL_S / MEM_USER_TTL_S)
  {"t":"forget"}                      clear this browser's memory (conversation + user facts), here and on disk
  {"t":"start","typed":true,"modes"}  a typed question (TYPED_INPUT=1): no ear, no audio; words come from "type"
  {"t":"type","rid","text","final"}   the text box as it is typed (whole text each keystroke). Finished words (followed
                                    by a space) stream into the tagger with keystroke timings, so the speculator works
                                    while typing; final=true is Enter: the endpoint. Starts a typed run if none is open.
  "fresh": true on start              a scripted conversation: start a new conversation (user facts are kept)
Server -> client: JSON events, each with "mode" ("headstart" | "baseline" | "ear" | "sys"):
  sys    {"t":"ready"}, {"t":"error","error"}, {"t":"run_done"}
  ear    {"t":"ear","kind":"partial|final|arc_command_complete|connected|fallback|warning|error","ear":"arc|deepgram",...}
         (ready and /api/health carry "ear": the ear new runs use, "ear_requested": env EAR; a run that falls back
         from Deepgram to Arc says so with a "fallback" ear event)
  per mode: words, frame, spec, endpoint, sources, answer, answer_error, metric, heard (open mic: a statement, not
         answered)   (see server/speculator.py, session.py). Each carries "q": the id of the question it belongs to.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import answer, config, embed, ear, lanes, safety
from . import cost as costmod
from . import memory as memmod
from . import persona as personamod
from . import arcstate as arcstatemod
from . import trainlog
from . import userindex
from .session import Session

ROOT = config.ROOT
WEB = ROOT / "web"
REQUEST_LOG = os.environ.get("REQUEST_LOG", "1") == "1"   # one line per connection / run / question on stdout
TYPED_INPUT = os.environ.get("TYPED_INPUT", "1") == "1"     # the text box: typed questions through the same pipeline
TYPED_TICK_S = float(os.environ.get("TYPED_TICK_S", "0.15"))   # speculator ticks while a typed question is open


def rlog(msg: str):
    """Request log (server.log in the macOS app): connections, runs, each question's start / end / error. Never keys;
    question words are cut to 80 characters."""
    if REQUEST_LOG:
        print(f"[req {time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] {msg}", flush=True)


def log_event(e: dict, mode: str, run_id: int):
    t = e.get("t")
    q = e.get("q")
    if t == "endpoint":
        rlog(f"run {run_id} {mode} q{q} endpoint by={e.get('by')} speech_end={e.get('speech_end_ms')}ms")
    elif t == "metric":
        words = str(e.get("question") or "")[:80]
        rlog(f"run {run_id} {mode} q{q} done ready={e.get('ready_ms')}ms path={e.get('path')} "
             f"lane={e.get('lane') or e.get('route') or '-'}" + (f" fallback={e['answer_fallback']}" if e.get("answer_fallback") else "")
             + (f" ERROR={e.get('error')}" if e.get("failed") else "") + f" | {words!r}")
    elif t == "heard":
        rlog(f"run {run_id} {mode} q{q} heard (not a request: {e.get('why')})")
    elif t in ("answer_error", "error"):
        rlog(f"run {run_id} {mode} q{q} {t}: {str(e.get('error'))[:200]}")
    elif t == "ear" and e.get("kind") in ("fallback", "error", "warning", "connected"):
        rlog(f"run {run_id} ear {e.get('kind')} {e.get('ear', '')} {str(e.get('error') or e.get('text') or e.get('reason') or e.get('message') or '')[:160]}")
    elif t == "run_done":
        rlog(f"run {run_id} done")
AUDIO = config.AUDIO_DIR
app = FastAPI(title="Head Start")
STATE = {"warm": False}


@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    t = time.perf_counter()
    if config.INDEX and config.INDEX_PERSIST:
        try:
            from .keystone import hook as _ks
            moved = _ks.migrate()
            if moved:
                print(f"[server] Keystone store v{_ks.STORE_VERSION}: the old store was moved to {moved}", flush=True)
        except Exception as e:  # noqa: BLE001  (a store that can't be moved must not stop the app)
            print(f"[server] Keystone migrate failed: {type(e).__name__}: {e}", flush=True)
    await loop.run_in_executor(None, ear.warmup)
    await embed.arun(embed.warmup)
    await embed.arun(personamod.warmup)
    lanes.index()
    await lanes.warm_web()
    await answer.warm()
    STATE.update(warm=True, warm_s=round(time.perf_counter() - t, 1))


def degraded() -> dict:
    """What is running on a fallback right now ({} = everything live). Shown in /api/health, the ready event, the UI."""
    w, a = lanes.web_health(), answer.health()
    out = {}
    if w["status"] == "cache":
        out["web"] = "cached (WEB_BACKEND=cache: recorded responses, not live)"
    elif w["status"] == "down":
        out["web"] = ("live web unreachable: serving recorded cache" if w["cache_entries"]
                      else "live web unreachable and no cache: web questions will have no sources")
    if a["status"] == "down":
        out["answer"] = ("answer model unreachable: extractive answers (top-source sentence)" if a["fallback"] != "none"
                         else "answer model unreachable")
    return out


def status_payload() -> dict:
    from . import notices
    w, a = lanes.web_health(), answer.health()
    return {**STATE, **ear.describe(), "web_backend": config.WEB_BACKEND, "web": w, "typed_input": TYPED_INPUT,
            "tagger_safety": safety.ENABLED,
            "answer": a, "degraded": degraded(), "notices": notices.build(w, a)}


@app.get("/api/health")
async def health(probe: int = 0):
    """probe=1: make one real request to the web backend and the answer model now (the preflight uses it)."""
    probes = {}
    if probe:
        w, a = await asyncio.gather(lanes.probe_web(), answer.probe())
        probes = {"probe": {"web_ms": w.get("probe_ms"), "answer_ms": a.get("probe_ms")}}
    return {"ok": True, **status_payload(), "answer_model": answer.model_name(), "local_docs": len(lanes.index()),
            "user_recordings": len(userindex.ready_dirs()),
            **probes}


def replayable(r: dict, audio_dir: Path, index_dir: Path) -> bool:
    """A Replay row is listed only when it can run here: its audio is installed (the public app bundles a subset) and,
    for a local-moments question, the local index is installed (the public app has none)."""
    if r.get("file") and not (audio_dir / r["file"]).exists():
        return False
    if r.get("kind") == "local" and not (index_dir / "docs.jsonl").exists():
        return False
    return True


@app.get("/api/questions")
async def questions(set: str = "all"):
    """set=web: eval/questions.jsonl; set=local: index/questions_local.jsonl (kind "local"); all: both."""
    rows = []
    srcs = {"web": [ROOT / "eval" / "questions.jsonl"], "local": [ROOT / "index" / "questions_local.jsonl"],
            "multi": [ROOT / "eval" / "multipart" / "questions.jsonl"]}     # planner set (rows carry "parts")
    for p in srcs.get(set, srcs["web"] + srcs["local"] + srcs["multi"]):
        if p.exists():
            rows += [r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip())
                     if replayable(r, AUDIO, config.INDEX_DIR)]
    return JSONResponse(rows)


@app.get("/api/conversations")
async def conversations():
    p = ROOT / "eval" / "conversations.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    return JSONResponse([r for r in rows if all((AUDIO / f).exists() for f in r.get("files") or [])])  # voiced here


@app.get("/api/results")
async def results():
    """The last local eval A/B (runs/ab_latest.json, untracked: absent in the app and a fresh clone). Served with its
    provenance (date, web backend, budget) and without its config block (it holds machine paths)."""
    from . import scorecard
    return JSONResponse(scorecard.ab_latest(ROOT / "runs" / "ab_latest.json"))


@app.get("/api/scorecard")
async def scorecard_():
    """The Results view: the organizers' six problems, one measured number each (server/scorecard.json, tracked, with
    the doc + section each number comes from), plus what this server runs now (budget, web lane)."""
    from . import scorecard
    return JSONResponse(scorecard.payload())


@app.get("/api/meter")
async def meter_prices():
    """Prices behind the cost meter's est. $ (server/config.py METER_*)."""
    from . import meter
    return JSONResponse({**meter.prices(), "cost_mode": costmod.profile().name, "cost_modes": list(costmod.MODES)})


@app.post("/api/cache/clear")
async def cache_clear():
    """Forget cached searches (the persisted search cache file too) and the router's card / fact lookups.
    The macOS app's "Clear Search Cache" menu item calls it."""
    from . import router, searchcache
    c = searchcache.cache()
    n = len(c)
    c.clear()
    try:
        config.CACHE_PATH.unlink(missing_ok=True)
    except OSError:
        pass
    router.reset_caches()
    rlog(f"cache cleared ({n} search entries)")
    return {"ok": True, "cleared": n}


# ------------------------------------------------------------------ the user's own recordings (server/userindex.py)
import weakref  # noqa: E402
LIVE_RUNS: weakref.WeakSet = weakref.WeakSet()      # runs in progress: indexing pauses while one runs (never competes with a live question)
userindex.set_busy_probe(lambda: any(not r.closed for r in list(LIVE_RUNS)))


def _local_request(request: Request) -> bool:
    """Mutating recording calls come from this server's own page (or the native app, which sends no Origin): a web
    page elsewhere must not make this Mac index (or delete) files."""
    o = request.headers.get("origin")
    if not o or o == "null":
        return o is None
    from urllib.parse import urlparse
    return urlparse(o).netloc == request.headers.get("host", "")


def _rec_error(msg: str, code: int = 400):
    return JSONResponse({"ok": False, "error": msg}, status_code=code)


@app.get("/api/recordings")
async def recordings_list():
    """Every recording (name, duration, moments, status, stage, progress 0-1, paused, stats). The page polls this
    while a job runs."""
    st = userindex.store()
    return {"ok": True, "recordings": st.list(), "busy": not st.idle(),
            "allowed": sorted(userindex.ALLOWED_EXT), "audio_vectors": config.USER_INDEX_AUDIO}


@app.post("/api/recordings")
async def recordings_add(request: Request):
    """{"path": "/abs/file.m4a", "name"?: "Team sync"} or {"paths": [...]}: index files already on this Mac, read in
    place (the native app passes paths). JSON only, same origin only."""
    if not _local_request(request):
        return _rec_error("cross-origin request refused", 403)
    if not request.headers.get("content-type", "").startswith("application/json"):
        return _rec_error("send JSON", 415)
    try:
        body = await request.json()
        assert isinstance(body, dict)
    except Exception:  # noqa: BLE001
        return _rec_error("bad JSON")
    paths = body.get("paths") if isinstance(body.get("paths"), list) else [body.get("path")]
    added, errors = [], []
    for p in paths[:50]:
        try:
            src = userindex.safe_source(p)
            added.append(userindex.store().add(src, name=body.get("name") if len(paths) == 1 else None,
                                               domain=None, kind="path"))
        except userindex.Unsupported as e:
            errors.append({"path": str(p)[-120:] if isinstance(p, str) else "", "error": str(e)})
        except OSError as e:
            errors.append({"path": str(p)[-120:], "error": f"cannot read: {e.strerror or e}"})
    rlog(f"recordings add: {len(added)} queued, {len(errors)} refused")
    return JSONResponse({"ok": bool(added), "added": added, "errors": errors}, status_code=200 if added else 400)


@app.post("/api/recordings/upload")
async def recordings_upload(request: Request, name: str = ""):
    """The web UI's drag-drop / picker: the file's bytes as the request body (any content type), its file name in
    ?name=. Saved to RECORDINGS_DIR/_incoming/ until indexed, then deleted."""
    if not _local_request(request):
        return _rec_error("cross-origin request refused", 403)
    try:
        ext = userindex.check_ext(name)
    except userindex.Unsupported as e:
        return _rec_error(str(e), 415)
    limit = config.USER_INDEX_MAX_MB * 1e6
    path, f = userindex.write_upload_tmp(ext)
    n = 0
    try:
        with f:
            async for chunk in request.stream():
                n += len(chunk)
                if n > limit:
                    raise ValueError(f"larger than {config.USER_INDEX_MAX_MB:.0f} MB")
                f.write(chunk)
        if n == 0:
            raise ValueError("empty file")
        rec = userindex.store().add(path, name=userindex.display_name(name), owned=True, kind="upload")
    except Exception as e:  # noqa: BLE001
        path.unlink(missing_ok=True)
        return _rec_error(f"upload failed: {e}"[:200])
    rlog(f"recordings upload: {ext} {n / 1e6:.1f} MB queued as {rec['id']}")
    return {"ok": True, "added": [rec]}


@app.delete("/api/recordings/{rid}")
async def recordings_remove(rid: str, request: Request):
    """Remove a recording and its moments (stops its job if it is being indexed)."""
    if not _local_request(request):
        return _rec_error("cross-origin request refused", 403)
    if not userindex.valid_id(rid):
        return _rec_error("bad id")
    if not await asyncio.get_running_loop().run_in_executor(None, userindex.store().remove, rid):
        return _rec_error("no such recording", 404)
    return {"ok": True}


@app.get("/audio/{name}")
async def audio(name: str):
    p = (AUDIO / name).resolve()
    if AUDIO.resolve() not in p.parents or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


_run_ids = itertools.count(1)


def _mic_debug(d) -> str:
    """The page's capture counters on a mic stop (web/app.js micDebug): frames from the capture node, seconds captured
    and sent, AudioContext state, track state, socket backlog. Logged so a short run shows where the audio stopped."""
    if not isinstance(d, dict):
        return ""
    keep = []
    for k, v in list(d.items())[:16]:
        if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float, str)):
            continue
        if isinstance(v, float):
            v = round(v, 2)
        keep.append(f"{k[:24]}={str(v)[:40]}")
    return " | page: " + " ".join(keep) if keep else ""


class Run:
    """One run: an ear feeding one Session per mode, and a sender task draining their events to the socket."""

    def __init__(self, ws: WebSocket, modes, domain, cost_mode: str | None = None, arcstate=None,
                 keyterms: list[str] | None = None, typed: bool = False):
        self.id = next(_run_ids)
        self.ws = ws
        LIVE_RUNS.add(self)
        self.q: asyncio.Queue = asyncio.Queue()
        t0 = time.perf_counter()
        self.sessions = [Session(self._emitter(m), mode=m, domain=domain, t0=t0, cost_mode=cost_mode, arcstate=arcstate)
                         for m in modes]
        self.typed = typed
        self.typing = TypedText() if typed else None
        self.final_sent = False
        self.final_tasks: list[asyncio.Task] = []
        self.ticker: asyncio.Task | None = None
        if typed:
            # a typed question: no ear (no audio, no cloud ear connection); Enter is the endpoint, so no semantic
            # (pause-based) early answer; no audio features for Arc state / the director
            for s in self.sessions:
                s.typed = True
                s.semantic = False
            self.ear = None
        else:
            self.ear = ear.make_ear(self.sessions, domain=domain, on_event=self._emitter("ear"), keyterms=keyterms,
                                    arcstate=arcstate)
        self.sender = asyncio.create_task(self._send())
        self.clock_set = False           # mic runs: the clock starts at the first audio frame, not at "start"
        self._stop: asyncio.Task | None = None
        self.closed = False
        self.samples = 0
        self.rid = None                  # the page's own id for this run (echoed on every event: late events of an
                                         # earlier run, e.g. its run_done, must not end the page's next run)
        if typed:
            self.set_clock(t0)
            self.ticker = asyncio.create_task(self._tick())

    # ------------------------------------------------------------------ typed questions (TYPED_INPUT)
    def _typed_ws(self, text: str, final: bool) -> list[dict]:
        return self.typing.update(text, time.perf_counter() - self.sessions[0].t0, final)

    async def type_text(self, text: str, final: bool = False):
        """The text box changed (the whole text). Finished words go to every session as a partial (the speculator
        searches while you type); final (Enter) closes the question: the endpoint, measured from Enter."""
        if self.closed or self.final_sent:
            return
        ws = self._typed_ws(text, final)
        if not final:
            for s in self.sessions:
                s.on_partial(ws)
            return
        self.final_sent = True
        if self.ticker is not None:
            self.ticker.cancel()
        if not ws:
            await self.stop()
            return
        enter = time.perf_counter() - self.sessions[0].t0
        ws[-1] = {**ws[-1], "end": round(max(ws[-1].get("end") or 0.0, enter), 3)}   # the endpoint is Enter
        self.q.put_nowait({"t": "ear", "mode": "ear", "kind": "final", "ear": "typed", "text": " ".join(w["text"] for w in ws),
                           "reason": "enter", **({"rid": self.rid} if self.rid is not None else {})})
        for s in self.sessions:
            s.end_of_audio()                 # nothing more is coming: a trailing "sorry," is not held for a continuation
            s.on_partial(ws)
        self.final_tasks = [asyncio.create_task(s.on_final(ws)) for s in self.sessions]
        asyncio.create_task(self._finish_typed())

    async def _finish_typed(self):
        await asyncio.gather(*self.final_tasks, return_exceptions=True)
        await self.stop()

    async def _tick(self):
        """While a typed question is open: re-send the finished words every TYPED_TICK_S (the speculator's stability
        timers count partials, as they do the ear's ~100 ms updates)."""
        try:
            while not self.closed and not self.final_sent:
                await asyncio.sleep(TYPED_TICK_S)
                ws = self.typing.last
                if ws and not self.final_sent:
                    for s in self.sessions:
                        s.on_partial(ws)
        except asyncio.CancelledError:
            pass

    def _emitter(self, mode):
        def emit(e):
            self.q.put_nowait({**e, "mode": mode, **({"rid": self.rid} if self.rid is not None else {})})
            if REQUEST_LOG:
                try:
                    log_event(e, mode, self.id)
                except Exception:          # logging must never break a run
                    pass
            trainlog.record(e, mode, self.id)    # TRAIN_COLLECT: redacted metric row (never raises)
        return emit

    def set_clock(self, t0: float):
        """Wall time of audio sample 0, for every session (latency metrics are measured from it)."""
        self.clock_set = True
        for s in self.sessions:
            s.set_t0(t0)

    async def feed(self, pcm: np.ndarray):
        if self.closed or self.ear is None:
            return
        self.samples += len(pcm)
        if not self.clock_set:
            # the first frame arrives once its samples were captured: sample 0 was len/16000 s ago (the AudioWorklet's
            # setup after "start" must not count as latency)
            self.set_clock(time.perf_counter() - len(pcm) / 16000)
        try:
            await self.ear.feed(pcm)
        except Exception as e:                      # one bad chunk must not end the run
            self.q.put_nowait({"t": "error", "mode": "sys", "error": f"ear: {type(e).__name__}: {e}"[:300]})

    async def _send(self):
        while True:
            e = await self.q.get()
            try:
                await self.ws.send_text(json.dumps(e, default=float))
            except Exception:
                return

    async def stop(self):
        """Flush the ear (answers the open utterance), close everything, report run_done. Idempotent: every caller
        awaits the same stop, and a caller being cancelled (a replay task) doesn't cancel the stop itself."""
        if self._stop is None:
            self._stop = asyncio.ensure_future(self._do_stop())
        await asyncio.shield(self._stop)

    async def _do_stop(self):
        rlog(f"run {self.id} stopping: " + ("typed" if self.typed else f"{self.samples / 16000:.1f}s of audio"))
        for s in self.sessions:
            s.end_of_audio()                     # a final held for its continuation is answered now
        if self.ticker is not None:
            self.ticker.cancel()
        try:
            if self.ear is not None:
                await self.ear.flush()
            elif self.final_tasks:
                await asyncio.gather(*self.final_tasks, return_exceptions=True)
        except Exception as e:
            self.q.put_nowait({"t": "error", "mode": "sys", "error": f"flush: {type(e).__name__}: {e}"[:300]})
        finally:
            if self.ear is not None:
                self.ear.close()
            self.closed = True
            LIVE_RUNS.discard(self)
        await asyncio.sleep(0.05)
        for s in self.sessions:
            s.close()
        self.q.put_nowait({"t": "run_done", "mode": "sys", **({"rid": self.rid} if self.rid is not None else {}),
                           "metrics": [s.history for s in self.sessions]})
        await asyncio.sleep(0.2)
        self.sender.cancel()

    def abort(self):
        """The client is gone: stop everything now, no flush, nothing more to send."""
        self.closed = True
        LIVE_RUNS.discard(self)
        if self._stop is not None and not self._stop.done():
            self._stop.cancel()
        if self.ticker is not None:
            self.ticker.cancel()
        if self.ear is not None:
            self.ear.close()
        for t in list(getattr(self.ear, "final_tasks", [])) + self.final_tasks:
            if not t.done():
                t.cancel()
        for s in self.sessions:
            s.close()
        self.sender.cancel()


class TypedText:
    """Words of a text box as it is typed, with keystroke timings (seconds on the run clock): a word starts at its
    first keystroke and ends when it was finished (the space after it). Only finished words are returned, so a word
    being typed ("Nvi") never reaches the tagger; Enter (final) finishes the last one. An edited word gets new times."""

    def __init__(self):
        self.first: dict[int, tuple[str, float]] = {}
        self.done: dict[int, tuple[str, float]] = {}
        self.last: list[dict] = []

    def update(self, text: str, now: float, final: bool = False) -> list[dict]:
        toks = str(text or "").split()
        for i, w in enumerate(toks):
            p = self.first.get(i)
            if p is None or not (w.startswith(p[0]) or p[0].startswith(w)):
                self.first[i] = (w, now)
            else:
                self.first[i] = (w, p[1])
        for i in [i for i in self.first if i >= len(toks)]:
            del self.first[i]
        n = len(toks) if (final or str(text or "")[-1:].isspace()) else max(0, len(toks) - 1)
        out = []
        for i in range(n):
            d = self.done.get(i)
            if d is None or d[0] != toks[i]:
                d = self.done[i] = (toks[i], now)
            t0 = self.first[i][1]
            out.append({"text": toks[i], "start": round(t0, 3), "end": round(max(d[1], t0), 3)})
        for i in [i for i in self.done if i >= n]:
            del self.done[i]
        self.last = out
        return out


async def replay(run: Run, path: Path):
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    t0 = time.perf_counter()
    run.set_clock(t0)
    for i in range(0, len(wav), 1600):
        target = t0 + (i + 1600) / 16000
        while time.perf_counter() < target:
            await asyncio.sleep(0.003)
        await run.feed(wav[i:i + 1600])
    # trailing silence so Arc can endpoint at the pause, like a live mic
    for _ in range(12):
        await asyncio.sleep(0.1)
        await run.feed(np.zeros(1600, np.float32))


async def _cancel(task: asyncio.Task | None):
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    conn_t0, conn_runs = time.perf_counter(), 0
    rlog(f"ws open from {getattr(getattr(ws, 'client', None), 'host', '?')}")
    run: Run | None = None
    task: asyncio.Task | None = None       # a replay streaming into `run`

    async def send(e):
        try:
            await ws.send_text(json.dumps(e))
        except Exception:
            pass

    memory: dict = {}          # per mode: conversation memory carried across mic runs on this connection
    client = {"id": None}      # this browser's id ("hello"): memory persists across reconnects
    store = memmod.Store()
    pstore = personamod.Store()
    prof = {"p": personamod.Profile()}   # this browser's persona (PERSONA=1; server/persona.py)

    async def send_persona(**kw):
        await send({"t": "persona", "mode": "sys", "on": personamod.ENABLED, "profile": prof["p"].summary(), **kw})
    # Arc state across presses (server/arcstate.py): one per connection = one conversation; per session in memory
    # unless ARC_STATE_PERSIST=1 (then <data dir>/arcstate/, keyed like memory by the client id)
    astate = arcstatemod.ArcState() if arcstatemod.ENABLED else None

    def persist():
        """Save the browser's memory (the first mode's: both modes hear the same words)."""
        if client["id"] and memory:
            m = next(iter(memory.values())).get("mem")
            if m is not None:
                store.save(client["id"], memmod.Memory.from_json(m))

    def stored_memory(fresh: bool = False) -> dict | None:
        """This connection's memory, else the browser's stored one (a reconnect). fresh: user facts only."""
        mem = None
        if client["id"]:
            mem = {"mem": store.load(client["id"]).to_json(), "at": 0}
        if fresh:
            if mem:
                m = memmod.Memory.from_json(mem["mem"])
                m.clear(user=False)
                mem = {"mem": m.to_json(), "at": 0}
            return mem
        return mem

    async def end_run():
        """Stop the current run: the replay feeding it first (so nothing is fed into a flushed ear), then the run."""
        nonlocal run, task
        await _cancel(task)
        task = None
        if run is not None:
            r, run = run, None
            await r.stop()
            if getattr(r, "kind", "") in ("mic", "typed"):
                for s in r.sessions:
                    memory[s.mode] = s.export_memory()
                persist()
                if r.kind == "mic":
                    arcstatemod.save(client["id"], astate)
            for s in r.sessions:
                if s.mode == "headstart" and personamod.ENABLED:
                    prof["p"] = s.persona
                    pstore.save(client["id"], s.persona)

    await send({"t": "ready", "mode": "sys", **status_payload()})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                b = msg["bytes"]
                if len(b) % 4:
                    await send({"t": "error", "mode": "sys",
                                "error": f"audio frame of {len(b)} bytes is not whole float32 samples: dropped"})
                    continue
                if run is not None and task is None and len(b):
                    await run.feed(np.frombuffer(b, dtype="<f4"))
                continue
            try:
                data = json.loads(msg.get("text") or "{}")
                if not isinstance(data, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as e:
                await send({"t": "error", "mode": "sys", "error": f"bad message: {e}"[:200]})
                continue
            t = data.get("t")
            if t == "hello":
                cid = data.get("client")
                client["id"] = cid if memmod.Store.valid(cid) else None
                if astate is not None and arcstatemod.PERSIST and client["id"] and not astate.turns:
                    astate = arcstatemod.load(client["id"])
                await send({"t": "memory", "mode": "sys", "persist": bool(client["id"] and config.MEM_PERSIST),
                            "graph": store.load(cid).graph() if client["id"] else None})
                prof["p"] = pstore.load(client["id"]) if client["id"] else personamod.Profile()
                await send_persona()
                continue
            if t in ("persona_clear", "persona_export"):
                if t == "persona_clear":
                    prof["p"] = personamod.Profile()
                    pstore.forget(client["id"])
                    for s in (run.sessions if run is not None else []):
                        s.set_persona(prof["p"])
                    await send_persona(cleared=True)
                else:
                    await send_persona(export=prof["p"].export())
                continue
            if t == "forget":
                memory.clear()
                if client["id"]:
                    store.forget(client["id"])
                if run is not None:
                    for s in run.sessions:
                        s.mem.clear()
                        s.carry, s.context, s.carry_pool = {}, "", {}
                if astate is not None:                # "forget" clears the voice state too
                    astate.reset("all")
                arcstatemod.forget(client["id"])
                await send({"t": "memory", "mode": "sys", "forgot": True, "graph": memmod.Memory().graph()})
                prof["p"] = personamod.Profile()
                pstore.forget(client["id"])
                if run is not None:
                    for s in run.sessions:
                        s.set_persona(prof["p"])
                await send_persona(cleared=True)
                continue
            if t == "arc_reset":                      # "Reset voice": Arc state only, memory stays
                if astate is not None:
                    astate.reset("all")
                arcstatemod.forget(client["id"])
                await send({"t": "arcstate", "mode": "sys", "reset": "all",
                            **(astate.summary() if astate is not None else {"on": False})})
                continue
            type_msg = None
            if t == "type":
                if not TYPED_INPUT:
                    await send({"t": "error", "mode": "sys", "error": "typed questions are off (TYPED_INPUT=0)"})
                    continue
                rid = data.get("rid")
                if (run is not None and getattr(run, "kind", "") == "typed" and not run.final_sent
                        and (rid is None or rid == run.rid)):
                    await run.type_text(str(data.get("text") or "")[:2000], bool(data.get("final")))
                    continue
                type_msg, t = data, "start"              # no open typed question: start one
                data = {**data, "t": "start", "typed": True}
            if t in ("start", "replay"):
                conn_runs += 1
                await end_run()
                if t == "replay":
                    p = AUDIO / Path(str(data.get("file", ""))).name
                    if not p.exists():
                        await send({"t": "error", "mode": "sys", "error": f"no such file {p.name}"})
                        continue
                modes = data.get("modes") or ["headstart"]   # the app: Onset only; the developer view asks for the baseline race
                typed = t == "start" and bool(data.get("typed")) and TYPED_INPUT
                rstate, kt = None, None
                if typed:
                    pass                              # typed: no voice, Arc state is not touched
                elif astate is not None and t == "start":
                    if astate.touch() == "idle":
                        await send({"t": "arcstate", "mode": "sys", "reset": "idle", **astate.summary()})
                    rstate = astate
                    if arcstatemod.KEYTERMS:
                        src = None if data.get("fresh") else (memory.get(modes[0]) or stored_memory())
                        try:
                            kt = astate.keyterms(memmod.Memory.from_json((src or {}).get("mem"))) or None
                        except Exception:
                            kt = None
                elif astate is not None:
                    rstate = arcstatemod.ArcState()   # a replay (demo file): its own state, never the user's voice
                run = Run(ws, modes, data.get("domain", "assistant"), cost_mode=costmod.valid(data.get("cost")),
                          arcstate=rstate, keyterms=kt, typed=typed)
                run.kind = "typed" if typed else "mic" if t == "start" else "replay"
                trainlog.bind(run.id, client.get("id"), run.kind)
                rid = data.get("rid")
                run.rid = rid if isinstance(rid, int) and not isinstance(rid, bool) else None
                rlog(f"run {run.id} start {run.kind}" + (f" {p.name}" if t == "replay" else "" if typed else
                     (" open-mic" if data.get("open") else " push-to-talk")) + f" modes={','.join(modes)}"
                     + ("" if typed else f" ear={ear.describe().get('ear')}")
                     + (f" arcstate=on keyterms={len(kt or [])}" if rstate is not None else ""))
                if t == "start":
                    fresh = bool(data.get("fresh"))
                    if fresh:
                        memory.clear()                # a scripted conversation starts a new one (user facts stay)
                    for s in run.sessions:            # push-to-talk: follow-ups keep the conversation
                        s.import_memory(memory.get(s.mode) or stored_memory(fresh))
                        if s.mode == "headstart":
                            s.set_persona(prof["p"])
                        if s.mem:
                            s.emit_memory(why="restored")
                        # open mic (clicked, keeps listening; not push-to-talk): statements are heard, not answered
                        s.open_mic = bool(data.get("open"))
                else:
                    memory.clear()                    # a replay starts a fresh conversation (user facts stay)
                    for s in run.sessions:
                        s.import_memory(stored_memory(fresh=True))
                        if s.mode == "headstart":
                            s.set_persona(prof["p"])
                if type_msg is not None:
                    await run.type_text(str(type_msg.get("text") or "")[:2000], bool(type_msg.get("final")))
                if t == "replay":
                    async def go(r=run, p=p):
                        await replay(r, p)
                        await r.stop()
                    task = asyncio.create_task(go())
            elif t == "stop":
                rlog(f"run {run.id if run else '-'} stop (client)" + _mic_debug(data.get("dbg")))
                await end_run()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        rlog(f"ws handler crashed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        raise
    finally:
        rlog(f"ws close after {time.perf_counter() - conn_t0:.0f}s, {conn_runs} run(s)"
             + (f", run {run.id} still open" if run is not None else ""))
        if task is not None and not task.done():
            task.cancel()
        if run is not None:
            run.abort()


if WEB.exists():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
