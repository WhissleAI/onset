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
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import answer, config, embed, ear, lanes
from . import memory as memmod
from .session import Session

ROOT = config.ROOT
WEB = ROOT / "web"
AUDIO = config.AUDIO_DIR
app = FastAPI(title="Head Start")
STATE = {"warm": False}


@app.on_event("startup")
async def startup():
    loop = asyncio.get_running_loop()
    t = time.perf_counter()
    await loop.run_in_executor(None, ear.warmup)
    await embed.arun(embed.warmup)
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
    return {**STATE, **ear.describe(), "web_backend": config.WEB_BACKEND, "web": lanes.web_health(),
            "answer": answer.health(), "degraded": degraded()}


@app.get("/api/health")
async def health(probe: int = 0):
    """probe=1: make one real request to the web backend and the answer model now (the preflight uses it)."""
    probes = {}
    if probe:
        w, a = await asyncio.gather(lanes.probe_web(), answer.probe())
        probes = {"probe": {"web_ms": w.get("probe_ms"), "answer_ms": a.get("probe_ms")}}
    return {"ok": True, **status_payload(), "answer_model": config.ANSWER_MODEL, "local_docs": len(lanes.index()),
            **probes}


@app.get("/api/questions")
async def questions(set: str = "all"):
    """set=web: eval/questions.jsonl; set=local: index/questions_local.jsonl (kind "local"); all: both."""
    rows = []
    srcs = {"web": [ROOT / "eval" / "questions.jsonl"], "local": [ROOT / "index" / "questions_local.jsonl"],
            "multi": [ROOT / "eval" / "multipart" / "questions.jsonl"]}     # planner set (rows carry "parts")
    for p in srcs.get(set, srcs["web"] + srcs["local"] + srcs["multi"]):
        if p.exists():
            rows += [r for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip())
                     if not r.get("parts") or (AUDIO / r["file"]).exists()]       # multi-part: only if voiced
    return JSONResponse(rows)


@app.get("/api/conversations")
async def conversations():
    p = ROOT / "eval" / "conversations.jsonl"
    return JSONResponse([json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else [])


@app.get("/api/results")
async def results():
    p = ROOT / "runs" / "ab_latest.json"
    return JSONResponse(json.loads(p.read_text()) if p.exists() else {})


@app.get("/api/meter")
async def meter_prices():
    """Prices behind the cost meter's est. $ (server/config.py METER_*)."""
    from . import meter
    return JSONResponse(meter.prices())


@app.get("/audio/{name}")
async def audio(name: str):
    p = (AUDIO / name).resolve()
    if AUDIO.resolve() not in p.parents or not p.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p)


class Run:
    """One run: an ear feeding one Session per mode, and a sender task draining their events to the socket."""

    def __init__(self, ws: WebSocket, modes, domain):
        self.ws = ws
        self.q: asyncio.Queue = asyncio.Queue()
        t0 = time.perf_counter()
        self.sessions = [Session(self._emitter(m), mode=m, domain=domain, t0=t0) for m in modes]
        self.ear = ear.make_ear(self.sessions, domain=domain, on_event=self._emitter("ear"))
        self.sender = asyncio.create_task(self._send())
        self.clock_set = False           # mic runs: the clock starts at the first audio frame, not at "start"
        self._stop: asyncio.Task | None = None
        self.closed = False

    def _emitter(self, mode):
        def emit(e):
            self.q.put_nowait({**e, "mode": mode})
        return emit

    def set_clock(self, t0: float):
        """Wall time of audio sample 0, for every session (latency metrics are measured from it)."""
        self.clock_set = True
        for s in self.sessions:
            s.set_t0(t0)

    async def feed(self, pcm: np.ndarray):
        if self.closed:
            return
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
        for s in self.sessions:
            s.end_of_audio()                     # a final held for its continuation is answered now
        try:
            await self.ear.flush()
        except Exception as e:
            self.q.put_nowait({"t": "error", "mode": "sys", "error": f"flush: {type(e).__name__}: {e}"[:300]})
        finally:
            self.ear.close()
            self.closed = True
        await asyncio.sleep(0.05)
        for s in self.sessions:
            s.close()
        self.q.put_nowait({"t": "run_done", "mode": "sys",
                           "metrics": [s.history for s in self.sessions]})
        await asyncio.sleep(0.2)
        self.sender.cancel()

    def abort(self):
        """The client is gone: stop everything now, no flush, nothing more to send."""
        self.closed = True
        if self._stop is not None and not self._stop.done():
            self._stop.cancel()
        self.ear.close()
        for t in getattr(self.ear, "final_tasks", []):
            if not t.done():
                t.cancel()
        for s in self.sessions:
            s.close()
        self.sender.cancel()


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
            if getattr(r, "kind", "") == "mic":
                for s in r.sessions:
                    memory[s.mode] = s.export_memory()
                persist()

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
                await send({"t": "memory", "mode": "sys", "persist": bool(client["id"] and config.MEM_PERSIST),
                            "graph": store.load(cid).graph() if client["id"] else None})
                continue
            if t == "forget":
                memory.clear()
                if client["id"]:
                    store.forget(client["id"])
                if run is not None:
                    for s in run.sessions:
                        s.mem.clear()
                        s.carry, s.context, s.carry_pool = {}, "", {}
                await send({"t": "memory", "mode": "sys", "forgot": True, "graph": memmod.Memory().graph()})
                continue
            if t in ("start", "replay"):
                await end_run()
                if t == "replay":
                    p = AUDIO / Path(str(data.get("file", ""))).name
                    if not p.exists():
                        await send({"t": "error", "mode": "sys", "error": f"no such file {p.name}"})
                        continue
                modes = data.get("modes") or ["headstart", "baseline"]
                run = Run(ws, modes, data.get("domain", "assistant"))
                run.kind = "mic" if t == "start" else "replay"
                if t == "start":
                    fresh = bool(data.get("fresh"))
                    if fresh:
                        memory.clear()                # a scripted conversation starts a new one (user facts stay)
                    for s in run.sessions:            # push-to-talk: follow-ups keep the conversation
                        s.import_memory(memory.get(s.mode) or stored_memory(fresh))
                        if s.mem:
                            s.emit_memory(why="restored")
                        # open mic (clicked, keeps listening; not push-to-talk): statements are heard, not answered
                        s.open_mic = bool(data.get("open"))
                else:
                    memory.clear()                    # a replay starts a fresh conversation (user facts stay)
                    for s in run.sessions:
                        s.import_memory(stored_memory(fresh=True))
                if t == "replay":
                    async def go(r=run, p=p):
                        await replay(r, p)
                        await r.stop()
                    task = asyncio.create_task(go())
            elif t == "stop":
                await end_run()
    except WebSocketDisconnect:
        pass
    finally:
        if task is not None and not task.done():
            task.cancel()
        if run is not None:
            run.abort()


if WEB.exists():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
