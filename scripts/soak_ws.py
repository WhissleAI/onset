"""Many questions on ONE websocket, the way the page sends them: does the Nth question still answer?

Mixes the three client paths:
  replay  {"t":"replay"}                                   (server streams eval/audio)
  ptt     {"t":"start"} + float32 PCM in real time + 0.5 s tail + {"t":"stop"}       (hold Space)
  open    {"t":"start","open":true} + several questions with pauses in ONE run + stop   (click Ask)

    .venv/bin/python scripts/soak_ws.py --port 51884 [--n 9] [--audio-dir DIR] [--modes headstart]
Exit 0 when every question got an answer (or an open-mic "heard"), 1 otherwise. Prints one line per question.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load(p: Path) -> np.ndarray:
    w, sr = sf.read(str(p), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        idx = np.arange(0, len(w), sr / 16000)
        w = np.interp(idx, np.arange(len(w)), w).astype(np.float32)
    return w


class Client:
    def __init__(self, ws, modes):
        self.ws, self.modes = ws, modes
        self.events: list[dict] = []
        self.reader = asyncio.create_task(self._read())

    async def _read(self):
        async for raw in self.ws:
            self.events.append(json.loads(raw))

    def mark(self) -> int:
        return len(self.events)

    async def wait(self, since: int, pred, timeout: float) -> list[dict]:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            got = [e for e in self.events[since:] if pred(e)]
            if got:
                return got
            if self.reader.done():
                raise ConnectionError("websocket closed")
            await asyncio.sleep(0.05)
        return []

    async def stream(self, pcm: np.ndarray, tail_s: float):
        t0 = time.perf_counter()
        pcm = np.concatenate([pcm, np.zeros(int(16000 * tail_s), np.float32)])
        for i in range(0, len(pcm), 1600):
            target = t0 + (i + 1600) / 16000
            while time.perf_counter() < target:
                await asyncio.sleep(0.003)
            await self.ws.send(pcm[i:i + 1600].astype("<f4").tobytes())


def answered(mode):
    return lambda e: e.get("mode") == mode and e.get("t") in ("metric", "heard")


async def main_async(a) -> int:
    audio = Path(a.audio_dir) if a.audio_dir else None
    if audio is None:
        from server import config
        audio = config.AUDIO_DIR
    qs = [json.loads(x) for x in (ROOT / "eval/questions.jsonl").read_text().splitlines() if x.strip()]
    qs = [q for q in qs if (audio / q["file"]).exists() and not q.get("parts")][: max(a.n, 3)]
    modes = a.modes.split(",")
    first = modes[0]
    plan = []
    for i, q in enumerate(qs[: a.n]):
        plan.append(("replay", "ptt")[i % 2] if i < a.n - 3 else "open")
    rows, ok_all = [], True
    async with websockets.connect(f"ws://127.0.0.1:{a.port}/ws", max_size=None, open_timeout=5) as ws:
        c = Client(ws, modes)
        if not await c.wait(0, lambda e: e.get("t") == "ready", 10):
            print("no ready event"); return 1
        await ws.send(json.dumps({"t": "hello", "client": "soak" + "0" * 20}))
        i = 0
        while i < len(plan):
            kind, q = plan[i], qs[i]
            t = time.perf_counter()
            if kind == "replay":
                m = c.mark()
                await ws.send(json.dumps({"t": "replay", "file": q["file"], "modes": modes}))
                got = await c.wait(m, answered(first), a.timeout)
                await c.wait(m, lambda e: e.get("t") == "run_done", 10)
                rows.append((i + 1, kind, q["text"], got, m)); i += 1
            elif kind == "ptt":
                m = c.mark()
                await ws.send(json.dumps({"t": "start", "modes": modes, "domain": "assistant"}))
                await c.stream(load(audio / q["file"]), tail_s=0.5)
                await ws.send(json.dumps({"t": "stop"}))
                got = await c.wait(m, answered(first), a.timeout)
                await c.wait(m, lambda e: e.get("t") == "run_done", 10)
                rows.append((i + 1, kind, q["text"], got, m)); i += 1
            else:   # open mic: the remaining questions in one run, 2.5 s of silence between them
                m = c.mark()
                await ws.send(json.dumps({"t": "start", "modes": modes, "domain": "assistant", "open": True}))
                starts = []
                for j in range(i, len(plan)):
                    starts.append(c.mark())
                    await c.stream(load(audio / qs[j]["file"]), tail_s=2.5)
                await asyncio.sleep(1.0)
                await ws.send(json.dumps({"t": "stop"}))
                await c.wait(m, lambda e: e.get("t") == "run_done", a.timeout)
                done = [e for e in c.events[m:] if answered(first)(e)]
                for k, j in enumerate(range(i, len(plan))):
                    rows.append((j + 1, "open", qs[j]["text"], done[k:k + 1], m))
                i = len(plan)
            new = [r for r in rows if not r[5:]]
            for r in new:
                rows[rows.index(r)] = r + (True,)
                met = r[3][0] if r[3] else {}
                ans = "".join(e.get("delta") or "" for e in c.events[r[4]:] if e.get("mode") == first
                              and e.get("t") == "answer" and met and e.get("q") == met.get("q"))
                st = "OK " if r[3] else "STUCK"
                ok_all &= bool(r[3])
                print(f"  [{st}] {r[0]:>2} {r[1]:<6} {r[2][:48]:<48} " +
                      (f"ready {met.get('ready_ms')} ms  {ans.strip()[:60]!r}" if met.get("t") == "metric" else
                       f"heard ({met.get('why')})" if met else f"no answer after {time.perf_counter() - t:.0f}s"), flush=True)
        errs = [e for e in c.events if e.get("t") in ("error", "answer_error")]
        if errs:
            print("  errors:", [str(e.get("error"))[:120] for e in errs[:5]])
        c.reader.cancel()
    print("ALL ANSWERED" if ok_all else "SOME QUESTIONS GOT NO ANSWER")
    return 0 if ok_all else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--modes", default="headstart,baseline")
    ap.add_argument("--timeout", type=float, default=30)
    a = ap.parse_args()
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
