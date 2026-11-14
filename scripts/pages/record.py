"""Record real Onset runs for the static demo (site/): every websocket event of a Race (Onset + Baseline) run, with
its arrival time, exactly as the browser receives it.

  # a server of your own (no paid web search: WEB_BACKEND=wikipedia), fresh data dir, then:
  .venv/bin/python scripts/pages/record.py --port 8812 --extra-audio <dir from make_audio.py> [--only facts,race]

Writes site/demo-data/runs/<id>.json and site/demo-data/audio/<file>.mp3 (ffmpeg, mono 48 kbps). The client side is the
same as web/app.js: one-file scenarios use the server's "replay"; multi-turn ones stream the PCM in real time
("start", 100 ms Float32 frames, 2.5 s between turns, 1.2 s tail, "stop"), so the server keeps memory across turns.
Events are sanitized (scripts/pages/sanitize.py) before they are written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sanitize import sanitize  # noqa: E402

OUT = ROOT / "site" / "demo-data"
MODES = ["headstart", "baseline"]


def load16k(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x.astype("<f4")


async def record(port: int, sc: dict, audio_dirs: list[Path], tail_s: float = 2.0, max_s: float = 60.0) -> dict:
    events: list[dict] = []
    pre: list[dict] = []
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
        t0 = None
        done = asyncio.Event()

        async def rx():
            async for m in ws:
                e = json.loads(m)
                if t0 is None:
                    pre.append(e)
                    continue
                events.append({"at": round((time.perf_counter() - t0) * 1000, 1), "e": e})
                if e.get("t") == "run_done":
                    done.set()

        rtask = asyncio.create_task(rx())
        await ws.send(json.dumps({"t": "hello", "client": "demo" + secrets.token_hex(8)}))
        await asyncio.sleep(1.0)
        t0 = time.perf_counter()
        if sc.get("typed"):
            # the demo page types the same keystrokes on the same clock (site/demo.js), as web/app.js sends them
            await ws.send(json.dumps({"t": "start", "typed": True, "modes": MODES, "rid": 1}))
            text, cms = sc["text"], sc.get("char_ms", 110)
            for i in range(1, len(text) + 1):
                await asyncio.sleep(cms / 1000)
                await ws.send(json.dumps({"t": "type", "rid": 1, "text": text[:i], "final": False}))
            await asyncio.sleep(sc.get("enter_ms", 500) / 1000)
            await ws.send(json.dumps({"t": "type", "rid": 1, "text": text, "final": True}))
        elif sc.get("file"):
            await ws.send(json.dumps({"t": "replay", "file": sc["file"], "modes": MODES, "rid": 1}))
        else:
            pcm = []
            for f in sc["files"]:
                p = next((d / f for d in audio_dirs if (d / f).exists()), None)
                if p is None:
                    raise FileNotFoundError(f)
                pcm.append(load16k(p))
            await ws.send(json.dumps({"t": "start", "modes": MODES, "fresh": True, "rid": 1,
                                      **({"open": True} if sc.get("open") else {})}))
            F, sent = 1600, 0

            async def pace(n):
                target = t0 + n / 16000
                while time.perf_counter() < target - 0.002:
                    await asyncio.sleep(min(0.02, target - time.perf_counter()))

            async def frame(f):
                nonlocal sent
                await pace(sent + F)
                await ws.send(np.ascontiguousarray(f, "<f4").tobytes())
                sent += len(f)

            turn_at = []
            for ti, x in enumerate(pcm):
                if ti:
                    for _ in range(25):
                        await frame(np.zeros(F, "<f4"))
                turn_at.append(round(sent / 16), )
                for i in range(0, len(x), F):
                    await frame(x[i:i + F])
            for _ in range(12):
                await frame(np.zeros(F, "<f4"))
            await ws.send(json.dumps({"t": "stop"}))
            sc = {**sc, "turn_at_ms": turn_at}
        try:
            await asyncio.wait_for(done.wait(), max_s)
        except asyncio.TimeoutError:
            print("  timeout waiting for run_done", file=sys.stderr)
        await asyncio.sleep(tail_s)
        rtask.cancel()
    return {"scenario": sc, "pre": pre, "events": events}


def to_mp3(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "24000",
                    "-b:a", "48k", "-map_metadata", "-1", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8812)
    ap.add_argument("--extra-audio", default="")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    spec = json.loads((HERE / "scenarios.json").read_text())
    only = {s for s in a.only.split(",") if s}
    dirs = [ROOT / "eval" / "audio"] + ([Path(a.extra_audio)] if a.extra_audio else [])
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "api").mkdir(parents=True, exist_ok=True)
    import urllib.request
    for name in ("meter", "health"):        # what the recording server ran (prices, ear, web lane)
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/api/{name}") as r:
            (OUT / "api" / f"{name}.json").write_text(json.dumps(sanitize(json.loads(r.read())), indent=1))
    for sc in spec["scenarios"]:
        if only and sc["id"] not in only:
            continue
        print("recording", sc["id"], flush=True)
        rec = asyncio.run(record(a.port, sc, dirs))
        rec = sanitize(rec)
        n = len(rec["events"])
        (OUT / "runs" / f"{sc['id']}.json").write_text(json.dumps(rec, separators=(",", ":")))
        for f in ([sc["file"]] if sc.get("file") else sc.get("files") or []):
            p = next(d / f for d in dirs if (d / f).exists())
            to_mp3(p, OUT / "audio" / (Path(f).stem + ".mp3"))
        last = rec["events"][-1]["at"] if n else 0
        print(f"  {n} events over {last / 1000:.1f} s", flush=True)


if __name__ == "__main__":
    main()
