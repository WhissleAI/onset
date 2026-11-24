"""Smoke test any running Onset engine (Mac or Windows): types a multi-question conversation into /ws the way the app
page does and checks every answer. Exit 0 = all passed.

    python scripts/smoke_typed.py --port 51990
    python scripts/smoke_typed.py --start        # starts macapp/server_main.py itself (env as the app sets it), then stops it

Each check is (question, words the answer must contain, any of). The chain covers facts, a follow-up that needs the
previous answer ("how old is he"), a short follow-up ("and canada"), a local card (units) and an open web question.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAIN = [
    ("Who is the prime minister of India", ["modi"]),
    ("how old is he", ["modi"]),
    ("what is the capital of australia", ["canberra"]),
    ("and canada", ["ottawa"]),
    ("how many ounces in a cup", ["8"]),
    ("what are five good books to read", []),            # any answer or results, no crash
]


async def ask(port: int) -> list[tuple[str, str, bool, float]]:
    import websockets
    out = []
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
        while json.loads(await ws.recv()).get("t") != "ready":
            pass
        for rid, (text, want) in enumerate(CHAIN, 1):
            await ws.send(json.dumps({"t": "start", "typed": True, "modes": ["headstart"], "rid": rid}))
            ev: list[dict] = []

            async def reader():
                async for raw in ws:
                    e = json.loads(raw)
                    if e.get("rid") not in (None, rid):
                        continue
                    ev.append(e)
                    if e.get("t") == "run_done":
                        return
            rd = asyncio.create_task(reader())
            t0 = time.time()
            for k in range(1, len(text) + 1, 3):
                await ws.send(json.dumps({"t": "type", "rid": rid, "text": text[:k]}))
                await asyncio.sleep(0.08)
            await ws.send(json.dumps({"t": "type", "rid": rid, "text": text, "final": True}))
            try:
                await asyncio.wait_for(rd, 60)
            except asyncio.TimeoutError:
                pass
            ans = "".join(e.get("delta", "") for e in ev if e.get("t") == "answer" and e.get("mode") == "headstart")
            met = next((e for e in ev if e.get("t") == "metric" and e.get("mode") == "headstart"), {})
            ans = ans or met.get("answer") or ""
            srcs = next((e.get("docs") or [] for e in reversed(ev) if e.get("t") == "sources"), [])
            ok = bool(ans or srcs) and all(w in ans.lower() for w in want)
            out.append((text, ans.replace("\n", " ")[:140], ok, time.time() - t0))
            await asyncio.sleep(1.0)
    return out


def start_server(port: int) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env.setdefault("ONNX_RUNTIME", "1")
    env.setdefault("WEB_BACKEND", "duckduckgo")
    env.setdefault("EAR", "arc")
    return subprocess.Popen([sys.executable, "-I", "-B", str(ROOT / "macapp" / "server_main.py"), "--port", str(port)],
                            cwd=str(ROOT), env=env, creationflags=0x08000000 if sys.platform == "win32" else 0)


def wait_warm(port: int, timeout: float = 300) -> bool:
    t = time.time()
    while time.time() - t < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
                if json.loads(r.read()).get("warm"):
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--start", action="store_true", help="start the engine from this folder (MODELS_DIR etc. from env)")
    a = ap.parse_args()
    proc = None
    port = a.port
    if a.start:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = port or s.getsockname()[1]
        proc = start_server(port)
    try:
        if not port:
            print("give --port of a running engine, or --start")
            return 2
        print(f"waiting for the engine on :{port} to warm up…")
        if not wait_warm(port):
            print("FAIL: the engine did not warm up (see its log)")
            return 1
        rows = asyncio.run(ask(port))
    finally:
        if proc:
            proc.terminate()
    bad = 0
    for text, ans, ok, secs in rows:
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {secs:4.1f}s  {text!r} -> {ans!r}")
    print(f"{len(rows) - bad}/{len(rows)} passed")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
