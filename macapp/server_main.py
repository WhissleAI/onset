"""Entry point the macOS app runs with its bundled Python: the same server as `uvicorn server.app:app`, plus

- a parent watchdog: if the app goes away without stopping us (crash, force quit), the server exits within ~1 s;
- port + host from the command line (the app picks a free localhost port; never 8770 unless it is free);
- quiet, key-free logging (keys come in as env vars from the Keychain and are never printed).

    python macapp/server_main.py --port 8791 [--parent-pid PID]
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def watch_parent(pid: int):
    while True:
        time.sleep(1.0)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            os._exit(0)
        except PermissionError:
            pass
        if os.getppid() == 1:           # re-parented to launchd: the app is gone
            os._exit(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--parent-pid", type=int, default=0)
    a = ap.parse_args()
    if a.parent_pid:
        threading.Thread(target=watch_parent, args=(a.parent_pid,), daemon=True, name="parent-watch").start()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import uvicorn
    from server import config
    keys = [k for k in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "DEEPGRAM_API_KEY", "TELEM_API_KEY") if os.environ.get(k)]
    print(f"[server] {config.APP_NAME}: port {a.port}, models {config.MODELS_DIR}, data {config.DATA_DIR}, "
          f"ear {config.EAR}, web {config.WEB_BACKEND}, keys set: {', '.join(keys) or 'none'}", flush=True)
    uvicorn.run("server.app:app", host=a.host, port=a.port, log_level="warning", ws_max_size=16 * 1024 * 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
