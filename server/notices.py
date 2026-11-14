"""Plain one-line notices for the top of the UI: shown only when something is actually wrong, each with what to do.

Built from the same health the `degraded` dict comes from (lanes.web_health(), answer.health()); `degraded` stays for
scripts/preflight.py and the tests, these are the words a person sees. Empty list = everything live = no banner.

Each notice: {"id", "level" ("warn"), "text", and optionally "action" ("settings": the Mac app opens its Settings
window), "action_label", "hint" (where to fix it when there is no Settings window: the web demo)}."""
from __future__ import annotations

import os
import re

_KEY_ERR = re.compile(r"api[_ ]?key|authenticat|unauthori[sz]ed|\b401\b|x-api-key|permission", re.I)


def _missing_key(a: dict) -> bool:
    """The answer model is down because there is no (valid) Anthropic key, not because the network is."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return True
    return bool(_KEY_ERR.search(str(a.get("last_error") or "")))


def build(web: dict, ans: dict) -> list[dict]:
    out: list[dict] = []
    if ans.get("status") == "down":
        if _missing_key(ans):
            out.append({"id": "answer-key", "level": "warn", "text": "Add an Anthropic key for full answers",
                        "action": "settings", "action_label": "Open Settings",
                        "hint": "set ANTHROPIC_API_KEY in .env, then restart the server"})
        else:
            out.append({"id": "answer-down", "level": "warn",
                        "text": "Answer service unreachable · showing the best source sentence",
                        "hint": "answers come back on their own when it is reachable"})
    st = web.get("status")
    if st == "cache":
        out.append({"id": "web-saved", "level": "warn", "text": "Live web off · using saved results",
                    "hint": "offline mode (WEB_BACKEND=cache)"})
    elif st == "down":
        if web.get("cache_entries"):
            out.append({"id": "web-down", "level": "warn", "text": "Live web unavailable · using saved results",
                        "hint": "retrying in the background"})
        else:
            out.append({"id": "web-down", "level": "warn",
                        "text": "Live web unavailable · web questions have no sources right now",
                        "hint": "retrying in the background"})
    return out
