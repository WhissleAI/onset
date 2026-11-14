"""Score live-data answers (eval/arcvalue/livedata.jsonl) against ground truth fetched NOW (run it right after the
replay; the run file's "created" time and this script's fetch time are both written out).

  .venv/bin/python eval/arcvalue/score_live.py runs/arcvalue/live.json     # -> runs/arcvalue/live_scored.json

Ground truth (the router verticals' own sources, i.e. what a card would say: for weather / currency this is circular
for the card arm by construction; clock and units are exact and independent):
  weather  Open-Meteo current temperature at the city (server/verticals: geocode + forecast); correct = the answer
           names a temperature within 3 C (5 F) of it, or within 2 C of today's forecast high / low.
  fx       ECB reference rate via Frankfurter (verticals/fx.rate); correct = a number within 3 % of amount x rate.
  time     zoneinfo clock now; correct = a clock time within 20 min (12-hour times compared mod 12 h).
  units    exact conversion; correct = a number within 2 % of it.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QS = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/arcvalue/livedata.jsonl").read_text().splitlines()
      if l.strip()}
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def nums(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def temps(text: str) -> list[tuple[float, str]]:
    """(value, unit) for every temperature said: 72°F, 22 °C, 15 degrees (unit '' = unknown)."""
    out = []
    for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(?:°|degrees?|deg)?\s*(F|C|fahrenheit|celsius)?\b", text or "", re.I):
        tail = text[m.end(1):m.end(1) + 12].lower()
        if not (m.group(2) or "°" in tail or "degree" in tail):
            continue
        u = (m.group(2) or "").lower()[:1]
        out.append((float(m.group(1)), u))
    return out


def clocks(text: str) -> list[tuple[int, int, bool]]:
    """(hour, minute, has_ampm) for every clock time said."""
    out = []
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?", text or "", re.I):
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
        if ap.startswith("p") and h < 12:
            h += 12
        if ap.startswith("a") and h == 12:
            h = 0
        out.append((h, mi, bool(ap)))
    return out


async def truth(q: dict) -> dict:
    from server.verticals import fx, geo, weather
    t = q["truth"]
    k = q["kind"]
    if k == "live_weather":
        p = await geo.geocode(t["city"])
        d = await weather.forecast(p)
        return {"now_c": d["now"]["t"], "hi_c": d["days"][0]["hi"], "lo_c": d["days"][0]["lo"], "source": d["source"],
                "as_of": d.get("as_of")}
    if k == "live_fx":
        r, as_of, src, _ = await fx.rate(t["src"], t["dst"])
        return {"value": t["amount"] * r, "rate": r, "as_of": as_of, "source": src}
    if k == "live_time":
        now = dt.datetime.now(ZoneInfo(t["tz"]))
        return {"h": now.hour, "m": now.minute, "iso": now.isoformat(timespec="minutes"), "source": "zoneinfo"}
    return {"value": t["value"], "alt": t.get("alt"), "source": "exact"}


def correct(q: dict, tr: dict, ans: str) -> bool:
    k = q["kind"]
    if k == "live_weather":
        for v, u in temps(ans):
            c = (v - 32) * 5 / 9 if u == "f" else v
            tol_now = 3.0 if u != "f" else 5 * 5 / 9
            if u == "" and abs(v - tr["now_c"]) > 3 and abs((v - 32) * 5 / 9 - tr["now_c"]) <= 3:
                return True                                   # unit not said: a Fahrenheit reading
            if abs(c - tr["now_c"]) <= tol_now or abs(c - tr["hi_c"]) <= 2 or abs(c - tr["lo_c"]) <= 2:
                return True
        return False
    if k == "live_time":
        want = tr["h"] * 60 + tr["m"]
        for h, mi, ap in clocks(ans):
            def circ(a):
                x = abs(a - want) % 1440
                return min(x, 1440 - x)
            got = h * 60 + mi
            d = circ(got) if ap else min(circ(got), circ(got + 720))
            if d <= 20:
                return True
        return False
    tol = 0.03 if k == "live_fx" else 0.02
    vals = [tr["value"]] + ([tr["alt"]] if tr.get("alt") else [])
    return any(abs(n - v) <= tol * abs(v) for n in nums(ans) for v in vals)


async def main(paths):
    from server import router
    router.reset_caches()
    truths = {}
    for qid, q in QS.items():
        try:
            truths[qid] = await truth(q)
        except Exception as e:
            truths[qid] = {"error": f"{type(e).__name__}: {e}"}
    fetched = time.strftime("%Y-%m-%dT%H:%M:%S")
    out = {"truth_fetched": fetched, "truth": truths, "runs": {}}
    for p in paths:
        res = json.loads(Path(p).read_text())
        rows = []
        for r in res["rows"]:
            q = QS.get(r["id"])
            if not q:
                continue
            tr = truths[r["id"]]
            ok = None if "error" in tr else correct(q, tr, r.get("answer") or "")
            rows.append({"id": r["id"], "mode": r["mode"], "kind": q["kind"], "heard": r.get("question_heard"),
                         "answer": r.get("answer"), "answer_by": r.get("answer_by"), "ready_ms": r.get("ready_ms"),
                         "correct": ok, "truth": tr})
        out["runs"][p] = {"created": res.get("created"), "rows": rows}
        for mode in sorted({x["mode"] for x in rows}):
            g = [x for x in rows if x["mode"] == mode]
            print(f"{p} {mode:9s} correct {sum(1 for x in g if x['correct'])}/{len(g)}  " + "  ".join(
                f"{k[5:]} {sum(1 for x in g if x['kind'] == k and x['correct'])}/{sum(1 for x in g if x['kind'] == k)}"
                for k in ("live_weather", "live_fx", "live_time", "live_units")))
    dst = Path(paths[0]).with_name(Path(paths[0]).stem + "_scored.json")
    dst.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", dst)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
