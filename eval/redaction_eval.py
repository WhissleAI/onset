"""Redaction eval: the Arc Tagger's profanity / toxicity heads (server/safety.py, TAGGER_SAFETY=1) vs the old hard-coded
word list, on eval/redaction/utterances.jsonl (60 labelled utterances: clean, profane, toxic, self-harm, PII and
false-positive traps such as "Dick Van Dyke", "Scunthorpe", "pissarro", "craps game").

Each utterance streams into a fresh tagger word by word, the way a Session feeds it (push + poll per word, endpoint at
the end). Reports
  - word-level mask precision / recall (a masked word must be one the label masks)
  - turn-level toxic flag (insult / hate / threat) and self-harm flag precision / recall
  - false positives on the trap + clean rows
  - added latency per streaming update: poll time with the safety heads emitted vs without

  .venv/bin/python eval/redaction_eval.py [--out eval/redaction/results.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import safety  # noqa: E402
from server.tagger import Tagger  # noqa: E402

BASE = "entity,number,pii,events,intent,act,safety,state"     # the Session tagger's default emit list


def stream(t: Tagger, words: list[str]) -> tuple[list, dict | None, list[float]]:
    t.reset()
    spans, cur, times = [], None, []
    for i, w in enumerate(words):
        t0 = time.perf_counter()
        t.push([w])
        r = t.poll()
        times.append((time.perf_counter() - t0) * 1000)
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            cur = r["current"]
    t.push([], endpoint=True)
    r = t.poll()
    a0 = r.get("a0", 0)
    spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
    for u in r.get("utterances") or []:
        cur = u
    return spans, cur, times


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": round(p, 3), "recall": round(r, 3), "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "eval/redaction/utterances.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "eval/redaction/results.json"))
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.data).read_text().splitlines() if x.strip()]
    t_on = Tagger(domain="assistant", emit=safety.emit_list(BASE), options=safety.TAGGER_OPTIONS)
    t_off = Tagger(domain="assistant", emit=BASE)
    arms = {"tagger": {"tp": 0, "fp": 0, "fn": 0, "flag": [0, 0, 0], "sh": [0, 0, 0], "fp_rows": [], "miss": []},
            "list": {"tp": 0, "fp": 0, "fn": 0, "flag": [0, 0, 0], "sh": [0, 0, 0], "fp_rows": [], "miss": []}}
    lat_on, lat_off = [], []
    sh_own = [0, 0, 0]
    per_row = []
    for r in rows:
        words = r["text"].split()
        toks = [safety.norm(w) for w in words]
        gold = {i for i, t in enumerate(toks) if t in set(r["mask"])}
        spans, cur, tm_on = stream(t_on, words)
        _, _, tm_off = stream(t_off, words)
        lat_on += tm_on
        lat_off += tm_off
        out = {"id": r["id"], "cat": r["cat"], "text": r["text"]}
        for arm, tag in (("tagger", True), ("list", False)):
            m = safety.mask_idx(words, spans, tagger=tag)
            flag = safety.turn_flag(words, spans, cur, tagger=tag, masked=m)
            # the old pipeline had no turn flags at all: the list arm gets none (self-harm cues are new too)
            if arm == "list":
                flag = None
            S = arms[arm]
            got = set(m)
            S["tp"] += len(got & gold)
            S["fp"] += len(got - gold)
            S["fn"] += len(gold - got)
            if got - gold:
                S["fp_rows"].append(f"{r['id']}: {' '.join(toks[i] for i in sorted(got - gold))}")
            if gold - got:
                S["miss"].append(f"{r['id']}: {' '.join(toks[i] for i in sorted(gold - got))}")
            gtox = r["flag"] in ("insult", "hate", "threat")
            ptox = flag in ("insult", "hate", "threat")
            S["flag"][0] += gtox and ptox
            S["flag"][1] += ptox and not gtox
            S["flag"][2] += gtox and not ptox
            gsh, psh = r["flag"] == "self_harm", flag == "self_harm"
            S["sh"][0] += gsh and psh
            S["sh"][1] += psh and not gsh
            S["sh"][2] += gsh and not psh
            out[arm] = {"masked": [toks[i] for i in sorted(got)], "flag": flag}
        # the tagger's own self-harm flags alone (no cue phrases), for honesty about where self-harm recall comes from
        own = any(x.get("on") and x.get("tag") in ("safety.crisis.self_harm", "safety.toxicity.self_harm")
                  for x in (cur or {}).get("safety") or [])
        sh_own[0] += own and r["flag"] == "self_harm"
        sh_own[1] += own and r["flag"] != "self_harm"
        sh_own[2] += (not own) and r["flag"] == "self_harm"
        per_row.append(out)
    res = {"n": len(rows), "arms": {}}
    for arm, S in arms.items():
        res["arms"][arm] = {"mask": prf(S["tp"], S["fp"], S["fn"]), "toxic_flag": prf(*S["flag"]),
                            "self_harm": prf(*S["sh"]), "false_masks": S["fp_rows"], "missed": S["miss"]}
    res["self_harm_tagger_flag_only"] = prf(*sh_own)
    q = lambda xs, p: round(statistics.quantiles(xs, n=100)[p - 1], 3)  # noqa: E731
    res["latency_ms_per_update"] = {"with_safety_p50": q(lat_on, 50), "with_safety_p95": q(lat_on, 95),
                                    "without_p50": q(lat_off, 50), "without_p95": q(lat_off, 95),
                                    "added_p50": round(q(lat_on, 50) - q(lat_off, 50), 3), "updates": len(lat_on)}
    res["rows"] = per_row
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
