"""Does Arc Transcriber (ARC_META=transcriber: Parakeet-TDT 0.6B tap + Arc heads, MPS, 0.5 s chunks / 1 s context) give
useful MID-UTTERANCE confidence on long questions, compared with Arc Mini (ARC_META=mini)?

For every real Flux trace longer than --min-dur s, both models run in external-transcript mode on the real audio + the
same Flux words, chunk by chunk (100 ms); at each Flux TurnInfo we snapshot what the model had said BY THAT AUDIO TIME
(its own algorithmic lag included; compute time not counted, i.e. optimistic for the Transcriber). Measured:
  - lag: audio time of the newest utterance read-out vs the message time (how stale the model's view is)
  - AUC of command_complete p for "these words are the whole question" over all updates, and over the decision
    window (updates after a pause >= 0.1 s, where an endpoint could fire)
  - AUC of for-me (device directed) is not meaningful here (every question is for the assistant): reported as the
    mean only
  - compute: real-time factor (compute / audio)

  .venv/bin/python eval/transcriber_eval.py [--min-dur 3.5] [--ids q13,q18]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import arcmeta  # noqa: E402
from server.ear import arc, flux_word  # noqa: E402

CC = "discourse.assistant.command_complete"


def run(model, tr):
    x, _ = sf.read(str(ROOT / "eval" / "audio" / tr["file"]), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    x = np.concatenate([x, np.zeros(int(1.6 * 16000), np.float32)])
    s = model.stream(domain="assistant", transcript="external", name_source="word", interim_source="deepgram",
                     partials=False, fallback=None)
    msgs = tr["msgs"]
    st = {"cc_p": None, "at": None, "for_me": None}
    snap, k, comp = {}, 0, 0.0
    for i in range(0, len(x), 1600):
        now = (i + 1600) / 16000
        while k < len(msgs) and msgs[k]["_t"] <= now:
            m = msgs[k]
            snap[k] = dict(st)
            if m.get("type") == "TurnInfo" and m.get("words"):
                s.add_words([dict(text=w["word"], start=w["start"], end=w["end"], is_final=m["event"] == "EndOfTurn")
                             for w in m["words"]])
            k += 1
        t0 = time.perf_counter()
        out = s.feed(x[i:i + 1600])
        comp += time.perf_counter() - t0
        lv = out.get("live") or {}
        u = lv.get("utt") or {}
        if u:
            st["cc_p"] = u.get("command_complete")
            st["for_me"] = u.get("device_directed")
            st["at"] = lv.get("t")
    s.flush()
    return snap, comp / (len(x) / 16000)


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    pts = [(a, b) for a, b in zip(y, p) if b is not None]
    if len({a for a, _ in pts}) < 2:
        return None
    return round(float(roc_auc_score([a for a, _ in pts], [b for _, b in pts])), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dur", type=float, default=3.5)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "runs" / "flux_traces.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if (r.get("dur_s") or 0) > a.min_dur and (not a.ids or r["id"] in a.ids.split(","))]
    print(len(rows), "long questions")
    models = {"mini": arc()}
    t = time.perf_counter()
    models["transcriber"] = arcmeta.transcriber()
    print(f"transcriber loaded in {time.perf_counter() - t:.1f} s")
    res = {m: {"y": [], "p": [], "yw": [], "pw": [], "lag": [], "for_me": [], "rtf": []} for m in models}
    for tr in rows:
        tis = [(i, m) for i, m in enumerate(tr["msgs"]) if m.get("type") == "TurnInfo" and m.get("words")]
        eots = [m for _, m in tis if m["event"] == "EndOfTurn"]
        if not eots:
            continue
        fin = eots[-1]
        fw = [flux_word(w["word"]) for w in fin["words"]]
        for name, mdl in models.items():
            snap, rtf = run(mdl, tr)
            R = res[name]
            R["rtf"].append(rtf)
            for i, m in tis:
                if m["event"] == "EndOfTurn":
                    continue
                ws = [flux_word(w["word"]) for w in m["words"]]
                y = int(m["turn_index"] == fin["turn_index"] and ws == fw)
                sm = snap.get(i, {})
                R["y"].append(y)
                R["p"].append(sm.get("cc_p"))
                pause = m["_t"] - m["words"][-1]["end"]
                if pause >= 0.1:
                    R["yw"].append(y)
                    R["pw"].append(sm.get("cc_p"))
                if sm.get("at") is not None:
                    R["lag"].append(m["_t"] - sm["at"])
                if sm.get("for_me") is not None:
                    R["for_me"].append(sm["for_me"])
        print(tr["id"], {n: round(res[n]["rtf"][-1], 3) for n in models}, flush=True)
    out = {}
    for name, R in res.items():
        out[name] = {"auc_cc_all_updates": auc(R["y"], R["p"]), "auc_cc_decision_window": auc(R["yw"], R["pw"]),
                     "n_updates": len(R["y"]), "n_window": len(R["yw"]),
                     "lag_s_p50": round(float(np.median(R["lag"])), 2) if R["lag"] else None,
                     "for_me_mean": round(float(np.mean(R["for_me"])), 3) if R["for_me"] else None,
                     "rtf_p50": round(float(np.median(R["rtf"])), 3)}
    print(json.dumps(out, indent=1))
    (ROOT / "runs" / "transcriber_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
