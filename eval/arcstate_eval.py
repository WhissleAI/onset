"""ArcState measurements (server/arcstate.py).

voice      Arc Mini's voice embedding (diarization="voice", tapped per speech frame) on the eval audio: one mean
           embedding per file, labelled by its TTS voice (eval/questions.jsonl, showcase, conversations). Reports
           same-voice vs other-voice cosine, the equal-error threshold, and the sequential enrollment test: for each
           conversation, its presses in order (same voice -> must stay S1) plus one press by another voice (must be
           flagged). Also the cost: Arc stream compute per second of audio with diarization "voice"+tap vs the default.
keyterms   Deepgram Flux with vs without the conversation's names as keyterms, on follow-ups that re-say a name from
           an earlier turn (eval/arcstate/keyterm_set.jsonl; audio made with macOS `say` by --make-audio): name
           accuracy (the name heard exactly), WER, and time from the end of audio to EndOfTurn. Controls: the same
           follow-ups with unrelated names as keyterms, and the plain question set with unrelated keyterms (no harm).

  nice -n 10 .venv/bin/python eval/arcstate_eval.py voice    [--limit N]      -> runs/arcstate/voice.json
  nice -n 10 .venv/bin/python eval/arcstate_eval.py keyterms [--make-audio]  -> runs/arcstate/keyterms.json
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "arcstate"
AUDIO = ROOT / "eval" / "audio"
KSET = ROOT / "eval" / "arcstate" / "keyterm_set.jsonl"
KAUDIO = ROOT / "eval" / "arcstate" / "audio"       # generated, git-ignored (no audio in the repo)


def load_wav(p):
    x, sr = sf.read(str(p), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    return x


def items_voice():
    out = []
    for f in (ROOT / "eval/questions.jsonl", ROOT / "eval/showcase/questions.jsonl"):
        for l in open(f):
            d = json.loads(l)
            out.append({"file": d["file"], "voice": d.get("voice") or "?", "conv": None})
    for l in open(ROOT / "eval/conversations.jsonl"):
        d = json.loads(l)
        for i, fn in enumerate(d["files"]):
            out.append({"file": fn, "voice": "say:" + d["voice"], "conv": d["id"], "turn": i})
    return [x for x in out if (AUDIO / x["file"]).exists()]


FRAMES: dict = {}


def run_voice(args):
    from server import arcstate
    from server.ear import arc
    a = arc()
    items = items_voice()[: args.limit or None]
    embs, cost = {}, {"voice": [0.0, 0.0], "slots": [0.0, 0.0]}
    for k, it in enumerate(items):
        x = load_wav(AUDIO / it["file"])
        x = np.concatenate([x, np.zeros(8000, np.float32)])
        for mode in (("voice", "slots") if k < 20 else ("voice",)):
            kw = {"diarization": "voice"} if mode == "voice" else {}
            s = a.stream(domain="assistant", transcript="external", name_source="word", interim_source="deepgram",
                         partials=False, fallback=None, **kw)
            tap = arcstate.Tap(s) if mode == "voice" else None
            t = time.perf_counter()
            fr = []
            for i in range(0, len(x), 1600):
                s.feed(x[i:i + 1600])
                if tap is not None:
                    fr += tap.drain(0.0)
            s.flush()
            cost[mode][0] += time.perf_counter() - t
            cost[mode][1] += len(x) / 16000
            if tap is not None:
                fr += tap.drain(0.0)
                if fr:
                    embs[it["file"]] = arcstate.utt_vector([e for _, e in fr]), len(fr)
                    FRAMES[it["file"]] = np.stack([e for _, e in fr]).astype(np.float16)
        if k % 20 == 0:
            print(k, len(items), flush=True)
    files = [it for it in items if it["file"] in embs]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "voice_embs.npz", files=np.array([it["file"] for it in files]),
             E=np.stack([embs[it["file"]][0] for it in files]), F=np.array([it["file"] for it in files]))
    if FRAMES:
        np.savez(OUT / "voice_frames.npz", **{k.replace(".", "_"): v for k, v in FRAMES.items()})
    E = np.stack([embs[it["file"]][0] for it in files])
    S = E @ E.T
    V = [it["voice"] for it in files]
    same, diff = [], []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            (same if V[i] == V[j] else diff).append(float(S[i, j]))
    same, diff = np.array(same), np.array(diff)
    best = None
    for thr in np.arange(0.2, 0.99, 0.01):
        far, frr = float((diff >= thr).mean()), float((same < thr).mean())
        if best is None or abs(far - frr) < abs(best[1] - best[2]):
            best = (round(float(thr), 2), far, frr)
    # sequential enrollment, as live: each conversation's presses in order + one press of another voice (inserted
    # second), then the rest. Uses the shipped threshold.
    thr = arcstate.VOICE_SAME
    by_conv = {}
    for it in files:
        if it["conv"]:
            by_conv.setdefault(it["conv"], []).append(it)
    others = [it for it in files if it["conv"] is None]
    ok_same = n_same = ok_other = n_other = 0
    for ci, (cid, its) in enumerate(sorted(by_conv.items())):
        its.sort(key=lambda r: r["turn"])
        other = next(o for o in others[ci % len(others):] + others if o["voice"] != its[0]["voice"])
        seq = [its[0], other] + its[1:]
        st = arcstate.ArcState()
        for it in seq:
            r = st.assign(embs[it["file"]][0])
            if it is other:
                n_other += 1
                ok_other += r["who"] == "other"
            elif it is not its[0]:
                n_same += 1
                ok_same += r["who"] == "you"
    res = {"files": len(files), "voices": len(set(V)), "frames_mean": float(np.mean([embs[f["file"]][1] for f in files])),
           "same_voice_cos": {"mean": float(same.mean()), "p5": float(np.percentile(same, 5))},
           "other_voice_cos": {"mean": float(diff.mean()), "p95": float(np.percentile(diff, 95))},
           "eer_thr": best[0], "eer_far": best[1], "eer_frr": best[2],
           "shipped_thr": thr, "far_at_thr": float((diff >= thr).mean()), "frr_at_thr": float((same < thr).mean()),
           "sequential": {"same_kept": f"{ok_same}/{n_same}", "other_flagged": f"{ok_other}/{n_other}"},
           "cost_s_per_audio_s": {m: round(c / max(a_, 1e-9), 4) for m, (c, a_) in cost.items()},
           "cost_note": "first 20 files run both ways (same files); voice = diarization voice + tap"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "voice.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------------------------------------- keyterms
def make_audio():
    KAUDIO.mkdir(parents=True, exist_ok=True)
    for l in open(KSET):
        d = json.loads(l)
        p = KAUDIO / f"{d['id']}.wav"
        if p.exists():
            continue
        aiff = p.with_suffix(".aiff")
        subprocess.run(["say", "-v", d.get("voice", "Samantha"), "-o", str(aiff), d["say"]], check=True)
        x, sr = sf.read(str(aiff), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        sf.write(str(p), x, 16000)
        aiff.unlink()


def norm_words(s):
    return [w for w in re.sub(r"[^\w' ]", " ", s.lower().replace("-", " ")).split() if w]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


async def flux_once(x, keyterms):
    import websockets
    from server import config, ear
    x = np.concatenate([x, np.zeros(int(1.6 * 16000), np.float32)])
    msgs = []
    async with websockets.connect(ear.flux_url(keyterms=keyterms),
                                  additional_headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}"}) as ws:
        t0 = time.perf_counter()

        async def rx():
            async for m in ws:
                d = json.loads(m)
                d["_t"] = time.perf_counter() - t0
                msgs.append(d)
        r = asyncio.create_task(rx())
        for i in range(0, len(x), 1600):
            await ws.send((np.clip(x[i:i + 1600], -1, 1) * 32767).astype("<i2").tobytes())
            await asyncio.sleep(max(0, t0 + (i + 1600) / 16000 - time.perf_counter()))
        await ws.send(json.dumps({"type": "CloseStream"}))
        try:
            await asyncio.wait_for(r, 4)
        except Exception:  # noqa: BLE001
            pass
    eots = [m for m in msgs if m.get("event") == "EndOfTurn"]
    text = " ".join(m.get("transcript", "") for m in eots)
    speech_end = (len(x) / 16000) - 1.6
    lat = (eots[-1]["_t"] - speech_end) if eots else None
    return text, lat, [m for m in msgs if m.get("type") in ("Error", "ConfigureFailure", "Warning")]


def kterms(names):
    from server import arcstate
    from server import memory as memmod
    m = memmod.Memory()
    for i, n in enumerate(names):
        m.ents[n.lower()] = {"key": n.lower(), "name": n, "type": "thing", "turns": [len(names) - i]}
    return arcstate.ArcState().keyterms(m)


def name_ok(name, text):
    return " ".join(norm_words(name)) in " ".join(norm_words(text))


async def run_keyterms(args):
    if args.make_audio:
        make_audio()
    rows = [json.loads(l) for l in open(KSET)]
    if args.limit:
        rows = rows[: args.limit]
    pool = sorted({n for r in rows for n in r["memory"]})
    res = []
    for i, r in enumerate(rows):
        x = load_wav(KAUDIO / f"{r['id']}.wav")
        rng = np.random.default_rng(i)
        unrelated = [n for n in rng.permutation(pool).tolist() if n not in r["memory"]][: len(r["memory"])]
        row = {"id": r["id"], "name": r["name"], "say": r["say"]}
        on = kterms(r["memory"])                         # what the app would send: arcstate.keyterms over memory
        row["keyterms"] = on
        for arm, kt in (("off", None), ("on", on), ("unrelated", kterms(unrelated))):
            text, lat, errs = await flux_once(x, kt)
            e, n = wer(r["say"], text)
            row[arm] = {"text": text, "name_ok": name_ok(r["name"], text), "err": e, "n": n,
                        "eot_ms": None if lat is None else round(lat * 1000), "errors": errs[:2]}
        res.append(row)
        print(r["id"], r["name"], "|", row["off"]["text"], "|", row["on"]["text"], flush=True)
    ctrl = []
    if args.control:
        qs = [json.loads(l) for l in open(ROOT / "eval/questions.jsonl")][: args.control]
        for i, q in enumerate(qs):
            x = load_wav(AUDIO / q["file"])
            kt = np.random.default_rng(100 + i).permutation(pool).tolist()[:6]
            row = {"id": q["id"]}
            for arm, k in (("off", None), ("unrelated", kt)):
                text, lat, _ = await flux_once(x, k)
                e, n = wer(q["text"], text)
                row[arm] = {"text": text, "err": e, "n": n, "eot_ms": None if lat is None else round(lat * 1000),
                            "inserted": [t for t in kt if k and name_ok(t, text) and not name_ok(t, q["text"])]}
            ctrl.append(row)
            print(q["id"], "|", row["off"]["text"], "|", row["unrelated"]["text"], flush=True)

    def agg(rows_, arm):
        e = sum(r[arm]["err"] for r in rows_)
        n = sum(r[arm]["n"] for r in rows_)
        lats = [r[arm]["eot_ms"] for r in rows_ if r[arm]["eot_ms"] is not None]
        out = {"wer": round(e / max(n, 1), 4), "eot_p50_ms": float(np.median(lats)) if lats else None}
        if "name_ok" in rows_[0][arm]:
            out["names_right"] = f"{sum(r[arm]['name_ok'] for r in rows_)}/{len(rows_)}"
        if "inserted" in rows_[0][arm]:
            out["inserted_names"] = sum(len(r[arm]["inserted"]) for r in rows_)
        return out
    summ = {"followups": {arm: agg(res, arm) for arm in ("off", "on", "unrelated")}, "n": len(res)}
    if ctrl:
        summ["control_questions"] = {arm: agg(ctrl, arm) for arm in ("off", "unrelated")}
        summ["control_n"] = len(ctrl)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "keyterms.json").write_text(json.dumps({"summary": summ, "rows": res, "control": ctrl}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["voice", "keyterms"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--make-audio", action="store_true")
    ap.add_argument("--control", type=int, default=0, help="also N plain questions with unrelated keyterms")
    a = ap.parse_args()
    if a.what == "voice":
        run_voice(a)
    else:
        asyncio.run(run_keyterms(a))
