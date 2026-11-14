"""Per-message feature snapshots of the 80 real Flux traces, for the confidence-fusion fits and evals.

Every TurnInfo message of runs/flux_traces.jsonl is replayed in arrival order through
  - the Arc Tagger exactly as Session feeds it (eval/flux_gate_eval.TagState), and
  - Arc Mini audio metadata in external-transcript mode on the real audio + the same words (eval/flux_gate_eval.ArcReplay)
and we save what a live Session would know when that message arrives: the words with Flux confidence, per-word revision
history (server/confidence.WordTrack), the tagger's spans (score / thr / ver / alt), intents / acts, command_complete,
the gate, Arc audio cc_p / cc events / for-me, Flux's end_of_turn_confidence and the pause since the last word ended.
Labels: `complete` (the message's words are the question's final words, same turn) and, per word, `final_ok` (this
word survives unchanged into the final transcript).

Run: .venv/bin/python eval/conf_features.py [--no-arc]   -> runs/conf_features.jsonl (~1 min with Arc)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from flux_gate_eval import ArcReplay, TagState, norm_words  # noqa: E402
from server import confidence, frame  # noqa: E402
from server.session import gate_open  # noqa: E402
from server.tagger import Tagger  # noqa: E402

CC_TAG = "discourse.assistant.command_complete"

OUT = ROOT / "runs" / "conf_features.jsonl"


def span_rec(s):
    return {"type": s["type"], "family": s.get("family"), "start": s["start"], "end": s["end"], "text": s.get("text"),
            "score": s.get("score"), "thr": s.get("thr"), "ver": s.get("ver"), "alt": s.get("alt")}


class ArcReplay2:
    """ArcReplay + the director-v2 streams, snapshotted at each message's arrival (what Arc had said before it):
    the external-mode Arc Mini metadata state (cc_p / cc event / for-me, as ArcReplay) + server.arcmeta.ArcExtra (the
    heads the assistant domain doesn't ship, audio level) + the Arc ear's own words (builtin decoder, a second stream on
    the same audio): `aw` = [[word, start, end], ...] of its open partial, `af` = its finalized utterances so far."""

    def __init__(self, ear_words: bool = True):
        from server.ear import arc
        self.a = arc()
        self.ear_words = ear_words

    def run(self, tr, audio_dir):
        import numpy as np
        import soundfile as sf
        from server.arcmeta import ArcExtra
        x, _ = sf.read(str(Path(audio_dir) / tr["file"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        x = np.concatenate([x, np.zeros(int(1.6 * 16000), np.float32)])
        s = self.a.stream(domain="assistant", transcript="external", name_source="word", interim_source="deepgram",
                          partials=False, fallback=None)
        b = self.a.stream(domain="assistant", transcript="builtin", partials=True) if self.ear_words else None
        xe = ArcExtra()
        msgs = tr["msgs"]
        state = {"cc_p": None, "cc_t": None, "for_me": None}
        aw, af = [], []
        snap = {}
        self.cc_emits = []
        k = 0
        for i in range(0, len(x), 1600):
            now = (i + 1600) / 16000
            while k < len(msgs) and msgs[k]["_t"] <= now:
                m = msgs[k]
                snap[k] = dict(state, x2=xe.state(), aw=[list(w) for w in aw], af=list(af[-3:]))
                if m.get("type") == "TurnInfo" and m.get("words"):
                    s.add_words([dict(text=w["word"], start=w["start"], end=w["end"],
                                      is_final=m["event"] == "EndOfTurn") for w in m["words"]])
                k += 1
            out = s.feed(x[i:i + 1600])
            xe.step(s, x[i:i + 1600])
            u = (out.get("live") or {}).get("utt") or {}
            if u:
                state["cc_p"] = u.get("command_complete")
                state["for_me"] = u.get("device_directed")
            for e in out.get("events") or []:
                if e.get("tag") == CC_TAG:
                    state["cc_t"] = e.get("t")
                    self.cc_emits.append((now, e.get("t")))
            if b is not None:
                ob = b.feed(x[i:i + 1600])
                for fu in ob.get("transcript") or []:
                    af.append([fu.get("text", ""), fu.get("start"), fu.get("end")])
                    aw = []
                p = ob.get("transcript_partial")
                if p is not None:
                    aw = [(w["text"], w.get("start"), w.get("end")) for w in (p.get("words") or [])] if not p.get("final") else []
        s.flush()
        if b is not None:
            b.flush()
        return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(ROOT / "runs" / "flux_traces.jsonl"))
    ap.add_argument("--no-arc", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--audio-dir", default=str(ROOT / "eval" / "audio"))
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th trace from i")
    ap.add_argument("--v2", action="store_true", help="director v2 streams: Arc extra heads + level, the Arc ear's words, "
                    "Flux word starts")
    ap.add_argument("--no-ear-words", action="store_true", help="--v2 without the Arc ear's own decoder (faster)")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.traces).read_text().splitlines() if x.strip()]
    si, sn = (int(v) for v in a.shard.split("/"))
    rows = rows[si::sn]
    done = set()
    if Path(a.out).exists():                  # resumable
        done = {json.loads(x)["id"] for x in Path(a.out).read_text().splitlines() if x.strip()}
    rows = [r for r in rows if r["id"] not in done]
    t = Tagger(domain="assistant")
    cc_scores = {}                                  # word index -> score of the newest command_complete event there
    _poll = t.poll

    def poll():
        r = _poll()
        for e in r.get("events") or []:
            if e["type"] == "discourse.assistant.command_complete" and e.get("word") is not None:
                cc_scores[e["word"]] = float(e.get("score") or 0)
        return r
    t.poll = poll
    ts = TagState(t)
    if a.v2:
        arc = ArcReplay2(ear_words=not a.no_ear_words)
    else:
        arc = None if a.no_arc else ArcReplay()
        if arc is not None:
            arc.audio_dir = a.audio_dir
    out = open(a.out, "a")
    for tr in rows:
        msgs = tr["msgs"]
        tis = [(i, m) for i, m in enumerate(msgs) if m.get("type") == "TurnInfo"]
        eots = [m for _, m in tis if m["event"] == "EndOfTurn" and norm_words(m)]
        if not eots:
            continue
        final = eots[-1]
        final_words = norm_words(final)
        last_end = float(final["words"][-1]["end"])
        snap = (arc.run(tr, a.audio_dir) if a.v2 else arc.run(tr)) if arc else {}
        cc_emits = list(arc.cc_emits) if arc else []
        ts.reset()
        cc_scores.clear()
        track = confidence.WordTrack()
        cur_turn = None
        recs = []
        for i, m in tis:
            if m["turn_index"] != cur_turn:
                ts.reset()
                cc_scores.clear()
                track = confidence.WordTrack()
                cur_turn = m["turn_index"]
            ws = norm_words(m)
            if not ws:
                continue
            ts.sync(ws)
            confs = [float(w.get("confidence") or 0) for w in m["words"] if norm_words({"words": [w]})]
            starts = [float(w["start"]) for w in m["words"] if norm_words({"words": [w]})]
            ends = [float(w["end"]) for w in m["words"] if norm_words({"words": [w]})]
            hist = track.update(ws)
            f = ts.frame()
            cc_score = cc_scores.get(ts.cc_at) if ts.cc_at is not None else None
            sm = snap.get(i, {})
            w_end = ends[-1]
            same_turn = m["turn_index"] == final["turn_index"]
            fin_toks = [frame.norm(w) for w in final_words]
            toks = [frame.norm(w) for w in ws]
            final_ok = [same_turn and k < len(fin_toks) and toks[k] == fin_toks[k] for k in range(len(toks))]
            rec = {
                "id": tr["id"], "text": tr["text"], "i": i, "t": m["_t"], "event": m["event"], "turn": m["turn_index"],
                "eot": float(m.get("end_of_turn_confidence") or 0), "words": ws, "conf": confs, "ends": ends,
                "revs": hist["revs"], "age": hist["age"], "pause": round(m["_t"] - w_end, 3),
                "complete": same_turn and ws == final_words, "final_ok": final_ok,
                "last_end": last_end, "final_t": final["_t"], "final_turn": final["turn_index"],
                "spans": [span_rec(s) for s in ts.spans],
                "intent": [{"tag": x["tag"], "p": x["p"], "thr": x.get("thr"), "on": x.get("on")}
                           for x in ((ts.current or {}).get("intent") or [])],
                "act": [{"tag": x["tag"], "p": x["p"], "thr": x.get("thr"), "on": x.get("on")}
                        for x in ((ts.current or {}).get("act") or [])],
                "cc_at": ts.cc_at, "cc_score": cc_score, "gate": bool(gate_open(f, ts.cc_at, ts.spans) and f.has_content()),
                "has_content": f.has_content(), "act_question": f.act_question, "content_idx": list(f.content_idx),
                "slots": [list(s) for s in f.slots], "query": f.query(), "fintent": f.intent, "fintent_p": f.intent_p,
                "arc_cc_p": sm.get("cc_p"), "arc_cc_t": sm.get("cc_t"), "arc_for_me": sm.get("for_me"),
            }
            # Arc cc event for exactly these words arriving before Flux's next message (the eager+arc wait path)
            nxt = next((mm["_t"] for _, mm in tis if mm["_t"] > m["_t"]), 1e9)
            if a.v2:
                rec["starts"] = starts
                rec["x2"], rec["aw"], rec["af"] = sm.get("x2"), sm.get("aw"), sm.get("af")
            rec["arc_wait_t"] = next((now for now, et in cc_emits
                                      if m["_t"] <= now < nxt and et is not None and et >= w_end - 0.6), None)
            recs.append(rec)
        for r in recs:
            out.write(json.dumps(r) + "\n")
        print(tr["id"], len(recs), flush=True)
    out.close()
    t.close()


if __name__ == "__main__":
    main()
