"""Phase 1 offline measure: retrieval-in-the-loop entity correction (server/heard.py) on real Deepgram Flux traces.

Each trace has the scripted text (what was said) and Flux's words with per-word confidence and revision history.
For every trace: final Flux words -> fused word confidence (server/confidence.py, from the revision history) -> Arc
Tagger spans (word by word, as live) -> frame -> heard.propose(). Each proposed fix is scored against the script:
  correct      the span was misheard and the fix spells what was said ("mari guri" -> Marie Curie)
  wrong        the span was misheard and the fix is another name
  false        the span was heard right and the fix changed it (the worst case)
Recall: of the misheard NAME spans (the script has a capitalised word there; letters differ), how many were fixed
right; "fixable" = the said name is in the candidate set at all (the leads index here: no speculative results offline).

  .venv/bin/python eval/loop_asr_eval.py [--traces PATH] [--limit N] [--out runs/loop/asr_eval.json]
"""
import argparse, collections, difflib, gzip, json, os, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOOP_ASR_FIX", "1")
from server import confidence, heard, config  # noqa: E402
from server import frame as framemod  # noqa: E402
from server.session import clean  # noqa: E402
from server.tagger import Tagger  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traces", default="/Users/karan/Desktop/work/whissle/arc_tagger_handoff/data/sources/flux_traces.jsonl.gz")
ap.add_argument("--srcs", default="director,eval80,pii")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=str(ROOT / "runs/loop/asr_eval.json"))
ap.add_argument("--no-conf", action="store_true", help="drop word confidence (an ear without it: Arc's own)")
args = ap.parse_args()

T = Tagger(domain="assistant")
GAZ = heard.leads()


def toks_of(text):
    return [t for t in (framemod.norm(w) for w in text.split()) if t]


def heard_words(tr):
    turns = json.loads(tr["turns"]) if isinstance(tr["turns"], str) else tr["turns"]
    ups = json.loads(tr["updates"]) if isinstance(tr["updates"], str) else tr["updates"]
    words = [w for _, turn in turns for w in turn]
    if not words and ups:
        words = ups[-1][3]
    # revision history: replay every update (earlier turns' final words as the prefix)
    finals = {ti: [clean(w[0]) for w in turn] for ti, turn in turns}
    track = confidence.WordTrack()
    h = {"revs": [], "age": []}
    for u in ups:
        ti = u[2]
        prefix = [w for k in sorted(finals) if k < ti for w in finals[k]]
        ws = prefix + [clean(w[0]) for w in u[3]]
        ws = [w for w in ws if w]
        if ws:
            h = track.update(ws)
    ws = [(clean(w[0]), w[3] if len(w) > 3 else None) for w in words]
    ws = [(w, c) for w, c in ws if w]
    return [w for w, _ in ws], [c for _, c in ws], h


def tag(words):
    T.reset()
    spans, current, events = [], None, set()
    for i in range(len(words)):
        T.push(words[i:i + 1])
        r = T.poll()
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            current = r["current"]
        for u in r.get("utterances") or []:
            current = u
        for e in r.get("events") or []:
            events.add(e["type"])
    return spans, current, events


def cap_positions(text):
    """Script token indices that are capitalised names (not the first word, not "I")."""
    out = set()
    k = 0
    for i, w in enumerate(text.split()):
        t = framemod.norm(w)
        if not t:
            continue
        if i > 0 and w[:1].isupper() and t != "i" and not t.startswith("i'"):
            out.add(k)
        k += 1
    return out


def same(a: str, b: str) -> bool:
    return heard.letters(heard.spoken(a)) == heard.letters(heard.spoken(b))


def main():
    rows = []
    stats = collections.Counter()
    ex = collections.defaultdict(list)
    srcs = set(args.srcs.split(","))
    t0 = time.time()
    n = 0
    with gzip.open(args.traces, "rt") as fh:
        for line in fh:
            tr = json.loads(line)
            if tr["src"] not in srcs or not tr.get("intended"):
                continue
            n += 1
            if args.limit and n > args.limit:
                break
            words, confs, h = heard_words(tr)
            if not words:
                continue
            spans, current, events = tag(words)
            f = framemod.build(words, spans, current, events, local_ok=False)
            wc = []
            if not args.no_conf and any(c is not None for c in confs) and len(h["revs"]) == len(words):
                fused = confidence.word_confs(words, confs, h["revs"], h["age"], spans)
                wc = heard.suspect_conf(confs, fused)
            fixes = heard.propose(f, wc, None)
            said = toks_of(tr["intended"])
            caps = cap_positions(tr["intended"])
            ht = [framemod.norm(w) for w in words]
            sm = difflib.SequenceMatcher(a=said, b=ht, autojunk=False)
            ops = sm.get_opcodes()
            # misheard name segments (denominator)
            mis = []
            for op, i1, i2, j1, j2 in ops:
                if op == "replace" and set(range(i1, i2)) & caps:
                    s_seg, h_seg = " ".join(said[i1:i2]), " ".join(ht[j1:j2])
                    if not same(s_seg, h_seg):
                        mis.append((i1, i2, j1, j2, s_seg, h_seg))
            stats["traces"] += 1
            stats["misheard_names"] += len(mis)
            for m in mis:
                sp = heard.spoken(m[4])
                if sp in GAZ.spoken:
                    stats["fixable"] += 1
                    if wc and m[3] > m[2] and max(wc[m[2]:m[3]]) < config.LOOP_KEEP_CONF:
                        stats["fixable_reachable"] += 1     # in the index AND no heard word is high-confidence
                        if len(ex["reachable"]) < 60:
                            ex["reachable"].append({"said": m[4], "heard": m[5], "conf": [round(c, 2) for c in wc[m[2]:m[3]]],
                                                    "q": " ".join(words)})
                if wc and m[3] > m[2] and max(wc[m[2]:m[3]]) < config.LOOP_KEEP_CONF:
                    stats["reachable"] += 1
            fixed_mis = set()
            for fx in fixes:
                # the script segment under the fix's heard span
                seg, misheard = None, False
                for op, i1, i2, j1, j2 in ops:
                    if j1 < fx.b and fx.a < j2 or (j1 == j2 and fx.a <= j1 < fx.b):
                        if op == "equal":
                            seg = (seg or []) + said[i1 + max(0, fx.a - j1): i1 + min(j2, fx.b) - j1]
                        else:
                            misheard = True
                            seg = (seg or []) + said[i1:i2]
                seg = " ".join(seg or [])
                ok = same(fx.name, seg) or (misheard and heard.letters(heard.spoken(fx.name)) in heard.letters(heard.spoken(seg)) and
                                            len(heard.letters(fx.name)) >= 4)
                if not misheard:
                    kind = "false"
                elif ok:
                    kind = "correct"
                else:
                    kind = "wrong"
                stats[kind] += 1
                for k, m in enumerate(mis):
                    if m[2] < fx.b and fx.a < m[3] and ok:
                        fixed_mis.add(k)
                if len(ex[kind]) < 40:
                    ex[kind].append({"id": tr["id"], "said": tr["intended"], "heard": " ".join(words),
                                     "fix": f"{fx.said} -> {fx.name}", "seg": seg, "score": round(fx.score, 3),
                                     "src": fx.src, "type": fx.kind,
                                     "conf": [round(c, 2) for c in wc[fx.a:fx.b]] if wc else None})
            stats["recalled"] += len(fixed_mis)
            if mis and len(ex["missed"]) < 60 and not fixed_mis:
                ex["missed"].append({"id": tr["id"], "said": tr["intended"], "heard": " ".join(words),
                                     "seg": [(m[4], m[5]) for m in mis],
                                     "conf": [round(c, 2) for c in wc] if wc else None})
    fx = stats["correct"] + stats["wrong"] + stats["false"]
    summary = {
        "traces": stats["traces"], "fixes": fx, "correct": stats["correct"], "wrong": stats["wrong"],
        "false_on_correct_words": stats["false"],
        "precision": round(stats["correct"] / fx, 3) if fx else None,
        "false_rate_per_1k_traces": round(1000 * stats["false"] / max(1, stats["traces"]), 2),
        "misheard_name_spans": stats["misheard_names"], "fixable_in_leads": stats["fixable"],
        "reachable_low_conf": stats["reachable"], "fixable_and_reachable": stats["fixable_reachable"],
        "recalled": stats["recalled"],
        "recall": round(stats["recalled"] / max(1, stats["misheard_names"]), 3),
        "recall_fixable": round(stats["recalled"] / max(1, stats["fixable"]), 3),
        "secs": round(time.time() - t0, 1),
        "config": {k: getattr(config, k) for k in ("LOOP_KEEP_CONF", "LOOP_SUSPECT_CONF", "LOOP_LOW_CONF",
                                                   "LOOP_FIX_MIN", "LOOP_FIX_MARGIN")} | {"no_conf": args.no_conf},
    }
    print(json.dumps(summary, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "examples": ex}, indent=1))


main()
