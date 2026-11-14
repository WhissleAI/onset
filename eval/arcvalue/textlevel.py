"""Text-level Arc-value checks (no audio, no network, no model calls): what Arc's read-out changes in what is SENT
and what is ANSWERED, vs the Arc-off arm (frame.plain: the raw transcript is the query, everything is answered).

  .venv/bin/python eval/arcvalue/textlevel.py            # -> runs/arcvalue/textlevel.json (+ a summary on stdout)

Inputs (all transcripts are what Deepgram Flux heard, not the scripts):
  privacy    arc_tagger_handoff/data/from_director/pii_lines.jsonl (1210 lines, 670 with gold hidden spans, per-word
             hide flags aligned to the Flux words) + the gold set's PII items (gold.jsonl, labels.pii).
             Leak = a hidden (identifying) word appears in a string that leaves the machine for a SEARCH: the web
             query, the entity query, the Tavily-shaped query, the free-API arguments of the router plan. Reported
             separately: hidden words in the answer-model prompt (question text).
  open mic   from_director/openmic_flux_turns.jsonl test split (86 scenes): one Flux turn = one ear turn; it is a
             request if it covers an addressed utterance. Arc = session.open_mic_skip on the turn's frame (the
             default open-mic gate); plain answers every turn. + gold.jsonl labels.respond.
  freshness  gold.jsonl labels.freshness. Arc's Tavily parameters = lanes.web_plan(intent, text, final, hints) (what
             the Tavily lane would send; the Wikipedia lane sends none of them) and the router lane (a live-data card
             is fresh by construction). Plain = topic general, no time filter.
  correction gold.jsonl labels.correction: does the query keep the repair and drop the reparandum?
  query      gold.jsonl labels.query: content-word recall / precision of the web query vs the gold query span.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from server import config, frame as framemod, lanes, narrow, router  # noqa: E402
from server.session import clean, open_mic_skip  # noqa: E402
from server.tagger import Tagger  # noqa: E402

DIRECTOR = Path("/Users/karan/Desktop/work/whissle/arc_tagger_handoff/data/from_director")
GOLD = Path("/Users/karan/Desktop/work/whissle/arc_tagger_handoff/data/gold/gold.jsonl")
OUT = ROOT / "runs" / "arcvalue" / "textlevel.json"

_T: Tagger | None = None


def arc_frame(words: list[str]) -> framemod.Frame:
    """The final frame the live session builds: words pushed one at a time (as partials arrive), then the endpoint."""
    global _T
    if _T is None:
        _T = Tagger(domain="assistant")
    t = _T
    t.reset()
    words = [w for w in (clean(x) for x in words) if w]
    spans, current, events, prev_intent = [], None, set(), None

    def take(r):
        nonlocal spans, current
        a0 = r.get("a0", 0)
        spans = [s for s in spans if s["end"] <= a0] + r.get("spans", [])
        if r.get("current") is not None:
            current = r["current"]
        for u in r.get("utterances") or []:
            current = u
        for e in r.get("events") or []:
            events.add(e["type"])
    for i in range(len(words)):
        t.push(words[i:i + 1])
        take(t.poll())
        f = framemod.build(words[:i + 1], spans, current, events, prev_intent)
        if f.intent.startswith("discourse.intent."):
            prev_intent = (f.intent, f.intent_p)
    t.endpoint()
    take(t.poll())
    return framemod.build(words, spans, current, set(), prev_intent)


def sent_strings(f: framemod.Frame) -> dict:
    """What leaves the machine for a search, per destination."""
    if f.plain:
        return {"web": f.query()}
    q = f.query() or " ".join(f.words)
    out = {"web": q, "entity": f.entity_query(), "tavily": narrow.shape_query(f, q)}
    try:
        p = router.plan(f)
        if p.lane in ("card", "facts") and p.vertical != "leads":       # leads = local index, never sent
            vals = []
            for v in p.args.values():
                if isinstance(v, str):
                    vals.append(v)
                elif hasattr(v, "subject"):
                    vals.append(str(getattr(v, "subject", "")))
            out["router"] = " ".join(vals)
    except Exception:
        pass
    return out


def toks(s: str) -> set:
    return {framemod.norm(w) for w in (s or "").split() if framemod.norm(w)}


def leak(words: list[str], hide_idx: set, sent: dict) -> tuple[bool, list]:
    """A hidden word leaks if its (normalized) form is in a sent string and it isn't also said un-hidden elsewhere
    (then the sent copy can't be told apart) and it isn't a stopword ("at" of an e-mail address)."""
    nw = [framemod.norm(w) for w in words]
    shown = {nw[i] for i in range(len(nw)) if i not in hide_idx}
    hidden = {nw[i] for i in hide_idx if nw[i] and nw[i] not in framemod.STOP and nw[i] not in shown}
    hits = []
    allsent = set()
    for dest, s in sent.items():
        h = sorted(hidden & toks(s))
        allsent |= set(h)
        if h:
            hits.append((dest, h))
    leak.share = len(allsent) / len(hidden) if hidden else 0.0     # share of the hidden words that went out
    return bool(hits), hits


# ------------------------------------------------------------------------------------------------- privacy
def privacy() -> dict:
    rows = [json.loads(x) for x in (DIRECTOR / "pii_lines.jsonl").read_text().splitlines() if x.strip()]
    res = {"arc": Counter(), "plain": Counter()}
    examples, over = [], []
    by_cat: dict = {}
    for r in rows:
        for turn in r.get("flux_turns") or []:
            words = turn["words"]
            hide = {i for i, h in enumerate(turn["hide"]) if h}
            has = bool(hide)
            fa = arc_frame(words)
            fp = framemod.plain(words)
            la, ha = leak(words, hide, sent_strings(fa))
            sa = leak.share
            lp, hp = leak(words, hide, sent_strings(fp))
            sp = leak.share
            prompt_a, _ = leak(words, hide, {"prompt": fa.resolved_text})
            prompt_p, _ = leak(words, hide, {"prompt": fp.text})
            c = by_cat.setdefault(r["cat"], Counter())
            c["turns"] += 1
            if has:
                for arm, lk, pr in (("arc", la, prompt_a), ("plain", lp, prompt_p)):
                    res[arm]["pii_turns"] += 1
                    res[arm]["leaked"] += lk
                    res[arm]["leaked_most"] += (sa if arm == "arc" else sp) >= 0.5   # >= half the hidden words sent
                    res[arm]["prompt_leaked"] += pr
                    res[arm]["test_pii_turns"] += r["split"] == "test"
                    res[arm]["test_leaked"] += lk and r["split"] == "test"
                c["pii"] += 1
                c["arc_leak"] += la
                c["plain_leak"] += lp
                c["arc_leak_most"] += sa >= 0.5
                c["plain_leak_most"] += sp >= 0.5
                if lp and not la and len(examples) < 40:
                    examples.append({"id": r["id"], "cat": r["cat"], "heard": " ".join(words), "plain_query": fp.query(),
                                     "arc_query": fa.query(), "arc_entity": fa.entity_query(),
                                     "redacted": list(fa.redacted)})
                if la:
                    c.setdefault("arc_leak_ids", [])
                    if len(c["arc_leak_ids"]) < 5:
                        c["arc_leak_ids"].append({"id": r["id"], "heard": " ".join(words), "sent": ha})
            else:
                res["arc"]["clean_turns"] += 1
                if fa.redact_tok:
                    res["arc"]["over_redacted"] += 1
                    c["over_redacted"] += 1
                    if len(over) < 15:
                        over.append({"id": r["id"], "cat": r["cat"], "heard": " ".join(words), "arc_query": fa.query(),
                                     "redacted": list(fa.redacted)})
    # gold set PII items (labels.pii spans; private_name included: Arc does not redact names by design)
    gold = [json.loads(x) for x in GOLD.read_text().splitlines() if x.strip()]
    g = {"arc": Counter(), "plain": Counter()}
    gex = []
    for r in gold:
        spans = r["labels"].get("pii") or []
        if not spans:
            continue
        words = r.get("raw_words") or r["words"]
        fa, fp = arc_frame(words), framemod.plain(words)
        for kind in ("all", "no_private_name"):
            hide = {i for a, b, t in spans for i in range(a, b) if kind == "all" or t != "private_name"}
            if not hide:
                continue
            la, ha = leak(words, hide, sent_strings(fa))
            lp, _ = leak(words, hide, sent_strings(fp))
            g["arc"][kind + "_n"] += 1
            g["plain"][kind + "_n"] += 1
            g["arc"][kind + "_leaked"] += la
            g["plain"][kind + "_leaked"] += lp
            if kind == "all" and la:
                gex.append({"id": r["id"], "heard": " ".join(words), "types": [t for *_, t in spans], "sent": ha})
    return {"pii_lines": {k: dict(v) for k, v in res.items()}, "by_cat": {k: dict(v) for k, v in by_cat.items()},
            "examples_arc_kept_out": examples, "over_redacted_examples": over,
            "gold": {k: dict(v) for k, v in g.items()}, "gold_arc_leaks": gex}


# ------------------------------------------------------------------------------------------------- open mic
def openmic() -> dict:
    utts = {}
    for x in (DIRECTOR / "openmic_utterances.jsonl").read_text().splitlines():
        r = json.loads(x)
        utts[(r["scene"], r["utt"])] = r
    turns = [json.loads(x) for x in (DIRECTOR / "openmic_flux_turns.jsonl").read_text().splitlines() if x.strip()]
    turns = [t for t in turns if t["split"] == "test"]
    c = {"arc": Counter(), "plain": Counter()}
    ex_saved, ex_missed = [], []
    kinds = Counter()
    for t in turns:
        words = [w[0] for w in t["words"]]
        if not [w for w in words if clean(w)]:
            continue
        cov = [utts.get((t["scene"], i)) for i in t["covers_utts"]]
        cov = [u for u in cov if u]
        req = any(u["addressed"] for u in cov)
        f = arc_frame(words)
        skip = open_mic_skip(f) is not None or not f.has_content()
        for arm, answered in (("arc", not skip), ("plain", True)):
            c[arm]["turns"] += 1
            c[arm]["requests"] += req
            c[arm]["non_requests"] += not req
            c[arm]["answered_non_request"] += answered and not req
            c[arm]["missed_request"] += (not answered) and req
            c[arm]["answered"] += answered
        if not req:
            for u in cov:
                kinds[(u["kind"], not skip)] += 1
            if skip and len(ex_saved) < 12:
                ex_saved.append({"scene": t["scene"], "heard": " ".join(words), "kinds": [u["kind"] for u in cov],
                                 "plain_query": framemod.plain(words).query()})
        elif skip and len(ex_missed) < 12:
            ex_missed.append({"scene": t["scene"], "heard": " ".join(words),
                              "addressed": [u["text"] for u in cov if u["addressed"]]})
    gold = [json.loads(x) for x in GOLD.read_text().splitlines() if x.strip()]
    g = {"arc": Counter(), "plain": Counter()}
    for r in gold:
        words = r.get("raw_words") or r["words"]
        req = r["labels"]["respond"] != "none"
        f = arc_frame(words)
        skip = open_mic_skip(f) is not None or not f.has_content()
        for arm, answered in (("arc", not skip), ("plain", True)):
            g[arm]["n"] += 1
            g[arm]["non_requests"] += not req
            g[arm]["answered_non_request"] += answered and not req
            g[arm]["missed_request"] += (not answered) and req
    return {"director_test": {k: dict(v) for k, v in c.items()}, "gold": {k: dict(v) for k, v in g.items()},
            "nonrequest_kinds_answered_by_arc": {f"{k}|{'answered' if a else 'held'}": n for (k, a), n in kinds.items()},
            "examples_held": ex_saved, "examples_missed": ex_missed}


# ------------------------------------------------------------------------------------------------- freshness
FRESH = {"live", "day", "week", "recent"}
STABLE = {"evergreen", "historical"}


def fresh_of(f) -> dict:
    p = lanes.web_plan(f.intent, f.resolved_text, final=True, hints=f.web_hints())
    try:
        rp = router.plan(f)
        lane, vert = rp.lane, rp.vertical
    except Exception:
        lane, vert = "", ""
    constrained = bool(p.get("time_range") or p.get("max_age_hours") or p["topic"] in ("news", "finance"))
    card = lane == "card" and vert in ("weather", "fx", "time", "sun")
    return {"topic": p["topic"], "time_range": p.get("time_range"), "max_age_hours": p.get("max_age_hours"),
            "lane": lane, "vertical": vert, "constrained": constrained, "card": card}


def freshness() -> dict:
    gold = [json.loads(x) for x in GOLD.read_text().splitlines() if x.strip()]
    c = {"arc": Counter(), "arc_or_card": Counter(), "plain": Counter()}
    by_label: dict = {}
    ex = []
    for r in gold:
        lab = r["labels"].get("freshness")
        if lab not in FRESH | STABLE:
            continue
        words = r.get("raw_words") or r["words"]
        f = arc_frame(words)
        a = fresh_of(f)
        need = lab in FRESH
        for arm, got in (("arc", a["constrained"]), ("arc_or_card", a["constrained"] or a["card"]), ("plain", False)):
            c[arm]["n"] += 1
            c[arm]["need_fresh"] += need
            c[arm]["correct"] += got == need
            c[arm]["fresh_hit"] += need and got
            c[arm]["stale_false_fresh"] += (not need) and got
        bl = by_label.setdefault(lab, Counter())
        bl["n"] += 1
        bl["arc_constrained"] += a["constrained"]
        bl["arc_card"] += a["card"]
        bl[f"tr_{a['time_range']}"] += 1
        bl[f"topic_{a['topic']}"] += 1
        if len(ex) < 400:
            ex.append({"id": r["id"], "heard": " ".join(words), "gold": lab, **a})
    return {"counts": {k: dict(v) for k, v in c.items()}, "by_label": {k: dict(v) for k, v in by_label.items()},
            "rows": ex}


# ------------------------------------------------------------------------------------------------- corrections + query
def corrections_and_query() -> dict:
    gold = [json.loads(x) for x in GOLD.read_text().splitlines() if x.strip()]
    corr = {"arc": Counter(), "plain": Counter()}
    rows = []
    qr = {"arc": [0, 0, 0], "plain": [0, 0, 0]}        # gold content words found, gold content words, sent words
    for r in gold:
        L = r["labels"]
        words = r["words"]
        fa, fp = arc_frame(r.get("raw_words") or words), framemod.plain(r.get("raw_words") or words)
        nw = [framemod.norm(w) for w in words]
        if L.get("query"):
            gq = {nw[i] for i in L["query"] if i < len(nw) and nw[i] and nw[i] not in framemod.STOP}
            for arm, f in (("arc", fa), ("plain", fp)):
                s = toks(f.query())
                qr[arm][0] += len(gq & s)
                qr[arm][1] += len(gq)
                qr[arm][2] += len(s)
        c = L.get("correction")
        if not c or not c.get("repair") or not c.get("reparandum"):
            continue
        rep = {nw[i] for i in range(*c["repair"]) if nw[i] and nw[i] not in framemod.STOP}
        bad = {nw[i] for i in range(*c["reparandum"]) if nw[i] and nw[i] not in framemod.STOP} - rep
        bad -= {nw[i] for i in L.get("query") or [] if i < len(nw)}      # a reparandum word the gold query keeps
        row = {"id": r["id"], "heard": " ".join(words)}
        for arm, f in (("arc", fa), ("plain", fp)):
            s = toks(f.query()) | toks(f.entity_query() if not f.plain else "")
            ok = rep <= s and not (bad & s)
            corr[arm]["n"] += 1
            corr[arm]["repair_kept"] += rep <= s
            corr[arm]["reparandum_sent"] += bool(bad & s)
            corr[arm]["clean"] += ok
            row[arm] = f.query()
        rows.append(row)
    qs = {arm: {"recall": round(v[0] / max(1, v[1]), 3), "precision": round(v[0] / max(1, v[2]), 3),
                "gold_words": v[1], "sent_words": v[2]} for arm, v in qr.items()}
    return {"correction": {k: dict(v) for k, v in corr.items()}, "correction_rows": rows, "query_vs_gold": qs}


def main():
    t = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": {"TAGGER_HINTS": config.TAGGER_HINTS,
           "NARROW_QUERY": config.NARROW_QUERY, "MEM_INLINE": config.MEM_INLINE},
           "privacy": privacy(), "openmic": openmic(), "freshness": freshness(),
           "gold_corrections": corrections_and_query()}
    OUT.write_text(json.dumps(res, indent=1, default=str))
    p = res["privacy"]
    print("privacy pii_lines", json.dumps(p["pii_lines"]))
    print("privacy gold", json.dumps(p["gold"]))
    print("openmic", json.dumps(res["openmic"]["director_test"]), json.dumps(res["openmic"]["gold"]))
    print("freshness", json.dumps(res["freshness"]["counts"]))
    print("corrections", json.dumps(res["gold_corrections"]["correction"]), json.dumps(res["gold_corrections"]["query_vs_gold"]))
    print(f"wrote {OUT} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
