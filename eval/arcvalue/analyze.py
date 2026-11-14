"""Arc-value tables + chart from runs/arcvalue/* (docs/ARC_VALUE.md).

  .venv/bin/python eval/arcvalue/analyze.py      # -> runs/arcvalue/summary.json, docs/img/arc_value.png, stdout tables

Arms: headstart = Onset (default mode), baseline = same Arc frame, search after the endpoint, plain = Arc off.
Definitions (per question-run, pooled over reps):
  accuracy      run_ab.score (every expect keyword in the answer); conversations: run_conv's substring scorer
  harmful       corrections: the answer names the corrected-away answer ("wrong" keyword) and not the meant one;
                elsewhere: wrong AND stated without hedging (no "sources don't cover" style admission)
  wasted        answers beyond one for one spoken question (a split turn answered twice, a phantom answer)
  paid/q        web calls that reach the paid backend (would be Tavily credits; the Wikipedia lane is free)
  usd/q         the cost meter's est. $ per question (answer-model tokens + paid calls priced as Tavily)
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from run_ab import pct, score  # noqa: E402

R = ROOT / "runs" / "arcvalue"
ARMS = ("headstart", "baseline", "plain")
LABEL = {"headstart": "Onset", "baseline": "Baseline", "plain": "Arc-off"}
_HEDGE = re.compile(r"\b(sources? (?:don't|do not|doesn't|does not)|not (?:covered|in the sources)|"
                    r"(?:don't|do not) (?:cover|mention|say)|no (?:information|source)|isn't (?:covered|mentioned)|"
                    r"can't (?:find|tell)|couldn't find|unclear)\b", re.I)


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def rows_of(*names, ids=None):
    out = []
    for n in names:
        d = load(n)
        if d:
            out += [dict(r, _file=n) for r in d["rows"] if ids is None or r["id"] in ids]
    return out


def stats(rows, wrong: dict | None = None) -> dict:
    """Per-arm numbers over run_ab rows."""
    out = {}
    for arm in ARMS:
        g = [r for r in rows if r["mode"] == arm]
        if not g:
            continue
        sc = [r for r in g if r.get("correct") is not None]
        ok = sum(1 for r in sc if r["correct"])
        harm = 0
        for r in sc:
            a = r.get("answer") or ""
            if wrong is not None:
                w = wrong.get(r["id"]) or []
                harm += bool(w) and bool(score(a, w)) and not r["correct"]
            else:
                harm += (not r["correct"]) and bool(a) and not _HEDGE.search(a)
        ready = [r["ready_ms"] for r in g if r.get("ready_ms") is not None]
        out[arm] = {
            "n": len(g), "scored": len(sc), "correct": ok, "acc": round(ok / len(sc), 3) if sc else None,
            "harmful": harm, "harm_rate": round(harm / len(sc), 3) if sc else None,
            "wasted": sum(max(0, (r.get("answers") or 1) - 1) for r in g),
            "p50": pct(ready, 50), "p90": pct(ready, 90),
            "paid_q": round(statistics.mean([(r["web_paid"] if r.get("web_paid") is not None else r.get("paid_calls") or 0)
                                             for r in g]), 2),
            "usd_q": round(statistics.mean([r.get("usd") or 0 for r in g]), 5),
            "errors": sum(1 for r in g if r.get("error") or r.get("ready_ms") is None),
        }
    return out


def qset(path):
    return {json.loads(l)["id"]: json.loads(l) for l in (ROOT / path).read_text().splitlines() if l.strip()}


def examples(rows, ids, n=6, prefer=("plain",)):
    """Paired transcripts + answers per question id (first rep), for the doc."""
    out = []
    for i in ids:
        ex = {"id": i}
        for arm in ARMS:
            r = next((x for x in rows if x["id"] == i and x["mode"] == arm), None)
            if r:
                ex[arm] = {"heard": r.get("question_heard"), "query": r.get("query"), "answer": r.get("answer"),
                           "correct": r.get("correct"), "ready_ms": r.get("ready_ms"), "answers": r.get("answers"),
                           "path": r.get("path"), "answer_by": r.get("answer_by")}
        out.append(ex)
        if len(out) >= n:
            break
    return out


def diff_ids(rows, better="headstart", worse="plain"):
    """Question ids where `better` beat `worse` in every rep both have."""
    ids = []
    for i in dict.fromkeys(r["id"] for r in rows):
        b = [r["correct"] for r in rows if r["id"] == i and r["mode"] == better and r.get("correct") is not None]
        w = [r["correct"] for r in rows if r["id"] == i and r["mode"] == worse and r.get("correct") is not None]
        if b and w and sum(b) / len(b) > sum(w) / len(w):
            ids.append(i)
    return ids


def conv_stats():
    d = load("conv.json")
    if not d:
        return None
    out = {}
    for arm in ARMS:
        rows = [r for r in d if r["mode"] == arm]
        if not rows:
            continue
        fu = [t for r in rows for t in r["turns"][1:]]
        t1 = [r["turns"][0] for r in rows if r["turns"]]
        sc = [t for t in fu if t["correct"] is not None]
        lat = [t["ready_ms"] for t in fu if t.get("ready_ms") is not None]
        out[arm] = {"followups": len(sc), "fu_correct": sum(1 for t in sc if t["correct"]),
                    "fu_acc": round(sum(1 for t in sc if t["correct"]) / len(sc), 3) if sc else None,
                    "turn1_acc": round(sum(1 for t in t1 if t["correct"]) / max(1, sum(1 for t in t1 if t["correct"] is not None)), 3),
                    "p50": pct(lat, 50), "p90": pct(lat, 90),
                    "paid_q": round(statistics.mean([t.get("paid") or 0 for t in fu]), 2) if fu else None,
                    "wrong_n_answers": sum(1 for r in rows if r["n_answers"] != r["n_turns"]),
                    "missing_turns": sum(max(0, r["n_turns"] - r["n_answers"]) for r in rows),
                    "extra_answers": sum(max(0, r["n_answers"] - r["n_turns"]) for r in rows)}
    return out


def multi_stats():
    d = load("multi_r1.json")
    if not d:
        return None
    qs = qset("eval/multipart/questions.jsonl")
    out = {}
    for arm in ARMS:
        g = [r for r in d["rows"] if r["mode"] == arm]
        if not g:
            continue
        po = pn = allok = 0
        for r in g:
            q = qs[r["id"]]
            a = r.get("answer") or ""
            pk = [bool(score(a, p["expect"])) for p in q["parts"]]
            ov = bool(score(a, q["overall"])) if q["overall"] else True
            po += sum(pk)
            pn += len(pk)
            allok += all(pk) and ov
        ready = [r["ready_ms"] for r in g if r.get("ready_ms") is not None]
        out[arm] = {"n": len(g), "part_acc": round(po / pn, 3), "parts": f"{po}/{pn}", "all_parts": allok,
                    "all_acc": round(allok / len(g), 3), "p50": pct(ready, 50), "p90": pct(ready, 90),
                    "paid_q": round(statistics.mean([(r["web_paid"] if r.get("web_paid") is not None else
                                                      r.get("paid_calls") or 0) for r in g]), 2),
                    "usd_q": round(statistics.mean([r.get("usd") or 0 for r in g]), 5),
                    "wasted": sum(max(0, (r.get("answers") or 1) - 1) for r in g)}
    return out


def live_stats():
    d = load("live_scored.json")
    if not d:
        return None
    rows = next(iter(d["runs"].values()))["rows"]
    out = {}
    for arm in ARMS:
        g = [r for r in rows if r["mode"] == arm]
        if g:
            ready = [r["ready_ms"] for r in g if r.get("ready_ms") is not None]
            out[arm] = {"n": len(g), "correct": sum(1 for r in g if r["correct"]),
                        "acc": round(sum(1 for r in g if r["correct"]) / len(g), 3),
                        "by_kind": {k: f"{sum(1 for r in g if r['kind'] == k and r['correct'])}/"
                                       f"{sum(1 for r in g if r['kind'] == k)}" for k in
                                    ("live_weather", "live_fx", "live_time", "live_units")},
                        "card": sum(1 for r in g if r.get("answer_by") == "card"), "p50": pct(ready, 50)}
    return out


def main():
    S = {}
    corr_q = qset("eval/arcvalue/corrections.jsonl")
    wrong = {k: v.get("wrong") or [] for k, v in corr_q.items()}
    fq = rows_of("fact58_r1.json", "fact58_r2.json")
    S["factoid58"] = stats(fq)
    S["factoid58_web"] = stats([r for r in fq if r.get("kind") != "local"])
    S["factoid58_local"] = stats([r for r in fq if r.get("kind") == "local"])
    sh = rows_of("showcase_r1.json", "showcase_r2.json")
    S["showcase"] = stats(sh)
    cr = rows_of("corr_r1.json", "corr_r2.json")
    S["corrections"] = stats(cr, wrong)
    S["corrections_new"] = stats([r for r in cr if r["id"].startswith("cx")], wrong)
    S["corrections_existing"] = stats([r for r in cr if not r["id"].startswith("cx")], wrong)
    restart = {k for k, v in corr_q.items() if "restart" in (v.get("style") or "")}
    S["restarts"] = stats([r for r in cr if r["id"] in restart], wrong)
    S["self_corrections"] = stats([r for r in cr if r["id"] not in restart], wrong)
    S["conversations"] = conv_stats()
    S["multipart"] = multi_stats()
    S["noisy_snr20"] = stats(rows_of("noisy_snr20.json"))
    S["noisy_snr10"] = stats(rows_of("noisy_snr10.json"))
    S["live"] = live_stats()
    tl = load("textlevel.json")
    S["textlevel"] = None
    if tl:
        p, o, f = tl["privacy"], tl["openmic"], tl["freshness"]["counts"]
        S["textlevel"] = {
            "privacy_leak": {a: round(p["pii_lines"][a]["leaked"] / p["pii_lines"][a]["pii_turns"], 3) for a in ("arc", "plain")},
            "privacy_leak_most": {a: round(p["pii_lines"][a]["leaked_most"] / p["pii_lines"][a]["pii_turns"], 3) for a in ("arc", "plain")},
            "gold_pii_leak": {a: round(p["gold"][a]["all_leaked"] / p["gold"][a]["all_n"], 3) for a in ("arc", "plain")},
            "openmic_answered_nonreq": {a: round(o["director_test"][a]["answered_non_request"] / o["director_test"][a]["non_requests"], 3) for a in ("arc", "plain")},
            "openmic_missed_req": {a: round(o["director_test"][a]["missed_request"] / o["director_test"][a]["requests"], 3) for a in ("arc", "plain")},
            "openmic_acc": {a: round(1 - (o["director_test"][a]["answered_non_request"] + o["director_test"][a]["missed_request"]) / o["director_test"][a]["turns"], 3) for a in ("arc", "plain")},
            "fresh_acc": {a: round(f[a]["correct"] / f[a]["n"], 3) for a in ("arc", "arc_or_card", "plain")},
            "fresh_recall": {a: round(f[a]["fresh_hit"] / f[a]["need_fresh"], 3) for a in ("arc", "arc_or_card", "plain")},
        }
    S["examples"] = {
        "corrections": examples(cr, diff_ids(cr)[:8], 8),
        "corrections_arc_worse": examples(cr, diff_ids(cr, "plain", "headstart")[:5], 5),
        "factoid": examples(fq, diff_ids(fq)[:6], 6),
        "factoid_arc_worse": examples(fq, diff_ids(fq, "plain", "headstart")[:6], 6),
        "showcase": examples(sh, diff_ids(sh)[:6], 6),
        "showcase_arc_worse": examples(sh, diff_ids(sh, "plain", "headstart")[:5], 5),
        "noisy": examples(rows_of("noisy_snr10.json"), diff_ids(rows_of("noisy_snr10.json"))[:5], 5),
        "multipart": examples(rows_of("multi_r1.json"), [r["id"] for r in rows_of("multi_r1.json") if r["mode"] == "plain"][:26], 26),
    }
    (R / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    for k, v in S.items():
        if k in ("examples",):
            continue
        print(k, json.dumps(v))
    chart(S)


def chart(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cats = []

    def add(label, d, key="acc"):
        if d and all(a in d for a in ARMS):
            cats.append((label, [d[a][key] for a in ARMS]))
    add("Factoids\n(58 set)", S["factoid58"])
    add("Showcase\n(scored 15)", S["showcase"])
    add("Corrections\n+ restarts", S["corrections"])
    if S.get("conversations"):
        add("Follow-ups\n(conversations)", S["conversations"], "fu_acc")
    if S.get("multipart"):
        add("Multi-part\n(per part)", S["multipart"], "part_acc")
    add("Noisy 20 dB\n(showcase)", S["noisy_snr20"])
    add("Noisy 10 dB\n(showcase)", S["noisy_snr10"])
    if S.get("live"):
        add("Live data\n(vs ground truth)", S["live"])
    tl = S.get("textlevel")
    if tl:
        cats.append(("Open mic\n(turn handled\nright, text)", [tl["openmic_acc"]["arc"], tl["openmic_acc"]["arc"],
                                                              tl["openmic_acc"]["plain"]]))
        cats.append(("Privacy\n(PII kept out,\ntext)", [1 - tl["privacy_leak"]["arc"], 1 - tl["privacy_leak"]["arc"],
                                                       1 - tl["privacy_leak"]["plain"]]))
        cats.append(("Freshness\n(right filter,\ntext)", [tl["fresh_acc"]["arc"], tl["fresh_acc"]["arc"],
                                                         tl["fresh_acc"]["plain"]]))
    colors = {"headstart": "#2a78d6", "baseline": "#eb6834", "plain": "#1baf7a"}
    ink, muted, grid, surf = "#0b0b0b", "#52514e", "#e6e5df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    x = np.arange(len(cats))
    w = 0.26
    for j, arm in enumerate(ARMS):
        vals = [c[1][j] for c in cats]
        bars = ax.bar(x + (j - 1) * w, vals, w - 0.03, color=colors[arm], label=LABEL[arm], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v * 100:.0f}", ha="center", va="bottom",
                    fontsize=6.5, color=muted)
    ax.set_xticks(x, [c[0] for c in cats], fontsize=7.5, color=ink)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100%"], fontsize=8, color=muted)
    ax.grid(axis="y", color=grid, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5)
    fig.text(0.01, 0.985, "What Arc's understanding adds: accuracy per category, three arms", fontsize=11.5,
             color=ink, va="top")
    fig.text(0.01, 0.945, "Wikipedia web lane, Flux ear, Haiku 5.5 answers; replays pooled over reps · text-level bars "
             "(open mic, privacy, freshness) have no Baseline/Onset split: both use the same Arc frame · "
             "docs/ARC_VALUE.md", fontsize=7.5, color=muted, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = ROOT / "docs" / "img" / "arc_value.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=surf)
    print("wrote", out)


if __name__ == "__main__":
    main()
