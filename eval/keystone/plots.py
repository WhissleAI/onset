"""Charts for docs/RETRIEVAL.md from runs/index/*.json -> docs/img/keystone_*.png (matplotlib, light surface).

    .venv/bin/python eval/keystone/plots.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/index"
IMG = ROOT / "docs/img"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"fresh": "#8a8984", "a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "e": "#eda100", "f": "#6250d6",
     "oracle": "#e87ba4"}
NAME = {"fresh": "no reuse", "a": "(a) today: search cache", "b": "(b) + memory cells", "c": "(c) + Arc-keyed edges",
        "d": "(d) + equivalence verifier/hash", "e": "(e) + activation, one structure", "f": "(f) + VOI gate",
        "oracle": "oracle (full information)"}


def style(ax, title=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")


def rolling(xs, w=100):
    out = []
    for i in range(len(xs)):
        win = [x for x in xs[max(0, i - w + 1):i + 1] if x is not None]
        out.append(sum(win) / len(win) if win else None)
    return out


def rolling_med(xs, w=100):
    return [statistics.median(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


def stream_chart(rep, out, pols=("fresh", "a", "b", "e", "f")):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
    for p in pols:
        if p not in rep["policies"]:
            continue
        rows = rep["policies"][p]["rows"]
        x = list(range(1, len(rows) + 1))
        paid = [r["paid"] for r in rows]
        cum = [sum(paid[:i + 1]) / (i + 1) for i in range(len(paid))]
        loc = [1.0 if r["path"] in ("key", "hash", "qnn", "dense") or r["path"].startswith("cache") else 0.0
               for r in rows]
        lat = [r["lat_ms"] for r in rows]
        acc = [None if r["correct"] is None else float(r["correct"]) for r in rows]
        kw = dict(color=C[p], linewidth=2 if p in ("a", "f") else 1.4, label=NAME[p])
        axs[0, 0].plot(x, cum, **kw)
        axs[0, 1].plot(x, rolling(loc), **kw)
        axs[1, 0].plot(x, rolling_med(lat), **kw)
        axs[1, 1].plot(x, rolling(acc), **kw)
    style(axs[0, 0], "Paid web calls per question (cumulative)")
    style(axs[0, 1], "Local hit rate (rolling 100)")
    style(axs[1, 0], "p50 retrieval latency, ms (rolling 100)")
    style(axs[1, 1], "Correct (keywords or same evidence; rolling 100)")
    axs[0, 0].set_ylim(0, None)
    axs[1, 0].set_ylim(0, None)
    for ax in axs[1]:
        ax.set_xlabel("question # in the stream", fontsize=8.5, color=INK2)
    axs[0, 0].legend(fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    plt.close(fig)


def voi_chart(sweep, sweep_dev, fit, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    ax = axs[0]
    for sw, lab, col, mk in ((sweep, "test stream", C["f"], "o"), (sweep_dev, "dev stream", "#9085e9", "s")):
        if not sw:
            continue
        pts = sw["points"]
        ax.plot([p["regret_rate"] * 100 for p in pts], [p["paid_per_q"] for p in pts], color=col, marker=mk,
                markersize=4, linewidth=1.6, label=f"Keystone + VOI, theta sweep ({lab})")
        ax.scatter([sw["a"]["regret_rate"] * 100], [sw["a"]["paid_per_q"]], color=C["a"], marker=mk, s=50, zorder=5,
                   label=f"today's search cache ({lab})")
        ax.scatter([0], [sw["fresh"]["paid_per_q"]], color=C["fresh"], marker=mk, s=50, zorder=5,
                   label=f"no reuse ({lab})")
    style(ax, "Operating curve: paid calls vs regret")
    ax.set_xlabel("regret: % of questions answered worse than paying would have", fontsize=8.5, color=INK2)
    ax.set_ylabel("paid calls / question", fontsize=8.5, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax = axs[1]
    for split, col, mk in (("dev", "#9085e9", "s"), ("test", C["f"], "o")):
        cal = fit[split]["calibration"]
        ax.plot([b["p_mean"] for b in cal], [b["y_rate"] for b in cal], color=col, marker=mk, markersize=4,
                linewidth=1.6, label=f"{split} (ECE {fit[split]['ece']:.3f}, Brier {fit[split]['brier']:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
    style(ax, "VOI gate calibration (per candidate)")
    ax.set_xlabel("predicted p(local answer is no worse than paying)", fontsize=8.5, color=INK2)
    ax.set_ylabel("observed rate", fontsize=8.5, color=INK2)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    plt.close(fig)


def eq_chart(ah, out):
    t = ah["test"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    ax = axs[0]
    series = [("emb_nn", "embedding NN (cosine threshold)", C["a"]), ("verifier", "NN + Arc verifier (learned)", C["f"]),
              ("hash", "learned 64-bit hash (Hamming radius)", C["b"]),
              ("learned_cos", "learned projection (cosine)", C["c"])]
    for k, lab, col in series:
        pts = [p for p in t.get(k, []) if p["precision"] is not None]
        ax.plot([p["coverage"] for p in pts], [p["precision"] for p in pts], color=col, marker="o", markersize=3.5,
                linewidth=1.6, label=lab)
    ax.scatter([t["key"]["coverage"]], [t["key"]["precision"]], color=C["e"], s=60, zorder=5,
               label="typed key rule (exact)")
    style(ax, "Same-answer reuse on held-out families")
    ax.set_xlabel("coverage (share of paraphrases that reuse a stored answer)", fontsize=8.5, color=INK2)
    ax.set_ylabel("precision (reused answer is the right one)", fontsize=8.5, color=INK2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax = axs[1]
    fa = t.get("near_miss_false_accept")
    if fa:
        labs = [f"emb >= {k}" for k in ("0.88", "0.9", "0.92")] + [f"verifier >= {k}" for k in ("0.5", "0.7")] + \
            ["hash r <= 4", "typed key"]
        vals = [fa["emb_nn"].get(k, fa["emb_nn"].get(float(k))) for k in ("0.88", "0.9", "0.92")] + \
            [fa["verifier"].get(k, fa["verifier"].get(float(k))) for k in ("0.5", "0.7")] + \
            [fa["hash"].get("4", fa["hash"].get(4)), fa["key"]]
        cols = [C["a"]] * 3 + [C["f"]] * 2 + [C["b"], C["e"]]
        ax.barh(range(len(vals)), [v * 100 for v in vals], color=cols, height=0.6)
        ax.set_yticks(range(len(vals)), labs, fontsize=8.5)
        for i, v in enumerate(vals):
            ax.text(v * 100 + 0.5, i, f"{v * 100:.1f}%", va="center", fontsize=8, color=INK2)
        ax.invert_yaxis()
    style(ax, f"Hard negatives: near-miss questions wrongly reused (n={fa['n'] if fa else 0})")
    ax.set_xlabel("false-accept rate, %", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    plt.close(fig)


def ann_chart(ann, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURF)
    ax = axs[0]
    sizes = sorted(int(n) for n in ann["latency"])
    meths = [("fp32_768", "exact fp32 768 (numpy)", C["a"]), ("fp32_256+rescore", "fp32 256 first pass + rescore",
                                                              C["f"]),
             ("bits_256+rescore", "binary 256 + rescore", C["b"]), ("hnsw_768", "HNSW (hnswlib, ef=64)", C["c"]),
             ("fp16_768", "exact fp16 768 (numpy, no BLAS)", C["fresh"])]
    for k, lab, col in meths:
        ys = [ann["latency"][str(n)].get(k, {}).get("p50_us") for n in sizes]
        ax.plot(sizes, [y / 1000 for y in ys], color=col, marker="o", markersize=4, linewidth=1.6, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "First-pass search latency vs store size (p50, one query)")
    ax.set_xlabel("cells", fontsize=8.5, color=INK2)
    ax.set_ylabel("ms", fontsize=8.5, color=INK2)
    ax.legend(fontsize=7.5, frameon=False)
    ax = axs[1]
    dims = [768, 512, 256, 128, 64]
    for q, col in (("fp32", C["a"]), ("int8", C["c"]), ("bits", C["b"])):
        ax.plot(dims, [ann["recall"][f"{d}_{q}"]["with_rescore_M64"] for d in dims], color=col, marker="o",
                markersize=4, linewidth=1.6, label=f"{q} first pass + rescore of 64")
    ax.plot(dims, [ann["recall"][f"{d}_fp32"]["first_pass_only"] for d in dims], color=C["a"], linestyle="--",
            linewidth=1.2, label="fp32 first pass only")
    ax.invert_xaxis()
    style(ax, "Recall@10 vs Matryoshka truncation (11k real lead vectors)")
    ax.set_xlabel("first-pass dims", fontsize=8.5, color=INK2)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURF)
    plt.close(fig)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    rep = json.loads((RUNS / "replay.json").read_text())
    stream_chart(rep, IMG / "keystone_stream.png")
    ld = lambda n: json.loads((RUNS / n).read_text()) if (RUNS / n).exists() else None  # noqa: E731
    voi_chart(ld("voi_sweep.json"), ld("voi_sweep_dev.json"), ld("voi_fit.json"), IMG / "keystone_voi.png")
    eq_chart(ld("ahash_eval.json"), IMG / "keystone_equivalence.png")
    ann_chart(ld("ann_bench.json"), IMG / "keystone_ann.png")
    print("charts ->", IMG)


if __name__ == "__main__":
    main()
