"""Charts for docs/RETRIEVAL.md §5 (Keystone director) from runs/index/kdir_results.json -> docs/img/keystone_kdir_*.png.

  keystone_kdir_curve.png    paid calls / q vs regret on the two test streams: each policy's operating curve (offline,
                             on the test stream's VOI-logged decisions) + the measured replay point of each policy
  keystone_kdir_stream.png   paid calls / q per 500-question window over the held-out test stream
  keystone_kdir_network.png  the network effect: per-user / + shared edges / + shared question cells / pooled

    .venv/bin/python eval/keystone/plots_kdir.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plots import GRID, INK, INK2, SURF, style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/index"
IMG = ROOT / "docs/img"
# categorical slots in fixed order (dataviz reference palette)
COL = {"voi": "#2a78d6", "kdir": "#eb6834", "imit": "#1baf7a", "cache": "#eda100", "fb": "#6250d6",
       "oracle": "#8a8984", "voi21": "#e87ba4"}
LBL = {"voi": "VOI gate (today)", "kdir": "KDIR (shipped trees)", "imit": "KDIR imitation",
       "cache": "search cache", "fb": "KDIR + feedback (online)", "oracle": "oracle", "voi21": "KDIR, 21 VOI features"}


def main():
    r = json.loads((RUNS / "kdir_results.json").read_text())
    SH = r.get("shipped", "3c_nodir")
    ab = r.get("ablation", {})
    cur = r.get("test_curves", {})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), facecolor=SURF)
    for ax, sk, title in zip(axes, ("old_test", "test"), ("old test stream (800 q, s1)", "held-out test stream (2,000 q)")):
        style(ax, title)
        for k, key in (("voi", "voi"), ("voi21", "2c_voi21_all"), ("kdir", SH), ("imit", "3d_imitation")):
            c = cur.get(f"{sk}:{key}")
            if c:
                xs = [p["regret_rate"] * 100 for p in c]
                ys = [p["paid_per_q"] for p in c]
                ax.plot(xs, ys, color=COL[k], lw=2, label=LBL[k] + " (curve)", alpha=0.9)
        for k, key, mk in (("cache", "a", "s"), ("voi", "voi", "o"), ("kdir", SH, "D"), ("fb", "5_ttl", "^"),
                           ("oracle", "oracle", "*")):
            a = ab.get(f"{sk}:{key}")
            if a:
                al = a["summary"]["all"]
                ax.scatter([al["regret_rate"] * 100], [al["paid_per_q"]], color=COL[k], s=110 if mk == "*" else 70, marker=mk, zorder=5,
                           edgecolor=SURF, linewidth=2, label=LBL[k] + " (replay)")
        ax.set_xlim(-0.2, 5)
        ax.set_xlabel("regret (% of questions answered wrong where paying was right)", fontsize=8.5, color=INK2)
        ax.set_ylabel("paid calls per question", fontsize=8.5, color=INK2)
    axes[1].legend(fontsize=7.2, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(IMG / "keystone_kdir_curve.png", dpi=150, facecolor=SURF)

    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURF)
    style(ax, "paid calls / q per 500-question window, held-out test stream")
    for k, key in (("voi", "voi"), ("kdir", SH), ("fb", "4_feedback_online"), ("oracle", "oracle")):
        a = ab.get(f"test:{key}")
        if a and a["summary"].get("windows"):
            w = a["summary"]["windows"]
            ax.plot([x["start"] + 250 for x in w], [x["paid_per_q"] for x in w], color=COL[k], lw=2, marker="o", ms=4,
                    label=LBL[k])
    ax.set_xlabel("question position in the stream", fontsize=8.5, color=INK2)
    ax.set_ylabel("paid calls / q", fontsize=8.5, color=INK2)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "keystone_kdir_stream.png", dpi=150, facecolor=SURF)

    net = r.get("network", {})
    if net:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), facecolor=SURF)
        stores = [("user", "per-user"), ("user_shared", "+ shared edges"), ("user_shared_cells", "+ shared Q→A cells"),
                  ("pooled", "pooled")]
        for ax, sk in zip(axes, ("old_test", "test")):
            style(ax, {"old_test": "old test stream (30 users)", "test": "held-out test stream (40 users)"}[sk])
            for i, (pol, k) in enumerate((("voi", "voi"), (SH, "kdir"))):
                ys = []
                for st, _ in stores:
                    if st == "pooled":
                        a = ab.get(f"{sk}:{pol}")
                        ys.append(a["summary"]["all"]["paid_per_q"] if a else 0)
                    else:
                        a = net.get(f"{sk}:{st}:{pol}")
                        ys.append(a["summary"]["all"]["paid_per_q"] if a else 0)
                xs = [j + (i - 0.5) * 0.36 for j in range(len(stores))]
                ax.bar(xs, ys, width=0.34, color=COL[k], label=LBL[k], edgecolor=SURF, linewidth=2)
                for x, y in zip(xs, ys):
                    ax.text(x, y + 0.008, f"{y:.3f}", ha="center", fontsize=7, color=INK)
            ax.set_xticks(range(len(stores)), [s for _, s in stores], fontsize=8, color=INK2)
            ax.set_ylabel("paid calls / q", fontsize=8.5, color=INK2)
        axes[0].legend(fontsize=8, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(IMG / "keystone_kdir_network.png", dpi=150, facecolor=SURF)
    print("charts ->", IMG)


if __name__ == "__main__":
    main()
