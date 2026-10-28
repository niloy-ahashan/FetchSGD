"""Compare all methods on mHealth at one upload budget: the seed-mean test
accuracy of IC (B1), MFedMC v2 (B2), PMFH-N, FL-FD and HybridSketchMFedMC v2
(ours, B3) against cumulative upload MB, on a single plot.

Reads the *_mean.csv files written by each method's seed-mean plot script, so
run those first:
  CommEfficient/CommEfficient/plot_seed_mean.py   (IC)
  MFedMC/mHealth_v2/plot_seed_mean.py             (MFedMC v2)
  PMFH/plot_seed_mean_mhealth.py                  (PMFH-N)
  FL-FD/mhealth/plot_seed_mean.py                 (FL-FD)
  HybridSketchMFedMC_v2/plot_seed_mean.py         (Hybrid v2, per budget folder)

The baselines' config per budget is fixed (the budget sweeps that exist); the
Hybrid config is chosen on the command line, e.g. for 5 MB:
  venv/bin/python plot_mhealth_budget_comparison.py --budget 5 \
    --hybrid_cols 5000 --hybrid_client 0.2 --hybrid_top 3 --hybrid_rounds 42
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = (5, 10, 15, 20, 25)
# Rounds per budget for the baselines whose budget is set by the round count.
MFEDMC_ROUNDS = {5: 2, 10: 3, 15: 5, 20: 6, 25: 8}
FLFD_ROUNDS = {5: 20, 10: 39, 15: 58, 20: 78, 25: 97}
# Categorical slots 1-5 of the validated reference palette, one per method in a
# fixed order so a method keeps its colour in every budget's plot.
COLORS = {"Hybrid v2 (ours)": "#2a78d6", "IC": "#eb6834", "MFedMC v2": "#1baf7a",
          "PMFH-N": "#eda100", "FL-FD": "#e87ba4"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=int, required=True, choices=BUDGETS, help="upload budget in MB")
    p.add_argument("--hybrid_cols", type=int, required=True, help="Hybrid sketch columns")
    p.add_argument("--hybrid_client", type=float, required=True, help="Hybrid client ratio per round")
    p.add_argument("--hybrid_top", type=int, required=True, help="Hybrid Top-k modalities")
    p.add_argument("--hybrid_rounds", type=int, required=True, help="Hybrid rounds")
    p.add_argument("--out_dir", default=os.path.join(HERE, "results", "mhealth_budget_comparison"))
    return p.parse_args()


def methods(args):
    """(label, glob, config text, script that writes the csv) per method."""
    b = args.budget
    ic_cols = b * 200
    return [
        ("Hybrid v2 (ours)",
         f"HybridSketchMFedMC_v2/results/{b}MB/HybridV2_mhealth_mm_mm_*MB_dirichlet*_it{args.hybrid_rounds}_"
         f"r3xc{args.hybrid_cols}_*_Top_{args.hybrid_top}_Client_{args.hybrid_client:g}_*_mean.csv",
         f"{round(args.hybrid_client * 10)}/10 clients · Top {args.hybrid_top} · 3×{args.hybrid_cols} · "
         f"{args.hybrid_rounds} rds",
         f"HybridSketchMFedMC_v2/plot_seed_mean.py --log_dir HybridSketchMFedMC_v2/result_updated/{b}MB"),
        ("IC", f"CommEfficient/results/IC_mhealth_mm_mm_*MB_*_r3xc{ic_cols}_*_mean.csv",
         f"10/10 clients · 3×{ic_cols} · 42 rds",
         f"CommEfficient/CommEfficient/plot_seed_mean.py --dataset mhealth_mm --cols {ic_cols} --rounds 42"),
        ("MFedMC v2", f"MFedMC/mHealth_v2/results/MFedMC_mHealth_v2_mm_*MB_*_it{MFEDMC_ROUNDS[b]}_*_mean.csv",
         f"1/10 clients · Top 1 · uncompressed · {MFEDMC_ROUNDS[b]} rds",
         f"MFedMC/mHealth_v2/plot_seed_mean.py --rounds {MFEDMC_ROUNDS[b]}"),
        ("PMFH-N", f"PMFH/results/PMFHfhN_mhealth_mm_mm_*MB_*_it{b // 5}_*_mean.csv",
         f"10/10 clients · 64-bit hash · {b // 5} rd{'s' if b > 5 else ''}",
         f"PMFH/plot_seed_mean_mhealth.py --rounds {b // 5}"),
        ("FL-FD", f"FL-FD/mhealth/results/FLFD_mhealth_mm_mm_*MB_*_it{FLFD_ROUNDS[b]}_*_mean.csv",
         f"8/10 clients · {FLFD_ROUNDS[b]} rds",
         f"FL-FD/mhealth/plot_seed_mean.py --rounds {FLFD_ROUNDS[b]}"),
    ]


def load(path):
    """(x, mean, std) from a seed-mean csv; MFedMC names its upload column cum_up_MB_mean."""
    with open(path, encoding="utf-8") as f:
        cols = f.readline().strip().split(",")
    arr = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
    xcol = "cum_up_MB" if "cum_up_MB" in cols else "cum_up_MB_mean"
    return (arr[:, cols.index(xcol)], arr[:, cols.index("mean_test_acc_pct")],
            arr[:, cols.index("std_test_acc_pct")])


def place_legend(ax, fig, curves, candidates, **kw):
    """Legend at the candidate spot covering the fewest (densely resampled) curve
    points, ties broken by the most clearance (as in HybridSketchMFedMC_v2/plot_configs_overlay.py)."""
    dense = []
    for x, y in curves:
        if len(x) == 1:
            dense.append(np.column_stack([x, y]))
            continue
        xs = np.linspace(x[0], x[-1], 400)
        dense.append(np.column_stack([xs, np.interp(xs, x, y)]))
    dense = np.vstack(dense)

    def score(leg):
        fig.canvas.draw()
        b, p = leg.get_window_extent(), ax.transData.transform(dense)
        dx = np.maximum(np.maximum(b.x0 - p[:, 0], p[:, 0] - b.x1), 0)
        dy = np.maximum(np.maximum(b.y0 - p[:, 1], p[:, 1] - b.y1), 0)
        return int(((dx == 0) & (dy == 0)).sum()), -float(np.hypot(dx, dy).min())

    best = None
    for loc, anchor in candidates:
        leg = ax.legend(loc=loc, bbox_to_anchor=anchor, **kw)
        s = score(leg)
        leg.remove()
        if best is None or s < best[0]:
            best = (s, loc, anchor)
    return ax.legend(loc=best[1], bbox_to_anchor=best[2], **kw)


def main():
    args = parse_args()
    rows, curves = [], []
    for label, pat, config, script in methods(args):
        hits = glob.glob(os.path.join(HERE, pat))
        if len(hits) != 1:
            raise SystemExit(f"{label}: expected one csv matching {pat}, found {len(hits)}"
                             + ("".join(f"\n  {h}" for h in hits) if hits else
                                f"\n  make it with: venv/bin/python {script}"))
        x, mean, std = load(hits[0])
        seeds = re.search(r"_seeds([0-9_]+)_mean\.csv$", hits[0]).group(1).replace("_", " ")
        rows.append(dict(label=label, config=config, seeds=seeds, x=x, mean=mean, std=std))

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    ax.axvline(args.budget, color="0.55", linestyle=":", linewidth=1.2, zorder=1)
    # label at the foot of the guide: the best curves end near the top at every budget
    ax.text(args.budget, 1.5, f" {args.budget} MB budget", color="0.4", fontsize=8, va="bottom", ha="left")
    # baselines first so ours is drawn on top
    for r in sorted(rows, key=lambda r: r["label"] == "Hybrid v2 (ours)"):
        ours = r["label"] == "Hybrid v2 (ours)"
        few = len(r["x"]) <= 10
        ax.plot(r["x"], r["mean"], color=COLORS[r["label"]], linewidth=2.4 if ours else 1.7,
                marker="o" if few else None, markersize=5, zorder=3 if ours else 2,
                label=f"{r['label']}: {r['mean'][-1]:.2f}% at {r['x'][-1]:.2f} MB  ({r['config']})")
        curves.append((r["x"], r["mean"]))

    ax.set_title(f"mHealth, ~{args.budget} MB upload budget: test accuracy vs cumulative upload "
                 f"(mean over seeds)", fontsize=10)
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, max(r["x"][-1] for r in rows) * 1.03)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    # legend order: ours first, then baselines in the fixed colour order
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: list(COLORS).index(labels[i].split(":")[0]))
    # corners plus mid-height spots on the right (the gap between the fast
    # risers and the slow ones that some budgets leave there); anchors in axes coords
    candidates = [("lower right", None), ("upper left", None), ("upper right", None)]
    candidates += [("center right", (0.99, y)) for y in np.arange(0.80, 0.29, -0.05)]
    place_legend(ax, fig, curves, candidates,
                 handles=[handles[i] for i in order], labels=[labels[i] for i in order],
                 fontsize=8, frameon=True, facecolor="white", edgecolor="0.7", framealpha=0.9)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"mhealth_{args.budget}MB_comparison.png")
    fig.savefig(out, bbox_inches="tight")

    csv = os.path.splitext(out)[0] + ".csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("method,config,seeds,final_cum_up_MB,final_mean_acc_pct,final_std_acc_pct\n")
        for r in sorted(rows, key=lambda r: list(COLORS).index(r["label"])):
            f.write(f"{r['label']},{r['config']},{r['seeds']},{r['x'][-1]:.2f},"
                    f"{r['mean'][-1]:.2f},{r['std'][-1]:.2f}\n")
            print(f"  {r['label']:<17} {r['mean'][-1]:6.2f}% ± {r['std'][-1]:5.2f}  at {r['x'][-1]:6.2f} MB  "
                  f"seeds {r['seeds']}")
    print(f"Plot saved to {out}")
    print(f"Data saved to {csv}")


if __name__ == "__main__":
    main()
