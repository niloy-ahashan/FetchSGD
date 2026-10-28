"""Plot the mean test accuracy over seeds of one Independent Compression (B1)
config against cumulative upload, read from the seed-tagged text logs in
result_updated/ (IC has no npz output; its per-round history only lives in
the console log table).

One curve = mean over the given seeds of test_acc at each round, with the
individual seeds drawn faintly underneath; x is the cumulative upload MB
(cum_up_MB), which is identical across seeds for IC, and y is fixed to
0-100 % in steps of 20. The std is not drawn; it is printed and written to
the .csv of round, cum_up_MB, mean, std next to the PNG.

Example (mHealth, 5 MB budget = 3x1000 sketch, 42 rounds):
  venv/bin/python CommEfficient/CommEfficient/plot_seed_mean.py \
    --dataset mhealth_mm --cols 1000 --rounds 42 --seeds 42 43 44 45 46
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "IC"
DATASET_LABELS = {"mhealth_mm": "mHealth", "uci_har_mm": "UCI HAR"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--dataset", default="uci_har_mm", help="dataset dir name in the file names")
    p.add_argument("--cols", type=int, default=None, help="only runs with this many sketch columns (--num_cols)")
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds (--num_epochs)")
    p.add_argument("--log_dir", default=os.path.join(os.path.dirname(HERE), "result_updated"))
    p.add_argument("--out", default=None, help="PNG path (default: CommEfficient/results/<config>_mean.png)")
    return p.parse_args()


def num_modalities(dataset):
    """Modality count from datasets/<dataset>/prepare_stats.json (mod_dims);
    2-modality caches (e.g. uci_har_mm: acc + gyro) have no mod_dims."""
    path = os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets", dataset, "prepare_stats.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            mod_dims = json.load(f).get("mod_dims")
        if mod_dims:
            return len(mod_dims)
    return 2


def find_log(args, seed):
    r = args.rounds if args.rounds else "*"
    c = args.cols if args.cols else "*"
    pat = os.path.join(args.log_dir, f"{PREFIX}_{args.dataset}_mm_*MB_dirichlet*_it{r}_r*xc{c}_*_seed{seed}.txt")
    hits = [f for f in glob.glob(pat) if "NAMB" not in f and f.endswith(f"_seed{seed}.txt")]
    if len(hits) > 1:
        raise SystemExit(f"Several logs match seed {seed}; pass --cols / --rounds to pick one:\n  "
                         + "\n  ".join(sorted(hits)))
    return hits[0] if hits else None


def table(path):
    """All rows of the per-round table, as {column: np.array}."""
    keys, rows = None, []
    for ln in open(path, encoding="utf-8"):
        parts = ln.split()
        if parts and parts[0] == "iter":
            keys, rows = parts, []
        elif keys and len(parts) == len(keys) and parts[0].isdigit():
            rows.append(list(map(float, parts)))
    if not rows:
        return None
    arr = np.array(rows)
    return {k: arr[:, i] for i, k in enumerate(keys)}


def main():
    args = parse_args()
    runs = {}
    for s in args.seeds:
        f = find_log(args, s)
        t = table(f) if f else None
        if t is None:
            raise SystemExit(f"No {PREFIX} log with a round table for seed {s}")
        runs[s] = (f, t)

    lengths = {len(t["iter"]) for _, t in runs.values()}
    if len(lengths) > 1:
        raise SystemExit(f"Seeds have different round counts: "
                         f"{ {s: len(t['iter']) for s, (_, t) in runs.items()} }")
    first = next(iter(runs.values()))[1]
    for s, (_, t) in runs.items():
        if not np.allclose(t["cum_up_MB"], first["cum_up_MB"]):
            raise SystemExit(f"Seed {s} has a different cum_up_MB column than seed {args.seeds[0]}")

    x = first["cum_up_MB"]
    acc = np.stack([100 * t["test_acc"] for _, t in runs.values()])
    mean = acc.mean(axis=0)
    std = acc.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)

    base = os.path.basename(next(iter(runs.values()))[0])
    stem = base[: base.rindex("_seed")]
    cfg = stem.split("_dirichlet")[1]
    rows_cols = next(p for p in cfg.split("_") if "xc" in p)[1:].replace("xc", "×")
    out = args.out or os.path.join(os.path.dirname(HERE), "results",
                                   f"{stem}_seeds{'_'.join(args.seeds)}_mean.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for row in acc:
        ax.plot(x, row, color="#4C72B0", alpha=0.18, linewidth=0.9)
    ax.plot(x, mean, color="#4C72B0", linewidth=2.0, label=f"mean of {len(runs)} seeds")
    ax.annotate(f"{mean[-1]:.2f}%", xy=(x[-1], mean[-1]),
                xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=9)
    dataset = DATASET_LABELS.get(args.dataset, args.dataset)
    clients = int(cfg.split("_clients")[1].split("_")[0])
    workers = int(cfg.split("_workers")[1].split("_")[0])
    ax.set_title(f"IC (FetchSGD) on {dataset}, {x[-1]:.2f} MB upload, {len(x)} rounds", fontsize=10)
    # top-left, under the legend: every curve starts near chance, so it stays clear
    ax.text(0.02, 0.88,
            f"Sketch size: {rows_cols}\n"
            f"Clients: {workers}/{clients} per round\n"
            f"Modalities: {num_modalities(args.dataset)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7"))
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, x[-1] * 1.02)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out)

    csv = os.path.splitext(out)[0] + ".csv"
    np.savetxt(csv, np.column_stack([first["iter"], x, mean, std]), delimiter=",",
               header="round,cum_up_MB,mean_test_acc_pct,std_test_acc_pct", comments="", fmt="%g")

    print(f"Seeds {' '.join(args.seeds)}: final round {int(first['iter'][-1])}, cum_up {x[-1]:.2f} MB, "
          f"test acc {mean[-1]:.2f} ± {std[-1]:.2f} %")
    print(f"Plot saved to {out}")
    print(f"Data saved to {csv}")


if __name__ == "__main__":
    main()
