"""Plot the mean test accuracy over seeds of HybridSketchMFedMC v2 (B3) runs
against cumulative upload, one plot per config, read from the seed-tagged
text logs (same style as the IC / PMFH / FL-FD plot_seed_mean scripts, so the
budget plots line up).

Every config found in --log_dir (logs grouped by their name up to _seed<N>)
gets its own plot: mean over seeds of test_acc at each round, with the
individual seeds drawn faintly underneath; x is the cumulative upload MB
(cum_up_MB), identical across seeds within a config (fixed sketch size and
client count per round), and y is fixed to 0-100 % in steps of 20. The std is
not drawn; it is printed and written to the .csv next to each PNG. Without
--seeds every seed found for a config is used (some configs have fewer).

Example (all 15 configs of the 5 MB budget sweep, into results/5MB/):
  venv/bin/python HybridSketchMFedMC_v2/plot_seed_mean.py \
    --log_dir HybridSketchMFedMC_v2/result_updated/5MB
"""
import argparse
import ast
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "HybridV2"
DATASET_LABELS = {"mhealth": "mHealth", "uci_har": "UCI HAR"}
CONFIG_RX = re.compile(r"_it(?P<it>\d+)_r(?P<rows>\d+)xc(?P<cols>\d+)_.*"
                       r"Top_(?P<top>\d+)_Client_(?P<client>[0-9.]+)_")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    p.add_argument("--seeds", nargs="+", default=None, help="default: every seed found per config")
    p.add_argument("--rounds", type=int, default=None, help="only configs with this many rounds")
    p.add_argument("--cols", type=int, default=None, help="only configs with this many sketch columns")
    p.add_argument("--top", type=int, default=None, help="only configs keeping this many modalities")
    p.add_argument("--client", type=float, default=None, help="only configs with this client ratio")
    p.add_argument("--out_dir", default=None,
                   help="default: results/<log_dir name> for a result_updated subfolder, else results/")
    return p.parse_args()


def find_configs(args):
    """{stem: {seed: path}} for every config in log_dir passing the filters."""
    groups = defaultdict(dict)
    for f in glob.glob(os.path.join(args.log_dir, f"{PREFIX}_*_seed*.txt")):
        base = os.path.basename(f)
        m = re.search(r"_seed(\d+)\.txt$", base)
        if "NAMB" in base or not m:
            continue
        stem = base[: m.start()]
        c = CONFIG_RX.search(stem)
        if c is None:
            continue
        if ((args.rounds is not None and int(c["it"]) != args.rounds)
                or (args.cols is not None and int(c["cols"]) != args.cols)
                or (args.top is not None and int(c["top"]) != args.top)
                or (args.client is not None and float(c["client"]) != args.client)):
            continue
        groups[stem][m.group(1)] = f
    return groups


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


def header(path):
    """Resolved hyperparameters (as strings) plus the modality count from the mod_dims line."""
    hp, n_mod = {}, None
    for ln in open(path, encoding="utf-8"):
        m = re.search(r"mod_dims=(\{.*?\})", ln)
        if m:
            n_mod = len(ast.literal_eval(m.group(1)))
        m = re.match(r"^  (\w+): (.+)$", ln)
        if m:
            hp[m.group(1)] = m.group(2).strip()
        if ln.split()[:1] == ["iter"]:
            break
    return hp, n_mod


def plot_config(stem, files, seeds, out_dir):
    runs = {}
    for s in seeds:
        t = table(files[s])
        if t is None:
            raise SystemExit(f"No round table in {files[s]}")
        runs[s] = t

    lengths = {len(t["iter"]) for t in runs.values()}
    if len(lengths) > 1:
        raise SystemExit(f"{stem}: seeds have different round counts: "
                         f"{ {s: len(t['iter']) for s, t in runs.items()} }")
    first = runs[seeds[0]]
    for s, t in runs.items():
        if not np.allclose(t["cum_up_MB"], first["cum_up_MB"]):
            raise SystemExit(f"{stem}: seed {s} has a different cum_up_MB column than seed {seeds[0]}")

    x = first["cum_up_MB"]
    acc = np.stack([100 * t["test_acc"] for t in runs.values()])
    mean = acc.mean(axis=0)
    std = acc.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)

    hp, n_mod = header(files[seeds[0]])
    n_clients = int(hp["num_clients"])
    per_round = max(1, round(float(hp["client_select_ratio"]) * n_clients))
    dataset = DATASET_LABELS.get(hp.get("dataset"), hp.get("dataset", "?"))

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for row in acc:
        ax.plot(x, row, color="#4C72B0", alpha=0.18, linewidth=0.9)
    ax.plot(x, mean, color="#4C72B0", linewidth=2.0, marker="o", markersize=2,
            label=f"mean of {len(runs)} seeds ({', '.join(seeds)})")
    ax.annotate(f"{mean[-1]:.2f}%", xy=(x[-1], mean[-1]),
                xytext=(-8, -16 if mean[-1] > 85 else 10), textcoords="offset points",
                ha="right", fontsize=9)
    ax.set_title(f"HybridSketchMFedMC v2 on {dataset}, {x[-1]:.2f} MB upload, "
                 f"{len(x)} round{'s' if len(x) > 1 else ''}", fontsize=10)
    info = (f"Sketch size: {hp['num_rows']}×{hp['num_cols']}\n"
            f"Clients: {per_round}/{n_clients} per round\n"
            f"Modalities: {hp['num_select_modalities']}/{n_mod} per client")
    box = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7")
    # Curves that end high pass through the top-left early, so their legend and
    # info box go bottom-right (as in the FL-FD plots); low ones keep IC's top-left.
    if mean[-1] >= 60:
        ax.text(0.98, 0.14, info, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=box)
        ax.legend(loc="lower right", fontsize=9, frameon=False)
    else:
        ax.text(0.02, 0.88, info, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox=box)
        ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, x[-1] * 1.02)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, f"{stem}_seeds{'_'.join(seeds)}_mean.png")
    fig.savefig(out)
    plt.close(fig)

    csv = os.path.splitext(out)[0] + ".csv"
    np.savetxt(csv, np.column_stack([first["iter"], x, mean, std]), delimiter=",",
               header="round,cum_up_MB,mean_test_acc_pct,std_test_acc_pct", comments="", fmt="%g")
    return x[-1], len(x), mean[-1], std[-1], out


def main():
    args = parse_args()
    log_dir = os.path.abspath(args.log_dir)
    sub = os.path.basename(log_dir)
    out_dir = args.out_dir or os.path.join(HERE, "results", *([] if sub == "result_updated" else [sub]))
    os.makedirs(out_dir, exist_ok=True)

    groups = find_configs(args)
    if not groups:
        raise SystemExit(f"No {PREFIX} logs in {args.log_dir} match the filters")
    for stem in sorted(groups):
        files = groups[stem]
        if args.seeds:
            missing = [s for s in args.seeds if s not in files]
            if missing:
                raise SystemExit(f"{stem}: no log for seed(s) {' '.join(missing)}")
            seeds = list(args.seeds)
        else:
            seeds = sorted(files, key=int)
        up, rounds, m, sd, out = plot_config(stem, files, seeds, out_dir)
        print(f"{os.path.basename(out)}\n  seeds {' '.join(seeds)}: {rounds} rounds, cum_up {up:.2f} MB, "
              f"test acc {m:.2f} ± {sd:.2f} %")
    print(f"{len(groups)} plot(s) saved to {out_dir}")


if __name__ == "__main__":
    main()
