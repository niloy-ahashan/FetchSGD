"""Plot the mean test accuracy over seeds of one FL-FD mHealth config against
cumulative upload, read from the seed-tagged text logs in result_updated/
(same style as CommEfficient/CommEfficient/plot_seed_mean.py for IC and
PMFH/plot_seed_mean_mhealth.py, so the budget plots line up).

One curve = mean over the given seeds of test_acc at each round, with the
individual seeds drawn faintly underneath; x is the cumulative upload MB
(cum_up_MB), which is identical across seeds for FL-FD (full model from a
fixed number of clients each round), and y is fixed to 0-100 % in steps of
20. The std is not drawn; it is printed and written to the .csv next to the PNG.

The upload budget is set by the number of rounds, so pick a config with
--rounds, e.g. (97 rounds, 25.11 MB):
  venv/bin/python FL-FD/mhealth/plot_seed_mean.py --rounds 97 --seeds 42 43 44 45 46
"""
import argparse
import ast
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "FLFD_mhealth_mm"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds")
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    p.add_argument("--out", default=None, help="PNG path (default: FL-FD/mhealth/results/<config>_mean.png)")
    return p.parse_args()


def find_log(args, seed):
    r = args.rounds if args.rounds else "*"
    pat = os.path.join(args.log_dir, f"{PREFIX}_mm_*MB_dirichlet*_it{r}_*_seed{seed}.txt")
    hits = [f for f in glob.glob(pat) if "NAMB" not in f and f.endswith(f"_seed{seed}.txt")]
    if len(hits) > 1:
        raise SystemExit(f"Several logs match seed {seed}; pass --rounds to pick one:\n  "
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
    first_file, first = next(iter(runs.values()))
    for s, (_, t) in runs.items():
        if not np.allclose(t["cum_up_MB"], first["cum_up_MB"]):
            raise SystemExit(f"Seed {s} has a different cum_up_MB column than seed {args.seeds[0]}")

    x = first["cum_up_MB"]
    acc = np.stack([100 * t["test_acc"] for _, t in runs.values()])
    mean = acc.mean(axis=0)
    std = acc.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)

    base = os.path.basename(first_file)
    stem = base[: base.rindex("_seed")]
    out = args.out or os.path.join(HERE, "results", f"{stem}_seeds{'_'.join(args.seeds)}_mean.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    hp, n_mod = header(first_file)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    for row in acc:
        ax.plot(x, row, color="#4C72B0", alpha=0.18, linewidth=0.9)
    ax.plot(x, mean, color="#4C72B0", linewidth=2.0, marker="o", markersize=2,
            label=f"mean of {len(runs)} seeds")
    ax.annotate(f"{mean[-1]:.2f}%", xy=(x[-1], mean[-1]),
                xytext=(-8, -16 if mean[-1] > 85 else 10), textcoords="offset points",
                ha="right", fontsize=9)
    ax.set_title(f"FL-FD on mHealth, {x[-1]:.2f} MB upload, "
                 f"{len(x)} round{'s' if len(x) > 1 else ''}", fontsize=10)
    # bottom-right, above the legend: FL-FD passes 80 % within a few MB, so the
    # top-left corner (used by the IC/PMFH plots) would cover the curve
    ax.text(0.98, 0.14,
            f"Clients: {hp['clients_per_round']}/{hp['num_clients']} per round\n"
            f"Modalities: {n_mod} (GAF channels)\n"
            f"Local epochs: {hp['local_epochs']}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.7"))
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, x[-1] * 1.02)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
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
