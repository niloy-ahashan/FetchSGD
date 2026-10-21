"""Average Independent Compression (B1) UCI HAR runs over seeds, in the
columns of the Results sheet.

One row = mean over the given seeds of the final round of each run, read from
the seed-tagged text logs in result_updated/ (IC has no npz output; its
per-round history only lives in the console log table):
  Avg. Test Accuracy (%) = last-round test_acc (8-sample shard mean),
  Avg. Time (s)          = last-round time_s (training time, as in the log table),
  Total Upload           = cumulative upload MB (cum_up_MB).
The per-round table has no test_loss column, so that column is "n/a", as in
PMFH's / Hybrid v2's summarizers.

Called automatically at the end of run_uci_har_independent_compression.sh
(when NUM_SEEDS/SEEDS is set); can also be run by hand, e.g.
  venv/bin/python CommEfficient/CommEfficient/summarize_seeds.py --seeds 42 43 44 45 46
"""
import argparse
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "IC"
METHOD_LABEL = "Independent Compression (B1)"
# Total modality count per dataset dir name, for the "#Modalities" column.
# IC never selects a modality subset, so this is only used for documentation;
# the column itself is always "-".
TOTAL_MODALITIES = {"uci_har_mm": 2}

# Matches the config portion of _txt_stem (mm_train_independent.py), from
# right after the wildcarded up_str through to "_seed"; up_str and seed are
# deliberately not captured here (same reasoning as Hybrid v2's summarizer).
CONFIG_RX = re.compile(
    r"_dirichlet(?P<alpha>[^_]+)_it(?P<it>\d+)_"
    r"r(?P<rows>\d+)xc(?P<cols>\d+)_k(?P<k>\d+)_"
    r"lr(?P<lr>[^_]+)_"
    r"clients(?P<clients>\d+)_workers(?P<workers>\d+)_seed"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--dataset", default="uci_har_mm", help="dataset dir name in the file names")
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds (--num_epochs)")
    p.add_argument("--since", type=float, default=0.0, help="only logs written after this unix time")
    p.add_argument("--log_dir", default=os.path.join(os.path.dirname(HERE), "result_updated"))
    return p.parse_args()


def find_log(args, seed):
    r = args.rounds if args.rounds else "*"
    pat = os.path.join(args.log_dir, f"{PREFIX}_{args.dataset}_mm_*MB_dirichlet*_it{r}_*_seed{seed}.txt")
    hits = [f for f in glob.glob(pat)
            if "NAMB" not in f and f.endswith(f"_seed{seed}.txt") and os.path.getmtime(f) >= args.since]
    return max(hits, key=os.path.getmtime) if hits else None


def last_row(path):
    """Last row of the per-round table, as {column: value}."""
    keys, row = None, None
    for ln in open(path, encoding="utf-8"):
        parts = ln.split()
        if parts and parts[0] == "iter":
            keys = parts
        elif keys and len(parts) == len(keys) and parts[0].isdigit():
            row = dict(zip(keys, map(float, parts)))
    return row


def main():
    args = parse_args()
    runs, missing = [], []
    for s in args.seeds:
        f = find_log(args, s)
        row = last_row(f) if f else None
        m = CONFIG_RX.search(os.path.basename(f)) if f else None
        if row is None or m is None:
            missing.append(s)
            continue
        runs.append(dict(
            seed=s, file=os.path.basename(f), config=m.groupdict(), rounds=int(row["iter"]),
            planned=int(m["it"]), workers=int(m["workers"]), clients=int(m["clients"]),
            acc=100 * row["test_acc"], time=row["time_s"], up=row["cum_up_MB"],
        ))
    if not runs:
        raise SystemExit(f"No {PREFIX} logs found for seeds {' '.join(args.seeds)}")
    configs = {tuple(sorted(r["config"].items())) for r in runs}
    if len(configs) > 1:
        raise SystemExit(f"Seeds come from different configs (dirichlet/it/rows/cols/k/lr/"
                         f"clients/workers): {sorted(configs)}; pass --rounds / --since to pick one")
    unfinished = [r["seed"] for r in runs if r["rounds"] != r["planned"]]
    if unfinished:
        raise SystemExit(f"Seed(s) {' '.join(unfinished)} stopped before the last round")
    rounds, k, n = runs[0]["rounds"], runs[0]["workers"], runs[0]["clients"]
    cfg = runs[0]["config"]

    def mean(key):
        return float(np.mean([r[key] for r in runs]))

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    head = ["Methods", "#Rows", "#Cols", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)", "Total Upload"]
    row = [METHOD_LABEL, cfg["rows"], cfg["cols"], f"{k}/{n} ({round(100 * k / n)}%)", "-",
           str(rounds), "n/a", f"{mean('acc'):.2f}", f"{mean('time'):.2f}", f"{mean('up'):.2f} MB"]

    used = [r["seed"] for r in runs]
    lines = [f"{METHOD_LABEL} ({args.dataset}), mean over {len(runs)} seed(s): {' '.join(used)}", "",
             "Per seed (final round):",
             f"{'seed':>6} {'test acc %':>11} {'time s':>9} {'upload MB':>10}  file"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['acc']:>11.2f} {r['time']:>9.2f} {r['up']:>10.2f}  {r['file']}")
    lines += ["", "Results-sheet row (mean over seeds):"]
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    lines += ["", f"std over seeds: test acc {std('acc'):.2f}, time {std('time'):.2f}, upload {std('up'):.4f}",
              "", "Tab-separated, paste into the sheet (columns A-J):", "\t".join(row)]
    if missing:
        lines += ["", f"No finished log for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)

    out = os.path.join(args.log_dir, f"{PREFIX}_{args.dataset}_mm_{mean('up'):.2f}MB_dirichlet{cfg['alpha']}_"
                                     f"it{rounds}_r{cfg['rows']}xc{cfg['cols']}_k{cfg['k']}_lr{cfg['lr']}_"
                                     f"clients{n}_workers{k}_seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
