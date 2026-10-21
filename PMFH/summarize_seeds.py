"""Average PMFH UCI HAR runs (main_uci_har_v2.py) over seeds, in the columns of the Results sheet.

One row = mean over the given seeds of the final round of each run, read from the
seed-tagged text logs in result_updated/ (the npz name has no seed, so it is not used):
  Avg. Test Accuracy (%) = last-round test_acc (trained head, 8-sample shard mean),
  Avg. Time (s)          = last-round time_s (training time, as in the log table),
  Total Upload           = cumulative upload MB (cum_up_MB).
The runner logs no test loss, so that column is "n/a".

Called automatically at the end of run_uci_har_pmfh_fedhead.sh; can also be run by hand, e.g.
  venv/bin/python PMFH/summarize_seeds.py --seeds 42 43 44 45 46
  venv/bin/python PMFH/summarize_seeds.py --prefix PMFHjh --seeds 42 43 --rounds 10
"""
import argparse
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = {"PMFHfh": "PMFH (fed. head)", "PMFHjh": "PMFH (joint head)", "PMFHv2": "PMFH-v2"}
CONFIG_RX = re.compile(r"_it(\d+)_bit(\d+)_h(\d+)_c(\d+)_clf([^_]+)_lr([^_]+)_E(\d+)_Ep(\d+)_"
                       r"clients(\d+)_workers(\d+)_seed")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--prefix", default="PMFHfh", choices=sorted(METHODS),
                   help="PMFHfh = fedhead, PMFHjh = joint head, PMFHv2 = v2")
    p.add_argument("--dataset", default="uci_har_mm", help="dataset dir name in the file names")
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds")
    p.add_argument("--since", type=float, default=0.0, help="only logs written after this unix time")
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    return p.parse_args()


def find_log(args, seed):
    r = args.rounds if args.rounds else "*"
    pat = os.path.join(args.log_dir, f"{args.prefix}_{args.dataset}_mm_*MB_*_it{r}_*_seed{seed}.txt")
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
        if row is None:
            missing.append(s)
            continue
        m = CONFIG_RX.search(os.path.basename(f))
        runs.append(dict(seed=s, file=os.path.basename(f), config=m.groups()[:-1], rounds=int(row["iter"]),
                         planned=int(m[1]), workers=int(m[10]), clients=int(m[9]),
                         acc=100 * row["test_acc"], time=row["time_s"], up=row["cum_up_MB"]))
    if not runs:
        raise SystemExit(f"No {args.prefix} logs found for seeds {' '.join(args.seeds)}")
    configs = {r["config"] for r in runs}
    if len(configs) > 1:
        raise SystemExit(f"Seeds come from different configs (it, bit, h, c, clf, lr, E, Ep, clients, "
                         f"workers): {sorted(configs)}; pass --rounds / --since to pick one")
    unfinished = [r["seed"] for r in runs if r["rounds"] != r["planned"]]
    if unfinished:
        raise SystemExit(f"Seed(s) {' '.join(unfinished)} stopped before the last round")
    rounds, k, n = runs[0]["rounds"], runs[0]["workers"], runs[0]["clients"]

    def mean(key):
        return float(np.mean([r[key] for r in runs]))

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    head = ["Methods", "#Rows", "#Cols", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)", "Total Upload"]
    row = [METHODS[args.prefix], "-", "-", f"{k}/{n} ({round(100 * k / n)}%)", "-", str(rounds),
           "n/a", f"{mean('acc'):.2f}", f"{mean('time'):.2f}", f"{mean('up'):.2f} MB"]

    used = [r["seed"] for r in runs]
    lines = [f"{METHODS[args.prefix]} ({args.dataset}), mean over {len(runs)} seed(s): {' '.join(used)}", "",
             "Per seed (final round):",
             f"{'seed':>6} {'test acc %':>11} {'time s':>9} {'upload MB':>10}  file"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['acc']:>11.2f} {r['time']:>9.2f} {r['up']:>10.2f}  {r['file']}")
    lines += ["", "Results-sheet row (mean over seeds):"]
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    lines += ["", f"std over seeds: test acc {std('acc'):.2f}, time {std('time'):.2f}",
              "", "Tab-separated, paste into the sheet (columns A-J):", "\t".join(row)]
    if missing:
        lines += ["", f"No finished log for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)

    _, bit, h, c = runs[0]["config"][:4]
    out = os.path.join(args.log_dir, f"{args.prefix}_{args.dataset}_mm_{mean('up'):.2f}MB_it{rounds}_"
                                     f"bit{bit}_h{h}_c{c}_seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
