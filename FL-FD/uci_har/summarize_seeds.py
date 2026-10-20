"""Average FL-FD UCI HAR runs over seeds, in the columns of the Results sheet (UCI tab).

One row = mean over the given seeds of the final round of each run:
  Avg. Test Loss / Accuracy = last-round shard test_loss / test_acc (the "Final test accuracy" line),
  Avg. Time (s)             = last-round time_s (training time, as in the log table),
  Total Upload              = cumulative upload MB.

Called automatically at the end of run_uci_har_features.sh; can also be run by hand, e.g.
  venv/bin/python FL-FD/uci_har/summarize_seeds.py --seeds 42 43 44 45 46
  venv/bin/python FL-FD/uci_har/summarize_seeds.py --seeds 42 43 --rounds 280 --clients_per_round 10
"""
import argparse
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--dataset", default="uci_har_mm", help="dataset dir name in the file names")
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds")
    p.add_argument("--clients_per_round", type=int, default=None, help="only runs with this K")
    p.add_argument("--variant", default="", help="'' (GAF difference), 'summation' or 'shared_fd64'")
    p.add_argument("--since", type=float, default=0.0, help="only npz files written after this unix time")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    return p.parse_args()


def find_npz(args, seed):
    k = args.clients_per_round if args.clients_per_round else "*"
    r = args.rounds if args.rounds else "*"
    var = f"_{args.variant}" if args.variant else ""
    pat = os.path.join(args.results_dir, f"FLFD_{args.dataset}_K{k}_E*_R{r}{var}_seed{seed}.npz")
    rx = re.compile(rf"FLFD_{re.escape(args.dataset)}_K\d+_E\d+_R\d+{re.escape(var)}_seed{re.escape(seed)}\.npz$")
    hits = [f for f in glob.glob(pat) if rx.search(os.path.basename(f)) and os.path.getmtime(f) >= args.since]
    return max(hits, key=os.path.getmtime) if hits else None


def time_from_log(args, npz_name, seed):
    """Fallback for npz files written before time_s was stored: last time_s in the text log."""
    m = re.search(r"_K(\d+)_E(\d+)_R(\d+)", npz_name)
    var = f"{args.variant}_" if args.variant else ""
    pat = os.path.join(args.log_dir, f"FLFD_{args.dataset}_mm_*MB_*_it{m[3]}_K{m[1]}_E{m[2]}_lr*_{var}clients*_seed{seed}.txt")
    logs = [f for f in glob.glob(pat) if "NAMB" not in f]
    if not logs:
        return float("nan")
    rows = [ln.split() for ln in open(max(logs, key=os.path.getmtime), encoding="utf-8")]
    rows = [r for r in rows if len(r) == 8 and r[0] == m[3]]
    return float(rows[-1][-1]) if rows else float("nan")


def main():
    args = parse_args()
    runs, missing = [], []
    for s in args.seeds:
        f = find_npz(args, s)
        if f is None:
            missing.append(s)
            continue
        d = np.load(f)
        t = float(d["time_s"][-1]) if "time_s" in d.files else time_from_log(args, os.path.basename(f), s)
        runs.append(dict(seed=s, file=os.path.basename(f), rounds=len(d["test_acc"]),
                         k=int(d["clients_per_round"]), loss=float(d["test_loss"][-1]),
                         acc=100 * float(d["test_acc"][-1]), time=t, up=float(d["total_upload_mib"])))
    if not runs:
        raise SystemExit(f"No FL-FD results found for seeds {' '.join(args.seeds)}")
    configs = {(r["rounds"], r["k"]) for r in runs}
    if len(configs) > 1:
        raise SystemExit(f"Seeds come from different configs (rounds, K): {sorted(configs)}; "
                         f"pass --rounds / --clients_per_round to pick one")
    rounds, k = configs.pop()

    def mean(key):
        return float(np.nanmean([r[key] for r in runs]))

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        v = v[~np.isnan(v)]
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    n = args.num_clients
    head = ["Methods", "#Rows", "#Cols", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)", "Total Upload"]
    row = ["FL-FD", "-", "-", f"{k}/{n} ({round(100 * k / n)}%)", "-", str(rounds),
           f"{mean('loss'):.4f}", f"{mean('acc'):.2f}", f"{mean('time'):.2f}" if not all(np.isnan(r["time"]) for r in runs) else "n/a", f"{mean('up'):.2f} MB"]

    used = [r["seed"] for r in runs]
    lines = [f"FL-FD ({args.dataset}{', ' + args.variant if args.variant else ''}), "
             f"mean over {len(runs)} seed(s): {' '.join(used)}", "",
             "Per seed (final round):",
             f"{'seed':>6} {'test loss':>10} {'test acc %':>11} {'time s':>9} {'upload MB':>10}  file"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['loss']:>10.4f} {r['acc']:>11.2f} {r['time']:>9.2f} {r['up']:>10.2f}  {r['file']}")
    lines += ["", "Results-sheet row (mean over seeds):"]
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    lines += ["", f"std over seeds: test loss {std('loss'):.4f}, test acc {std('acc'):.2f}, time {std('time'):.2f}",
              "", "Tab-separated, paste into the sheet (columns A-J):", "\t".join(row)]
    no_time = [r["seed"] for r in runs if np.isnan(r["time"])]
    if no_time:
        lines += ["", f"WARNING: no time for seed(s) {' '.join(no_time)} (run before time_s was saved and no "
                      f"finished log); Avg. Time is over the other {len(runs) - len(no_time)} seed(s) only"]
    if missing:
        lines += ["", f"No results file for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)

    var = f"_{args.variant}" if args.variant else ""
    out = os.path.join(args.log_dir, f"FLFD_{args.dataset}_mm_{mean('up'):.2f}MB_it{rounds}_K{k}{var}_"
                                     f"seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
