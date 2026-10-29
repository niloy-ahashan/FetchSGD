"""Average FedMobile runs over seeds, in the columns of the Results sheet.

One row = mean over the given seeds of the final round of each run:
  Avg. Test Loss / Accuracy = last-round shard test_loss / test_acc (the "Final test accuracy" line),
  Avg. Time (s)             = last-round time_s (training time, as in the log table),
  Total Upload              = cumulative upload MB.

Called automatically at the end of run_uci_har*.sh / run_mhealth*.sh; can also be run by hand, e.g.
  MFedMC/.venv/bin/python FedMobile/torch_port/summarize_seeds.py --seeds 42 43 44 45 46
  MFedMC/.venv/bin/python FedMobile/torch_port/summarize_seeds.py --dataset mhealth_mm_random_group \
      --rounds 150 --variant miss0r0.8 --seeds 42 43 44 45 46
  MFedMC/.venv/bin/python FedMobile/torch_port/summarize_seeds.py --rounds 16 --local_epochs 5 --variant '' --seeds 42
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
    p.add_argument("--local_epochs", type=int, default=None, help="only runs with this many local epochs (E in the name)")
    p.add_argument("--variant", default=None,
                   help="name tag(s) between R<rounds>_ and _seed, e.g. 'miss0r0.8' or 'traingen'; "
                        "'' = plain runs only; default = any (use with --since)")
    p.add_argument("--since", type=float, default=0.0, help="only npz files written after this unix time")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    return p.parse_args()


def find_npz(args, seed):
    r = args.rounds if args.rounds else r"\d+"
    e = args.local_epochs if args.local_epochs else r"\d+"
    if args.variant is None:
        var = r"(?:_[^/]+?)?"
    else:
        var = f"_{re.escape(args.variant)}" if args.variant else ""
    rx = re.compile(rf"^FedMobile_{re.escape(args.dataset)}_K\d+_E{e}_R{r}{var}_seed{re.escape(seed)}\.npz$")
    hits = [f for f in glob.glob(os.path.join(args.results_dir, f"FedMobile_{args.dataset}_*_seed{seed}.npz"))
            if rx.match(os.path.basename(f)) and os.path.getmtime(f) >= args.since]
    return max(hits, key=os.path.getmtime) if hits else None


def _tag(name):
    """Variant tag(s) of an npz name ('' for a plain run)."""
    m = re.search(r"_R\d+(?:_(.+?))?_seed\d+\.npz$", name)
    return (m[1] or "") if m else ""


def main():
    args = parse_args()
    runs, missing = [], []
    for s in args.seeds:
        f = find_npz(args, s)
        if f is None:
            missing.append(s)
            continue
        d = np.load(f)
        runs.append(dict(seed=s, file=os.path.basename(f), tag=_tag(os.path.basename(f)),
                         rounds=len(d["test_acc"]), k=int(d["clients_per_round"]),
                         epochs=int(d["local_epochs"]), loss=float(d["test_loss"][-1]),
                         acc=100 * float(d["test_acc"][-1]), time=float(d["time_s"][-1]),
                         up=float(d["total_upload_mib"]), kept=float(d["kept"].sum(1).mean())))
    if not runs:
        raise SystemExit(f"No FedMobile results found for seeds {' '.join(args.seeds)}")
    configs = {(r["rounds"], r["epochs"], r["tag"]) for r in runs}
    if len(configs) > 1:
        raise SystemExit(f"Seeds come from different configs (rounds, local epochs, tag): {sorted(configs)}; "
                         f"pass --rounds / --local_epochs / --variant (or --since) to pick one")
    rounds, epochs, tag = configs.pop()
    k = runs[0]["k"]

    def mean(key):
        return float(np.mean([r[key] for r in runs]))

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    n = args.num_clients
    head = ["Methods", "#Rows", "#Cols", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)", "Total Upload"]
    row = ["FedMobile", "-", "-", f"{k}/{n} ({round(100 * k / n)}%)", "-", str(rounds),
           f"{mean('loss'):.4f}", f"{mean('acc'):.2f}", f"{mean('time'):.2f}", f"{mean('up'):.2f} MB"]

    used = [r["seed"] for r in runs]
    lines = [f"FedMobile ({args.dataset}{', ' + tag if tag else ''}), mean over {len(runs)} seed(s): {' '.join(used)}",
             "", "Per seed (final round):",
             f"{'seed':>6} {'test loss':>10} {'test acc %':>11} {'time s':>9} {'upload MB':>10} {'kept/rnd':>9}  file"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['loss']:>10.4f} {r['acc']:>11.2f} {r['time']:>9.2f} {r['up']:>10.2f} "
                     f"{r['kept']:>9.2f}  {r['file']}")
    lines += ["", "Results-sheet row (mean over seeds):"]
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    lines += ["", f"std over seeds: test loss {std('loss'):.4f}, test acc {std('acc'):.2f}, time {std('time'):.2f}",
              "", "Tab-separated, paste into the sheet (columns A-J):", "\t".join(row)]
    if missing:
        lines += ["", f"No results file for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)

    out = os.path.join(args.log_dir, f"FedMobile_{args.dataset}_mm_{mean('up'):.2f}MB_it{rounds}_K{k}_E{epochs}"
                                     f"{'_' + tag if tag else ''}_seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
