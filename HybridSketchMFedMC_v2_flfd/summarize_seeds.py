"""Average HybridSketchMFedMC v2 + FL-FD-model runs over seeds, in the columns of the Results sheet.

One row = mean over the given seeds of the final round of each run:
  Avg. Test Loss / Accuracy = last-round test_loss / test_acc from the npz (8-sample shard mean, as Hybrid v2),
  Avg. Time (s)             = last-round time_s from the seed's text log,
  Total Upload              = cumulative upload MB.

Called automatically at the end of run_uci_har.sh; by hand, e.g.
  venv/bin/python HybridSketchMFedMC_v2_flfd/summarize_seeds.py --seeds 42 43 44 45 46 --rounds 2084
"""
import argparse
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "HybridV2FD"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--model", default=None, help="branch or flfd (default: any)")
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds (--num_epochs)")
    p.add_argument("--since", type=float, default=0.0, help="only npz files written after this unix time")
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    return p.parse_args()


def find_npz(args, seed):
    model = args.model or "*"
    it = args.rounds or "*"
    hits = [f for f in glob.glob(os.path.join(args.results_dir, f"{PREFIX}_{model}_*_it{it}_*_seed{seed}.npz"))
            if os.path.getmtime(f) >= args.since]
    return max(hits, key=os.path.getmtime) if hits else None


def time_from_log(args, npz_path):
    stem = os.path.basename(npz_path)[:-4]
    pat = stem.replace("_mm_dirichlet", "_mm_*MB_dirichlet", 1)
    logs = [f for f in glob.glob(os.path.join(args.log_dir, pat + ".txt")) if "NAMB" not in f]
    if not logs:
        return float("nan")
    last = float("nan")
    for line in open(max(logs, key=os.path.getmtime), encoding="utf-8"):
        cols = line.split()
        if len(cols) == 8 and cols[0].isdigit():
            last = float(cols[-1])
    return last


def main():
    args = parse_args()
    runs, missing = [], []
    for s in args.seeds:
        f = find_npz(args, s)
        if f is None:
            missing.append(s)
            continue
        d = np.load(f)
        name = os.path.basename(f)
        m = re.search(r"_r(\d+)xc(\d+)_k(\d+)_", name)
        runs.append(dict(seed=s, file=name, model=name.split("_")[1], rounds=len(d["test_acc"]),
                         rows=int(m[1]), cols=int(m[2]), k=int(m[3]),
                         clients=int(round(float(d["client_selected"].sum(1).mean()))),
                         n_clients=int(d["client_selected"].shape[1]), n_mod=int(d["modality_selected"].shape[2]),
                         mods=int(round(float(d["modality_selected"].sum(2)[d["client_selected"] > 0].mean()))),
                         loss=float(d["test_loss"][-1]), acc=100 * float(d["test_acc"][-1]),
                         time=time_from_log(args, f), up=float(d["total_upload_mib"])))
    if not runs:
        raise SystemExit(f"No {PREFIX} results found for seeds {' '.join(args.seeds)}")
    configs = {(r["model"], r["rounds"], r["rows"], r["cols"], r["k"], r["clients"]) for r in runs}
    if len(configs) > 1:
        raise SystemExit(f"Seeds come from different configs (model, rounds, rows, cols, k, clients): "
                         f"{sorted(configs)}; pass --model / --rounds to pick one")
    model, rounds, rows, cols, k, clients = configs.pop()
    n, nm, mods = runs[0]["n_clients"], runs[0]["n_mod"], runs[0]["mods"]

    def mean(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        v = v[np.isfinite(v)]
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    head = ["Methods", "#Rows", "#Cols", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)", "Total Upload"]
    row = [f"Ours (FL-FD {model} model)", str(rows), str(cols), f"{clients}/{n} ({round(100 * clients / n)}%)",
           f"{mods}/{nm} ({round(100 * mods / nm)}%)", str(rounds), f"{mean('loss'):.4f}", f"{mean('acc'):.2f}",
           f"{mean('time'):.2f}", f"{mean('up'):.2f} MB"]
    used = [r["seed"] for r in runs]
    lines = [f"HybridSketchMFedMC v2 + FL-FD {model} model (uci_har_mm), k = {k}, "
             f"mean over {len(runs)} seed(s): {' '.join(used)}", "",
             "Per seed (final round):",
             f"{'seed':>6} {'test loss':>10} {'test acc %':>11} {'time s':>9} {'upload MB':>10}  file"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['loss']:>10.4f} {r['acc']:>11.2f} {r['time']:>9.2f} {r['up']:>10.2f}  {r['file']}")
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines += ["", "Results-sheet row (mean over seeds):",
              "  ".join(h.ljust(w) for h, w in zip(head, widths)),
              "  ".join(v.ljust(w) for v, w in zip(row, widths)), "",
              f"std over seeds: test loss {std('loss'):.4f}, test acc {std('acc'):.2f}, time {std('time'):.2f}",
              "", "Tab-separated, paste into the sheet (columns A-J):", "\t".join(row)]
    no_time = [r["seed"] for r in runs if not np.isfinite(r["time"])]
    if no_time:
        lines += ["", f"WARNING: no finished log for seed(s) {' '.join(no_time)}; Avg. Time is over the others"]
    if missing:
        lines += ["", f"No results file for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)
    out = os.path.join(args.log_dir, f"{PREFIX}_{model}_uci_har_mm_mm_{mean('up'):.2f}MB_it{rounds}_r{rows}xc{cols}_"
                                     f"k{k}_seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
