"""Average MFedMC v2 (B2) mHealth runs over seeds, in the columns of the
Results sheet.

One row = mean over the given seeds of the final round of each run, read from
the seed-tagged text logs in result_updated/ (the runner's npz name has no
seed and is therefore not used here for the mean; run_mhealth_mfedmc_
sketchfusion_protocol_v2.sh still rescues a seed-tagged copy of it, for
manual inspection):
  Avg. Test Accuracy (%) = last-round test_acc (8-sample shard mean, IC convention),
  Avg. Time (s)          = last-round time_s (cumulative training time, as in the log table),
  Total Upload           = cumulative upload MB (cum_up_MB),
  Total Download         = cumulative download MB (cum_down_MB).
The per-round table has no test_loss column (unlike IC's own npz), so that
column is "n/a", as in PMFH's and HybridSketchMFedMC_v2 (B3)'s summarizers.

Called automatically at the end of run_mhealth_mfedmc_sketchfusion_protocol_v2.sh
(when NUM_SEEDS/SEEDS is set); can also be run by hand, e.g.
  MFedMC/.venv/bin/python MFedMC/mHealth_v2/summarize_seeds.py --seeds 42 43 44 45 46
"""
import argparse
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "MFedMC_mHealth_v2"
METHOD_LABEL = "MFedMC v2 (B2)"
# Total modality count per dataset dir name, for the "#Modalities" column.
TOTAL_MODALITIES = {"mhealth_mm_random_group": 4, "mhealth_mm": 4}

# Matches the config portion of _txt_stem (main.py), from right after the
# wildcarded up_str through to "_seed"; up_str and seed are deliberately not
# captured here (see module docstring). No sketch rows/cols/k here (unlike
# B3) since B2 uploads uncompressed.
CONFIG_RX = re.compile(
    r"_dirichlet(?P<alpha>[^_]+)_it(?P<it>\d+)_"
    r"Top_(?P<top>\d+)_"
    r"Client_(?P<ratio>[0-9.]+)_"
    r"lr(?P<lr>[^_]+)_(?P<loss>highloss|lowloss)_"
    r"clients(?P<clients>\d+)_workers(?P<workers>\d+)_seed"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", required=True)
    p.add_argument("--dataset", default="mhealth_mm_random_group", help="dataset dir name (for #Modalities lookup)")
    p.add_argument("--rounds", type=int, default=None, help="only runs with this many rounds (--iterations)")
    p.add_argument("--since", type=float, default=0.0, help="only logs written after this unix time")
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    return p.parse_args()


def find_log(args, seed):
    r = args.rounds if args.rounds else "*"
    pat = os.path.join(args.log_dir, f"{PREFIX}_mm_*MB_dirichlet*_it{r}_*_seed{seed}.txt")
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
            top=int(m["top"]), acc=100 * row["test_acc"], time=row["time_s"],
            up=row["cum_up_MB"], down=row["cum_down_MB"],
        ))
    if not runs:
        raise SystemExit(f"No {PREFIX} logs found for seeds {' '.join(args.seeds)}")
    configs = {tuple(sorted(r["config"].items())) for r in runs}
    if len(configs) > 1:
        raise SystemExit(f"Seeds come from different configs (dirichlet/it/Top/"
                         f"Client/lr/loss-pref/clients/workers): {sorted(configs)}; "
                         f"pass --rounds / --since to pick one")
    unfinished = [r["seed"] for r in runs if r["rounds"] != r["planned"]]
    if unfinished:
        raise SystemExit(f"Seed(s) {' '.join(unfinished)} stopped before the last round")
    rounds, k, n, top = runs[0]["rounds"], runs[0]["workers"], runs[0]["clients"], runs[0]["top"]
    cfg = runs[0]["config"]

    def mean(key):
        return float(np.mean([r[key] for r in runs]))

    def std(key):
        v = np.array([r[key] for r in runs], dtype=float)
        return float(v.std(ddof=1)) if len(v) > 1 else 0.0

    total_mods = TOTAL_MODALITIES.get(args.dataset)
    if top == 0:
        mods_str = "-"  # 0 = all modalities, no selection
    elif total_mods is None:
        mods_str = str(top)
    else:
        mods_str = f"{top}/{total_mods} ({round(100 * top / total_mods)}%)"

    head = ["Methods", "#Clients", "#Modalities", "Total Rounds",
            "Avg. Test Loss", "Avg. Test Accuracy (%)", "Avg. Time (s)",
            "Total Upload (MB)", "Total Download (MB)"]
    row = [METHOD_LABEL, f"{k}/{n} ({round(100 * k / n)}%)", mods_str,
           str(rounds), "n/a", f"{mean('acc'):.2f}", f"{mean('time'):.2f}",
           f"{mean('up'):.2f}", f"{mean('down'):.2f}"]

    used = [r["seed"] for r in runs]
    lines = [f"{METHOD_LABEL} ({args.dataset}), mean over {len(runs)} seed(s): {' '.join(used)}", "",
             "Per seed (final round):",
             f"{'seed':>6} {'test acc %':>11} {'time s':>9} {'upload MB':>10} {'download MB':>12}  file"]
    for r in runs:
        lines.append(f"{r['seed']:>6} {r['acc']:>11.2f} {r['time']:>9.2f} {r['up']:>10.2f} {r['down']:>12.2f}  {r['file']}")
    lines += ["", "Results-sheet row (mean over seeds):"]
    widths = [max(len(h), len(v)) for h, v in zip(head, row)]
    lines.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    lines.append("  ".join(v.ljust(w) for v, w in zip(row, widths)))
    lines += ["", f"std over seeds: test acc {std('acc'):.2f}, time {std('time'):.2f}, "
                  f"upload {std('up'):.4f}, download {std('down'):.4f}",
              "", "Tab-separated, paste into the sheet:", "\t".join(row)]
    if missing:
        lines += ["", f"No finished log for seed(s): {' '.join(missing)} (not in the mean)"]
    text = "\n".join(lines)
    print("\n" + text)

    out = os.path.join(args.log_dir, f"{PREFIX}_mm_{mean('up'):.2f}MB_dirichlet{cfg['alpha']}_"
                                     f"it{rounds}_Top_{top}_Client_{cfg['ratio']}_lr{cfg['lr']}_"
                                     f"{cfg['loss']}_clients{n}_workers{k}_"
                                     f"seeds{'_'.join(used)}_summary.txt")
    os.makedirs(args.log_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
