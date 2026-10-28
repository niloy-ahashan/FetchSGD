"""Overlay the seed-mean test-accuracy curves of every HybridSketchMFedMC v2
config in one folder (e.g. the 5 MB budget sweep) on a single plot, against
cumulative upload MB. Logs are found and grouped exactly as in
plot_seed_mean.py (every seed found per config unless --seeds is given).

Encoding (15 curves need two channels; a hue per curve would exceed the
8-slot categorical palette): colour = clients per round x Top-k modalities,
line style = sketch columns. The legend is sorted by final accuracy and
carries the value, and a .csv of the final values is written next to the PNG.

Example:
  venv/bin/python HybridSketchMFedMC_v2/plot_configs_overlay.py \
    --log_dir HybridSketchMFedMC_v2/result_updated/5MB
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_seed_mean import CONFIG_RX, HERE, find_configs, header, table

# Categorical slots 1-5 of the reference palette (validated: CVD and
# normal-vision separation pass in light mode), in fixed order.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Line styles by sketch-width rank within one plot (the sweeps span 2500-30000
# columns but each budget folder uses only three widths, and three distinct
# styles read better than seven fixed ones); the in-plot key names the widths.
STYLES = ["--", "-", ":", "-."]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log_dir", default=os.path.join(HERE, "result_updated"))
    p.add_argument("--seeds", nargs="+", default=None, help="default: every seed found per config")
    p.add_argument("--out", default=None,
                   help="PNG path (default: results/<log_dir name>/HybridV2_<dataset>_<log_dir name>_all_configs_mean.png)")
    # find_configs() filters; kept unset here so every config is drawn
    p.set_defaults(rounds=None, cols=None, top=None, client=None)
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = os.path.abspath(args.log_dir)
    sub = os.path.basename(log_dir)
    groups = find_configs(args)
    if not groups:
        raise SystemExit(f"No HybridV2 logs in {args.log_dir}")

    curves = []
    for stem, files in groups.items():
        seeds = list(args.seeds) if args.seeds else sorted(files, key=int)
        missing = [s for s in seeds if s not in files]
        if missing:
            raise SystemExit(f"{stem}: no log for seed(s) {' '.join(missing)}")
        tabs = [table(files[s]) for s in seeds]
        if len({len(t["iter"]) for t in tabs}) > 1:
            raise SystemExit(f"{stem}: seeds have different round counts")
        if not all(np.allclose(t["cum_up_MB"], tabs[0]["cum_up_MB"]) for t in tabs):
            raise SystemExit(f"{stem}: seeds have different cum_up_MB columns")
        acc = np.stack([100 * t["test_acc"] for t in tabs])
        c = CONFIG_RX.search(stem)
        hp, n_mod = header(files[seeds[0]])
        n_clients = int(hp["num_clients"])
        curves.append(dict(
            x=tabs[0]["cum_up_MB"], mean=acc.mean(axis=0),
            std=acc.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(acc.shape[1]),
            clients=max(1, round(float(c["client"]) * n_clients)), n_clients=n_clients,
            top=int(c["top"]), n_mod=n_mod, rows=int(c["rows"]), cols=int(c["cols"]),
            rounds=int(c["it"]), seeds=seeds, dataset=hp.get("dataset", "?")))

    # colour per (clients, top) group in a fixed order, never by rank
    keys = sorted({(cv["clients"], -cv["top"]) for cv in curves})
    if len(keys) > len(COLORS):
        raise SystemExit(f"{len(keys)} clients x Top groups; only {len(COLORS)} colours defined")
    color_of = {k: COLORS[i] for i, k in enumerate(keys)}
    curves.sort(key=lambda cv: -cv["mean"][-1])

    widths = sorted({cv["cols"] for cv in curves})
    if len(widths) > len(STYLES):
        raise SystemExit(f"{len(widths)} sketch widths; only {len(STYLES)} line styles defined")
    dash_of = dict(zip(widths, STYLES))

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    for cv in curves:
        ax.plot(cv["x"], cv["mean"], color=color_of[(cv["clients"], -cv["top"])],
                linestyle=dash_of[cv["cols"]], linewidth=1.6,
                label=f"{cv['mean'][-1]:5.2f}%  {cv['clients']}/{cv['n_clients']} · "
                      f"Top {cv['top']} · {cv['rows']}×{cv['cols']} · {cv['rounds']}"
                      + ("" if len(cv["seeds"]) == 5 else f" (n={len(cv['seeds'])})"))

    label = {"mhealth": "mHealth", "uci_har": "UCI HAR"}.get(curves[0]["dataset"], curves[0]["dataset"])
    ax.set_title(f"HybridSketchMFedMC v2 on {label}: all {len(curves)} configs at the {sub} budget "
                 f"(mean over seeds)", fontsize=10)
    ax.set_xlabel("Cumulative upload (MB)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, max(cv["x"][-1] for cv in curves) * 1.02)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    # Both legends sit inside the axes. Where the curves leave room differs per
    # budget (slow-rising sweeps leave the top-left empty, fast ones the
    # bottom-right), so each legend takes the candidate spot that covers the
    # fewest points of the (densely resampled) curves.
    dense = []
    for cv in curves:
        xs = np.linspace(cv["x"][0], cv["x"][-1], 400)
        dense.append(np.column_stack([xs, np.interp(xs, cv["x"], cv["mean"])]))
    dense = np.vstack(dense)

    def covered(leg):
        fig.canvas.draw()
        """(curve points under the legend, -clearance in px to the nearest curve point):
        lower is better, so among spots covering nothing the one with most room wins."""
        b, p = leg.get_window_extent(), ax.transData.transform(dense)
        dx = np.maximum(np.maximum(b.x0 - p[:, 0], p[:, 0] - b.x1), 0)
        dy = np.maximum(np.maximum(b.y0 - p[:, 1], p[:, 1] - b.y1), 0)
        return int(((dx == 0) & (dy == 0)).sum()), -float(np.hypot(dx, dy).min())

    def place(candidates, **kw):
        best = None
        for loc, ncol, title in candidates:
            leg = ax.legend(loc=loc, ncol=ncol, title=title, **kw)
            score = covered(leg)
            leg.remove()
            if best is None or score < best[0]:
                best = (score, loc, ncol, title)
        _, loc, ncol, title = best
        return ax.legend(loc=loc, ncol=ncol, title=title, **kw), loc

    legend_box = dict(frameon=True, facecolor="white", edgecolor="0.7", framealpha=0.9, fontsize=7.5)
    title_1col = "Final acc · clients/round · Top-k · sketch · rounds\n(n = seeds, if fewer than 5)"
    title_2col = "Final acc · clients/round · Top-k · sketch · rounds   (n = seeds, if fewer than 5)"
    leg1, loc1 = place([("upper left", 1, title_1col), ("lower right", 2, title_2col),
                  ("upper left", 2, title_2col), ("center right", 1, title_1col)],
                 title_fontsize=7.5, borderpad=0.5, labelspacing=0.3, handlelength=2.2, **legend_box)
    ax.add_artist(leg1)
    style_handles = [Line2D([], [], color="0.35", linestyle=dash_of[c], linewidth=1.6,
                            label=f"sketch 3×{c}") for c in widths]
    corners = [c for c in ("lower right", "upper left", "center right", "lower center") if c != loc1]
    place([(c, 1, "Line style") for c in corners], handles=style_handles, title_fontsize=8, **legend_box)

    out = args.out or os.path.join(HERE, "results", sub,
                                   f"HybridV2_{curves[0]['dataset']}_{sub}_all_configs_mean.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # leg1 lives outside the axes and is only an added artist, so name it for the tight bbox
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=[leg1])

    csv = os.path.splitext(out)[0] + ".csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("clients_per_round,top,rows,cols,rounds,seeds,final_cum_up_MB,final_mean_acc_pct,final_std_acc_pct\n")
        for cv in curves:
            f.write(f"{cv['clients']},{cv['top']},{cv['rows']},{cv['cols']},{cv['rounds']},"
                    f"{' '.join(cv['seeds'])},{cv['x'][-1]:.2f},{cv['mean'][-1]:.2f},{cv['std'][-1]:.2f}\n")
    print(f"{len(curves)} configs overlaid")
    print(f"Plot saved to {out}")
    print(f"Data saved to {csv}")


if __name__ == "__main__":
    main()
