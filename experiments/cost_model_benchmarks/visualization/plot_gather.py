"""Visualization — Gather / Scatter-Add Bandwidth.

Single plot with three curves (contiguous, clustered, random) showing
gather (or scatter-add) runtime vs. k (number of rows gathered).

Usage::

    python -m visualization.plot_gather \\
        --contiguous data/gather_contiguous_*.json \\
        --clustered  data/gather_clustered_*.json \\
        --random     data/gather_random_*.json \\
        --operation  gather \\
        --output     figures/gather
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "contiguous": "#1f77b4",
    "clustered":  "#2ca02c",
    "random":     "#d62728",
}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "text.usetex": False,
})


def load_gather_file(paths: list, timing_key: str):
    """Merge multiple JSON files, return sorted (k, median, q25, q75) arrays."""
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for meas in data["measurements"]:
            trials = np.array(meas[timing_key])
            rows.append((
                meas["params"]["k"],
                float(np.median(trials)),
                float(np.percentile(trials, 25)),
                float(np.percentile(trials, 75)),
            ))
    rows.sort(key=lambda r: r[0])
    k_arr   = np.array([r[0] for r in rows])
    med_arr = np.array([r[1] for r in rows])
    q25_arr = np.array([r[2] for r in rows])
    q75_arr = np.array([r[3] for r in rows])
    return k_arr, med_arr, q25_arr, q75_arr


def parse_args():
    p = argparse.ArgumentParser(description="Plot gather/scatter-add bandwidth")
    p.add_argument("--contiguous", nargs="+", default=[], metavar="FILE")
    p.add_argument("--clustered",  nargs="+", default=[], metavar="FILE")
    p.add_argument("--random",     nargs="+", default=[], metavar="FILE")
    p.add_argument("--operation",  choices=["gather", "scatter_add"], default="gather")
    p.add_argument("--output",     type=str, default="figures/gather")
    return p.parse_args()


def main():
    args = parse_args()
    timing_key = (
        "gather_trials_seconds" if args.operation == "gather"
        else "scatter_add_trials_seconds"
    )

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for dist_name, files in [
        ("contiguous", args.contiguous),
        ("clustered",  args.clustered),
        ("random",     args.random),
    ]:
        if not files:
            continue
        k, med, q25, q75 = load_gather_file(files, timing_key)
        color = COLORS[dist_name]
        ax.errorbar(
            k * 1e-6, med * 1e3,
            yerr=[(med - q25) * 1e3, (q75 - med) * 1e3],
            fmt="o-", markersize=3, color=color, linewidth=0.9,
            capsize=2, elinewidth=0.8, label=dist_name.capitalize(),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    op_label = "Gather $x[\\mathrm{idx}]$" if args.operation == "gather" \
        else "Scatter-add (backward)"
    ax.set_xlabel("k (millions of rows gathered)")
    ax.set_ylabel(f"{op_label} time (ms)")
    ax.set_title(f"Buffer-Copy Bandwidth: {op_label}", fontsize=9)
    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.4)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
    print(f"[plot_gather] Saved {out}.pdf and {out}.png")
    print(
        f"Caption: {op_label} time vs. gather size $k$ for three index "
        "distributions: contiguous (best case, cache-friendly), clustered "
        "(METIS-partitioned halo pattern), and random (worst case). "
        "Error bars span IQR. The gap between contiguous/clustered and random "
        "quantifies the cache-miss penalty relevant to poorly-partitioned graphs."
    )


if __name__ == "__main__":
    main()
