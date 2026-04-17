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
    "clustered": "#2ca02c",
    "random": "#d62728",
}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "text.usetex": False,
    }
)


def fitted_gather(
    min_val,
    max_val,
    feature_dim,
    hbm_bandwidth_bytes_per_sec,
    l2_bandwidth_bytes_per_sec,
    launch_overhead_seconds,
    L2_thresh,
    HBM_thresh,
):
    """Return a function that models gather time as a function of k."""

    def time_model(b, overhead, inv_bw_L2, inv_bw_HBM, L2_thresh, HBM_thresh):
        # 1. Bucket the bytes into their respective physical regimes
        # Bytes processed exclusively at L2 speeds
        bytes_L2 = np.clip(b, 0, L2_thresh)
        # Bytes processed exclusively at HBM speeds
        bytes_HBM = np.maximum(0, b - HBM_thresh)

        # 2. Apply the specific bandwidth (slope) to each bucket
        t_mem = (bytes_L2 * inv_bw_L2) + (bytes_HBM * inv_bw_HBM)

        # 3. Floor the total time by the kernel launch overhead
        return np.maximum(overhead, t_mem)

    x = np.linspace(min_val, max_val, 100)
    inv_bw_L2 = 1.0 / l2_bandwidth_bytes_per_sec
    inv_bw_HBM = 1.0 / hbm_bandwidth_bytes_per_sec
    y = (
        time_model(
            x * feature_dim * 4.0,
            launch_overhead_seconds,
            inv_bw_L2,
            inv_bw_HBM,
            L2_thresh,
            HBM_thresh,
        )
        * 1e3
    )
    return x, y


def load_gather_file(paths: list, timing_key: str):
    """Merge multiple JSON files, return sorted (k, median, q25, q75) arrays."""
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for meas in data["measurements"]:
            trials = np.array(meas[timing_key])
            rows.append(
                (
                    meas["params"]["k"],
                    float(np.median(trials)),
                    float(np.percentile(trials, 25)),
                    float(np.percentile(trials, 75)),
                )
            )
    rows.sort(key=lambda r: r[0])
    k_arr = np.array([r[0] for r in rows])
    med_arr = np.array([r[1] for r in rows])
    q25_arr = np.array([r[2] for r in rows])
    q75_arr = np.array([r[3] for r in rows])
    return k_arr, med_arr, q25_arr, q75_arr


def parse_args():
    p = argparse.ArgumentParser(description="Plot gather/scatter-add bandwidth")
    p.add_argument("--contiguous", nargs="+", default=[], metavar="FILE")
    p.add_argument("--clustered", nargs="+", default=[], metavar="FILE")
    p.add_argument("--random", nargs="+", default=[], metavar="FILE")
    p.add_argument("--operation", choices=["gather", "scatter_add"], default="gather")
    p.add_argument("--fitted", type=str, default=None, metavar="FILE")
    p.add_argument("--output", type=str, default="figures/gather")
    return p.parse_args()


def main():
    args = parse_args()
    timing_key = (
        "gather_trials_seconds"
        if args.operation == "gather"
        else "scatter_add_trials_seconds"
    )

    fig, ax = plt.subplots(figsize=(5, 3.5))

    if args.fitted:
        with open(args.fitted) as f:
            primitives = json.load(f)
    min_k, max_k = 1e3, 1e9
    for dist_name, files in [
        ("contiguous", args.contiguous),
        ("clustered", args.clustered),
        ("random", args.random),
    ]:
        if not files:
            continue
        k, med, q25, q75 = load_gather_file(files, timing_key)
        min_k = k[0]
        max_k = k[-1]
        color = COLORS[dist_name]
        ax.errorbar(
            k * 1e-6,
            med * 1e3,
            yerr=[(med - q25) * 1e3, (q75 - med) * 1e3],
            fmt="o",
            markersize=3,
            color=color,
            linewidth=0.9,
            capsize=2,
            elinewidth=0.8,
            label=dist_name.capitalize(),
        )

        if args.fitted:
            hbm_bw = primitives["gather"][dist_name]["gather"][
                "bandwidth_bytes_per_sec"
            ]
            l2_bw = primitives["gather"][dist_name]["gather"][
                "L2_bandwidth_bytes_per_sec"
            ]
            overhead = primitives["gather"][dist_name]["gather"][
                "launch_overhead_seconds"
            ]
            thresh = primitives["gather"][dist_name]["gather"]["L2_inflection_bytes"]
            hbm_thresh = primitives["gather"][dist_name]["gather"][
                "HBM_inflection_bytes"
            ]
            x, y = fitted_gather(
                min_k,
                max_k,
                feature_dim=512,
                hbm_bandwidth_bytes_per_sec=hbm_bw,
                l2_bandwidth_bytes_per_sec=l2_bw,
                launch_overhead_seconds=overhead,
                L2_thresh=thresh,
                HBM_thresh=hbm_thresh,
            )
            ax.plot(
                x * 1e-6,
                y,
                "--",
                color=color,
                linewidth=1.2,
                label=f"{dist_name.capitalize()}-Expected",
                alpha=0.4,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    op_label = (
        "Gather $x[\\mathrm{idx}]$"
        if args.operation == "gather"
        else "Scatter-add (backward)"
    )
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
