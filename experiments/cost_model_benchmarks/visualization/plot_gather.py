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
    bandwidth_bytes_per_sec,
    launch_overhead_seconds,
):
    """Sample the fitted Hockney gather curve over a range of k values.

    Mirrors ``time_model`` in ``analysis/fit_primitives.py::fit_gather``:
    ``T(bytes) = launch_overhead + bytes / B_gather``. Returns (k, T_ms).
    """
    # Log-spaced, since the k sweep itself is log-spaced and the plot's x
    # axis is logarithmic — linspace would leave the low decades unsampled
    # and render the curve as a straight chord across them.
    x = np.logspace(np.log10(min_val), np.log10(max_val), 200)
    y = (launch_overhead_seconds + x * feature_dim * 4.0 / bandwidth_bytes_per_sec) * 1e3
    return x, y


def load_gather_file(paths: list, timing_key: str):
    """Merge multiple JSON files, return sorted (k, median, q25, q75) arrays
    plus the feature_dim the runs were measured at.

    feature_dim must come from the data, not a hardcoded constant: it sets the
    bytes-per-index (``k * feature_dim * 4``) used to evaluate the fitted
    curve, so a wrong value tilts the overlay by exactly that ratio.
    """
    rows = []
    feature_dims = set()
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        feature_dims.add(data["config"]["feature_dim"])
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
    if len(feature_dims) > 1:
        raise ValueError(
            f"Merged gather files have mismatched feature_dim values "
            f"{sorted(feature_dims)}; the bytes-per-index conversion is only "
            "well-defined for one. Plot them separately."
        )
    return k_arr, med_arr, q25_arr, q75_arr, feature_dims.pop()


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
        k, med, q25, q75, feature_dim = load_gather_file(files, timing_key)
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
            fit = primitives["gather"][dist_name][args.operation]
            x, y = fitted_gather(
                min_k,
                max_k,
                feature_dim=feature_dim,
                bandwidth_bytes_per_sec=fit["bandwidth_bytes_per_sec"],
                launch_overhead_seconds=fit["launch_overhead_seconds"],
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
