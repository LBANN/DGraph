"""Visualization — Ping-Pong Bandwidth / Latency.

Produces a log-log plot of one-way transfer time vs. message size for intra-
and inter-node measurements, with fitted lines overlaid and a residuals inset.

Usage::

    python -m visualization.plot_pingpong \\
        --intra  data/pingpong_intra_*.json \\
        --inter  data/pingpong_inter_*.json \\
        --primitives data/fitted_primitives.json \\
        --output figures/pingpong
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
COLORS = {"intra": "#1f77b4", "inter": "#d62728"}
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "text.usetex": False,
})


def load_measurements(files: list) -> tuple:
    """Return (byte_sizes, medians, q25, q75) from a list of JSON files."""
    byte_sizes, medians, q25s, q75s = [], [], [], []
    for p in files:
        with open(p) as f:
            data = json.load(f)
        for meas in data["measurements"]:
            trials = np.array(meas["trials_seconds"])
            byte_sizes.append(meas["params"]["message_bytes"])
            medians.append(float(np.median(trials)))
            q25s.append(float(np.percentile(trials, 25)))
            q75s.append(float(np.percentile(trials, 75)))
    order = np.argsort(byte_sizes)
    return (
        np.array(byte_sizes)[order],
        np.array(medians)[order],
        np.array(q25s)[order],
        np.array(q75s)[order],
    )


def fitted_line(bytes_arr: np.ndarray, primitives: dict, mode: str) -> np.ndarray:
    net = primitives.get("network", {}).get(mode, None)
    if net is None:
        return None
    t_L = net["latency_seconds"]
    B = net["bandwidth_bytes_per_sec"]
    return t_L + bytes_arr / B


def parse_args():
    p = argparse.ArgumentParser(description="Plot ping-pong results")
    p.add_argument("--intra", nargs="+", default=[], metavar="FILE")
    p.add_argument("--inter", nargs="+", default=[], metavar="FILE")
    p.add_argument("--primitives", type=str, default=None)
    p.add_argument("--output", type=str, default="figures/pingpong")
    return p.parse_args()


def main():
    args = parse_args()

    primitives = {}
    if args.primitives:
        with open(args.primitives) as f:
            primitives = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    ax_main, ax_res = axes

    for mode, files in [("intra", args.intra), ("inter", args.inter)]:
        if not files:
            continue
        xb, med, q25, q75 = load_measurements(files)
        color = COLORS[mode]
        label = "NVLink (intra)" if mode == "intra" else "InfiniBand (inter)"

        yerr_lo = med - q25
        yerr_hi = q75 - med
        ax_main.errorbar(
            xb * 1e-6, med * 1e3,
            yerr=[yerr_lo * 1e3, yerr_hi * 1e3],
            fmt="o", markersize=3, color=color, label=label,
            capsize=2, linewidth=0.8, elinewidth=0.8,
        )

        fit = fitted_line(xb, primitives, mode)
        if fit is not None:
            ax_main.plot(xb * 1e-6, fit * 1e3, "--", color=color,
                         linewidth=1.2, label=f"{label} fit")
            # Residuals
            residuals = (med - fit) / med * 100  # percent
            ax_res.plot(xb * 1e-6, residuals, "o-", markersize=3, color=color,
                        linewidth=0.8, label=label)

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_xlabel("Message size (MB)")
    ax_main.set_ylabel("One-way transfer time (ms)")
    ax_main.legend(loc="upper left")
    ax_main.grid(True, which="both", linestyle=":", linewidth=0.4)

    ax_res.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax_res.set_xscale("log")
    ax_res.set_xlabel("Message size (MB)")
    ax_res.set_ylabel("Residual (%)")
    ax_res.legend()
    ax_res.grid(True, which="both", linestyle=":", linewidth=0.4)

    fig.suptitle("Network Ping-Pong: Transfer Time vs. Message Size", fontsize=10)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
    print(f"[plot_pingpong] Saved {out}.pdf and {out}.png")
    print(
        "Caption: Log-log plot of one-way network transfer time vs. message size "
        "for intra-node (NVLink) and inter-node (InfiniBand) communication. "
        "Points show medians; error bars span the 25th--75th percentile (IQR). "
        "Dashed lines show linear-latency fits $T = t_L + s/B$. "
        "Right panel shows residuals (\\%) from the fit."
    )


if __name__ == "__main__":
    main()
