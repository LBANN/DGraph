"""Visualization — Predicted vs. Measured Scatter (Headline Figure).

Reads ``data/predictions.json`` and produces a predicted-vs-measured scatter
plot.  Points are colored by graph type (or fit/held-out split).

Usage::

    python -m visualization.plot_validation \\
        --predictions data/predictions.json \\
        --color-by    split \\
        --output      figures/validation
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "text.usetex": False,
})


def parse_args():
    p = argparse.ArgumentParser(description="Plot predicted vs. measured scatter")
    p.add_argument("--predictions", type=str, required=True)
    p.add_argument("--color-by", choices=["split", "graph", "world_size"],
                   default="split")
    p.add_argument("--output", type=str, default="figures/validation")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.predictions) as f:
        data = json.load(f)

    entries = data["predictions"]
    mape_fit  = data["aggregate"]["mape_fit_set"]
    mape_held = data["aggregate"]["mape_held_out"]

    meas  = np.array([e["measured_median_seconds"] * 1e3 for e in entries])
    pred  = np.array([e["predicted_seconds"] * 1e3        for e in entries])

    # Color groups
    if args.color_by == "split":
        groups = {
            "Fit set":     [i for i, e in enumerate(entries) if e["in_fit_set"]],
            "Held-out":    [i for i, e in enumerate(entries) if not e["in_fit_set"]],
        }
        palette = {"Fit set": "#1f77b4", "Held-out": "#d62728"}
    elif args.color_by == "graph":
        graph_types = sorted(set(e["config"].get("graph", "unknown") for e in entries))
        palette = dict(zip(graph_types, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]))
        groups = {g: [i for i, e in enumerate(entries)
                      if e["config"].get("graph", "unknown") == g]
                  for g in graph_types}
    else:  # world_size
        ws_vals = sorted(set(e["config"].get("world_size", 1) for e in entries))
        colors = plt.cm.viridis(np.linspace(0, 1, len(ws_vals)))
        palette = {f"K={w}": c for w, c in zip(ws_vals, colors)}
        groups = {f"K={w}": [i for i, e in enumerate(entries)
                             if e["config"].get("world_size", 1) == w]
                  for w in ws_vals}

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    all_vals = np.concatenate([meas, pred])
    lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="Ideal (y=x)")

    # 10% error bands
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                    alpha=0.08, color="gray")

    for label, idxs in groups.items():
        if not idxs:
            continue
        color = palette[label]
        ax.scatter(meas[idxs], pred[idxs], s=22, color=color,
                   alpha=0.85, label=label, zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured $T_{\\mathrm{layer}}$ (ms)")
    ax.set_ylabel("Predicted $T_{\\mathrm{layer}}$ (ms)")
    ax.set_title("Cost Model Validation: Predicted vs. Measured", fontsize=9)

    # Annotate MAPE
    mape_text = (
        f"Fit MAPE = {mape_fit*100:.1f}%\n"
        f"Held-out MAPE = {mape_held*100:.1f}%"
    )
    ax.text(0.04, 0.96, mape_text, transform=ax.transAxes,
            verticalalignment="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.4)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
    print(f"[plot_validation] Saved {out}.pdf and {out}.png")
    print(
        "Caption: Predicted vs. measured layer time $T_{\\mathrm{layer}}$ for all "
        "benchmarked configurations. Dashed diagonal: perfect prediction. "
        "Shaded band: $\\pm10\\%$ error region. "
        f"In-sample MAPE = {mape_fit*100:.1f}\\%, "
        f"held-out MAPE = {mape_held*100:.1f}\\%. "
        "Colors distinguish "
        + ("fit-set vs. held-out configurations." if args.color_by == "split"
           else f"configurations by {args.color_by}.")
    )


if __name__ == "__main__":
    main()
