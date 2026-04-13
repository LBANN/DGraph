"""Visualization — GNN Compute Primitive Runtime.

Two-panel figure: GCN-like (left) vs. edge-conditioned (right).
Each panel shows forward runtime vs. the swept variable (vertices or edges)
with the fitted linear model overlaid.

Usage::

    python -m visualization.plot_compute \\
        --gcn-vertex  data/compute_gcn_vswp.json \\
        --gcn-edge    data/compute_gcn_eswp.json \\
        --edge-vertex data/compute_edge_vswp.json \\
        --edge-edge   data/compute_edge_eswp.json \\
        --primitives  data/fitted_primitives.json \\
        --output      figures/compute
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"gcn": "#2ca02c", "edge": "#ff7f0e"}
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


def load_compute_file(path: str, timing_key: str = "forward_trials_seconds"):
    """Returns lists of (sweep_value, median, q25, q75)."""
    with open(path) as f:
        data = json.load(f)
    sweep = data["config"]["sweep"]
    rows = []
    for meas in data["measurements"]:
        trials = np.array(meas[timing_key])
        rows.append(
            (
                meas["params"]["sweep_value"],
                meas["params"]["num_vertices"],
                meas["params"]["num_edges"],
                float(np.median(trials)),
                float(np.percentile(trials, 25)),
                float(np.percentile(trials, 75)),
            )
        )
    rows.sort(key=lambda r: r[0])
    return sweep, rows


def fitted_compute(sweep_vals, fixed_val, sweep, model_type, primitives):
    params = primitives.get("compute", {}).get(model_type, {}).get("forward", None)
    if params is None:
        return None
    a, b, c = params["coeff_V"], params["coeff_E"], params["intercept"]
    if sweep == "vertices":
        V_arr = np.array(sweep_vals, dtype=float)
        E_arr = np.full_like(V_arr, fixed_val)
    else:
        E_arr = np.array(sweep_vals, dtype=float)
        V_arr = np.full_like(E_arr, fixed_val)
    return a * V_arr + b * E_arr + c


def plot_one_panel(ax, rows, sweep, fixed_val, model_type, primitives, color, title):
    xvals = [r[0] for r in rows]
    meds = np.array([r[3] for r in rows]) * 1e3
    lo = np.array([r[3] - r[4] for r in rows]) * 1e3
    hi = np.array([r[5] - r[3] for r in rows]) * 1e3

    ax.errorbar(
        xvals,
        meds,
        yerr=[lo, hi],
        fmt="o",
        markersize=4,
        color=color,
        capsize=2,
        linewidth=0.8,
        elinewidth=0.8,
        label="Measured (IQR)",
    )

    fit = fitted_compute(xvals, fixed_val, sweep, model_type, primitives)
    if fit is not None:
        ax.plot(xvals, fit * 1e3, "--", color=color, linewidth=1.2, label="Fit")

    xlabel = "|V| (vertices)" if sweep == "vertices" else "|E| (edges)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Forward time (ms)")
    ax.set_title(title, fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()
    ax.grid(True, which="both", linestyle=":", linewidth=0.4)


def parse_args():
    p = argparse.ArgumentParser(description="Plot GNN compute primitive results")
    p.add_argument("--gcn-vertex", type=str, default=None)
    p.add_argument("--gcn-edge", type=str, default=None)
    p.add_argument("--edge-vertex", type=str, default=None)
    p.add_argument("--edge-edge", type=str, default=None)
    p.add_argument("--primitives", type=str, default=None)
    p.add_argument("--output", type=str, default="figures/compute")
    return p.parse_args()


def main():
    args = parse_args()
    primitives = {}
    if args.primitives:
        with open(args.primitives) as f:
            primitives = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    panel_map = [
        ("gcn", "edges", args.gcn_edge, axes[0], "GCN-like"),
        ("edge", "edges", args.edge_edge, axes[1], "Edge-conditioned"),
    ]

    for model_type, sweep_label, path, ax, title in panel_map:
        if path is None:
            ax.set_visible(False)
            continue
        sweep, rows = load_compute_file(path)
        fixed_val = rows[0][2] if sweep == "vertices" else rows[0][1]  # E or V fixed
        plot_one_panel(
            ax,
            rows,
            sweep,
            fixed_val,
            model_type,
            primitives,
            COLORS[model_type],
            title,
        )

    # fig.suptitle("GNN Compute Primitive: Forward Runtime vs. Graph Size", fontsize=10)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
    print(f"[plot_compute] Saved {out}.pdf and {out}.png")
    print(
        "Caption: Forward runtime of a single GNN layer vs. subgraph size "
        "(vertex sweep top row, edge sweep bottom row) for GCN-like (left) "
        "and edge-conditioned (right) message functions. "
        "Dashed lines: fitted model $T_{\\mathrm{comp}} = a|V| + b|E| + c$. "
        "Error bars span IQR over 50+ trials."
    )


if __name__ == "__main__":
    main()
