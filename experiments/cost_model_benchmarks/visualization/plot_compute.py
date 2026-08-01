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

from analysis import compute_forms

COLORS = {"gcn": "#2ca02c", "edge": "#ff7f0e", "gcn_spmm":"#613cd1"}
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
    """Returns (sweep, rows, feature_dim).

    feature_dim must come from the data: T_comp's fitted form is a function of
    F now, so evaluating the fit for the overlay needs the F this file was
    measured at. One file is always a single (model, F) cell.
    """
    with open(path) as f:
        data = json.load(f)
    sweep = data["config"]["sweep"]
    feature_dim = data["config"]["feature_dim"]
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
    return sweep, rows, feature_dim


def fitted_compute(sweep_vals, fixed_val, sweep, model_type, primitives, feature_dim):
    """Evaluate the fitted T_comp curve for the overlay.

    Delegates to analysis.compute_forms rather than unpacking coefficients by
    name: fit_compute chooses its design matrix per layer type and per whether
    F was swept, so the coefficient *names* vary ("coeff_V"/"coeff_E" only
    exist in the fixed-F legacy form; the F-dependent forms use
    "coeff_EF2"/"coeff_EF"/"coeff_VF"). Reading them directly here is what
    raised KeyError: 'coeff_V' once the F sweep made the form edge_centric.
    """
    params = primitives.get("compute", {}).get(model_type, {}).get("forward", None)
    if params is None:
        return None
    if sweep == "vertices":
        V_arr = np.array(sweep_vals, dtype=float)
        E_arr = np.full_like(V_arr, fixed_val)
    else:
        E_arr = np.array(sweep_vals, dtype=float)
        V_arr = np.full_like(E_arr, fixed_val)
    # evaluate() returns a scalar, so map it over the sweep.
    return np.array([
        compute_forms.evaluate(params, v, e, feature_dim)
        for v, e in zip(V_arr, E_arr)
    ])


def plot_one_panel(ax, rows, sweep, fixed_val, model_type, primitives, color, title,
                   feature_dim):
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

    fit = fitted_compute(xvals, fixed_val, sweep, model_type, primitives, feature_dim)
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
    p.add_argument("--spmm-vertex", type=str, default=None,
                   help="Vertex sweep for the SpMM GCN layer (--model gcn_spmm)")
    p.add_argument("--spmm-edge", type=str, default=None,
                   help="Edge sweep for the SpMM GCN layer (--model gcn_spmm)")
    p.add_argument("--primitives", type=str, default=None)
    p.add_argument("--output", type=str, default="figures/compute")
    return p.parse_args()


def main():
    args = parse_args()
    primitives = {}
    if args.primitives:
        with open(args.primitives) as f:
            primitives = json.load(f)

    # Columns = layer variant, rows = sweep direction. The previous version
    # built a 1x2 grid from the *edge* sweeps only, leaving --gcn-vertex and
    # --edge-vertex parsed but never plotted, while the caption described a
    # two-row layout that did not exist. Only columns with at least one file
    # are drawn, so passing a subset still produces a sensible figure.
    columns = [
        ("gcn", "GCN-like (per-edge)", args.gcn_vertex, args.gcn_edge),
        ("edge", "Edge-conditioned", args.edge_vertex, args.edge_edge),
        ("gcn_spmm", "GCN (SpMM)", args.spmm_vertex, args.spmm_edge),
    ]
    columns = [c for c in columns if c[2] or c[3]]
    if not columns:
        raise SystemExit(
            "[plot_compute] No input files given. Pass at least one of "
            "--gcn-vertex/--gcn-edge/--edge-vertex/--edge-edge/"
            "--spmm-vertex/--spmm-edge."
        )

    ncols = len(columns)
    fig, axes = plt.subplots(2, ncols, figsize=(3.5 * ncols, 6), squeeze=False)

    panel_map = []
    for col, (model_type, title, vpath, epath) in enumerate(columns):
        panel_map.append((model_type, vpath, axes[0][col], f"{title} — |V| sweep"))
        panel_map.append((model_type, epath, axes[1][col], f"{title} — |E| sweep"))

    for model_type, path, ax, title in panel_map:
        if path is None:
            ax.set_visible(False)
            continue
        sweep, rows, feature_dim = load_compute_file(path)
        fixed_val = rows[0][2] if sweep == "vertices" else rows[0][1]  # E or V fixed
        plot_one_panel(
            ax,
            rows,
            sweep,
            fixed_val,
            model_type,
            primitives,
            COLORS[model_type],
            f"{title} (F={feature_dim})",
            feature_dim,
        )

    # fig.suptitle("GNN Compute Primitive: Forward Runtime vs. Graph Size", fontsize=10)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
    print(f"[plot_compute] Saved {out}.pdf and {out}.png")
    forms = sorted({
        primitives.get("compute", {}).get(m, {}).get("forward", {}).get("form")
        for m, _, _, _ in columns
    } - {None})
    form_expr = {
        "legacy_VE": "$a|V| + b|E| + c$",
        "edge_centric": "$a|E|F^2 + b|E|F + c|V|F + d$",
        "vertex_centric": "$a|V|F^2 + b|E|F + c|V|F + d$",
    }
    # Older fitted_primitives.json files predate the "form" key; evaluate()
    # treats those as legacy_VE, so say so rather than emitting a bare phrase.
    fitted_desc = (
        " / ".join(form_expr.get(f, f) for f in forms)
        if forms
        else form_expr["legacy_VE"]
    )
    print(
        "Caption: Forward runtime of a single GNN layer vs. subgraph size "
        "(vertex sweep top row, edge sweep bottom row) for "
        + ", ".join(t for _, t, _, _ in columns)
        + " message functions. Dashed lines: fitted model "
        + fitted_desc
        + ". Error bars span IQR over 50+ trials."
    )


if __name__ == "__main__":
    main()
