"""Visualization — T_global vs. K (Tipping Point).

Shows total training throughput or per-layer time as a function of the number
of GPUs K for a fixed graph.  Overlays the cost-model prediction (solid) and
measured values (dashed with markers).  Annotates K* — the point beyond which
adding more GPUs yields diminishing returns.

T_global(K) is computed as:

    T_global(K) = T_layer(K) * num_layers * num_epochs

For the tipping-point annotation K* is the largest K where the speedup
relative to K=1 is still within 10% of linear.

Usage::

    python -m visualization.plot_tipping_point \\
        --predictions data/predictions.json \\
        --num-layers 3 \\
        --num-epochs 100 \\
        --graph erdos_renyi \\
        --output figures/tipping_point
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
    p = argparse.ArgumentParser(description="Plot T_global vs. K tipping point")
    p.add_argument("--predictions", type=str, required=True)
    p.add_argument("--num-layers",  type=int, default=3)
    p.add_argument("--num-epochs",  type=int, default=100)
    p.add_argument("--graph",       type=str, default=None,
                   help="Filter to this graph type (optional)")
    p.add_argument("--model",       type=str, default=None,
                   help="Restrict to one layer type (gcn | edge | gcn_spmm). "
                        "Required when predictions.json pools several, since a "
                        "scaling curve is per-configuration.")
    p.add_argument("--num-vertices", type=int, default=None,
                   help="Restrict to one graph size, for the same reason.")
    p.add_argument("--output",      type=str, default="figures/tipping_point")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.predictions) as f:
        data = json.load(f)

    entries = data["predictions"]

    # Filter by graph type if requested
    if args.graph:
        entries = [e for e in entries
                   if e["config"].get("graph", "") == args.graph]

    if args.model:
        entries = [e for e in entries if e["config"].get("model", "") == args.model]
    if args.num_vertices:
        entries = [e for e in entries
                   if e["config"].get("num_vertices", 0) == args.num_vertices]

    # Group by world_size
    ws_to_meas  = {}
    ws_to_pred  = {}
    ws_to_cfg   = {}
    for e in entries:
        K = e["config"].get("world_size", 1)
        ws_to_meas.setdefault(K, []).append(e["measured_median_seconds"])
        ws_to_pred.setdefault(K, []).append(e["predicted_seconds"])
        c = e["config"]
        ws_to_cfg.setdefault(K, []).append(
            (c.get("model"), c.get("num_vertices"), c.get("graph"),
             c.get("feature_dim"), c.get("partitioner"))
        )

    if not ws_to_meas:
        print("[plot_tipping_point] No data found. Exiting.")
        return

    # A scaling curve is only meaningful for ONE configuration measured at
    # several K. Previously this grouped by world_size alone and took a median
    # over whatever else was in predictions.json — which silently averaged
    # across N (1 ms and 49 ms points at the same K), and, once the drivers
    # began sweeping E2E_MODELS, across layer types too. Refuse instead: the
    # median of gcn and gcn_spmm is not a runtime of anything.
    ambiguous = {K: sorted(set(cfgs)) for K, cfgs in ws_to_cfg.items()
                 if len(set(cfgs)) > 1}
    if ambiguous:
        K0 = sorted(ambiguous)[0]
        varying = [
            name for i, name in enumerate(
                ["model", "num_vertices", "graph", "feature_dim", "partitioner"])
            if len({c[i] for c in ambiguous[K0]}) > 1
        ]
        raise ValueError(
            f"predictions.json holds {len(ambiguous[K0])} different "
            f"configurations at world_size={K0}, differing in "
            f"{', '.join(varying)}. A tipping-point curve needs a single "
            f"configuration swept over K. Narrow it with --model / "
            f"--num-vertices / --graph. Configurations at K={K0}: "
            f"{ambiguous[K0]}"
        )

    K_vals   = sorted(ws_to_meas.keys())
    if K_vals[0] != 1:
        raise ValueError(
            f"No world_size=1 run found (smallest K in data is {K_vals[0]}). "
            "Ideal-scaling and scaling-efficiency numbers are only meaningful "
            "relative to a true single-GPU baseline — run bench_end_to_end.py "
            "(or bench_crossover.py's --no-dist mode) at world_size=1 for this "
            "graph/config before plotting the tipping point."
        )
    T_meas   = np.array([np.median(ws_to_meas[K]) for K in K_vals]) * args.num_layers * args.num_epochs
    T_pred   = np.array([np.median(ws_to_pred[K]) for K in K_vals]) * args.num_layers * args.num_epochs
    K_arr    = np.array(K_vals, dtype=float)

    # Ideal linear scaling from K=1
    T_single = T_meas[0]  # K=1 reference
    T_ideal  = T_single / K_arr

    # Identify K*: largest K where speedup is ≥ 90% of ideal
    speedup      = T_single / T_meas
    ideal_speedup = K_arr
    efficiency   = speedup / ideal_speedup
    kstar_idx    = np.where(efficiency >= 0.9)[0]
    K_star       = K_arr[kstar_idx[-1]] if len(kstar_idx) else K_arr[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # --- Left panel: T_global vs K ---
    ax1.plot(K_arr, T_ideal, "k:", linewidth=1, label="Ideal linear scaling")
    ax1.plot(K_arr, T_pred,  "-",  color="#1f77b4", linewidth=1.5, label="Predicted")
    ax1.plot(K_arr, T_meas,  "o--", color="#d62728", markersize=5, linewidth=1.2,
             label="Measured")
    ax1.axvline(K_star, color="gray", linestyle="--", linewidth=0.8)
    ax1.text(K_star * 1.05, ax1.get_ylim()[1] * 0.95,
             f"$K^* = {int(K_star)}$", fontsize=8, color="gray", va="top")
    ax1.set_xlabel("Number of GPUs ($K$)")
    ax1.set_ylabel(f"$T_{{\\mathrm{{global}}}}$ (s)\n"
                   f"({args.num_layers} layers × {args.num_epochs} epochs)")
    ax1.set_title("Training Time vs. GPU Count", fontsize=9)
    ax1.legend()
    ax1.grid(True, linestyle=":", linewidth=0.4)

    # --- Right panel: scaling efficiency ---
    ax2.plot(K_arr, efficiency * 100, "o-", color="#2ca02c", markersize=5, linewidth=1.2,
             label="Scaling efficiency")
    ax2.axhline(90, color="gray", linestyle="--", linewidth=0.8, label="90% threshold")
    ax2.axvline(K_star, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Number of GPUs ($K$)")
    ax2.set_ylabel("Scaling efficiency (%)")
    ax2.set_title("Strong Scaling Efficiency", fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(True, linestyle=":", linewidth=0.4)

    fig.suptitle("Tipping-Point Analysis: When Does Adding GPUs Stop Helping?", fontsize=10)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
    print(f"[plot_tipping_point] Saved {out}.pdf and {out}.png  K*={int(K_star)}")
    print(
        f"Caption: (Left) Total training time $T_{{\\mathrm{{global}}}}$ vs. GPU count $K$ "
        f"for {args.num_layers}-layer GNN trained for {args.num_epochs} epochs. "
        "Solid blue: cost-model prediction. Red dashed: measured. "
        "Dotted black: ideal linear speedup. "
        f"$K^* = {int(K_star)}$ marks the last GPU count with $\\geq 90\\%$ scaling efficiency. "
        "(Right) Scaling efficiency $= \\text{{speedup}} / K$."
    )


if __name__ == "__main__":
    main()
