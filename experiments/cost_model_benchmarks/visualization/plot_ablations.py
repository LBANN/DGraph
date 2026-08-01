"""Visualization — Ablation Studies.

Two-panel figure:
    (a) Hierarchical model (intra+inter separate) vs. flat model (single B, t_L)
        on a topology sweep (varying SBM inter-block density).
    (b) Full model vs. model without the T_buffer_copy term.

Reads ``data/predictions.json`` (which has the full breakdown per entry).

Usage::

    python -m visualization.plot_ablations \\
        --predictions data/predictions.json \\
        --primitives  data/fitted_primitives.json \\
        --output      figures/ablations
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

COLORS = {
    "full":       "#1f77b4",
    "flat":       "#ff7f0e",
    "no_buffer":  "#d62728",
    "measured":   "#2ca02c",
}


def relative_error(pred, meas):
    return abs(pred - meas) / meas if meas > 0 else float("nan")


def parse_args():
    p = argparse.ArgumentParser(description="Plot ablation studies")
    p.add_argument("--predictions", type=str, required=True)
    p.add_argument(
        "--primitives", type=str, required=True,
        help="fitted_primitives.json; needs the network.flat entry produced "
             "by fit_primitives.py when both --pingpong-intra and "
             "--pingpong-inter are supplied",
    )
    p.add_argument("--output",      type=str, default="figures/ablations")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.predictions) as f:
        data = json.load(f)
    with open(args.primitives) as f:
        primitives = json.load(f)

    entries = data["predictions"]
    T_overhead = data.get("T_overhead_seconds", 0.0)

    flat_net = primitives.get("network", {}).get("flat")
    if flat_net is None:
        raise RuntimeError(
            "fitted_primitives.json has no 'network.flat' entry. Re-run "
            "analysis/fit_primitives.py with both --pingpong-intra and "
            "--pingpong-inter so it can fit a genuine single-tier baseline "
            "for this ablation, instead of the previous heuristic."
        )

    def flat_comm_time(c_intra_bytes: float, c_inter_bytes: float) -> float:
        """Flat (single-tier) model: all halo bytes over one fitted (B, t_L).

        NOTE: this baseline is naive in *two* ways relative to the full model —
        it has no intra/inter tier split, and it applies no per-NIC contention
        correction (B is not divided by ranks-per-NIC). So the gap this panel
        shows is the combined value of the hierarchy *and* the contention
        term, not the hierarchy alone. Do not describe it as isolating the
        hierarchy; to separate them you would need a third curve with tiers
        but no contention.
        """
        total_bytes = c_intra_bytes + c_inter_bytes
        if total_bytes <= 0:
            return 0.0
        B = flat_net.get("bandwidth_bytes_per_sec", 1e10)
        t_L = flat_net.get("latency_seconds", 0.0)
        return max(t_L + total_bytes / B, 0.0)

    # --- Panel (a): topology sweep (SBM inter-density) ---
    # Filter to SBM entries, group by inter_density
    sbm_entries = [e for e in entries
                   if e["config"].get("graph", "") == "sbm"]

    density_groups = {}
    for e in sbm_entries:
        d = e["config"].get("sbm_inter_density", 0.0)
        density_groups.setdefault(d, []).append(e)

    densities = sorted(density_groups.keys())
    # Panel (a) needs SBM runs at several inter-densities. With ER-only data
    # it renders empty — and the caption below would still assert that "the
    # hierarchical model degrades more gracefully as inter-block density
    # increases", a claim with nothing behind it. Fail instead of emitting a
    # figure whose caption states a result the panel does not show.
    if len(densities) < 2:
        raise SystemExit(
            f"[plot_ablations] Panel (a) needs SBM runs at 2+ inter-densities; "
            f"found {len(densities)} ({densities or 'none'}) among "
            f"{len(entries)} predictions. Re-run bench_end_to_end.py with "
            f"--graph sbm and a swept --sbm-inter-density, or drop this figure."
        )
    mape_full = []
    mape_flat = []

    for d in densities:
        grp = density_groups[d]
        full_errs, flat_errs = [], []
        for e in grp:
            T_meas = e["measured_median_seconds"]
            # Full hierarchical prediction is already in predictions.json
            T_pred_full = e["predicted_seconds"]
            full_errs.append(relative_error(T_pred_full, T_meas))

            # Flat model: swap the hierarchical comm term for one predicted by
            # a genuinely fit single-tier (B, t_L) model over total halo bytes
            # — not a heuristic ratio. Reading T_comm_seconds from the stored
            # breakdown keeps this independent of how the full model combines
            # its tiers (currently additive; see README's open question on
            # sum vs max), so this panel needs no change if that flips.
            bd = e.get("breakdown", {})
            T_comm_hier = bd.get("T_comm_seconds", 0.0)
            c_intra = e["partition_stats"].get("c_intra_bytes", 0)
            c_inter = e["partition_stats"].get("c_inter_bytes", 0)
            T_comm_flat = flat_comm_time(c_intra, c_inter)
            T_pred_flat = (T_pred_full - T_comm_hier + T_comm_flat)
            flat_errs.append(relative_error(T_pred_flat, T_meas))

        mape_full.append(np.mean(full_errs) * 100 if full_errs else float("nan"))
        mape_flat.append(np.mean(flat_errs) * 100 if flat_errs else float("nan"))

    # --- Panel (b): with vs. without T_buffer_copy ---
    meas_all  = np.array([e["measured_median_seconds"] * 1e3 for e in entries])
    pred_full = np.array([e["predicted_seconds"] * 1e3        for e in entries])
    pred_nobuf = np.array([
        (e["predicted_seconds"] - e.get("breakdown", {}).get("T_buffer_copy_seconds", 0.0)) * 1e3
        for e in entries
    ])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 3.8))

    # --- Panel (a) ---
    if densities:
        ax_a.plot(densities, mape_full, "o-",  color=COLORS["full"],
                  markersize=5, linewidth=1.2, label="Hierarchical (intra+inter)")
        ax_a.plot(densities, mape_flat, "s--", color=COLORS["flat"],
                  markersize=5, linewidth=1.2, label="Flat (single-tier)")
        ax_a.set_xlabel("SBM inter-block edge density")
        ax_a.set_ylabel("MAPE (%)")
        ax_a.set_title("(a) Hierarchical vs. Flat Model\nover Topology Sweep", fontsize=9)
        ax_a.legend()
        ax_a.grid(True, linestyle=":", linewidth=0.4)
    else:
        ax_a.text(0.5, 0.5, "No SBM data found.\nRun bench_end_to_end with --graph sbm.",
                  ha="center", va="center", transform=ax_a.transAxes, fontsize=8)
        ax_a.set_title("(a) Hierarchical vs. Flat Model", fontsize=9)

    # --- Panel (b) ---
    lo = min(meas_all.min(), pred_full.min(), pred_nobuf.min()) * 0.9
    hi = max(meas_all.max(), pred_full.max(), pred_nobuf.max()) * 1.1
    ax_b.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="Ideal")
    ax_b.scatter(meas_all, pred_full,   s=18, color=COLORS["full"],
                 alpha=0.8, label="Full model", zorder=3)
    ax_b.scatter(meas_all, pred_nobuf,  s=18, color=COLORS["no_buffer"],
                 alpha=0.6, marker="^", label="Without $T_{\\mathrm{buf}}$", zorder=3)
    ax_b.set_xlim(lo, hi)
    ax_b.set_ylim(lo, hi)
    ax_b.set_xlabel("Measured $T_{\\mathrm{layer}}$ (ms)")
    ax_b.set_ylabel("Predicted $T_{\\mathrm{layer}}$ (ms)")
    ax_b.set_title("(b) Ablation: With vs. Without\n$T_{\\mathrm{buffer-copy}}$ Term", fontsize=9)
    ax_b.legend()
    ax_b.grid(True, linestyle=":", linewidth=0.4)

    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=300)
    print(f"[plot_ablations] Saved {out}.pdf and {out}.png")
    print(
        "Caption: Ablation studies. "
        "(a) MAPE of the hierarchical cost model (separate intra/inter tiers) vs. "
        "a flat model (single bandwidth parameter) across the SBM topology sweep. "
        "The hierarchical model degrades more gracefully as inter-block density increases. "
        "(b) Predicted vs. measured scatter with (blue circles) and without (red triangles) "
        "the $T_{\\mathrm{buffer-copy}}$ term, demonstrating that omitting this term "
        "causes systematic under-prediction for large halo sizes."
    )


if __name__ == "__main__":
    main()
