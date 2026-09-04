# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
"""
Paper-quality figure: DGraph GCN scaling and compute/communication breakdown
on an OGB dataset, from experiments/OGB/benchmark_results/*_timing_report.json.

Usage:
    python plot_scaling_breakdown.py --dataset=products \
        --node_count=2449029 --edge_count=61859140
    python plot_scaling_breakdown.py --dataset=papers100M \
        --node_count=111059956 --edge_count=1615685872 \
        --world_sizes="[2,4]"
"""

import json
import os
from typing import Optional, Sequence

import fire
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style: serif fonts, colorblind-safe palette (Okabe & Ito, 2008), editable
# text in vector output (fonttype 42 embeds real glyphs, not paths).
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": "#333333",
        "text.color": "#1a1a1a",
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#1a1a1a",
        "ytick.color": "#1a1a1a",
    }
)

COMPUTE_COLOR = "#0072B2"  # blue
COMM_COLOR = "#E69F00"  # orange
IDEAL_COLOR = "#999999"  # neutral gray
ACTUAL_COLOR = "#009E73"  # bluish green
OOM_COLOR = "#D55E00"  # vermillion

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "benchmark_results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

DATASET_DISPLAY_NAME = {
    "arxiv": "OGB-Arxiv",
    "products": "OGB-Products",
    "papers100M": "OGB-Papers100M",
}


def compute_comm_totals(report):
    compute = sum(v for k, v in report.items() if k.startswith("process-"))
    comm = sum(v for k, v in report.items() if k.startswith("feature-exchange-"))
    return compute, comm


def human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main(
    dataset: str = "products",
    node_count: Optional[int] = None,
    edge_count: Optional[int] = None,
    world_sizes: Sequence[int] = (1, 2, 4),
    out_name: Optional[str] = None,
):
    os.makedirs(FIG_DIR, exist_ok=True)
    world_sizes = list(world_sizes)

    reports = {}
    missing = []
    for ws in world_sizes:
        path = os.path.join(RESULTS_DIR, f"ogbn_{dataset}-{ws}_timing_report.json")
        if os.path.exists(path):
            with open(path) as f:
                reports[ws] = json.load(f)
        else:
            missing.append(ws)

    if not reports:
        raise FileNotFoundError(
            f"No timing reports found for dataset={dataset!r} in {RESULTS_DIR}"
        )

    present_ws = [ws for ws in world_sizes if ws in reports]
    baseline_ws = present_ws[0]

    total_time = np.array([reports[ws]["total-training-time"] for ws in present_ws])
    compute_time = np.array(
        [compute_comm_totals(reports[ws])[0] for ws in present_ws]
    )
    comm_time = np.array([compute_comm_totals(reports[ws])[1] for ws in present_ws])

    speedup = total_time[baseline_ws == np.array(present_ws)][0] / total_time
    ideal_speedup = np.array(present_ws, dtype=float) / baseline_ws

    fig, (ax_scale, ax_break) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # ---- Panel (a): scaling ------------------------------------------------
    # x-axis spans every requested world size (including missing/OOM ones) so
    # gaps in the sweep are visible rather than silently compressed away.
    all_x = np.arange(len(world_sizes))
    present_x = [world_sizes.index(ws) for ws in present_ws]

    ax_scale.plot(
        all_x,
        np.array(world_sizes, dtype=float) / baseline_ws,
        linestyle="--",
        color=IDEAL_COLOR,
        linewidth=1.6,
        label="Ideal Speedup",
        zorder=2,
    )
    ax_scale.plot(
        present_x,
        speedup,
        linestyle="-",
        color=ACTUAL_COLOR,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.0,
        linewidth=2.0,
        label="DGraph GCN",
        zorder=3,
    )
    for xi, s, t in zip(present_x, speedup, total_time):
        ax_scale.annotate(
            f"{s:.2f}$\\times$\n({t:,.0f} ms)",
            (xi, s),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
        )
    for ws in missing:
        xi = world_sizes.index(ws)
        ideal_y = ws / baseline_ws
        ax_scale.plot(
            xi, ideal_y, marker="x", color=OOM_COLOR, markersize=10, markeredgewidth=2.5, zorder=4
        )
        ax_scale.annotate(
            "OOM",
            (xi, ideal_y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=11,
            color=OOM_COLOR,
        )

    ax_scale.set_xticks(all_x)
    ax_scale.set_xticklabels([str(ws) for ws in world_sizes])
    ax_scale.set_xlabel("Number of GPUs")
    ax_scale.set_ylabel("Speedup")
    # ax_scale.set_title("(a) Training Time Scaling")
    y_top = max((world_sizes[-1] / baseline_ws), speedup.max() if len(speedup) else 1.0)
    ax_scale.set_ylim(0, y_top * 1.25)
    ax_scale.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax_scale.set_axisbelow(True)
    ax_scale.spines["top"].set_visible(False)
    ax_scale.spines["right"].set_visible(False)
    ax_scale.legend(frameon=False, loc="upper left")

    # ---- Panel (b): compute / communication breakdown ---------------------
    bar_width = 0.55
    ax_break.bar(
        present_x,
        compute_time,
        bar_width,
        label="Compute",
        color=COMPUTE_COLOR,
        zorder=3,
    )
    ax_break.bar(
        present_x,
        comm_time,
        bar_width,
        bottom=compute_time,
        label="Data Movement",
        color=COMM_COLOR,
        zorder=3,
    )

    fwd_total = compute_time + comm_time
    for xi, m, ft in zip(present_x, comm_time, fwd_total):
        comm_pct = 100.0 * m / ft
        ax_break.annotate(
            f"{comm_pct:.1f}%",
            (xi, ft),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=10.5,
        )
    for ws in missing:
        xi = world_sizes.index(ws)
        ax_break.annotate(
            "OOM",
            (xi, 0),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=11,
            color=OOM_COLOR,
        )

    ax_break.set_xticks(all_x)
    ax_break.set_xticklabels([str(ws) for ws in world_sizes])
    ax_break.set_xlabel("Number of GPUs")
    ax_break.set_ylabel("Forward Pass Time (ms)")
    # ax_break.set_title("(b) Compute / Communication Breakdown")
    ax_break.set_ylim(0, fwd_total.max() * 1.28 if len(fwd_total) else 1.0)
    ax_break.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax_break.set_axisbelow(True)
    ax_break.spines["top"].set_visible(False)
    ax_break.spines["right"].set_visible(False)
    ax_break.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.9,
        loc="upper right",
        fontsize=11,
        handlelength=1.4,
        borderpad=0.4,
    )

    display_name = DATASET_DISPLAY_NAME.get(dataset, f"OGB-{dataset.capitalize()}")
    size_bits = []
    if node_count is not None:
        size_bits.append(f"{human_count(node_count)} Nodes")
    if edge_count is not None:
        size_bits.append(f"{human_count(edge_count)} Edges")
    size_str = f" ({', '.join(size_bits)})" if size_bits else ""
    fig.suptitle(
        f"DGraph GCN on {display_name}{size_str}",
        fontsize=15,
        y=1.02,
    )
    fig.tight_layout()

    stem = out_name or f"{dataset}_scaling_breakdown"
    png_path = os.path.join(FIG_DIR, f"{stem}.png")
    pdf_path = os.path.join(FIG_DIR, f"{stem}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    fire.Fire(main)
