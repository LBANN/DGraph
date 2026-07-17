import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from fire import Fire  # noqa: E402

# Fixed categorical color/marker assignment by world_size, matching the tab10-based
# convention already used in experiments/cost_model_benchmarks/visualization/plot_crossover.py.
# World_size=1 (single-GPU baseline) is always drawn the same way so it reads as the
# reference line across every plot.
WORLD_SIZE_STYLE = {
    1: {"color": "#1f77b4", "marker": "o", "label": "1 GPU (baseline)"},
    2: {"color": "#ff7f0e", "marker": "s", "label": "2 GPUs"},
    4: {"color": "#2ca02c", "marker": "^", "label": "4 GPUs"},
    8: {"color": "#9467bd", "marker": "D", "label": "8 GPUs"},
}
OOM_COLOR = "#d62728"


def load_results(paths) -> list:
    """Load and flatten measurements across multiple result JSON files, tagging
    each with its run config."""
    records = []
    for path in paths:
        with open(path) as f:
            payload = json.load(f)
        config = payload["config"]
        for m in payload["measurements"]:
            records.append({"config": config, "measurement": m})
    return records


def _style(world_size: int) -> dict:
    return WORLD_SIZE_STYLE.get(
        world_size,
        {"color": "#7f7f7f", "marker": "x", "label": f"{world_size} GPUs"},
    )


def _filter(records, mesh_level: int) -> list:
    return sorted(
        [r for r in records if r["config"]["mesh_level"] == mesh_level],
        key=lambda r: r["config"]["world_size"],
    )


def plot_latency_vs_world_size(records, mesh_level: int, output_path: str) -> None:
    rows = _filter(records, mesh_level)
    if not rows:
        return

    plt.figure(figsize=(8, 6), dpi=150)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    world_sizes, medians, p95s, oom_ws = [], [], [], []
    for r in rows:
        m = r["measurement"]
        ws = r["config"]["world_size"]
        if m.get("oom"):
            oom_ws.append(ws)
            continue
        world_sizes.append(ws)
        medians.append(m["end_to_end_latency_ms"]["p50"])
        p95s.append(m["end_to_end_latency_ms"]["p95"])

    if world_sizes:
        colors = [_style(ws)["color"] for ws in world_sizes]
        plt.errorbar(
            world_sizes,
            medians,
            yerr=[np.array(p95s) - np.array(medians), np.zeros(len(medians))],
            fmt="none",
            ecolor="#7f7f7f",
            alpha=0.5,
            capsize=4,
        )
        for ws, med in zip(world_sizes, medians):
            style = _style(ws)
            plt.scatter([ws], [med], color=style["color"], marker=style["marker"],
                        s=100, zorder=3, label=style["label"])
        plt.plot(world_sizes, medians, color="#7f7f7f", linestyle="-", linewidth=1,
                  alpha=0.6, zorder=2)

    for ws in oom_ws:
        plt.axvline(x=ws, color=OOM_COLOR, linestyle="--", alpha=0.6)
        plt.text(ws, plt.ylim()[1] * 0.9 if plt.ylim()[1] else 1, "OOM",
                  color=OOM_COLOR, rotation=90, va="top", ha="right")

    plt.title(f"GraphCast Inference Latency vs World Size (mesh_level={mesh_level})")
    plt.xlabel("World size (# GPUs)")
    plt.ylabel("End-to-end latency, p50 (ms)")
    all_ws = sorted(set(world_sizes) | set(oom_ws))
    plt.xticks(all_ws, [str(w) for w in all_ws])
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[plot_scaling] saved {output_path}")


def plot_memory_vs_world_size(records, mesh_level: int, output_path: str) -> None:
    rows = _filter(records, mesh_level)
    if not rows:
        return

    plt.figure(figsize=(8, 6), dpi=150)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    world_sizes, peak_gb, oom_ws = [], [], []
    for r in rows:
        m = r["measurement"]
        ws = r["config"]["world_size"]
        if m.get("oom"):
            oom_ws.append(ws)
            continue
        world_sizes.append(ws)
        peak_gb.append(m["peak_memory_bytes"]["max"] / (1024 ** 3))

    for ws, mem in zip(world_sizes, peak_gb):
        style = _style(ws)
        plt.scatter([ws], [mem], color=style["color"], marker=style["marker"],
                    s=100, zorder=3, label=style["label"])
    if world_sizes:
        plt.plot(world_sizes, peak_gb, color="#7f7f7f", linestyle="-", linewidth=1,
                  alpha=0.6, zorder=2)

    for ws in oom_ws:
        plt.axvline(x=ws, color=OOM_COLOR, linestyle="--", alpha=0.6)

    plt.title(f"GraphCast Peak GPU Memory vs World Size (mesh_level={mesh_level})")
    plt.xlabel("World size (# GPUs)")
    plt.ylabel("Peak memory across ranks (GB)")
    all_ws = sorted(set(world_sizes) | set(oom_ws))
    plt.xticks(all_ws, [str(w) for w in all_ws])
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[plot_scaling] saved {output_path}")


PHASE_GROUPS = {
    "embed": ["model/embed"],
    "encoder": ["encoder/halo_exchange", "encoder/edge_block", "encoder/node_block", "encoder/grid_mlp"],
    "processor": None,  # filled dynamically: all processor/layer_*/* keys
    "decoder": ["decoder/halo_exchange", "decoder/edge_block", "decoder/node_block"],
    "final_prediction": ["model/final_prediction"],
}
PHASE_COLORS = {
    "embed": "#1f77b4",
    "encoder": "#ff7f0e",
    "processor": "#2ca02c",
    "decoder": "#9467bd",
    "final_prediction": "#8c564b",
}


def plot_phase_breakdown_stacked(records, mesh_level: int, output_path: str) -> None:
    rows = _filter(records, mesh_level)
    rows = [r for r in rows if not r["measurement"].get("oom")]
    if not rows:
        return

    plt.figure(figsize=(9, 6), dpi=150)
    world_sizes = [r["config"]["world_size"] for r in rows]
    x = np.arange(len(world_sizes))

    bottom = np.zeros(len(rows))
    for group_name, fixed_keys in PHASE_GROUPS.items():
        heights = []
        for r in rows:
            breakdown = r["measurement"]["phase_breakdown_ms"]
            if fixed_keys is not None:
                keys = fixed_keys
            else:
                keys = [k for k in breakdown if k.startswith("processor/layer_")]
            total = sum((breakdown.get(k) or {}).get("mean") or 0.0 for k in keys)
            heights.append(total)
        heights = np.array(heights)
        plt.bar(x, heights, bottom=bottom, color=PHASE_COLORS[group_name],
                label=group_name, width=0.6, edgecolor="white", linewidth=0.5)
        bottom += heights

    plt.title(f"GraphCast Per-Phase Time Breakdown (mesh_level={mesh_level})")
    plt.xlabel("World size (# GPUs)")
    plt.ylabel("Mean time per phase (ms)")
    plt.xticks(x, [str(w) for w in world_sizes])
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[plot_scaling] saved {output_path}")


def plot_throughput_vs_world_size(records, mesh_level: int, output_path: str) -> None:
    rows = _filter(records, mesh_level)
    rows = [r for r in rows if not r["measurement"].get("oom")]
    if not rows:
        return

    plt.figure(figsize=(8, 6), dpi=150)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    world_sizes = [r["config"]["world_size"] for r in rows]
    throughput = [r["measurement"]["throughput_grid_points_per_sec"] for r in rows]

    for ws, tp in zip(world_sizes, throughput):
        style = _style(ws)
        plt.scatter([ws], [tp], color=style["color"], marker=style["marker"],
                    s=100, zorder=3, label=style["label"])
    plt.plot(world_sizes, throughput, color="#7f7f7f", linestyle="-", linewidth=1,
              alpha=0.6, zorder=2, label="Actual")

    if throughput and world_sizes[0] == 1:
        ideal = [throughput[0] * ws for ws in world_sizes]
        plt.plot(world_sizes, ideal, color="#7f7f7f", linestyle="--", linewidth=1,
                  alpha=0.8, label="Ideal linear scaling")

    plt.title(f"GraphCast Inference Throughput vs World Size (mesh_level={mesh_level})")
    plt.xlabel("World size (# GPUs)")
    plt.ylabel("Throughput (grid points / sec)")
    plt.xticks(world_sizes, [str(w) for w in world_sizes])
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[plot_scaling] saved {output_path}")


def plot_crossover_vs_mesh_level(records, output_path: str) -> None:
    """bench_crossover.py-style plot: latency vs mesh_level, one line per world_size
    (including the world_size=1 baseline), with sign-change interpolation marking
    where each world_size first beats the baseline."""
    world_sizes = sorted({r["config"]["world_size"] for r in records})
    if 1 not in world_sizes or len(world_sizes) < 2:
        return

    plt.figure(figsize=(9, 6), dpi=150)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    baseline_by_level = {}
    for r in records:
        if r["config"]["world_size"] == 1 and not r["measurement"].get("oom"):
            baseline_by_level[r["config"]["mesh_level"]] = r["measurement"]["end_to_end_latency_ms"]["p50"]

    for ws in world_sizes:
        rows = sorted(
            [r for r in records if r["config"]["world_size"] == ws and not r["measurement"].get("oom")],
            key=lambda r: r["config"]["mesh_level"],
        )
        if not rows:
            continue
        levels = [r["config"]["mesh_level"] for r in rows]
        latencies = [r["measurement"]["end_to_end_latency_ms"]["p50"] for r in rows]
        style = _style(ws)
        plt.plot(levels, latencies, color=style["color"], marker=style["marker"],
                  linewidth=2, label=style["label"])

        if ws != 1:
            for i in range(1, len(levels)):
                l0, l1 = levels[i - 1], levels[i]
                if l0 not in baseline_by_level or l1 not in baseline_by_level:
                    continue
                diff_prev = latencies[i - 1] - baseline_by_level[l0]
                diff_curr = latencies[i] - baseline_by_level[l1]
                if diff_prev * diff_curr < 0:
                    m_ws = (latencies[i] - latencies[i - 1]) / (l1 - l0)
                    m_base = (baseline_by_level[l1] - baseline_by_level[l0]) / (l1 - l0)
                    if m_ws != m_base:
                        x_cross = l0 + (baseline_by_level[l0] - latencies[i - 1]) / (m_ws - m_base)
                        y_cross = latencies[i - 1] + m_ws * (x_cross - l0)
                        plt.plot(x_cross, y_cross, marker="*", color=style["color"],
                                  markersize=15, zorder=5)
                        plt.annotate(
                            f"{ws} GPUs beat baseline\n~mesh_level {x_cross:.1f}",
                            xy=(x_cross, y_cross), xytext=(10, 20),
                            textcoords="offset points", fontsize=9, fontweight="bold",
                            color=style["color"],
                            arrowprops=dict(arrowstyle="->", color=style["color"]),
                        )
                    break

    plt.title("GraphCast Distributed vs Single-GPU Crossover")
    plt.xlabel("mesh_level")
    plt.ylabel("End-to-end latency, p50 (ms)")
    plt.yscale("log")
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[plot_scaling] saved {output_path}")


def main(inputs: str, output_dir: str = "figures") -> None:
    """inputs: comma-separated list and/or glob patterns of result JSON paths."""
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for part in inputs.split(","):
        part = part.strip()
        matched = glob.glob(part)
        paths.extend(matched if matched else [part])
    paths = sorted(set(paths))
    if not paths:
        print(f"[plot_scaling] no files matched inputs={inputs!r}")
        return

    records = load_results(paths)
    mesh_levels = sorted({r["config"]["mesh_level"] for r in records})

    for mesh_level in mesh_levels:
        plot_latency_vs_world_size(
            records, mesh_level, os.path.join(output_dir, f"latency_vs_world_size_L{mesh_level}.png")
        )
        plot_memory_vs_world_size(
            records, mesh_level, os.path.join(output_dir, f"memory_vs_world_size_L{mesh_level}.png")
        )
        plot_phase_breakdown_stacked(
            records, mesh_level, os.path.join(output_dir, f"phase_breakdown_L{mesh_level}.png")
        )
        plot_throughput_vs_world_size(
            records, mesh_level, os.path.join(output_dir, f"throughput_vs_world_size_L{mesh_level}.png")
        )

    plot_crossover_vs_mesh_level(records, os.path.join(output_dir, "crossover_vs_mesh_level.png"))


if __name__ == "__main__":
    # Usage:
    # python plot_scaling.py --inputs "benchmark_results/*.json" --output_dir figures/
    Fire(main)
