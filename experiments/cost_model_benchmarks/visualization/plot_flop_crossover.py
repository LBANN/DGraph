import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_crossover(measurements):
    """
    Calculates the exact crossover point (num_global_edges) where multi-GPU
    execution becomes faster than single-GPU execution using linear interpolation.
    """
    # Sort measurements strictly by graph size
    measurements = sorted(measurements, key=lambda x: x["params"]["num_global_edges"])

    vertices = []
    single_times = []
    multi_times = []

    for m in measurements:
        v = m["params"]["num_global_edges"]

        # Extract median times, ignore OOM or missing single GPU data
        if m.get("single_gpu_oom", False) or not m.get("single_gpu_trials_seconds"):
            continue

        s_time = np.median(m["single_gpu_trials_seconds"])
        m_time = np.median(
            m["multi_gpu_trials_seconds_max"]
        )  # Use max time for synchronous bottleneck

        vertices.append(v)
        single_times.append(s_time)
        multi_times.append(m_time)

    for i in range(1, len(vertices)):
        diff_prev = multi_times[i - 1] - single_times[i - 1]
        diff_curr = multi_times[i] - single_times[i]

        # Sign change indicates the lines crossed
        if diff_prev * diff_curr < 0:
            x1, x2 = vertices[i - 1], vertices[i]
            y1_s, y2_s = single_times[i - 1], single_times[i]
            y1_m, y2_m = multi_times[i - 1], multi_times[i]

            m_s = (y2_s - y1_s) / (x2 - x1)
            m_m = (y2_m - y1_m) / (x2 - x1)

            if m_s != m_m:
                x_cross = x1 + (y1_m - y1_s) / (m_s - m_m)
                return x_cross

    return None  # No crossover found in this dataset


def plot_crossover_dynamics(
    file_pattern="results/crossover_*_world_F*.json",
    save_path="results/crossover_vs_features.png",
):
    """
    Parses benchmark files and plots the crossover point as a function of feature dimension,
    with separate lines for each distributed world size.
    """
    # Structure: data[world_size][feature_dim] = crossover_point
    plot_data = defaultdict(dict)

    # Locate files
    files = glob.glob(file_pattern)
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    # Regex to extract parameters from filename
    pattern = re.compile(r"crossover_(\d+)_world_F(\d+)\.json")

    for filepath in files:
        match = pattern.search(filepath)
        if match:
            world_size = int(match.group(1))
            feature_dim = int(match.group(2))

            try:
                with open(filepath, "r") as f:
                    payload = json.load(f)

                crossover_point = calculate_crossover(payload.get("measurements", []))
                if crossover_point is not None:
                    plot_data[world_size][feature_dim] = crossover_point
            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    if not plot_data:
        print("No valid crossover points could be calculated from the provided files.")
        return

    # Initialize Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, which="both", ls="-", alpha=0.3)

    markers = ["o", "s", "^", "D", "v", "p"]

    # Plot lines per world size
    for idx, (world_size, dim_data) in enumerate(sorted(plot_data.items())):
        # Sort by feature dimension for sequential line plotting
        sorted_dims = sorted(dim_data.keys())
        crossovers = [dim_data[d] for d in sorted_dims]

        plt.plot(
            sorted_dims,
            crossovers,
            marker=markers[idx % len(markers)],
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=f"{world_size} GPUs",
        )

    # Formatting
    plt.title(
        "GNN Communication Bottleneck:\nCrossover Threshold vs. Feature Dimension",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Feature Dimension (F)", fontsize=12)
    plt.ylabel("Crossover Point (Number of Edges)", fontsize=12)

    # Depending on the range of your feature dims, a log scale might be preferable for X
    # plt.xscale("log", base=2)
    plt.yscale("log")  # Y-axis (vertices) usually scales exponentially

    plt.legend(title="World Size", loc="upper left", framealpha=0.9)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Scaling dynamics visualization saved successfully to {save_path}")


if __name__ == "__main__":
    plot_crossover_dynamics()
