import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Plot Crossover Benchmark Results")
    p.add_argument(
        "--input", type=str, required=True, help="Path to benchmark JSON file"
    )
    p.add_argument(
        "--output",
        type=str,
        default="crossover_analysis.png",
        help="Path to save the generated plot",
    )
    return p.parse_args()


def plot_crossover_benchmark(payload, save_path="crossover_analysis.png"):
    """
    Parses benchmark payload and plots Single GPU vs Multi-GPU execution times,
    annotating the crossover point where distributed training becomes faster.
    """
    # Sort measurements strictly by graph size (num_vertices)
    measurements = sorted(
        payload["measurements"], key=lambda x: x["params"]["num_vertices"]
    )

    vertices = []
    single_gpu_times = []
    multi_gpu_times = []
    single_gpu_oom_points = []

    world_size = payload["config"]["world_size"]
    model_name = payload["config"]["model"]
    partitioner = payload["config"]["partitioner"]
    feature_dim = payload["config"]["feature_dim"]

    for m in measurements:
        v = m["params"]["num_global_edges"]
        vertices.append(v)

        # Multi-GPU: Use the max time across ranks as it represents the true synchronous bottleneck
        multi_time = np.median(m["multi_gpu_trials_seconds_max"])
        multi_gpu_times.append(multi_time)

        # Single-GPU: Handle OOM scenarios safely
        if m.get("single_gpu_oom", False) or not m["single_gpu_trials_seconds"]:
            single_gpu_times.append(np.nan)
            single_gpu_oom_points.append(v)
        else:
            single_gpu_times.append(np.median(m["single_gpu_trials_seconds"]))

    vertices = np.array(vertices)
    single_gpu_times = np.array(single_gpu_times)
    multi_gpu_times = np.array(multi_gpu_times)

    # Initialize Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Plot Valid Single GPU points
    valid_mask = ~np.isnan(single_gpu_times)
    plt.plot(
        vertices[valid_mask],
        single_gpu_times[valid_mask],
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#1f77b4",
        label="Single GPU",
    )

    # Plot Multi GPU points
    plt.plot(
        vertices,
        multi_gpu_times,
        marker="s",
        linestyle="-",
        linewidth=2,
        color="#ff7f0e",
        label=f"Distributed ({world_size} GPUs)",
    )

    # Annotate OOM boundaries
    for oom_v in single_gpu_oom_points:
        plt.axvline(x=oom_v, color="#d62728", linestyle="--", alpha=0.6)
        plt.text(
            oom_v * 1.02,
            plt.ylim()[1] * 0.9,
            "Single GPU OOM",
            color="#d62728",
            verticalalignment="top",
        )

    # Calculate and Annotate Crossover Point
    crossover_found = False
    for i in range(1, len(vertices)):
        if valid_mask[i - 1] and valid_mask[i]:
            diff_prev = multi_gpu_times[i - 1] - single_gpu_times[i - 1]
            diff_curr = multi_gpu_times[i] - single_gpu_times[i]

            # A sign change indicates the lines crossed
            if diff_prev * diff_curr < 0:
                x1, x2 = vertices[i - 1], vertices[i]
                y1_s, y2_s = single_gpu_times[i - 1], single_gpu_times[i]
                y1_m, y2_m = multi_gpu_times[i - 1], multi_gpu_times[i]

                # Linear interpolation for precise intersection coordinates
                m_s = (y2_s - y1_s) / (x2 - x1)
                m_m = (y2_m - y1_m) / (x2 - x1)

                if m_s != m_m:
                    x_cross = x1 + (y1_m - y1_s) / (m_s - m_m)
                    y_cross = y1_s + m_s * (x_cross - x1)

                    plt.plot(
                        x_cross,
                        y_cross,
                        marker="*",
                        color="#2ca02c",
                        markersize=15,
                        zorder=5,
                    )
                    plt.annotate(
                        f"Crossover:\n~{int(x_cross):,} edges",
                        xy=(x_cross, y_cross),
                        xytext=(-20, 40),
                        textcoords="offset points",
                        fontsize=10,
                        fontweight="bold",
                        color="#2ca02c",
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="arc3,rad=.2",
                            color="#2ca02c",
                        ),
                    )
                    crossover_found = True
                    break

    # Formatting
    plt.title(
        f"GNN Distributed Scaling Crossover\nModel: {model_name} | Partitioner: {partitioner} | Feature Dim: {feature_dim}",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Graph Size (Number of Edges)", fontsize=12)
    plt.ylabel("Execution Time (Seconds)", fontsize=12)
    plt.xscale(
        "log"
    )  # Using log scale for x-axis as graph sizes usually scale exponentially
    plt.yscale("linear")

    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    # Output
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    if crossover_found:
        print(f"Crossover point detected at approximately {int(x_cross):,} vertices.")
    else:
        print("No crossover point detected in the provided dataset.")


# Example usage assuming `payload` is already loaded in your environment:
# plot_crossover_benchmark(payload)


def main():
    args = parse_args()
    with open(args.input, "r") as f:
        payload = json.load(f)

    plot_crossover_benchmark(payload, save_path=args.output)


if __name__ == "__main__":
    main()
