import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import seaborn as sns

color_goups = ["pre-processing", "Conv", "Gather", "Scatter", "Final"]


def plot_stacked_bar_from_dicts(
    data, x_ticks=None, plot_filename="stacked_bar_plot.png", title="Stacked Bar Plot"
):
    """
    Generates a stacked bar plot from a list of dictionaries.

    Each dictionary in the list represents a bar. The keys of the dictionary
    are the 'operations', and the values are lists of floats. The height of
    each segment in a bar is the mean of the list of floats.

    Args:
        data (list[dict]): A list of dictionaries, where each dictionary
                           has the same set of keys.
        plot_filename (str): The name of the file to save the plot to.
    """
    if not data:
        print("Input data is empty. Cannot generate a plot.")
        return

    # Extract operation names from the first dictionary
    operations = []
    op_names = []
    cgroup_map = {}
    for key in data[0].keys():
        if key == "cache_generation_time":
            continue
        if key == "pre-processing":
            continue
        key_name = key.replace("_", " ")
        operations.append(key)
        op_names.append(key_name)
        for group in color_goups:
            if group in key:
                cgroup_map[key] = group
                break

    # Calculate the means for each operation in each dictionary
    processed_data = []
    for item in data:
        stats = {f"{op}_mean": np.mean(item[op][1:]) for op in operations}
        stats.update({f"{op}_std": np.std(item[op][1:]) for op in operations})
        processed_data.append(stats)

    df = pd.DataFrame(processed_data)
    df_means = df[[f"{op}_mean" for op in operations]]
    df_means.columns = op_names

    df_std = df[[f"{op}_std" for op in operations]]
    df_std.columns = op_names

    print("DataFrame of means:")
    print(df_means)
    print("DataFrame of standard deviations:")
    print(df_std)

    plt.rc("axes", titlesize=28)
    plt.rc("axes", labelsize=26)
    plt.rc("xtick", labelsize=24)
    plt.rc("ytick", labelsize=24)
    plt.rc("legend", fontsize=22)
    plt.rc("legend", title_fontsize=24)

    fig, ax = plt.subplots(figsize=(15, 18))

    # Use a seaborn color palette
    # colors = sns.color_palette("rocket", n_colors=len(color_goups))
    # color_map = {group: colors[i] for i, group in enumerate(color_goups)}
    # colors = [color_map[cgroup_map[op]] for op in operations]
    colors = sns.color_palette("viridis", n_colors=len(operations))

    df_means.plot(kind="bar", stacked=True, color=colors, ax=ax)

    df_cumulative_means = df_means.cumsum(axis=1)
    x_coords = np.arange(len(df_means))
    for op in reversed(op_names):  # Reverse to draw from top to bottom
        y_coords = df_cumulative_means[op]
        errors = df_std[op]
        ax.errorbar(
            x_coords,
            y_coords,
            yerr=errors,
            fmt="none",
            ecolor="black",
            capsize=4,
            elinewidth=1.5,
            capthick=1.5,
        )
    if x_ticks is not None:
        ax.set_xticks(np.arange(len(x_ticks)))
        ax.set_xticklabels(x_ticks)
    # Add labels and title
    plt.title(title)
    plt.xlabel("Ranks")
    plt.ylabel("Runtime (milliseconds)")
    plt.xticks(rotation=0)
    plt.legend(title="Operations")

    # Ensure layout is tight
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(plot_filename)
    # Also save as a PDF
    pdf_filename = os.path.splitext(plot_filename)[0] + ".pdf"
    plt.savefig(pdf_filename)
    print(f"Plot saved to {plot_filename}")


def main():
    log_dir = "logs"
    title = "GCN Training Time Breakdown on OGBN-Products on Matrix (H100)"

    ranks = [8, 16, 32]
    files = [f"products_timing_report_world_size_{i}_cache_True.json" for i in ranks]
    all_data = []
    for file in files:
        file_path = os.path.join(log_dir, file)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                all_data.append(data)
        else:
            print(f"File {file_path} does not exist. Skipping.")
            continue
    print(f"Loaded data for {len(all_data)} configurations.")
    # print(all_data)
    if all_data:
        plot_stacked_bar_from_dicts(
            all_data,
            x_ticks=ranks,
            plot_filename="products_bar_plot.png",
            title=title,
        )
        print("Plotting completed.")


if __name__ == "__main__":
    main()
