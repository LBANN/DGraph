import numpy as np
import matplotlib.pyplot as plt


def get_stats(_file, warmup=10):
    _data = np.load(_file)
    _data = _data[warmup:]
    _mean = np.mean(_data)
    _std = np.std(_data)
    return _mean, _std


def generate_plots(
    backend: str,
):
    ops = [
        "gather",
        "scatter",
    ]

    means = []
    stds = []

    for op in ops:
        _mean, _std = get_stats(f"logs/{backend.upper()}_{op}_times_0.npy")
        means.append(_mean)
        stds.append(_std)

    x = np.arange(len(means))
    plt.errorbar(x, means, yerr=stds, fmt="o")
    plt.xticks(x, [ops[i] for i in x])
    plt.xlabel("GPU")
    plt.ylabel("Time (ms)")
    plt.title(f"{backend.upper()} Benchmarks Results")
    plt.savefig(f"{backend}_benchmark_results.png")


if __name__ == "__main__":
    generate_plots("nccl")
    generate_plots("nvshmem")
