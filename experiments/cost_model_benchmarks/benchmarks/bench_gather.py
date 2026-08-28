"""Benchmark 1.4 — Buffer-Copy / Gather-Scatter Bandwidth.

Single-GPU benchmark.  Measures the effective bandwidth of ``x[idx]`` (gather)
and the corresponding ``scatter_add_`` (backward) for three index distributions:

* ``contiguous`` — contiguous block starting at a random offset
* ``clustered``  — *c* cluster centres, each with a contiguous block of size
                   ``--cluster-size``; simulates a well-partitioned graph halo
* ``random``     — uniformly random indices (worst-case for cache)

Sweeps *k* (number of gathered rows) from ``--min-k`` to ``--max-k``.

Usage::

    python -m benchmarks.bench_gather \\
        --distribution clustered \\
        --min-k 1000 --max-k 10000000 --steps 20 \\
        --N 20000000 --feature-dim 128 --cluster-size 64 \\
        --warmup 10 --trials 50 \\
        --output data/gather_clustered.json --seed 42
"""

import argparse

import numpy as np
import torch

from benchmarks.common import (
    collect_metadata,
    cuda_timed,
    seed_everything,
    write_result,
)


# ---------------------------------------------------------------------------
# Index generators
# ---------------------------------------------------------------------------

def contiguous_idx(N: int, k: int, device: torch.device) -> torch.Tensor:
    start = torch.randint(0, max(1, N - k), (1,)).item()
    return torch.arange(start, start + k, device=device)


def clustered_idx(N: int, k: int, cluster_size: int,
                  device: torch.device, rng: np.random.Generator) -> torch.Tensor:
    """Draw cluster centres, then take a contiguous block around each."""
    num_clusters = max(1, k // cluster_size)
    centres = rng.integers(0, N, size=num_clusters)
    idx_parts = []
    for c in centres:
        start = int(np.clip(c, 0, N - cluster_size))
        idx_parts.append(torch.arange(start, start + cluster_size, device=device))
    idx = torch.cat(idx_parts)[:k]
    return idx


def random_idx(N: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.randperm(N, device=device)[:k]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Gather/scatter bandwidth benchmark")
    p.add_argument("--distribution", choices=["contiguous", "clustered", "random"],
                   required=True)
    p.add_argument("--min-k", type=int, default=1_000)
    p.add_argument("--max-k", type=int, default=10_000_000)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--N", type=int, default=20_000_000,
                   help="Total number of rows in the source tensor x")
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--cluster-size", type=int, default=64,
                   help="Rows per cluster (only used with --distribution clustered)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = args.feature_dim

    # Pre-allocate the source tensor and gradient buffer once
    x = torch.randn(args.N, F, device=device)
    grad_y_template = torch.ones(1, F, device=device)  # resized per k

    # Sweep k values
    k_values = np.unique(
        np.round(
            np.logspace(
                np.log10(args.min_k),
                np.log10(args.max_k),
                num=args.steps,
            )
        ).astype(int)
    ).tolist()
    k_values = [min(int(k), args.N) for k in k_values]

    measurements = []
    for k in k_values:
        # Build index tensor
        if args.distribution == "contiguous":
            idx = contiguous_idx(args.N, k, device)
        elif args.distribution == "clustered":
            idx = clustered_idx(args.N, k, args.cluster_size, device, rng)
        else:
            idx = random_idx(args.N, k, device)

        # Expand idx for scatter_add_: shape [k, F]
        idx_expanded = idx.unsqueeze(1).expand(-1, F)
        grad_y = torch.ones(k, F, device=device)
        grad_x = torch.zeros_like(x)

        # --- Forward gather ---
        def gather_fn():
            _ = x[idx]

        gather_times = cuda_timed(gather_fn, warmup=args.warmup, trials=args.trials)

        # --- Backward scatter-add ---
        # grad_x.zero_() is deliberately NOT in the timed region. grad_x is
        # [N, F] (10+ GB at the default N), so zeroing it is an O(N) HBM
        # write costing ~3.1 ms -- while the scatter-add being measured is
        # O(k) and costs ~13 us at small k. Timing them together put a
        # k-independent 3.1 ms floor under every measurement (233x the real
        # cost at k=1000), which fit_gather then absorbed into
        # launch_overhead_seconds. That silently made the fitted scatter
        # model a function of N, even though it is applied to O(k) byte
        # counts (send_total * F * 4) in the cost model.
        #
        # Not re-zeroing means grad_x accumulates across trials. That is
        # harmless here: scatter_add_ runtime does not depend on the
        # accumulated values, and grad_y is all-ones so the sums stay small.
        def scatter_fn():
            grad_x.scatter_add_(0, idx_expanded, grad_y)

        scatter_times = cuda_timed(scatter_fn, warmup=args.warmup, trials=args.trials)

        measurements.append({
            "params": {
                "k": k,
                "N": args.N,
                "feature_dim": F,
                "distribution": args.distribution,
                "cluster_size": args.cluster_size if args.distribution == "clustered" else None,
            },
            "gather_trials_seconds": gather_times,
            "scatter_add_trials_seconds": scatter_times,
        })

        med_g = sorted(gather_times)[len(gather_times) // 2]
        med_s = sorted(scatter_times)[len(scatter_times) // 2]
        bytes_moved = k * F * 4
        bw_g = bytes_moved / med_g / 1e9
        bw_s = bytes_moved / med_s / 1e9
        print(
            f"[gather/{args.distribution}] k={k:>9}  "
            f"gather {1e3*med_g:.2f} ms ({bw_g:.1f} GB/s)  "
            f"scatter {1e3*med_s:.2f} ms ({bw_s:.1f} GB/s)"
        )

    payload = {
        "benchmark": "gather",
        "metadata": collect_metadata(),
        "config": {
            "distribution": args.distribution,
            "min_k": args.min_k,
            "max_k": args.max_k,
            "steps": args.steps,
            "N": args.N,
            "feature_dim": F,
            "cluster_size": args.cluster_size,
            "warmup": args.warmup,
            "trials": args.trials,
            "seed": args.seed,
        },
        "measurements": measurements,
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
