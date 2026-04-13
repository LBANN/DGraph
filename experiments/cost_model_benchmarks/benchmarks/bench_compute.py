"""Benchmark 1.3 — GNN Layer Compute Primitive.

Single-GPU benchmark.  Fits f_comp(|Ṽ|, |Ẽ|) for two message-function
variants:

* ``gcn``  — GCN-like: φ(h_b) = W h_b  (source-only linear transform)
* ``edge`` — Edge-conditioned: φ(h_b, h_a, e_ba) = MLP([h_b, h_a, e_ba])
             with a 2-layer MLP (hidden dim = feature_dim)

Two sweep modes (controlled by ``--sweep``):

* ``vertices`` — vary |V| with |E| fixed at ``--fixed-value``
* ``edges``    — vary |E| with |V| fixed at ``--fixed-value``

Usage::

    python -m benchmarks.bench_compute \\
        --model edge --sweep vertices \\
        --min 1000 --max 100000 --steps 15 \\
        --fixed-value 500000 --feature-dim 128 \\
        --warmup 10 --trials 50 \\
        --output data/compute_edge_vswp.json --seed 42
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from benchmarks.common import (
    collect_metadata,
    cuda_timed,
    seed_everything,
    write_result,
)


# ---------------------------------------------------------------------------
# Synthetic graph generation
# ---------------------------------------------------------------------------


def erdos_renyi_edges(
    num_vertices: int, num_edges: int, device: torch.device
) -> torch.Tensor:
    """Return an edge index tensor of shape [2, num_edges] (random, with replacement)."""
    src = torch.randint(0, num_vertices, (num_edges,), device=device)
    dst = torch.randint(0, num_vertices, (num_edges,), device=device)
    return torch.stack([src, dst], dim=0)


# ---------------------------------------------------------------------------
# GNN layers
# ---------------------------------------------------------------------------


class GCNLayer(nn.Module):
    """GCN-like: aggregate neighbour source features with a linear transform."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [V, F], edge_index: [2, E]
        src, dst = edge_index[0], edge_index[1]
        # Message: transform source features
        msg = self.linear(x[src])  # [E, F]
        # Aggregate: scatter-add to destination
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return out


class EdgeConditionedLayer(nn.Module):
    """Edge-conditioned: φ(h_b, h_a, e_ba) = MLP([h_b, h_a, e_ba])."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # x: [V, F], edge_index: [2, E], edge_attr: [E, F]
        src, dst = edge_index[0], edge_index[1]
        msg_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)  # [E, 3F]
        msg = self.mlp(msg_input)  # [E, F]
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="GNN compute primitive benchmark")
    p.add_argument("--model", choices=["gcn", "edge"], required=True)
    p.add_argument("--sweep", choices=["vertices", "edges"], required=True)
    p.add_argument("--min", type=int, default=1_000, dest="sweep_min")
    p.add_argument("--max", type=int, default=1_000_000, dest="sweep_max")
    p.add_argument("--steps", type=int, default=15)
    p.add_argument(
        "--fixed-value",
        type=int,
        default=500_000,
        help="Fixed |E| when sweeping vertices, or fixed |V| when sweeping edges",
    )
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    F = args.feature_dim

    # Build model
    if args.model == "gcn":
        model = GCNLayer(F).to(device)
    else:
        model = EdgeConditionedLayer(F).to(device)

    # Sweep points
    sweep_vals = np.unique(
        np.round(
            np.logspace(
                np.log10(args.sweep_min),
                np.log10(args.sweep_max),
                num=args.steps,
            )
        ).astype(int)
    ).tolist()

    measurements = []
    for val in sweep_vals:
        if args.sweep == "vertices":
            num_v, num_e = val, args.fixed_value
        else:
            num_v, num_e = args.fixed_value, val

        # Synthetic data
        x = torch.randn(num_v, F, device=device, requires_grad=True)
        edge_index = erdos_renyi_edges(num_v, num_e, device)
        edge_attr = (
            torch.randn(num_e, F, device=device) if args.model == "edge" else None
        )

        # Forward timing
        def fwd():
            if args.model == "gcn":
                model(x, edge_index)
            else:
                model(x, edge_index, edge_attr)

        fwd_times = cuda_timed(fwd, warmup=args.warmup, trials=args.trials)

        # Backward timing (run fwd first to get a graph)
        if args.model == "gcn":
            out = model(x, edge_index)
        else:
            out = model(x, edge_index, edge_attr)
        loss_ref = out.sum()

        def bwd():
            if x.grad is not None:
                x.grad.zero_()
            if args.model == "gcn":
                out_inner = model(x, edge_index)
            else:
                out_inner = model(x, edge_index, edge_attr)
            out_inner.sum().backward()

        bwd_times = cuda_timed(bwd, warmup=args.warmup, trials=args.trials)

        measurements.append(
            {
                "params": {
                    "num_vertices": num_v,
                    "num_edges": num_e,
                    "sweep_var": args.sweep,
                    "sweep_value": val,
                    "model": args.model,
                    "feature_dim": F,
                },
                "forward_trials_seconds": fwd_times,
                "backward_trials_seconds": bwd_times,
            }
        )
        med_fwd = sorted(fwd_times)[len(fwd_times) // 2]
        med_bwd = sorted(bwd_times)[len(bwd_times) // 2]
        print(
            f"[compute/{args.model}] |V|={num_v:>8}  |E|={num_e:>9}  "
            f"fwd {1e3*med_fwd:.2f} ms  bwd {1e3*med_bwd:.2f} ms"
        )

    payload = {
        "benchmark": "compute",
        "metadata": collect_metadata(),
        "config": {
            "model": args.model,
            "sweep": args.sweep,
            "sweep_min": args.sweep_min,
            "sweep_max": args.sweep_max,
            "steps": args.steps,
            "fixed_value": args.fixed_value,
            "feature_dim": F,
            "warmup": args.warmup,
            "trials": args.trials,
            "seed": args.seed,
        },
        "measurements": measurements,
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    main()
