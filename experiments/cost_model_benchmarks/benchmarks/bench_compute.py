"""Benchmark 1.3 — GNN Layer Compute Primitive.

Single-GPU benchmark.  Fits f_comp(|Ṽ|, |Ẽ|) for two message-function
variants:

* ``gcn``      — GCN-like, applied per edge: φ(h_b) = W h_b. Costs |E|F².
* ``edge``     — Edge-conditioned: φ(h_b, h_a, e_ba) = MLP([h_b, h_a, e_ba])
                 with a 2-layer MLP (hidden dim = feature_dim). Costs |E|F².
* ``gcn_spmm`` — A real GCN: transform-then-aggregate, dense [V,F]x[F,F]
                 followed by a sparse matmul. Costs |V|F² + |E|F.

Two sweep modes (controlled by ``--sweep``):

* ``vertices`` — vary |V| with |E| fixed at ``--fixed-value``
* ``edges``    — vary |E| with |V| fixed at ``--fixed-value``

Sweeping ``--feature-dim`` across several runs is what makes ``T_comp``'s F
dependence identifiable: at a single F the F² and F terms are collinear, so
compute-bound and bandwidth-bound layers are indistinguishable. See
``analysis/compute_forms.py``.

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


class GCNSpMMLayer(nn.Module):
    """A real GCN: transform-then-aggregate via sparse matmul.

    ``GCNLayer`` above transforms *per edge* (``|E| * F^2``); this transforms
    the vertex matrix once (``|V| * F^2``) and aggregates with an SpMM
    (``|E| * F``). At average degree 20 the per-edge form does ~20x the FLOPs,
    so the two need separate fits — see analysis/compute_forms.py.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj_sparse, self.linear(x))


def build_sparse_adj(edge_index: torch.Tensor, num_vertices: int,
                     device: torch.device, sparse_format: str) -> torch.Tensor:
    """Square [V, V] aggregation matrix: row = dst, column = src.

    Built once per sweep point, outside the timed region — in the distributed
    setting the adjacency comes prebuilt from the comm pattern, so including
    its construction here would measure something the model never pays for.
    """
    indices = torch.stack([edge_index[1], edge_index[0]])
    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(
        indices, values, size=(num_vertices, num_vertices), device=device
    ).coalesce()
    return adj.to_sparse_csr() if sparse_format == "csr" else adj


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
    p.add_argument("--model", choices=["gcn", "edge", "gcn_spmm"], required=True)
    p.add_argument("--sparse-format", choices=["csr", "coo"], default="csr",
                   help="Sparse layout for --model gcn_spmm. CSR is faster; "
                        "switch to coo if torch.sparse.mm's backward rejects "
                        "CSR on this PyTorch build.")
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
    elif args.model == "gcn_spmm":
        model = GCNSpMMLayer(F).to(device)
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

        # A sweep over feature dims will eventually hit a cell that does not
        # fit (edge-conditioned at F=512 with |E|=1e6 is the tightest). Skip
        # that point and keep going: an unguarded OOM here loses every
        # measurement already collected in this run.
        x = edge_index = edge_attr = adj = None
        try:
            # Synthetic data
            x = torch.randn(num_v, F, device=device, requires_grad=True)
            edge_index = erdos_renyi_edges(num_v, num_e, device)
            edge_attr = (
                torch.randn(num_e, F, device=device) if args.model == "edge" else None
            )
            adj = (
                build_sparse_adj(edge_index, num_v, device, args.sparse_format)
                if args.model == "gcn_spmm"
                else None
            )

            def call_model():
                if args.model == "gcn":
                    return model(x, edge_index)
                if args.model == "gcn_spmm":
                    return model(x, adj)
                return model(x, edge_index, edge_attr)

            # Forward timing
            fwd_times = cuda_timed(call_model, warmup=args.warmup, trials=args.trials)

            # NOTE: this closure times grad-zero + FORWARD + backward, not the
            # backward pass alone — the forward must be re-run inside the timed
            # region to rebuild the autograd graph consumed by each .backward()
            # call. So "backward_trials_seconds" below is really the combined
            # forward+backward cost (which is what the assembled cost model's
            # T_comp needs, since bench_end_to_end.py times the same op set).
            # Do not subtract or add "forward_trials_seconds" to it.
            def bwd():
                if x.grad is not None:
                    x.grad.zero_()
                call_model().sum().backward()

            bwd_times = cuda_timed(bwd, warmup=args.warmup, trials=args.trials)
        except torch.cuda.OutOfMemoryError:
            print(
                f"[compute/{args.model}] OOM: |V|={num_v:,} |E|={num_e:,} F={F} "
                "does not fit; skipping this point and continuing"
            )
            del x, edge_index, edge_attr, adj
            torch.cuda.empty_cache()
            continue

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
            f"[compute/{args.model}] |V|={num_v:>8}  |E|={num_e:>9}  F={F:>4}  "
            f"fwd {1e3*med_fwd:.2f} ms  bwd {1e3*med_bwd:.2f} ms"
        )

        del x, edge_index, edge_attr, adj
        torch.cuda.empty_cache()

    if not measurements:
        raise RuntimeError(
            f"Every sweep point OOMed for model={args.model} F={F} "
            f"(|V|/|E| from {args.sweep_min:,} to {args.sweep_max:,}, fixed "
            f"{args.fixed_value:,}). Writing an empty file would silently "
            "produce a fit with no data behind it. Lower --max, --fixed-value, "
            "or --feature-dim."
        )

    payload = {
        "benchmark": "compute",
        "metadata": collect_metadata(),
        "config": {
            "model": args.model,
            "sparse_format": args.sparse_format if args.model == "gcn_spmm" else None,
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
