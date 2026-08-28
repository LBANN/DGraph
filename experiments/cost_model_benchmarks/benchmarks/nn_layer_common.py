import torch
import torch.nn as nn
import torch.distributed as dist

# ===========================================================================
# GNN layers (same as bench_compute.py for consistency)
# ===========================================================================


class GCNLayer(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n_local = x.shape[0]
        # Only update local vertices (dst < n_local guard not needed since
        # edge_index already restricts to local dst)
        msg = self.linear(x[src])
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return out


def create_sparse_adj(
    edge_index: torch.Tensor,
    num_rows: int,
    num_cols: int,
    device: torch.device,
    sparse_format: str = "csr",
) -> torch.Tensor:
    """Build the sparse aggregation matrix from an ``[2, E]`` edge index.

    Row = destination (the locally-owned aggregation target, ``edge_index[1]``),
    column = source (the neighbour, possibly a halo vertex, ``edge_index[0]``) —
    the convention ``DGraph.distributed.commInfo`` establishes and that
    ``experiments/OGB/GCN.py::create_sparse_adj`` follows. The matrix is
    therefore rectangular in the distributed case:
    ``[num_local, num_local + num_halo]``.

    ``sparse_format="coo"`` is a fallback: CSR is faster for SpMM, but
    ``torch.sparse.mm``'s backward for CSR operands is narrower than for COO
    across PyTorch versions. If a run dies in backward, switch to coo before
    concluding anything about the layer.
    """
    indices = torch.stack([edge_index[1], edge_index[0]])
    values = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(
        indices, values, size=(num_rows, num_cols), device=device
    ).coalesce()
    return adj.to_sparse_csr() if sparse_format == "csr" else adj


class GCNSpMMLayer(nn.Module):
    """A real GCN: transform-then-aggregate, via a sparse matmul.

    ``GCNLayer`` above applies the linear transform *per edge*
    (``linear(x[src])``), costing ``|E| * F^2``. This layer transforms the
    ``[V, F]`` vertex matrix once (``|V| * F^2``) and then aggregates with an
    SpMM (``|E| * F``), which is what a GCN actually is. At average degree 20
    the per-edge form does ~20x the FLOPs, so the two are worth fitting
    separately rather than treating one as a stand-in for the other.

    The SpMM consumes the augmented ``[n_local + n_halo, F]`` feature matrix
    and emits only the ``n_local`` rows, so the tensor does not grow from
    layer to layer.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj_sparse, self.linear(x))


class EdgeConditionedLayer(nn.Module):
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
        src, dst = edge_index[0], edge_index[1]
        msg = self.mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        return out
