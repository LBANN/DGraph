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
