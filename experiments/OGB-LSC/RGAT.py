import torch
import torch.nn as nn
import torch.distributed as dist
from .distributed_layers import DistributedBatchNorm1D


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class CommAwareGAT(nn.Module):
    def __init__(
        self, in_channels, out_channels, comm, heads=1, bias=True, residual=False
    ):
        super(CommAwareGAT, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels, bias=False)
        self.comm = comm
        self.project_message = nn.Linear(2 * out_channels, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.residual = residual
        self.heads = heads
        if self.residual:
            self.res_net = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(
        self, x, edge_index, rank_mapping, gather_cache=None, scatter_cache=None
    ):
        h = self.conv1(x)
        _src_indices = edge_index[:, 0, :]
        _dst_indices = edge_index[:, 1, :]
        _src_rank_mappings = torch.cat(
            [rank_mapping[0].unsqueeze(0), rank_mapping[0].unsqueeze(0)], dim=0
        )
        _dst_rank_mappings = torch.cat(
            [rank_mapping[0].unsqueeze(0), rank_mapping[1].unsqueeze(0)], dim=0
        )
        h_i = self.comm.gather(h, _dst_indices, _dst_rank_mappings, cache=gather_cache)
        h_j = self.comm.gather(h, _src_indices, _src_rank_mappings, cache=gather_cache)
        messages = torch.cat([h_i, h_j], dim=-1)
        edge_scores = self.leaky_relu(self.project_message(messages)).squeeze(-1)
        numerator = torch.exp(edge_scores)
        denominator = self.comm.scatter(
            numerator, _dst_indices, _dst_rank_mappings, h.size(1), cache=scatter_cache
        )
        denominator = self.comm.gather(
            denominator, _src_indices, _src_rank_mappings, cache=gather_cache
        )
        alpha_ij = numerator / (denominator + 1e-16)
        attention_messages = h_j * alpha_ij.unsqueeze(-1)
        out = self.comm.scatter(
            attention_messages,
            _src_indices,
            _src_rank_mappings,
            h.size(1),
            cache=scatter_cache,
        )
        if self.residual:
            out = out + self.res_net(x)
        if self.bias is not None:
            out = out + self.bias

        return out


class CommAwareRGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        num_relations,
        num_layers,
        heads,
        comm,
        dropout=0.5,
    ):
        super(CommAwareRGAT, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.comm = comm
        relation_specific_convs = nn.ModuleList()
        for _ in range(num_relations):
            relation_specific_convs.append(
                CommAwareGAT(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    bias=True,
                    residual=True,
                    comm=comm,
                )
            )
        self.layers.append(relation_specific_convs)

        for _ in range(num_layers - 1):
            relation_specific_convs = nn.ModuleList()
            for _ in range(num_relations):
                relation_specific_convs.append(
                    CommAwareGAT(
                        hidden_channels * heads,
                        hidden_channels,
                        heads=heads,
                        bias=True,
                        residual=True,
                        comm=comm,
                    )
                )
            self.layers.append(relation_specific_convs)

        for _ in range(num_layers):
            self.bn_layers.append(DistributedBatchNorm1D(hidden_channels))
        self.skip_layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skip_layers.append(nn.Linear(hidden_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            DistributedBatchNorm1D(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )
