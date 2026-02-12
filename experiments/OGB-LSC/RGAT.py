# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)

import torch
import torch.nn as nn
import torch.distributed as dist
from distributed_layers import DistributedBatchNorm1D
from DGraph import Communicator
import os.path as osp
from CacheGenerator import get_cache
import os
from typing import Any, List, Optional, overload
from DGraph.distributed.nccl import (
    NCCLBackendEngine,
    NCCLGraphCommPlan,
    NCCLEdgeConditionedGraphCommPlan,
)


def print_on_rank_zero(*args, **kwargs):
    dist.barrier()
    if dist.get_rank() == 0:
        print(*args, **kwargs)
    dist.barrier()


def print_on_all_ranks(*args, **kwargs):
    dist.barrier()
    for rank in range(dist.get_world_size()):
        if dist.get_rank() == rank:
            print(*args, **kwargs)
        dist.barrier()


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
        self,
        in_channels: int,
        out_channels: int,
        comm: Communicator,
        heads: int = 1,
        bias: bool = True,
        residual: bool = False,
        hetero: bool = False,
    ):
        super(CommAwareGAT, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels, bias=False)
        self.comm = comm
        self.project_message = nn.Linear(2 * out_channels, 1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = residual
        self.heads = heads
        self.hetero = hetero
        if self.residual:
            self.res_net = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    @overload
    def forward(
        self,
        x: torch.Tensor,
        comm_plan: NCCLEdgeConditionedGraphCommPlan,
        *,
        x_j: Optional[torch.Tensor] = None,
    ): ...

    @overload
    def forward(
        self,
        x: torch.Tensor,
        *,
        edge_index: Any,
        rank_mapping: Any,
        x_j: Optional[torch.Tensor] = None,
        src_gather_cache: Optional[Any] = None,
        dest_gather_cache: Optional[Any] = None,
        dest_scatter_cache: Optional[Any] = None,
    ): ...

    def forward(
        self,
        x,
        comm_plan=None,
        *,
        edge_index=None,
        rank_mapping=None,
        x_j=None,
        src_gather_cache=None,
        dest_gather_cache=None,
        dest_scatter_cache=None,
    ):
        """Forward method that can use either a communication plan or COO format

        Args:
            x: Node features tensor
            comm_plan: Communication plan object (if available)
            edge_index: Edge index tensor in COO format
            rank_mapping: Rank mapping tensors
            x_j: Optional source node features tensor (for hetero graphs)
            src_gather_cache: Optional cache for source gather communication
            dest_gather_cache: Optional cache for destination gather communication
            dest_scatter_cache: Optional cache for destination scatter communication

        Returns:
            out: Output node features tensor
        """
        if comm_plan is not None:
            return self._forward_comm_plan(x, comm_plan, x_j=x_j)

        return self._forward_coo(
            x,
            edge_index=edge_index,
            rank_mapping=rank_mapping,
            x_j=x_j,
            src_gather_cache=src_gather_cache,
            dest_gather_cache=dest_gather_cache,
            dest_scatter_cache=dest_scatter_cache,
        )

    def _process_messages(
        self,
        h,
        h_j,
    ):
        messages = torch.cat([h, h_j], dim=-1)
        edge_scores = self.leaky_relu(self.project_message(messages))
        numerator = torch.exp(edge_scores)
        return numerator

    def _calc_attention_messages(
        self,
        neighbor_features,
        numerator,
        denominator,
    ):
        alpha_ij = numerator / (denominator + 1e-16)
        attention_messages = neighbor_features * alpha_ij
        return attention_messages

    def _apply_res_and_bias(self, out, x):
        if self.residual:
            out = out + self.res_net(x)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_comm_plan(
        self, x, comm_plan: NCCLEdgeConditionedGraphCommPlan, x_j=None
    ):
        h = self.conv1(x)

        source_graph_plan = comm_plan.source_graph_plan
        if self.hetero:
            assert x_j is not None
            h_j = self.conv1(x_j)
            assert comm_plan.dest_graph_plan is not None
            dest_graph_plan = comm_plan.dest_graph_plan
        else:
            h_j = h
            dest_graph_plan = source_graph_plan

        assert isinstance(self.comm._Communicator__backend_engine, NCCLBackendEngine)

        h_i = self.comm.gather(h, comm_plan=dest_graph_plan)

        h_j = self.comm.gather(h_j, comm_plan=source_graph_plan)

        numerator = self._process_messages(h_i, h_j)

        denominator = self.comm.scatter(numerator, comm_plan=dest_graph_plan)

        denominator = self.comm.gather(denominator, comm_plan=dest_graph_plan)
        attention_messages = self._calc_attention_messages(h_j, numerator, denominator)

        out = self.comm.scatter(attention_messages, comm_plan=dest_graph_plan)

        out = self._apply_res_and_bias(out, x)

        return out

    def _forward_coo(
        self,
        x,
        edge_index,
        rank_mapping,
        x_j=None,
        src_gather_cache=None,
        dest_gather_cache=None,
        dest_scatter_cache=None,
    ):
        h = self.conv1(x)
        if self.hetero:
            assert x_j is not None
            h_j = self.conv1(x_j)
        else:
            h_j = h

        _src_indices = edge_index[:, 0, :]
        _dst_indices = edge_index[:, 1, :]
        _src_rank_mappings = torch.cat(
            [rank_mapping[0].unsqueeze(0), rank_mapping[0].unsqueeze(0)], dim=0
        )
        _dst_rank_mappings = torch.cat(
            [rank_mapping[0].unsqueeze(0), rank_mapping[1].unsqueeze(0)], dim=0
        )

        h_i = self.comm.gather(
            h, _dst_indices, _dst_rank_mappings, cache=dest_gather_cache
        )

        h_j = self.comm.gather(
            h_j, _src_indices, _src_rank_mappings, cache=src_gather_cache
        )

        numerator = self._process_messages(h_i, h_j)

        denominator = self.comm.scatter(
            numerator,
            _dst_indices,
            _dst_rank_mappings,
            h.size(1),
            cache=dest_scatter_cache,
        )

        denominator = self.comm.gather(
            denominator, _src_indices, _src_rank_mappings, cache=dest_gather_cache
        )

        attention_messages = self._calc_attention_messages(h_j, numerator, denominator)

        out = self.comm.scatter(
            attention_messages,
            _dst_indices,
            _dst_rank_mappings,
            h.size(1),
            cache=dest_scatter_cache,
        )

        out = self._apply_res_and_bias(out, x)

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
        relation_specific_convs = []

        for _ in range(num_relations):
            relation_specific_convs.append(
                CommAwareGAT(
                    in_channels,
                    hidden_channels,
                    heads=heads,
                    bias=True,
                    residual=True,
                    comm=comm,
                    hetero=True,
                )
            )
        self.layers.append(nn.ModuleList(relation_specific_convs))

        for _ in range(num_layers - 1):
            relation_specific_convs = []
            for _ in range(num_relations):
                relation_specific_convs.append(
                    CommAwareGAT(
                        hidden_channels,
                        hidden_channels,
                        heads=heads,
                        bias=True,
                        residual=True,
                        comm=comm,
                        hetero=True,
                    )
                )
            self.layers.append(nn.ModuleList(relation_specific_convs))

        for _ in range(num_layers):
            self.bn_layers.append(
                DistributedBatchNorm1D(hidden_channels, recompute=True)
            )

        self.skip_layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.skip_layers.append(nn.Linear(hidden_channels, hidden_channels))

        self.in_place_relus = nn.ModuleList()
        for _ in range(num_layers * 3):
            self.in_place_relus.append(nn.ReLU(inplace=True))

        self.in_place_dropouts = nn.ModuleList()
        for _ in range(num_layers * 3):
            self.in_place_dropouts.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            DistributedBatchNorm1D(hidden_channels, recompute=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=False),
            nn.Linear(hidden_channels, out_channels),
        )
        self.num_relations = num_relations

    def forward(self, xs, edge_types, comm_plans: List[NCCLGraphCommPlan]):

        assert len(edge_types) == len(comm_plans)
        outs = xs

        for i in range(self.num_layers):
            temp_outs = [self.skip_layers[i](outs[feat]) for feat in range(len(outs))]

            for j, (edge_type, comm_plan) in enumerate(zip(edge_types, comm_plans)):
                if j > 0:
                    break

                src_edge_type, dst_edge_type = edge_type

                temp_outs[dst_edge_type] += self.layers[i][j](  # type: ignore
                    outs[dst_edge_type], x_j=outs[src_edge_type], comm_plan=comm_plan
                )
            outs = [
                self.bn_layers[i](temp_outs[feat]) for feat in range(len(temp_outs))
            ]
            for feat in range(len(outs)):
                outs[feat] = self.in_place_relus[i * 3 + feat](outs[feat])
                outs[feat] = self.in_place_dropouts[i * 3 + feat](outs[feat])

        dummy_prameters_use = bool(int(os.getenv("RGAT_DUMMY_ALL_PARAMS_USE", "0")))
        if dummy_prameters_use:
            # Dummy operation to touch all outs to avoid DDP's 'unused parameters'
            dummy = torch.zeros(1, device=outs[0].device, dtype=outs[0].dtype)
            for t in outs:
                dummy = dummy + (
                    t[0].sum() * 0.0
                )  # zero-valued scalar that depends on t
            outs[0][0] = outs[0][0] + dummy

        return self.mlp(outs[0])
