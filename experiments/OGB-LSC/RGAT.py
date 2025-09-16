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
import os.path as osp
from CacheGenerator import get_cache


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
        in_channels,
        out_channels,
        comm,
        heads=1,
        bias=True,
        residual=False,
        hetero=False,
    ):
        super(CommAwareGAT, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels, bias=False)
        self.comm = comm
        self.project_message = nn.Linear(2 * out_channels, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
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

    def forward(
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

        messages = torch.cat([h_i, h_j], dim=-1)
        edge_scores = self.leaky_relu(self.project_message(messages))
        numerator = torch.exp(edge_scores)

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
        alpha_ij = numerator / (denominator + 1e-16)
        attention_messages = h_j * alpha_ij
        out = self.comm.scatter(
            attention_messages,
            _dst_indices,
            _dst_rank_mappings,
            h.size(1),
            cache=dest_scatter_cache,
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
        use_cache=True,
        cache_file_path="rgat_cache",
    ):
        super(CommAwareRGAT, self).__init__()
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        self.comm = comm
        self.use_cache = use_cache
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
                        hidden_channels * heads,
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
        self.num_relations = num_relations
        self._setup_caches(cache_file_path)

    def _setup_caches(self, cache_file_path):
        num_relations = self.num_relations
        comm = self.comm
        # Caching for RGAT is a little bit tricky. There are three types of communication
        # 1. Source gather (gathering source node features from source ranks)
        # 2. Destination gather (gathering destination node features from destination ranks)
        # 3. Destination scatter (scattering the messages to destination ranks)
        # That gets repeated for each relation type.
        # So we will have 3 * num_relations cache files

        self.src_gather_cache_files = [
            (
                f"{cache_file_path}_src_gather_cache_rel_{rel}_rank"
                + f"_{comm.get_world_size()}_{comm.get_rank()}.pt"
            )
            for rel in range(num_relations)
        ]

        self.dest_scatter_cache_files = [
            (
                f"{cache_file_path}_dest_scatter_cache_rel_{rel}_rank"
                + f"_{comm.get_world_size()}_{comm.get_rank()}.pt"
            )
            for rel in range(num_relations)
        ]
        self.dest_gather_cache_files = [
            (
                f"{cache_file_path}_dest_gather_cache_rel_{rel}_rank"
                + f"_{comm.get_world_size()}_{comm.get_rank()}.pt"
            )
            for rel in range(num_relations)
        ]
        self.src_gather_caches = []
        self.dest_scatter_caches = []
        self.dest_gather_caches = []

        if self.use_cache:
            for caches in zip(
                self.src_gather_cache_files,
                self.dest_scatter_cache_files,
                self.dest_gather_cache_files,
            ):
                (
                    src_gather_cache_file,
                    dest_scatter_cache_file,
                    dest_gather_cache_file,
                ) = caches
                if (
                    osp.exists(src_gather_cache_file)
                    and osp.exists(dest_scatter_cache_file)
                    and osp.exists(dest_gather_cache_file)
                ):
                    _src_gather_cache = torch.load(
                        src_gather_cache_file, weights_only=False
                    )
                    _dest_scatter_cache = torch.load(
                        dest_scatter_cache_file, weights_only=False
                    )
                    _dest_gather_cache = torch.load(
                        dest_gather_cache_file, weights_only=False
                    )
                    self.src_gather_caches.append(_src_gather_cache)
                    self.dest_scatter_caches.append(_dest_scatter_cache)
                    self.dest_gather_caches.append(_dest_gather_cache)
                else:
                    self.src_gather_caches.append(None)
                    self.dest_scatter_caches.append(None)
                    self.dest_gather_caches.append(None)

    def forward(self, xs, adjts, edge_types, rank_mappings):
        assert len(adjts) == len(edge_types)
        assert len(adjts) == self.num_relations

        outs = xs

        for i in range(self.num_layers):
            temp_outs = [self.skip_layers[i](outs[feat]) for feat in range(len(outs))]
            for j, (edge_index, edge_type, rank_mapping) in enumerate(
                zip(adjts, edge_types, rank_mappings)
            ):

                if self.use_cache:
                    caches = get_cache(
                        src_gather_cache=self.src_gather_caches[j],
                        dest_gather_cache=self.dest_gather_caches[j],
                        dest_scatter_cache=self.dest_scatter_caches[j],
                        src_gather_cache_file=self.src_gather_cache_files[j],
                        dest_scatter_cache_file=self.dest_scatter_cache_files[j],
                        dest_gather_cache_file=self.dest_gather_cache_files[j],
                        rank=self.comm.get_rank(),
                        world_size=self.comm.get_world_size(),
                        src_indices=edge_index[:, 0, :],
                        dest_indices=edge_index[:, 1, :],
                        edge_location=rank_mapping[0],
                        src_data_mappings=rank_mapping[0],
                        dest_data_mappings=rank_mapping[1],
                        num_input_rows=outs[edge_type[0]].size(0),
                        num_output_rows=outs[edge_type[1]].size(0),
                    )
                    src_gather_cache, dest_scatter_cache, dest_gather_cache = caches
                else:
                    src_gather_cache = None
                    dest_scatter_cache = None
                    dest_gather_cache = None

                src_edge_type, dst_edge_type = edge_type
                self.comm.barrier()
                if self.comm.get_rank() == 0:
                    print(
                        f"Layer {i} Relation {j} started on rank {self.comm.get_rank()}"
                    )
                    print(
                        f"Edge index shape: {edge_index.shape}"
                        f" Edge type: {edge_type}",
                        f" src tensor shape: {outs[src_edge_type].shape}",
                        f" dst tensor shape: {outs[dst_edge_type].shape}",
                    )
                self.comm.barrier()
                temp_outs[dst_edge_type] += self.layers[i][j](
                    outs[dst_edge_type],
                    edge_index,
                    rank_mapping,
                    x_j=outs[src_edge_type],
                    src_gather_cache=src_gather_cache,
                    dest_gather_cache=dest_gather_cache,
                    dest_scatter_cache=dest_scatter_cache,
                )
                self.comm.barrier()
                if self.comm.get_rank() == 0:
                    print(f"Layer {i} Relation {j} done on rank {self.comm.get_rank()}")
                self.comm.barrier()
            outs = [
                self.bn_layers[i](temp_outs[feat]) for feat in range(len(temp_outs))
            ]
            outs = [torch.relu(outs[feat]) for feat in range(len(outs))]
            outs = [
                torch.dropout(outs[feat], p=self.dropout, train=self.training)
                for feat in range(len(outs))
            ]

        return self.mlp(outs[0])
