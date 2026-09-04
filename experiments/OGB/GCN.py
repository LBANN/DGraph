# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
from typing import Union

import torch
import torch.nn as nn

from DGraph.utils.TimingReport import TimingReport


def create_sparse_adj(
    edge_index,
    num_local_nodes,
    num_halo_nodes,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Converts an edge_index of shape [num_edges, 2] into a PyTorch sparse tensor.

    Follows ``DGraph.distributed.commInfo``'s edge-list convention: column 1 is
    the destination, which is always a locally-owned vertex, and column 0 is the
    source/neighbour, which may be a halo vertex. Hence the adjacency is
    rectangular: [num_local_nodes, num_local_nodes + num_halo_nodes].
    """
    # PyTorch sparse tensors expect indices in shape [2, num_edges]
    indices = torch.stack(
        [
            edge_index[:, 1],  # Rows: Targets (strictly < num_local_nodes)
            edge_index[:, 0],  # Cols: Sources (< num_local_nodes + num_halo_nodes)
        ]
    )

    # Adjacency values are 1 for unweighted graphs
    values = torch.ones(edge_index.size(0), dtype=dtype, device=device)

    # Create the sparse COO tensor
    adj_sparse = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_local_nodes, num_local_nodes + num_halo_nodes),
        device=device,
    )

    # Convert to CSR format for faster spmm operations
    if not adj_sparse.is_coalesced():
        adj_sparse = adj_sparse.coalesce()

    adj_sparse_csr = adj_sparse.to_sparse_csr()

    return adj_sparse_csr


class GCNLayer(nn.Module):
    """GCN layer using a sparse adjacency matmul for message passing.

    Transform-then-aggregate: a dense [V, F] x [F, F] linear followed by an
    SpMM against the rectangular local adjacency. The SpMM consumes the
    augmented [num_local + num_halo, F] feature matrix and emits only the
    num_local rows, so the tensor does not grow from layer to layer.
    """

    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, adj_sparse):
        """
        Args:
            x: Node features tensor of shape [num_local + num_halo, in_channels]
            adj_sparse: Sparse adjacency of shape
                [num_local, num_local + num_halo]
        """
        # 1. Feature transformation (Dense)
        x = self.linear(x)

        # 2. Message Passing via Sparse Matrix Multiplication
        # This single line replaces x_j, out.zeros(), and scatter_add()
        out = torch.sparse.mm(adj_sparse, x)

        # 3. Activation
        out = self.act(out)

        return out


class GCNModel(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        halo_exchanger,
    ):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_channels, hidden_channels))
        self.convs.append(GCNLayer(hidden_channels, out_channels))
        self.halo_exchanger = halo_exchanger

    def forward(self, x, comm_pattern):
        # comm_pattern.local_edge_list must already have been converted to a
        # sparse CSR adjacency via create_sparse_adj().
        edge_index = comm_pattern.local_edge_list
        counter = 1
        for conv in self.convs[:-1]:
            with TimingReport(f"feature-exchange-{counter}"):
                boundary_features = self.halo_exchanger(x, comm_pattern)

            with TimingReport(f"process-{counter}"):
                x = torch.cat([x, boundary_features], dim=0)
                x = conv(x, edge_index)

            counter += 1

        with TimingReport(f"feature-exchange-{counter}"):
            boundary_features = self.halo_exchanger(x, comm_pattern)

        with TimingReport(f"process-{counter}"):
            x = torch.cat([x, boundary_features], dim=0)
            x = self.convs[-1](x, edge_index)
        return x
