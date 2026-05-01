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
import torch
import torch.nn as nn

from DGraph.distributed import HaloExchange, CommunicationPattern
from DGraph.utils.TimingReport import TimingReport
from DGraph import Communicator
import torch.distributed as dist
import sys

try:
    from torch_geometric.nn.conv import GCNConv

    PYG_AVAILABLE = True
except ImportError:
    print(
        "torch_geometric not found, skipping import of GCNConv. Make sure to install torch_geometric if you want to use it."
    )
    PYG_AVAILABLE = False


import torch
import torch.nn as nn


class GCNLayer_impl(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer_impl, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, edge_index):
        source_vertices = edge_index[:, 0]
        target_vertices = edge_index[:, 1]
        x = self.linear(x)

        x_j = x[target_vertices, :]
        out_channels = x_j.size(1)
        out = torch.zeros(x.size(0), out_channels, dtype=x.dtype, device=x.device)
        scatter_index = (
            source_vertices.unsqueeze(-1).expand(-1, out_channels).to(x.device)
        )
        out = out.scatter_add(0, scatter_index, x_j)
        out = self.act(out)
        return out


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        if PYG_AVAILABLE:
            self.conv = GCNConv(in_channels, out_channels)
        else:
            self.conv = GCNLayer_impl(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GCNModel(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, halo_exchanger
    ):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNLayer(hidden_channels, hidden_channels))
        self.convs.append(GCNLayer(hidden_channels, out_channels))
        self.halo_exchanger = halo_exchanger

    def forward(self, x, comm_pattern):
        edge_index = comm_pattern.local_edge_list
        for conv in self.convs[:-1]:
            boundary_features = self.halo_exchanger(x, comm_pattern)
            x = torch.cat([x, boundary_features], dim=0)
            x = conv(x, edge_index)

        boundary_features = self.halo_exchanger(x, comm_pattern)
        x = torch.cat([x, boundary_features], dim=0)
        x = self.convs[-1](x, edge_index)
        return x


class GraphConvLayer(nn.Module):
    def __init__(self, message_dim, out_channels):
        super(GraphConvLayer, self).__init__()
        self.conv = nn.Linear(message_dim, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, num_local_nodes, edge_features=None):
        source_vertices = edge_index[:, 0]
        target_vertices = edge_index[:, 1]

        assert (source_vertices < num_local_nodes).all(), (
            f"Graph routing error: Found source_vertices >= num_local_nodes ({num_local_nodes}). "
            "Boundary nodes must only act as targets (x_j) in this aggregation scheme!"
        )

        x_i = x[source_vertices, :]
        x_j = x[target_vertices, :]

        if edge_features is not None:
            x_ij = torch.cat([x_i, x_j, edge_features], dim=1)
        else:
            x_ij = torch.cat([x_i, x_j], dim=1)

        m_ij = self.conv(x_ij)
        m_ij = self.act(m_ij)

        out_channels = m_ij.size(1)

        # 1. Allocate ONLY for the local nodes (the sources we are aggregating to)
        out = torch.zeros(num_local_nodes, out_channels, dtype=x.dtype, device=x.device)

        # 2. Scatter messages back to the SOURCE vertices
        scatter_index = (
            source_vertices.unsqueeze(-1).expand(-1, out_channels).to(x.device)
        )

        # 3. Perform the aggregation
        out = out.scatter_add(0, scatter_index, m_ij)

        return out


class CommAwareGCN(nn.Module):
    """
    Least interesting GNN model to test distributed training
    but good enough for the purpose of testing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: int,
        num_classes: int,
        halo_exchanger: HaloExchange,
        comm: Communicator,
    ):
        super(CommAwareGCN, self).__init__()
        self.halo_exchanger = halo_exchanger

        self.conv1 = GraphConvLayer(2 * in_channels, hidden_dims)

        self.conv2 = GraphConvLayer(2 * hidden_dims, hidden_dims)

        self.fc = nn.Linear(hidden_dims, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.comm = comm

    def forward(
        self, local_node_features: torch.Tensor, comm_pattern: CommunicationPattern
    ):

        num_local_nodes = local_node_features.shape[0]

        with TimingReport("feature-exchange-1"):
            boundary_features = self.halo_exchanger(local_node_features, comm_pattern)

        with TimingReport("process-1"):
            x = torch.cat([local_node_features, boundary_features], dim=0)
            x = self.conv1(x, comm_pattern.local_edge_list, num_local_nodes)

        with TimingReport("feature-exchange-2"):
            boundary_features = self.halo_exchanger(x, comm_pattern)

        with TimingReport("process-2"):
            x = torch.cat([x, boundary_features], dim=0)
            x = self.conv2(x, comm_pattern.local_edge_list, num_local_nodes)

        with TimingReport("final-fc"):
            x = self.fc(x)
        # x = self.softmax(x)
        return x
