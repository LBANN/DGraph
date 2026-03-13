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

from DGraph.distributed import DGraphMessagePassing, HaloExchange, CommunicationPattern
from DGraph.utils.TimingReport import TimingReport
from DGraph import Communicator


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_features=None):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]

        if edge_features is not None:
            x_ij = torch.cat([x_i, x_j, edge_features], dim=1)
        else:
            x_ij = torch.cat([x_i, x_j], dim=1)

        x = self.conv(x_ij)
        x = self.act(x)

        x = torch.scatter_add(x, 0, edge_index[0].unsqueeze(1), x)

        return x


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

        self.conv1 = GraphConvLayer(in_channels, hidden_dims)

        self.conv2 = GraphConvLayer(hidden_dims, hidden_dims)

        self.fc = nn.Linear(hidden_dims, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.comm = comm

    def forward(self, node_features: torch.Tensor, comm_pattern: CommunicationPattern):

        with TimingReport("feature-exchange-1"):
            boundary_features = self.halo_exchanger(node_features, comm_pattern)

        with TimingReport("process-1"):
            x = torch.cat([node_features, boundary_features], dim=0)
            x = self.conv1(x, comm_pattern.local_edge_list)

        with TimingReport("feature-exchange-2"):
            boundary_features = self.halo_exchanger(x, comm_pattern)

        with TimingReport("process-2"):
            x = torch.cat([x, boundary_features], dim=0)
            x = self.conv2(x, comm_pattern.local_edge_list)

        with TimingReport("final-fc"):
            x = self.fc(x)
        # x = self.softmax(x)
        return x
