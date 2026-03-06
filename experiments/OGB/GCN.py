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
from DGraph.distributed.nccl._NCCLCommPlan import NCCLGraphCommPlan
from DGraph.utils.TimingReport import TimingReport


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class CommAwareGCN(nn.Module):
    """
    GNN model that uses NCCLGraphCommPlan for distributed gather-scatter.
    """

    def __init__(self, in_channels, hidden_dims, num_classes, comm):
        super(CommAwareGCN, self).__init__()

        self.conv1 = ConvLayer(in_channels, hidden_dims)
        self.conv2 = ConvLayer(hidden_dims, hidden_dims)
        self.fc = nn.Linear(hidden_dims, num_classes)
        self.comm = comm

    def forward(
        self,
        node_features: torch.Tensor,
        comm_plan: NCCLGraphCommPlan,
    ):
        """
        Args:
            node_features: Local node features (batch, num_local_nodes, features)
            comm_plan: Pre-computed NCCLGraphCommPlan for gather-scatter
        """
        TimingReport.start("Gather_1")
        x = self.comm.gather(node_features, comm_plan=comm_plan)
        TimingReport.stop("Gather_1")

        TimingReport.start("Conv_1")
        x = self.conv1(x)
        TimingReport.stop("Conv_1")

        TimingReport.start("Scatter_1")
        x = self.comm.scatter(x, comm_plan=comm_plan)
        TimingReport.stop("Scatter_1")

        TimingReport.start("Gather_2")
        x = self.comm.gather(x, comm_plan=comm_plan)
        TimingReport.stop("Gather_2")

        TimingReport.start("Conv_2")
        x = self.conv2(x)
        TimingReport.stop("Conv_2")

        TimingReport.start("Scatter_2")
        x = self.comm.scatter(x, comm_plan=comm_plan)
        TimingReport.stop("Scatter_2")

        TimingReport.start("Final_FC")
        x = self.fc(x)
        TimingReport.stop("Final_FC")

        return x
