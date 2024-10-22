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
    Least interesting GNN model to test distributed training
    but good enough for the purpose of testing.
    """

    def __init__(self, in_channels, hidden_dims, num_classes, comm):
        super(CommAwareGCN, self).__init__()

        self.conv1 = ConvLayer(in_channels, hidden_dims)
        self.conv2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc = nn.Linear(hidden_dims, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.comm = comm

    def forward(self, node_features, edge_index):
        x = self.comm.gather(node_features, edge_index[0])
        x = self.conv1(x)
        x = self.comm.scatter(x, edge_index[1])
        x = self.comm.gather(x, edge_index[0])
        x = self.conv2(x)
        x = self.comm.scatter(x, edge_index[1])
        x = self.fc(x)
        x = self.softmax(x)
        return x
