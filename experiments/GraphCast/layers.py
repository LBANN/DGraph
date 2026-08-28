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
from DGraph.utils.TimingReport import TimingReport

"""
Local only layers for mesh processing. These layers do not perform any communication and can be used in both GraphCast and MeshGraphNet."""


class MeshGraphMLP(nn.Module):
    """MLP for graph processing"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        """
        Initializes a MeshGraphMLP instance.

        Args:
            input_dim (int): The dimensionality of the input features.
            output_dim (int): The dimensionality of the output features.
            hidden_dim (int, optional): The dimensionality of the hidden layers. Defaults to 512.
            hidden_layers (int, optional): The number of hidden layers. Defaults to 1.
            activation_fn (nn.Module, optional): The activation function to use. Defaults to nn.SiLU().
            norm_type (str, optional): The type of normalization to apply. Defaults to "LayerNorm".
        """
        super(MeshGraphMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        norm_layer = getattr(nn, norm_type)
        layers = [
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
        ]
        for _ in range(hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                activation_fn,
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(norm_layer(output_dim))
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the MLP

        Args:
            x: Node or edge features

        Returns:
            The transformed tensor
        """
        with TimingReport("MeshGraphMLP/forward"):
            return self._model(x)


class MeshNodeBlock(nn.Module):
    """Node block for mesh processing. Used in GraphCast and MeshGraphNet."""

    def __init__(
        self,
        input_node_dim: int,
        input_edge_dim: int,
        output_node_dim: int,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        aggregation_type: str = "sum",
    ):
        """
        Initializes a MeshNodeBlock instance.

        Args:
            input_node_dim (int): The dimensionality of the input node features.
            input_edge_dim (int): The dimensionality of the input edge features.
            output_node_dim (int): The dimensionality of the output node features.
            comm (CommunicatorBase): The communicator to use for distributed training.
            hidden_dim (int, optional): The dimensionality of the hidden layers. Defaults to 512.
            aggregation_type (str, optional): The type of aggregation to use. Defaults to "sum".
        """
        super(MeshNodeBlock, self).__init__()
        assert aggregation_type in ["sum"], "Only sum aggregation is supported for now."
        self.aggregation_type = aggregation_type
        self.mesh_mlp = MeshGraphMLP(
            input_dim=input_node_dim + input_edge_dim,
            output_dim=output_node_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_hidden_layers,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        src_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the node block

        Args:
            node_features: The node features
            edge_features: The edge features
            src_indices: The source indices
            dst_indices: The destination indices

        Returns:
            The updated node features
        """
        num_local_nodes = node_features.shape[0]
        with TimingReport("MeshNodeBlock/scatter_add"):
            aggregated_edge_features = torch.zeros(
                num_local_nodes,
                edge_features.shape[-1],
                device=edge_features.device,
                dtype=edge_features.dtype,
            )
            aggregated_edge_features.scatter_add_(
                0,
                src_indices.unsqueeze(-1).expand(-1, edge_features.shape[-1]),
                edge_features,
            )

        with TimingReport("MeshNodeBlock/mlp"):
            x = torch.cat([node_features, aggregated_edge_features], dim=-1)
            node_features_new = self.mesh_mlp(x) + node_features
        return node_features_new


class MeshEdgeBlock(nn.Module):
    """Edge block for mesh processing. Used in GraphCast and MeshGraphNet."""

    def __init__(
        self,
        input_src_node_dim: int,
        input_dst_node_dim: int,
        input_edge_dim: int,
        output_edge_dim: int,
        hidden_dim: int = 512,
        num_hidden_layers: int = 1,
        aggregation_type: str = "sum",
    ):
        """
        Args:
            input_node_dim (int): The dimensionality of the input node features.
            input_edge_dim (int): The dimensionality of the input edge features.
            output_edge_dim (int): The dimensionality of the output edge features.
            hidden_dim (int, optional): The dimensionality of the hidden layers. Defaults to 512.
            aggregation_type (str, optional): The type of aggregation to use. Defaults to "sum".
        """

        # TODO: Add concat trick for edge features - S.Z
        super(MeshEdgeBlock, self).__init__()
        assert aggregation_type in ["sum"], "Only sum aggregation is supported for now."
        self.aggregation_type = aggregation_type
        self.mesh_mlp = MeshGraphMLP(
            input_dim=input_src_node_dim + input_dst_node_dim + input_edge_dim,
            output_dim=output_edge_dim,
            hidden_dim=hidden_dim,
            hidden_layers=num_hidden_layers,
        )

    def forward(
        self,
        src_node_features: torch.Tensor,
        dst_node_features: torch.Tensor,
        edge_features: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the edge block

        Args:
            node_features: The node features
            edge_features: The edge features
            src_indices: The source indices
            dst_indices: The destination indices

        Returns:
            The updated edge features
        """
        with TimingReport("MeshEdgeBlock/gather"):
            src_node_features = src_node_features[src_indices]
            dst_node_features = dst_node_features[dst_indices]

        with TimingReport("MeshEdgeBlock/mlp"):
            concatenated_features = torch.cat(
                [src_node_features, dst_node_features, edge_features], dim=-1
            )
            edge_features_new = self.mesh_mlp(concatenated_features) + edge_features
        return edge_features_new
