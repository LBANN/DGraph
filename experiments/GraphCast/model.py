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
from typing import Optional
from torch import Tensor
from layers import Processor, MLP
from graphcast_config import Config


class GraphCastEncoder(nn.Module):
    """Encoder for the GraphCast model. The encoder is responsible for taking grid
    information and encoding it into the multi-mesh, which the processor uses."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.grid_feature_embedder = MLP(73, 512)
        self.mesh_feature_embedder = MLP(73, 512)
        self.grid2mesh_edge_embedder = MLP(73, 512)
        self.mesh2grid_edge_embedder = MLP(73, 512)
        self.mesh2mesh_edge_embedder = MLP(73, 512)

    def forward(
        self,
        grid_features: Tensor,
        mesh_features: Tensor,
        mesh2mesh_edge_features: Tensor,
        grid2mesh_edge_features: Tensor,
        grid2mesh_edge_indices,
        mesh2grid_edge_features: Tensor,
        mesh2grid_edge_indices,
    ):
        embedded_grid_features = self.grid_feature_embedder(grid_features)
        embedded_mesh_features = self.mesh_feature_embedder(mesh_features)
        embedded_grid2mesh_edge_features = self.grid2mesh_edge_embedder(
            grid2mesh_edge_features
        )
        embedded_mesh2grid_edge_features = self.mesh2grid_edge_embedder(
            mesh2grid_edge_features
        )
        embedded_mesh2mesh_edge_features = self.mesh2mesh_edge_embedder(
            mesh2mesh_edge_features
        )

        return (
            embedded_grid_features,
            embedded_mesh_features,
            embedded_mesh2mesh_edge_features,
            embedded_grid2mesh_edge_features,
            grid2mesh_edge_indices,
            embedded_mesh2grid_edge_features,
            mesh2grid_edge_indices,
        )


class GraphCastProcessor(nn.Module):
    """Processor for the GraphCast model. The processor is responsible for
    processing the multi-mesh and updating the state of the forecast."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.processor = Processor(512, 512, 9, 512, 2, 512, 2)

    def forward(
        self,
        embedded_mesh_features: Tensor,
        embedded_mesh2mesh_edge_features: Tensor,
        mesh2mesh_edge_indices,
        embedded_grid2mesh_edge_features: Tensor,
        grid2mesh_edge_indices,
        embedded_mesh2grid_edge_features: Tensor,
        mesh2grid_edge_indices,
    ):
        return self.processor(
            embedded_mesh_features,
            embedded_mesh2mesh_edge_features,
            mesh2mesh_edge_indices,
            embedded_grid2mesh_edge_features,
            grid2mesh_edge_indices,
            embedded_mesh2grid_edge_features,
            mesh2grid_edge_indices,
        )


class GraphCastDecoder(nn.Module):
    """ """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.decoder = MLP(512, 73)

    def forward(self, x):
        return self.decoder(x)


class DGraphCast(nn.Module):
    """Main weather prediction model from the paper"""

    def __init__(self, cfg: Config, *args, **kwargs):
        super().__init__()
        self.encoder = GraphCastEncoder(*args, **kwargs)
        self.processor = GraphCastProcessor(*args, **kwargs)
        self.decoder = GraphCastDecoder(*args, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.processor(x)
        x = self.decoder(x)
        return x


class GraphWeatherForecaster(nn.Module):
    """Main weather prediction model from the paper"""

    def __init__(
        self,
        lat_lons: list,
        resolution: int = 2,
        feature_dim: int = 78,
        aux_dim: int = 24,
        output_dim: Optional[int] = None,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
    ):
        """
        Graph Weather Model based off https://arxiv.org/pdf/2202.07575.pdf

        Args:
            lat_lons: List of latitude and longitudes for the grid
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            feature_dim: Input feature size
            aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc)
            output_dim: Optional, output feature size, useful if want only subset of variables in
            output
            node_dim: Node hidden dimension
            edge_dim: Edge hidden dimension
            num_blocks: Number of message passing blocks in the Processor
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Use gradient checkpointing to reduce model memory
        """
        super().__init__()
        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim

        self.encoder = GraphCastEncoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=feature_dim + aux_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
        )
        self.processor = GraphCastProcessor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
        )
        self.decoder = GraphCastDecoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons_heights

        Returns:
            The next state in the forecast
        """
        x, edge_idx, edge_attr = self.encoder(features)
        x = self.processor(x, edge_idx, edge_attr)
        x = self.decoder(x, features[..., : self.feature_dim])
        return x
