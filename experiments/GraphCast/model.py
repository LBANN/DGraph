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
from typing import Optional, Tuple
from torch import Tensor
from layers import MeshEdgeBlock, MeshGraphMLP, MeshNodeBlock
from graphcast_config import Config
from data_utils.graphcast_graph import DistributedGraphCastGraph


class GraphCastEmbedder(nn.Module):

    def __init__(self, cfg, *args, **kwargs):
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__()
        grid_input_dim = cfg.model.grid_input_dim
        mesh_input_dim = cfg.model.mesh_input_dim

        input_edge_dim = cfg.model.input_edge_dim
        hidden_dim = cfg.model.hidden_dim

        self.grid_input_dim = grid_input_dim
        self.mesh_input_dim = mesh_input_dim
        self.hidden_dim = hidden_dim

        # MLP for grid node features
        self.grid_feature_embedder = MeshGraphMLP(
            input_dim=grid_input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=1,
        )
        # MLP for mesh node features
        self.mesh_feature_embedder = MeshGraphMLP(
            input_dim=mesh_input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=1,
        )
        # MLP for grid2mesh edge features
        self.grid2mesh_edge_embedder = MeshGraphMLP(
            input_dim=input_edge_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=1,
        )
        # MLP for mesh2grid edge features
        self.mesh2grid_edge_embedder = MeshGraphMLP(
            input_dim=input_edge_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=1,
        )
        # MLP for mesh2mesh edge features
        self.mesh2mesh_edge_embedder = MeshGraphMLP(
            input_dim=input_edge_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=1,
        )

    def forward(
        self,
        grid_features: Tensor,
        mesh_features: Tensor,
        mesh2mesh_edge_features: Tensor,
        grid2mesh_edge_features: Tensor,
        mesh2grid_edge_features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            embedded_mesh2grid_edge_features,
        )


class GraphCastEncoder(nn.Module):
    """Encoder for the GraphCast model. The encoder is responsible for taking grid
    information and encoding it into the multi-mesh, which the processor uses."""

    def __init__(self, cfg, comm, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        edge_block_invars = (
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            comm,
            cfg.model.hidden_dim,
        )
        self.edge_mlp = MeshEdgeBlock(*edge_block_invars)

        node_block_invars = (
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            comm,
            cfg.model.hidden_dim,
        )
        self.mesh_node_mlp = MeshNodeBlock(*node_block_invars)
        self.grid_node_mlp = MeshGraphMLP(
            input_dim=cfg.model.hidden_dim, output_dim=cfg.model.hidden_dim
        )

    def forward(
        self,
        grid_node_features,
        mesh_node_features,
        grid2mesh_edge_features,
        grid2mesh_edge_indices_src,
        grid2mesh_edge_indices_dst,
    ):
        e_feats = self.edge_mlp(
            mesh_node_features,
            grid_node_features,
            grid2mesh_edge_features,
            grid2mesh_edge_indices_src,
            grid2mesh_edge_indices_dst,
        )

        n_feats = self.node_mlp(
            mesh_node_features,
            e_feats,
            grid2mesh_edge_indices_dst,
            grid2mesh_edge_indices_dst,
        )

        mesh_node_features = mesh_node_features + n_feats
        grid_node_features = grid_node_features + self.grid_node_mlp(grid_node_features)

        return grid_node_features, mesh_node_features


class GraphCastProcessor(nn.Module):
    """Processor for the GraphCast model. The processor is responsible for
    processing the multi-mesh and updating the state of the forecast."""

    def __init__(self, cfg, comm, *args, **kwargs):
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__()
        processor_layers = cfg.model.processor_layers
        node_block_invars = (
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            comm,
            cfg.model.hidden_dim,
        )
        edge_block_invars = (
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            comm,
            cfg.model.hidden_dim,
        )
        edge_layers = []
        node_layers = []
        for _ in range(processor_layers):
            edge_layers.append(MeshEdgeBlock(*edge_block_invars))
        for _ in range(processor_layers):
            node_layers.append(MeshNodeBlock(*node_block_invars))

        self.edge_processors = nn.ModuleList(edge_layers)
        self.node_processors = nn.ModuleList(node_layers)

    def forward(
        self,
        embedded_mesh_features: Tensor,
        embedded_mesh2mesh_edge_features: Tensor,
        mesh2mesh_edge_indices_src: Tensor,
        mesh2mesh_edge_indices_dst: Tensor,
    ):
        e_feats = embedded_mesh2mesh_edge_features
        n_feats = embedded_mesh_features
        for edge_layer, node_layer in zip(self.edge_processors, self.node_processors):
            e_feats = edge_layer(
                n_feats,
                n_feats,
                e_feats,
                mesh2mesh_edge_indices_src,
                mesh2mesh_edge_indices_dst,
            )
            n_feats = node_layer(
                n_feats, e_feats, mesh2mesh_edge_indices_src, mesh2mesh_edge_indices_dst
            )
        return n_feats, e_feats


class GraphCastDecoder(nn.Module):
    """Decoder for the GraphCast model. The decoder is responsible for taking the latent
    state of the mesh graph and decoding it to a regular grid. Unlike the processor,
    the decoder works on the bipartite graph between the mesh to the grid.
    """

    def __init__(self, cfg, comm, *args, **kwargs):
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__()
        edge_block_invars = (
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            cfg.model.hidden_dim,
            comm,
            cfg.model.hidden_dim,
        )
        self.edge_mlp = MeshEdgeBlock(*edge_block_invars)
        dst_node_input_dim = cfg.model.hidden_dim
        dst_node_output_dim = cfg.model.hidden_dim
        m2g_edge_output_dim = cfg.model.hidden_dim
        self.node_mlp = MeshGraphMLP(
            input_dim=dst_node_input_dim + m2g_edge_output_dim,
            output_dim=dst_node_output_dim,
            hidden_dim=cfg.model.hidden_dim,
            hidden_layers=1,
        )

    def forward(
        self,
        mesh2grid_edge_features,
        grid_node_features,
        mesh_node_features,
        mesh2graph_edge_indices_src,
        mesh2graph_edge_indices_dst,
    ):
        e_feats = self.edge_mlp(
            grid_node_features,
            mesh_node_features,
            mesh2grid_edge_features,
            mesh2graph_edge_indices_src,
            mesh2graph_edge_indices_dst,
        )
        n_feats = self.node_mlp(
            mesh_node_features,
            e_feats,
            mesh2graph_edge_indices_src,
            mesh2graph_edge_indices_dst,
        )

        n_feats = grid_node_features + n_feats
        return n_feats


class DGraphCast(nn.Module):
    """Main weather prediction model from the paper"""

    def __init__(self, cfg: Config, comm, *args, **kwargs):
        super().__init__()
        self.hidden_dim = cfg.model.hidden_dim
        self.output_grid_dim = cfg.model.output_grid_dim
        self.comm = comm
        self.embedder = GraphCastEmbedder(cfg=cfg, comm=comm, *args, **kwargs)
        self.encoder = GraphCastEncoder(cfg=cfg, comm=comm, *args, **kwargs)
        self.processor = GraphCastProcessor(*args, **kwargs)
        self.decoder = GraphCastDecoder(*args, **kwargs)
        self.final_prediction = MeshGraphMLP(
            input_dim=self.hidden_dim, output_dim=self.output_grid_dim
        )

    def forward(
        self, input_grid_features: Tensor, static_graph: DistributedGraphCastGraph
    ) -> Tensor:

        input_grid_features = input_grid_features.squeeze(0)
        input_mesh_features = static_graph.mesh_graph_node_features
        mesh2mesh_edge_features = static_graph.mesh_graph_edge_features
        grid2mesh_edge_features = static_graph.grid2mesh_graph_edge_features
        mesh2grid_edge_features = static_graph.mesh2grid_graph_edge_features
        mesh2mesh_edge_indices_src = static_graph.mesh_graph_src_indices
        mesh2mesh_edge_indices_dst = static_graph.mesh_graph_dst_indices
        mesh2grid_edge_indices_src = static_graph.mesh2grid_graph_src_indices
        mesh2grid_edge_indices_dst = static_graph.mesh2grid_graph_dst_indices
        grid2mesh_edge_indices_src = static_graph.grid2mesh_graph_src_indices
        grid2mesh_edge_indices_dst = static_graph.grid2mesh_graph_dst_indices

        out = self.embedder(
            input_grid_features,
            input_mesh_features,
            mesh2mesh_edge_features,
            grid2mesh_edge_features,
            mesh2grid_edge_features,
        )
        (
            embedded_grid_features,
            embedded_mesh_features,
            embedded_mesh2mesh_edge_features,
            embedded_grid2mesh_edge_features,
            embedded_mesh2grid_edge_features,
        ) = out
        encoded_grid_features, encoded_mesh_features = self.encoder(
            embedded_grid_features,
            embedded_mesh_features,
            embedded_grid2mesh_edge_features,
            grid2mesh_edge_indices_src,
            grid2mesh_edge_indices_dst,
        )

        out = self.processor(
            encoded_mesh_features,
            embedded_mesh2mesh_edge_features,
            mesh2mesh_edge_indices_src,
            mesh2mesh_edge_indices_dst,
        )
        processed_mesh_node_features, _ = out
        x = self.decoder(
            embedded_mesh2grid_edge_features,
            encoded_grid_features,
            processed_mesh_node_features,
            mesh2grid_edge_indices_src,
            mesh2grid_edge_indices_dst,
        )
        output = self.final_prediction(x)
        return output
