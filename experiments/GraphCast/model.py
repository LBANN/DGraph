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
from typing import Tuple
from torch import Tensor
from layers import MeshEdgeBlock, MeshGraphMLP, MeshNodeBlock
from graphcast_config import Config
from data_utils.graphcast_graph import DistributedGraphCastGraph
from DGraph.distributed import HaloExchange
from DGraph.distributed.commInfo import CommunicationPattern
from DGraph.utils.TimingReport import TimingReport


class GraphCastEmbedder(nn.Module):

    def __init__(self, cfg: Config, *args, **kwargs):
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__()
        grid_input_dim = cfg.model.input_grid_dim
        mesh_input_dim = cfg.model.input_mesh_dim

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

    def __init__(self, cfg: Config, comm, *args, **kwargs) -> None:
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__(*args, **kwargs)
        hidden_dim = cfg.model.hidden_dim

        self.exchanger = HaloExchange(comm)

        self.edge_mlp = MeshEdgeBlock(
            input_src_node_dim=hidden_dim,
            input_dst_node_dim=hidden_dim,
            input_edge_dim=hidden_dim,
            output_edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.mesh_node_mlp = MeshNodeBlock(
            input_node_dim=hidden_dim,
            input_edge_dim=hidden_dim,
            output_node_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.grid_node_mlp = MeshGraphMLP(
            input_dim=hidden_dim, output_dim=hidden_dim, hidden_dim=hidden_dim
        )

    def forward(
        self,
        grid_node_features: Tensor,
        mesh_node_features: Tensor,
        grid2mesh_edge_features: Tensor,
        comm_pattern: CommunicationPattern,
    ) -> Tuple[Tensor, Tensor]:
        # local_edge_list: [E, 2] with [neighbor=grid/halo, central=mesh] (col1 is
        # always the locally-owned aggregation target per CommunicationPattern's contract)
        edge_index = comm_pattern.local_edge_list
        src_indices = edge_index[:, 0]  # grid/halo (neighbor, message source)
        dst_indices = edge_index[:, 1]  # mesh (central, aggregation target)
        num_local = comm_pattern.num_local_vertices

        with TimingReport("encoder/halo_exchange"):
            halo_features = self.exchanger(grid_node_features, comm_pattern)
            augmented_grid = torch.cat([grid_node_features, halo_features], dim=0)

        with TimingReport("encoder/edge_block"):
            e_feats = self.edge_mlp(
                src_node_features=augmented_grid,
                dst_node_features=mesh_node_features,
                edge_features=grid2mesh_edge_features,
                src_indices=src_indices,
                dst_indices=dst_indices,
            )

        with TimingReport("encoder/node_block"):
            n_feats = self.mesh_node_mlp(
                node_features=mesh_node_features,
                edge_features=e_feats,
                src_indices=dst_indices,
            )

        with TimingReport("encoder/grid_mlp"):
            grid_node_features = grid_node_features + self.grid_node_mlp(grid_node_features)

        mesh_node_features = mesh_node_features + n_feats
        return grid_node_features, mesh_node_features


class GraphCastProcessor(nn.Module):
    """Processor for the GraphCast model. The processor is responsible for
    processing the multi-mesh and updating the state of the forecast."""

    def __init__(self, cfg: Config, comm, *args, **kwargs):
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__()
        hidden_dim = cfg.model.hidden_dim
        processor_layers = cfg.model.processor_layers

        self.exchanger = HaloExchange(comm)

        self.edge_processors = nn.ModuleList(
            [
                MeshEdgeBlock(
                    input_src_node_dim=hidden_dim,
                    input_dst_node_dim=hidden_dim,
                    input_edge_dim=hidden_dim,
                    output_edge_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(processor_layers)
            ]
        )
        self.node_processors = nn.ModuleList(
            [
                MeshNodeBlock(
                    input_node_dim=hidden_dim,
                    input_edge_dim=hidden_dim,
                    output_node_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(processor_layers)
            ]
        )

    def forward(
        self,
        embedded_mesh_features: Tensor,
        embedded_mesh2mesh_edge_features: Tensor,
        comm_pattern: CommunicationPattern,
    ) -> Tuple[Tensor, Tensor]:
        e_feats = embedded_mesh2mesh_edge_features
        n_feats = embedded_mesh_features

        # local_edge_list: [E, 2] with [neighbor=mesh_src, central=mesh_dst] (col1 is
        # always the locally-owned aggregation target per CommunicationPattern's contract)
        edge_index = comm_pattern.local_edge_list
        src_indices = edge_index[:, 0]  # neighbor (message source)
        dst_indices = edge_index[:, 1]  # central (aggregation target)
        num_local = comm_pattern.num_local_vertices

        for i, (edge_layer, node_layer) in enumerate(
            zip(self.edge_processors, self.node_processors)
        ):
            with TimingReport(f"processor/layer_{i}/halo_exchange"):
                halo_features = self.exchanger(n_feats, comm_pattern)
                augmented = torch.cat([n_feats, halo_features], dim=0)

            with TimingReport(f"processor/layer_{i}/edge_block"):
                e_feats = edge_layer(
                    src_node_features=augmented,
                    dst_node_features=augmented,
                    edge_features=e_feats,
                    src_indices=src_indices,
                    dst_indices=dst_indices,
                )

            with TimingReport(f"processor/layer_{i}/node_block"):
                n_feats = node_layer(
                    node_features=augmented[:num_local],
                    edge_features=e_feats,
                    src_indices=dst_indices,
                )

        return n_feats, e_feats


class GraphCastDecoder(nn.Module):
    """Decoder for the GraphCast model. The decoder is responsible for taking the latent
    state of the mesh graph and decoding it to a regular grid. Unlike the processor,
    the decoder works on the bipartite graph between the mesh to the grid.
    """

    def __init__(self, cfg: Config, comm, *args, **kwargs):
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__()
        hidden_dim = cfg.model.hidden_dim

        self.exchanger = HaloExchange(comm)

        self.edge_mlp = MeshEdgeBlock(
            input_src_node_dim=hidden_dim,
            input_dst_node_dim=hidden_dim,
            input_edge_dim=hidden_dim,
            output_edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.node_mlp = MeshNodeBlock(
            input_node_dim=hidden_dim,
            input_edge_dim=hidden_dim,
            output_node_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        mesh2grid_edge_features: Tensor,
        grid_node_features: Tensor,
        mesh_node_features: Tensor,
        comm_pattern: CommunicationPattern,
    ) -> Tensor:
        """
        Args:
            mesh2grid_edge_features (Tensor): The edge features from the mesh to the grid
            grid_node_features (Tensor): The grid node features
            mesh_node_features (Tensor): The mesh node features
            comm_pattern (CommunicationPattern): Precomputed communication pattern
                for the mesh2grid bipartite graph (partitioned by grid vertex placement).

        Returns:
            (Tensor): The updated grid node features
        """
        # local_edge_list: [E, 2] with [neighbor=mesh/halo, central=grid] (col1 is
        # always the locally-owned aggregation target per CommunicationPattern's contract)
        edge_index = comm_pattern.local_edge_list
        src_indices = edge_index[:, 0]  # mesh/halo (neighbor, message source)
        dst_indices = edge_index[:, 1]  # grid (central, aggregation target)
        num_local = comm_pattern.num_local_vertices

        with TimingReport("decoder/halo_exchange"):
            # Mesh nodes are the neighbors (sources); grid nodes are the central (destination).
            halo_mesh_features = self.exchanger(mesh_node_features, comm_pattern)
            augmented_mesh = torch.cat([mesh_node_features, halo_mesh_features], dim=0)

        with TimingReport("decoder/edge_block"):
            e_feats = self.edge_mlp(
                src_node_features=augmented_mesh,      # mesh features (local + halo)
                dst_node_features=grid_node_features,  # grid features (destination side)
                edge_features=mesh2grid_edge_features,
                src_indices=src_indices,
                dst_indices=dst_indices,
            )

        with TimingReport("decoder/node_block"):
            n_feats = self.node_mlp(
                node_features=grid_node_features[:num_local],  # local grid nodes being updated
                edge_features=e_feats,
                src_indices=dst_indices,
            )

        return grid_node_features + n_feats


class DGraphCast(nn.Module):
    """Main weather prediction model from the paper"""

    def __init__(self, cfg: Config, comm, *args, **kwargs):
        """
        Args:
            cfg: Config object
            comm: Communicator object
        """
        super().__init__()
        self.hidden_dim = cfg.model.hidden_dim
        self.output_grid_dim = cfg.model.output_grid_dim
        self.embedder = GraphCastEmbedder(cfg=cfg, *args, **kwargs)
        self.encoder = GraphCastEncoder(cfg=cfg, comm=comm, *args, **kwargs)
        self.processor = GraphCastProcessor(cfg=cfg, comm=comm, *args, **kwargs)
        self.decoder = GraphCastDecoder(cfg=cfg, comm=comm, *args, **kwargs)
        self.final_prediction = MeshGraphMLP(
            input_dim=self.hidden_dim,
            output_dim=self.output_grid_dim,
            hidden_dim=self.hidden_dim,
        )

    def forward(
        self, input_grid_features: Tensor, static_graph: DistributedGraphCastGraph
    ) -> Tensor:
        """
        Args:
            input_grid_features (Tensor): The input grid features
            static_graph (DistributedGraphCastGraph): The static graph object

        Returns:
            (Tensor): The predicted output grid
        """
        input_grid_features = input_grid_features.squeeze(0)
        input_mesh_features = static_graph.mesh_graph_node_features
        mesh2mesh_edge_features = static_graph.mesh_graph_edge_features
        grid2mesh_edge_features = static_graph.grid2mesh_graph_edge_features
        mesh2grid_edge_features = static_graph.mesh2grid_graph_edge_features
        comm_patterns = static_graph.distributed_comm_patterns

        with TimingReport("model/embed"):
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

        with TimingReport("model/encode"):
            encoded_grid_features, encoded_mesh_features = self.encoder(
                embedded_grid_features,
                embedded_mesh_features,
                embedded_grid2mesh_edge_features,
                comm_patterns.grid2mesh,
            )

        with TimingReport("model/process"):
            processed_mesh_node_features, _ = self.processor(
                encoded_mesh_features,
                embedded_mesh2mesh_edge_features,
                comm_patterns.mesh,
            )

        with TimingReport("model/decode"):
            x = self.decoder(
                embedded_mesh2grid_edge_features,
                encoded_grid_features,
                processed_mesh_node_features,
                comm_patterns.mesh2grid,
            )

        with TimingReport("model/final_prediction"):
            output = self.final_prediction(x)
        output = input_grid_features + output
        return output
