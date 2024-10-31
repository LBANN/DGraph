from typing import Tuple

import einops
import h3
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """MLP for graph processing"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: Optional[str] = "LayerNorm",
        use_checkpointing: bool = False,
    ):
        """
        MLP

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type one of 'LayerNorm', 'GraphNorm',
                'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing or not
        """

        super(MLP, self).__init__()
        self.use_checkpointing = use_checkpointing

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the MLP

        Args:
            x: Node or edge features

        Returns:
            The transformed tensor
        """
        if self.use_checkpointing:
            out = checkpoint(self.model, x, use_reentrant=False)
        else:
            out = self.model(x)
        return out


class NodeProcessor(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
    ):
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """

        super(NodeProcessor, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        u=None,
        batch=None,
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        row, col = edge_index
        scatter_dim = 0
        output_size = x.size(scatter_dim)
        # aggregate edge message by target
        out = scatter_sum(edge_attr, col, dim=scatter_dim, dim_size=output_size)
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        out += x  # residual connection

        return out


class Processor(torch.nn.Module):
    """Processor for latent graphD"""

    def __init__(
        self,
        input_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: str = "LayerNorm",
    ):
        """
        Latent graph processor

        Args:
            input_dim: Input dimension for the node
            edge_dim: Edge input dimension
            num_blocks: Number of message passing blocks
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        # Build the default graph
        # Take features from encoder and put into processor graph
        self.input_dim = input_dim

        self.graph_processor = GraphProcessor(
            num_blocks,
            input_dim,
            edge_dim,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
        )

    def forward(self, x: torch.Tensor, edge_index, edge_attr) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            x: Torch tensor containing node features
            edge_index: Connectivity of graph, of shape [2, Num edges] in COO format
            edge_attr: Edge attribues in [Num edges, Features] shape

        Returns:
            torch Tensor containing the values of the nodes of the graph
        """
        out, _ = self.graph_processor(x, edge_index, edge_attr)
        return out


class Encoder(torch.nn.Module):
    """Encoder graph model"""

    def __init__(
        self,
        lat_lons: list,
        resolution: int = 2,
        input_dim: int = 78,
        output_dim: int = 256,
        output_edge_dim: int = 256,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
    ):
        """
        Encode the lat/lon data inot the isohedron graph

        Args:
            lat_lons: List of (lat,lon) points
            resolution: H3 resolution level
            input_dim: Input node dimension
            output_dim: Output node dimension
            output_edge_dim: Edge dimension
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing to use less memory
        """
        super().__init__()
        self.output_dim = output_dim
        self.num_latlons = len(lat_lons)
        self.base_h3_grid = sorted(
            list(h3.uncompact(h3.get_res0_indexes(), resolution))
        )
        self.base_h3_map = {h_i: i for i, h_i in enumerate(self.base_h3_grid)}
        self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        self.h3_mapping = {}
        h_index = len(self.base_h3_grid)
        for h in self.base_h3_grid:
            if h not in self.h3_mapping:
                h_index -= 1
                self.h3_mapping[h] = h_index + self.num_latlons
        # Now have the h3 grid mapping, the bipartite graph of edges connecting lat/lon to h3 nodes
        # Should have vertical and horizontal difference
        self.h3_distances = []
        for idx, h3_point in enumerate(self.h3_grid):
            lat_lon = lat_lons[idx]
            distance = h3.point_dist(lat_lon, h3.h3_to_geo(h3_point), unit="rads")
            self.h3_distances.append([np.sin(distance), np.cos(distance)])
        self.h3_distances = torch.tensor(self.h3_distances, dtype=torch.float)
        # Compress to between 0 and 1

        # Build the default graph
        # lat_nodes = torch.zeros((len(lat_lons_heights), input_dim), dtype=torch.float)
        # h3_nodes = torch.zeros((h3.num_hexagons(resolution), output_dim), dtype=torch.float)
        # Get connections between lat nodes and h3 nodes
        edge_sources = []
        edge_targets = []
        for node_index, lat_node in enumerate(self.h3_grid):
            edge_sources.append(node_index)
            edge_targets.append(self.h3_mapping[lat_node])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

        # Use homogenous graph to make it easier
        self.graph = Data(edge_index=edge_index, edge_attr=self.h3_distances)

        self.latent_graph = self.create_latent_graph()

        # Extra starting ones for appending to inputs, could 'learn' good starting points
        self.h3_nodes = torch.nn.Parameter(
            torch.zeros((h3.num_hexagons(resolution), input_dim), dtype=torch.float)
        )
        # Output graph

        self.node_encoder = MLP(
            input_dim,
            output_dim,
            hidden_dim_processor_node,
            hidden_layers_processor_node,
            mlp_norm_type,
            self.use_checkpointing,
        )
        self.edge_encoder = MLP(
            2,
            output_edge_dim,
            hidden_dim_processor_edge,
            hidden_layers_processor_edge,
            mlp_norm_type,
            self.use_checkpointing,
        )
        self.latent_edge_encoder = MLP(
            2,
            output_edge_dim,
            hidden_dim_processor_edge,
            hidden_layers_processor_edge,
            mlp_norm_type,
            self.use_checkpointing,
        )
        self.graph_processor = GraphProcessor(
            1,
            output_dim,
            output_edge_dim,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type,
        )

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:
            Torch tensors of node features, latent graph edge index, and latent edge attributes
        """
        batch_size = features.shape[0]
        self.h3_nodes = self.h3_nodes.to(features.device)
        self.graph = self.graph.to(features.device)
        self.latent_graph = self.latent_graph.to(features.device)
        features = torch.cat(
            [features, einops.repeat(self.h3_nodes, "n f -> b n f", b=batch_size)],
            dim=1,
        )
        # Cat with the h3 nodes to have correct amount of nodes, and in right order
        features = einops.rearrange(features, "b n f -> (b n) f")
        out = self.node_encoder(features)  # Encode to 256 from 78
        edge_attr = self.edge_encoder(
            self.graph.edge_attr
        )  # Update attributes based on distance
        # Copy attributes batch times
        edge_attr = einops.repeat(edge_attr, "e f -> (repeat e) f", repeat=batch_size)
        # Expand edge index correct number of times while adding the proper number to the edge index
        edge_index = torch.cat(
            [
                self.graph.edge_index + i * torch.max(self.graph.edge_index) + i
                for i in range(batch_size)
            ],
            dim=1,
        )
        out, _ = self.graph_processor(out, edge_index, edge_attr)  # Message Passing
        # Remove the extra nodes (lat/lon) from the output
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        _, out = torch.split(out, [self.num_latlons, self.h3_nodes.shape[0]], dim=1)
        out = einops.rearrange(out, "b n f -> (b n) f")
        return (
            out,
            torch.cat(
                [
                    self.latent_graph.edge_index
                    + i * torch.max(self.latent_graph.edge_index)
                    + i
                    for i in range(batch_size)
                ],
                dim=1,
            ),
            self.latent_edge_encoder(
                einops.repeat(
                    self.latent_graph.edge_attr,
                    "e f -> (repeat e) f",
                    repeat=batch_size,
                )
            ),
        )  # New graph

    def create_latent_graph(self):
        """
        Copies over and generates a Data object for the processor to use

        Returns:
            The connectivity and edge attributes for the latent graph
        """
        # Get connectivity of the graph
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        for h3_index in self.base_h3_grid:
            h_points = h3.k_ring(h3_index, 1)
            for h in h_points:  # Already includes itself
                distance = h3.point_dist(
                    h3.h3_to_geo(h3_index), h3.h3_to_geo(h), unit="rads"
                )
                edge_attrs.append([np.sin(distance), np.cos(distance)])
                edge_sources.append(self.base_h3_map[h3_index])
                edge_targets.append(self.base_h3_map[h])
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        return Data(edge_index=edge_index, edge_attr=edge_attrs)
