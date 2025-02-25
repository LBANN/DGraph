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
from typing import Optional
import torch
import torch.distributed as dist

# Graph object to store and keep track of distributed graph data

# TODO: This is a simple implementation. We want to extend this in the future to
# support more complex graph data structures and behave like PyTorch's DTensor.


class DistributedGraph:
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        node_loc: torch.Tensor,
        edge_loc: torch.Tensor,
        edge_dest_rank_mapping: torch.Tensor,
        num_nodes: int,
        num_edges: int,
        world_size: int,
        edge_features: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        test_mask: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
    ):
        """A Distributed Graph Object to store and keep track of a single graph
        distributed across multiple ranks.

        Args:
            node_features (torch.Tensor): Node features tensor of
                                          shape (num_nodes, num_node_features)
            edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges)
            labels (torch.Tensor): Node labels tensor of shape (num_nodes,)
            node_loc (torch.Tensor): Tensor representing rank mapping for each node
                                    of shape (num_nodes,)
            edge_loc (torch.Tensor): Tensor representing rank mapping for each edge
                                    of shape (num_edges,)
            num_nodes (int): Number of nodes in the graph
            num_edges (int): Number of edges in the graph
        """

        assert node_features.dim() == 2, "Invalid node features shape. Expect 2D tensor"
        assert edge_index.dim() == 2, "Invalid edge index shape. Expect 2D tensor"
        assert node_loc.dim() == 1, "Invalid node_loc shape. Expect 1D tensor"
        assert edge_loc.dim() == 1, "Invalid edge_loc shape. Expect 1D tensor"
        assert node_features.shape[0] == num_nodes, (
            "Invalid node features shape. "
            + f"Expected shape: (num_nodes, num_node_features) got {node_features.shape}"
        )
        assert edge_index.shape[1] == num_edges, (
            "Invalid edge index shape. "
            + f"Expected shape: (2, num_edges) got {edge_index.shape}"
        )
        assert node_loc.shape[0] == num_nodes, (
            "Invalid node_loc shape. "
            + f"Expected shape: (num_nodes,) got {node_loc.shape}"
        )
        assert edge_loc.shape[0] == num_edges, (
            "Invalid edge_loc shape. "
            + f"Expected shape: (num_edges,) got {edge_loc.shape}"
        )

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.graph_labels = graph_labels
        self.world_size = world_size
        self._nodes_per_rank = node_loc.bincount()
        self._edges_per_rank = edge_loc.bincount()
        self.node_loc = node_loc
        self.edge_loc = edge_loc

        self.max_node_per_rank = int(self._nodes_per_rank.max().item())
        self.max_edge_per_rank = int(self._edges_per_rank.max().item())
        self.edge_dest_rank_mapping = edge_dest_rank_mapping
        self.rank_mappings = torch.cat(
            [self.edge_loc.unsqueeze(0), self.edge_dest_rank_mapping.unsqueeze(0)],
            dim=0,
        )

    def get_nodes_per_rank(self) -> torch.Tensor:
        """Returns the number of nodes per rank based on the rank mapping

        Returns:
            torch.Tensor: Number of nodes per rank tensor
        """
        return self._nodes_per_rank

    def get_edges_per_rank(self) -> torch.Tensor:
        """Returns the number of edges per rank based on the rank mapping

        Returns:
            torch.Tensor: Number of edges per rank tensor
        """
        return self._edges_per_rank

    def get_max_node_per_rank(self) -> int:
        """Returns the maximum number of nodes per rank according to the rank mapping

        Returns:
            int: Maximum number of nodes per rank
        """

        if self.max_node_per_rank is None:
            nodes_per_rank = self.node_loc.bincount()
            self.max_node_per_rank = int(nodes_per_rank.max().item())
        return self.max_node_per_rank

    def get_max_edge_per_rank(self) -> int:
        """Returns the maximum number of edges per rank according to the rank mapping

        Returns:
            int: Maximum number of edges per rank
        """

        if self.max_edge_per_rank is None:
            edges_per_rank = self.edge_loc.bincount()
            self.max_edge_per_rank = int(edges_per_rank.max().item())
        return self.max_edge_per_rank

    def get_local_node_features(self, rank) -> torch.Tensor:
        """
        Returns the local node features for the current rank based on the rank mapping

        Returns:
            torch.Tensor: Local node features tensor of shape
                         (num_local_nodes, num_node_features)
        """

        rank_mask = self.node_loc == rank
        local_node_features = self.node_features[rank_mask, :]
        return local_node_features

    def get_global_node_features(self) -> torch.Tensor:
        """
        Returns the global node features

        Returns:
            torch.Tensor: Global node features tensor of
                  shape (num_nodes, num_node_features)
        """
        return self.node_features

    def get_local_edge_indices(self, rank) -> torch.Tensor:
        """
        Returns the local edge indices for the current rank based on the rank mapping

        Returns:
            torch.Tensor: Local edge indices tensor of shape (2, num_local_edges)
        """

        rank_mask = self.edge_loc == rank
        local_edge_index = self.edge_index[:, rank_mask]
        return local_edge_index

    def get_global_edge_indices(self) -> torch.Tensor:
        """
        Returns the global edge indices in COO format

        Returns:
            torch.Tensor: Global edge indices tensor of shape (2, num_edges)
        """
        return self.edge_index

    def get_global_rank_mappings(self) -> torch.Tensor:
        """
        Returns the global rank mappings for the edge indices in the form
        (source_rank, destination_rank) for each edge.

        Returns:
            torch.Tensor: Global rank mappings tensor of shape (2, num_edges)
        """
        return self.rank_mappings

    def get_local_rank_mappings(self, rank) -> torch.Tensor:
        """
        Returns the rank mappings for local edges in this rank in the form
        (source_rank, destination_rank) for each edge.

        Returns:
            torch.Tensor: Local rank mappings tensor of shape (2, num_local_edges)
        """

        local_edge_mask = self.edge_loc == rank
        return self.rank_mappings[:, local_edge_mask]

    def get_local_labels(self, rank) -> torch.Tensor:
        """
        Returns the local labels for the current rank based on the rank mapping

        Returns:
            torch.Tensor: Local labels tensor
        """
        rank_mask = self.node_loc == rank
        local_labels = self.labels[rank_mask]

        return local_labels

    def get_global_labels(self) -> torch.Tensor:
        return self.labels

    def get_local_mask(self, mask: str, rank) -> torch.Tensor:
        """
        Returns the local mask for the given mask type based on the rank mapping.
        The local slice is calculated based on the range of nodes local to the
        current rank.

        Args:
            mask (str): Mask type. Can be "train", "val", or "test"

        Returns:
            torch.Tensor: Local mask tensor
        """

        nodes_per_rank = self.node_loc.bincount()

        local_node_start = 0 if rank == 0 else nodes_per_rank[:rank].sum().item()
        local_node_end = nodes_per_rank[: rank + 1].sum().item()

        if mask == "train":
            assert self.train_mask is not None, "Train mask not found"
            _mask = self.train_mask
        elif mask == "val":
            assert self.val_mask is not None, "Val mask not found"
            _mask = self.val_mask
        elif mask == "test":
            assert self.test_mask is not None, "Test mask not found"
            _mask = self.test_mask
        else:
            raise ValueError(f"Invalid mask {mask}")

        _mask = _mask.int()

        _mask_rank = (_mask < local_node_end) & (_mask >= local_node_start)

        local_mask = _mask[_mask_rank] % self._nodes_per_rank[rank]
        return local_mask

    def _get_index_to_rank_mapping(self, _indices):
        """Returns the rank mapping for the given indices"""
        return self.node_loc[_indices.int()]

    def get_sender_receiver_ranks(self):
        """Returns the sender and receiver ranks for each edge"""
        return self.edge_loc, self.edge_dest_rank_mapping
