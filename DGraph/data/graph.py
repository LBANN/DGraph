from typing import Optional
import torch


# Graph object to store and keep track of distributed graph data

# TODO: This is a simple implementation. We want to extend this in the future to
# support more complex graph data structures and behave like PyTorch's DTensor.


class DistributedGraph:
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.LongTensor,
        num_nodes: int,
        num_edges: int,
        rank: int,
        world_size: int,
        edge_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        test_mask: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
        pre_calculate_mapping: bool = False,
    ):
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
        self.rank = rank
        self.world_size = world_size
        self._nodes_per_rank = num_nodes // world_size
        self._edges_per_rank = num_edges // world_size
        if pre_calculate_mapping:
            self._precalculate_index_rank_mappings(edge_index)

    def _get_local_shape(self, dim: int) -> int:
        return dim // self.world_size

    def _get_padded_shape(self, dim: int) -> int:
        return self._get_local_shape(dim) * self.world_size

    def _get_global_start_index(self, dim: int) -> int:
        return self._get_local_shape(dim) * self.rank

    def _get_global_end_index(self, dim: int) -> int:
        return self._get_local_shape(dim) * (self.rank + 1)

    def get_global_node_feature_shape(self) -> tuple:
        """
        Returns the global shape of the node features tensor
        """
        node_feature_shape = self.node_features.shape
        padded_shape = (node_feature_shape[0] // self.world_size) * self.world_size
        return (padded_shape, *node_feature_shape[1:])

    def get_local_node_feature_shape(self) -> tuple:
        """
        Returns the local shape of the node features tensor
        """
        node_feature_shape = self.node_features.shape
        padded_shape = node_feature_shape[0] // self.world_size
        return (padded_shape, *node_feature_shape[1:])

    def get_global_edge_index_shape(self) -> tuple:
        """
        Returns the global shape of the edge index tensor
        """
        edge_index_shape = self.edge_index.shape
        padded_shape = (edge_index_shape[0] // self.world_size) * self.world_size
        return (padded_shape, *edge_index_shape[1:])

    def get_local_edge_index_shape(self) -> tuple:
        """
        Returns the local shape of the edge index tensor
        """
        edge_index_shape = self.edge_index.shape
        padded_shape = edge_index_shape[0] // self.world_size
        return (padded_shape, *edge_index_shape[1:])

    def _get_local_slice(
        self,
        _tensor: torch.Tensor,
        _start: int,
        _end: int,
    ) -> torch.Tensor:
        # This looks useless now but can be useful in the future. The plan is to
        # implement a distributed graph class for each backend (NCCL, MPI, etc.)
        # and this method will be useful then.
        return _tensor[_start:_end]

    def _get_global_slice(
        self,
        _tensor: torch.Tensor,
    ):
        # This looks useless now but can be useful in the future. The plan is to
        # implement a distributed graph class for each backend (NCCL, MPI, etc.)
        # and this method will be useful then.
        return _tensor

    def get_local_node_features(self):
        """"""

        global_node_feature_shape = self.get_local_node_feature_shape()
        _start_index = self._get_global_start_index(global_node_feature_shape[0])
        _end_index = self._get_global_end_index(global_node_feature_shape[0])

        return self._get_local_slice(self.node_features, _start_index, _end_index)

    def get_global_node_features(self):
        """"""
        return self._get_global_slice(self.node_features)

    def get_local_edge_indices(self):
        global_edge_index_shape = self.get_global_edge_index_shape()
        _start_index = self._get_global_start_index(global_edge_index_shape[0])
        _end_index = self._get_global_end_index(global_edge_index_shape[0])

        return self._get_local_slice(self.edge_index, _start_index, _end_index)

    def get_global_edge_indices(self):
        return self._get_global_slice(self.edge_index)

    def get_global_rank_mappings(self):

        return self.rank_mappings

    def get_local_rank_mappings(self):

        _start_index = self._get_global_start_index(self.rank_mappings.shape[0])
        _end_index = self._get_global_end_index(self.rank_mappings.shape[0])

        return self._get_local_slice(self.rank_mappings, _start_index, _end_index)

    def _make_push_graph_data(self):
        """Two-sided communication backends (NCCL) can only do push operations.
        This method prepares the additional data for push operations."""

        # We assume for the sake of simplicity that the graph data
        # is contiguous and equally divided among the ranks.
        # In the future, this can be made more general.
        self.node_to_rank_mapping = (
            torch.arange(0, self.num_nodes) // self._nodes_per_rank
        )
        self.edge_to_rank_mapping = (
            torch.arange(0, self.num_edges) // self._edges_per_rank
        )
        self.local_node_mapping = torch.arange(0, self.num_nodes) % self._nodes_per_rank
        self.local_edge_mapping = torch.arange(0, self.num_edges) % self._edges_per_rank

    def _get_index_to_rank_mapping(self, _indices):
        """Returns the rank mapping for the given indices"""
        return self.node_to_rank_mapping[_indices.int()]

    def _precalculate_index_rank_mappings(self, _indices):
        """Precalculates the rank mappings for the given indices and caches them.
        This is needed for two-sided communication backends like NCCL."""
        self._source_sender_ranks = self._get_index_to_rank_mapping(_indices[0, :])
        self._source_receiver_ranks = self._get_index_to_rank_mapping(_indices[1, :])
        self.rank_mappings = torch.stack(
            [self._source_sender_ranks, self._source_receiver_ranks], dim=0
        )

    def get_sender_receiver_ranks(self):
        """Returns the sender and receiver ranks for each edge"""
        return self._source_sender_ranks, self._source_receiver_ranks
