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
from torch.utils.data import Dataset
from DGraph.Communicator import CommunicatorBase
from ogb.nodeproppred import NodePropPredDataset
from DGraph.data.graph import DistributedGraph
import numpy as np
import os

SUPPORTED_DATASETS = datasets = [
    "ogbn-arxiv",
    "ogbn-proteins",
    "ogbn-papers100M",
    "ogbn-products",
]


def process_homogenous_data(
    graph_data, labels, rank, world_Size, split_idx, partition_file, *args, **kwargs
) -> DistributedGraph:
    """For processing homogenous graph with node features, edge index and labels"""
    assert "node_feat" in graph_data, "Node features not found"
    assert "edge_index" in graph_data, "Edge index not found"
    assert "num_nodes" in graph_data, "Number of nodes not found"
    assert os.path.exists(partition_file), f"Partition file {partition_file} not found"
    assert graph_data["edge_feat"] is None, "Edge features not supported"

    node_features = torch.Tensor(graph_data["node_feat"]).float()
    edge_index = torch.Tensor(graph_data["edge_index"]).long()
    num_nodes = graph_data["num_nodes"]
    labels = torch.Tensor(labels).long()
    # For bidirectional graphs the number of edges are double counted
    num_edges = edge_index.shape[1]

    train_nodes = torch.from_numpy(split_idx["train"])
    valid_nodes = torch.from_numpy(split_idx["valid"])
    test_nodes = torch.from_numpy(split_idx["test"])

    partition_data = np.load(partition_file, allow_pickle=True)
    node_rank_mapping = partition_data["node_rank_mapping"]
    edge_rank_mapping = partition_data["edge_rank_mapping"]

    graph_obj = DistributedGraph(
        node_features=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_edges=num_edges,
        rank=rank,
        world_size=world_Size,
        labels=labels,
        train_mask=train_nodes,
        val_mask=valid_nodes,
        test_mask=test_nodes,
    )
    return graph_obj


class DistributedOGBWrapper(Dataset):
    def __init__(
        self,
        dname: str,
        comm_object: CommunicatorBase,
        cache_partitioning_file=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        assert (
            dname in SUPPORTED_DATASETS
        ), f"Dataset {dname} not supported. Supported datasets: {SUPPORTED_DATASETS}"

        assert comm_object._is_initialized, "Communicator not initialized"
        assert (
            cache_partitioning_file is not None
        ), "Either run partitioning or provide a cache directory"

        self.dname = dname
        self.comm_object = comm_object

        assert self.comm_object._is_initialized, "Communicator not initialized"

        self._rank = self.comm_object.get_rank()
        self._world_size = self.comm_object.get_world_size()

        self.dataset = NodePropPredDataset(
            name=dname,
        )
        graph_data, labels = self.dataset[0]

        self.split_idx = self.dataset.get_idx_split()
        assert self.split_idx is not None, "Split index not found"

        graph_obj = process_homogenous_data(
            graph_data,
            labels,
            self._rank,
            self._world_size,
            self.split_idx,
            partition_file=cache_partitioning_file,
            *args,
            **kwargs,
        )

        self.graph_obj = graph_obj

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        local_node_features = self.graph_obj.get_local_node_features()
        labels = self.graph_obj.get_local_labels()

        # TODO: Move this to a backend-specific collator in the future
        if self.comm_object.backend == "nccl":
            # Return Graph object with Rank placement data

            # NOTE: Two-sided comm needs all the edge indices not the local ones
            edge_indices = self.graph_obj.get_global_edge_indices()
            rank_mappings = self.graph_obj.get_global_rank_mappings()

            # A hack to get the local edge indices, will fix this later with
            # TensorDict as the underlying data object rather than tensors.
            # This is just for quick spin up of the experiments
            edge_indices.__setattr__("local_num_edges", self.graph_obj._edges_per_rank)
            rank_mappings.__setattr__("local_num_edges", self.graph_obj._edges_per_rank)

        else:
            # One-sided communication, no need for rank placement data

            edge_indices = self.graph_obj.get_local_edge_indices()
            rank_mappings = self.graph_obj.get_local_rank_mappings()

        return local_node_features, edge_indices, rank_mappings, labels
