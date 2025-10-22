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
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from DGraph.Communicator import CommunicatorBase
from ogb.nodeproppred import NodePropPredDataset
from DGraph.data.graph import DistributedGraph
from DGraph.data.graph import get_round_robin_node_rank_map
import numpy as np
import os
import torch.distributed as dist

SUPPORTED_DATASETS = [
    "ogbn-arxiv",
    "ogbn-proteins",
    "ogbn-papers100M",
    "ogbn-products",
]

num_classes = {
    "ogbn-arxiv": 40,
    "ogbn-proteins": 112,
    "ogbn-papers100M": 172,
    "ogbn-products": 47,
}


def node_renumbering(node_rank_placement) -> Tuple[torch.Tensor, torch.Tensor]:
    """The nodes are renumbered based on the rank mappings so the node features and
    numbers are contiguous."""

    contiguous_rank_mapping, renumbered_nodes = torch.sort(node_rank_placement)
    return renumbered_nodes, contiguous_rank_mapping


def edge_renumbering(
    edge_indices, renumbered_nodes, vertex_mapping, edge_features=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    src_indices = edge_indices[0, :]
    dst_indices = edge_indices[1, :]
    src_indices = renumbered_nodes[src_indices]
    dst_indices = renumbered_nodes[dst_indices]

    edge_src_rank_mapping = vertex_mapping[src_indices]
    edge_dest_rank_mapping = vertex_mapping[dst_indices]

    sorted_src_rank_mapping, sorted_indices = torch.sort(edge_src_rank_mapping)
    dst_indices = dst_indices[sorted_indices]
    src_indices = src_indices[sorted_indices]

    sorted_dest_rank_mapping = edge_dest_rank_mapping[sorted_indices]

    if edge_features is not None:
        # Sort the edge features based on the sorted indices
        edge_features = edge_features[sorted_indices]

    return (
        torch.stack([src_indices, dst_indices], dim=0),
        sorted_src_rank_mapping,
        sorted_dest_rank_mapping,
        edge_features,
    )


def process_homogenous_data(
    graph_data,
    labels,
    rank: int,
    world_Size: int,
    split_idx: dict,
    node_rank_placement: torch.Tensor,
    *args,
    **kwargs,
) -> DistributedGraph:
    """For processing homogenous graph with node features, edge index and labels"""
    assert "node_feat" in graph_data, "Node features not found"
    assert "edge_index" in graph_data, "Edge index not found"
    assert "num_nodes" in graph_data, "Number of nodes not found"
    assert graph_data["edge_feat"] is None, "Edge features not supported"

    node_features = torch.Tensor(graph_data["node_feat"]).float()
    edge_index = torch.Tensor(graph_data["edge_index"]).long()
    num_nodes = graph_data["num_nodes"]
    labels = torch.Tensor(labels).long()
    # For bidirectional graphs the number of edges are double counted
    num_edges = edge_index.shape[1]

    assert node_rank_placement.shape[0] == num_nodes, "Node mapping mismatch"
    assert "train" in split_idx, "Train mask not found"
    assert "valid" in split_idx, "Validation mask not found"
    assert "test" in split_idx, "Test mask not found"

    train_nodes = torch.from_numpy(split_idx["train"])
    valid_nodes = torch.from_numpy(split_idx["valid"])
    test_nodes = torch.from_numpy(split_idx["test"])

    # Renumber the nodes and edges to make them contiguous
    renumbered_nodes, contiguous_rank_mapping = node_renumbering(node_rank_placement)
    node_features = node_features[renumbered_nodes]

    # Sanity check to make sure we placed the nodes in the correct spots

    assert torch.all(node_rank_placement[renumbered_nodes] == contiguous_rank_mapping)

    # First renumber the edges
    # Then we calculate the location of the source and destination vertex of each edge
    # based on the rank mapping
    # Then we sort the edges based on the source vertex rank mapping
    # When determining the location of the edge, we use the rank of the source vertex
    # as the location of the edge

    edge_index, edge_rank_mapping, edge_dest_rank_mapping, _ = edge_renumbering(
        edge_index, renumbered_nodes, contiguous_rank_mapping, edge_features=None
    )

    train_nodes = renumbered_nodes[train_nodes]
    valid_nodes = renumbered_nodes[valid_nodes]
    test_nodes = renumbered_nodes[test_nodes]

    labels = labels[renumbered_nodes]

    graph_obj = DistributedGraph(
        node_features=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_loc=contiguous_rank_mapping.long(),
        edge_loc=edge_rank_mapping.long(),
        edge_dest_rank_mapping=edge_dest_rank_mapping.long(),
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
        dir_name: Optional[str] = None,
        node_rank_placement: Optional[torch.Tensor] = None,
        force_reprocess: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        assert (
            dname in SUPPORTED_DATASETS
        ), f"Dataset {dname} not supported. Supported datasets: {SUPPORTED_DATASETS}"

        assert comm_object._is_initialized, "Communicator not initialized"

        self.dname = dname
        self.num_classes = num_classes[dname]
        self.comm_object = comm_object

        assert self.comm_object._is_initialized, "Communicator not initialized"

        self._rank = self.comm_object.get_rank()
        self._world_size = self.comm_object.get_world_size()

        comm_object.barrier()
        # Load the dataset on rank 0
        if comm_object.get_rank() == 0:
            self.dataset = NodePropPredDataset(
                name=dname,
            )
        # Block until rank 0 loads and processe the data
        # For the first time, the code downloads and processes the data
        # doing that on all ranks causes a race condition
        comm_object.barrier()
        # Load the dataset on all other ranks
        # This is to use the processed data that was generated by rank 0
        # This should account for a race condition

        if comm_object.get_rank() != 0:
            self.dataset = NodePropPredDataset(
                name=dname,
            )
        comm_object.barrier()
        graph_data, labels = self.dataset[0]

        self.split_idx = self.dataset.get_idx_split()
        assert self.split_idx is not None, "Split index not found"

        dir_name = dir_name if dir_name is not None else os.getcwd() + "/data"

        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        cached_graph_file = f"{dir_name}/{dname}_graph_data_{self._world_size}.pt"

        if os.path.exists(cached_graph_file) and not force_reprocess:
            graph_obj = torch.load(cached_graph_file)
        else:
            if node_rank_placement is None:
                if self._rank == 0:
                    print(f"Node rank placement not provided, generating a round robin placement")
                node_rank_placement = get_round_robin_node_rank_map(
                    graph_data["num_nodes"], self._world_size
                )

            graph_obj = process_homogenous_data(
                graph_data,
                labels,
                self._rank,
                self._world_size,
                self.split_idx,
                node_rank_placement=node_rank_placement,
                *args,
                **kwargs,
            )

            if self._rank == 0:
                print(f"Saving the processed graph data to {cached_graph_file}")
                torch.save(graph_obj, cached_graph_file)

        self.graph_obj = graph_obj

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        rank = self.comm_object.get_rank()
        local_node_features = self.graph_obj.get_local_node_features(rank=rank)
        labels = self.graph_obj.get_local_labels(rank=rank)

        # TODO: Move this to a backend-specific collator in the future
        if self.comm_object.backend == "nccl":
            # Return Graph object with Rank placement data

            # NOTE: Two-sided comm needs all the edge indices not the local ones
            edge_indices = self.graph_obj.get_global_edge_indices()
            rank_mappings = self.graph_obj.get_global_rank_mappings()
        else:
            # One-sided communication, no need for rank placement data

            edge_indices = self.graph_obj.get_local_edge_indices(rank=rank)
            rank_mappings = self.graph_obj.get_local_rank_mappings(rank=rank)

        return local_node_features, edge_indices, rank_mappings, labels
