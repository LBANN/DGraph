# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
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
"""
Dataset class that returns OGB graphs as torch tensors with pre-computed
NCCLGraphCommPlan for distributed gather-scatter operations.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import os

import torch
from torch.utils.data import Dataset
from ogb.nodeproppred import NodePropPredDataset

from DGraph.Communicator import CommunicatorBase
from DGraph.data.graph import DistributedGraph, get_round_robin_node_rank_map
from DGraph.data.ogbn_datasets import process_homogenous_data
from DGraph.distributed.nccl._NCCLCommPlan import (
    NCCLGraphCommPlan,
    COO_to_NCCLCommPlan,
)


SUPPORTED_DATASETS = [
    "ogbn-arxiv",
    "ogbn-proteins",
    "ogbn-papers100M",
    "ogbn-products",
]

NUM_CLASSES = {
    "ogbn-arxiv": 40,
    "ogbn-proteins": 112,
    "ogbn-papers100M": 172,
    "ogbn-products": 47,
}


@dataclass
class NCCLGraphData:
    """Container for graph data and communication plan.

    Attributes:
        node_features: Local node features for this rank (num_local_nodes, feature_dim)
        edge_index: Global edge index in COO format (2, num_edges)
        labels: Local labels for this rank (num_local_nodes,)
        rank_mappings: Rank mappings for edges (2, num_edges) where [0] is src rank, [1] is dst rank
        comm_plan: Pre-computed NCCLGraphCommPlan for gather-scatter operations
        train_mask: Local training node indices
        val_mask: Local validation node indices
        test_mask: Local test node indices
        num_local_nodes: Number of nodes on this rank
        num_local_edges: Number of edges owned by this rank
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    labels: torch.Tensor
    rank_mappings: torch.Tensor
    comm_plan: NCCLGraphCommPlan
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_local_nodes: int
    num_local_edges: int

    def to(self, device: torch.device) -> "NCCLGraphData":
        """Move all tensors to the specified device."""
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.labels = self.labels.to(device)
        self.rank_mappings = self.rank_mappings.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.comm_plan = self.comm_plan.to(device)
        return self


class NCCLOGBDataset(Dataset):
    """Dataset class for OGB node property prediction with NCCLGraphCommPlan.

    Loads OGB data, partitions nodes across ranks, and pre-computes a
    NCCLGraphCommPlan for efficient distributed gather-scatter operations.
    Follows the same structure as the OGB-LSC DistributedHeteroGraphDataset.

    Args:
        dname: Dataset name (e.g., "ogbn-arxiv", "ogbn-products")
        comm_object: Initialized communicator object
        dir_name: Directory for caching processed data and comm plans
        node_rank_placement: Optional pre-computed node-to-rank mapping
        force_reprocess: If True, recompute even if cached data exists
    """

    def __init__(
        self,
        dname: str,
        comm_object: CommunicatorBase,
        dir_name: Optional[str] = None,
        node_rank_placement: Optional[torch.Tensor] = None,
        force_reprocess: bool = False,
    ) -> None:
        super().__init__()
        assert dname in SUPPORTED_DATASETS, (
            f"Dataset {dname} not supported. Supported: {SUPPORTED_DATASETS}"
        )
        assert comm_object._is_initialized, "Communicator not initialized"
        assert comm_object.backend == "nccl", (
            f"This dataset requires NCCL backend, got {comm_object.backend}"
        )

        self.dname = dname
        self.num_classes = NUM_CLASSES[dname]
        self.comm_object = comm_object
        self._rank = comm_object.get_rank()
        self._world_size = comm_object.get_world_size()

        dir_name = dir_name if dir_name is not None else os.path.join(os.getcwd(), "data")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        self.graph_obj = self._load_graph_data(
            dname, dir_name, node_rank_placement, force_reprocess
        )

        comm_plan_file = os.path.join(
            dir_name,
            f"{dname}_nccl_comm_plan_rank{self._rank}_world{self._world_size}.pt",
        )

        if os.path.exists(comm_plan_file) and not force_reprocess:
            if self._rank == 0:
                print(f"Loading cached comm plan from {comm_plan_file}")
            self._load_comm_plans(comm_plan_file)
        else:
            if self._rank == 0:
                print("Computing NCCLGraphCommPlan...")
            self._generate_comm_plans(comm_plan_file)

        self.comm_object.barrier()

    def _load_graph_data(
        self,
        dname: str,
        dir_name: str,
        node_rank_placement: Optional[torch.Tensor],
        force_reprocess: bool,
    ) -> DistributedGraph:
        """Load and partition OGB graph data across ranks."""
        cached_graph_file = os.path.join(
            dir_name, f"{dname}_graph_data_{self._world_size}.pt"
        )

        # Rank 0 downloads first to avoid race conditions
        self.comm_object.barrier()
        if self._rank == 0:
            dataset = NodePropPredDataset(name=dname)
        self.comm_object.barrier()
        if self._rank != 0:
            dataset = NodePropPredDataset(name=dname)
        self.comm_object.barrier()

        graph_data, labels = dataset[0]
        split_idx = dataset.get_idx_split()
        assert split_idx is not None, "Split index not found"

        if os.path.exists(cached_graph_file) and not force_reprocess:
            if self._rank == 0:
                print(f"Loading cached graph data from {cached_graph_file}")
            graph_obj = torch.load(cached_graph_file, weights_only=False)
        else:
            if node_rank_placement is None:
                if self._rank == 0:
                    print("Node rank placement not provided, using round-robin")
                node_rank_placement = get_round_robin_node_rank_map(
                    graph_data["num_nodes"], self._world_size
                )

            graph_obj = process_homogenous_data(
                graph_data,
                labels,
                self._rank,
                self._world_size,
                split_idx,
                node_rank_placement=node_rank_placement,
            )

            if self._rank == 0:
                print(f"Saving processed graph data to {cached_graph_file}")
                torch.save(graph_obj, cached_graph_file)

        self.comm_object.barrier()
        return graph_obj

    def _generate_comm_plans(self, fname: str) -> None:
        """Generate NCCLGraphCommPlan for the graph and save to disk.

        For the homogeneous OGB graph we create one plan: a source-vertex
        gather plan for each locally-owned edge (edges whose source vertex
        lives on this rank).
        """
        edge_index = self.graph_obj.get_global_edge_indices()
        src_vertices = edge_index[0]

        nodes_per_rank = self.graph_obj.get_nodes_per_rank()
        offsets = torch.zeros(self._world_size + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(nodes_per_rank, dim=0)

        src_start = offsets[self._rank].item()
        src_end = offsets[self._rank + 1].item()
        local_edge_mask = (src_vertices >= src_start) & (src_vertices < src_end)
        local_edge_list = torch.nonzero(local_edge_mask, as_tuple=True)[0]

        self.comm_plan = COO_to_NCCLCommPlan(
            rank=self._rank,
            world_size=self._world_size,
            global_edges_vertex_ids=src_vertices,
            local_edge_list=local_edge_list,
            offset=offsets,
        )

        self._save_comm_plans(fname)

    def _save_comm_plans(self, filepath: str) -> None:
        """Save comm plan to disk."""
        torch.save({"comm_plan": self.comm_plan}, filepath)

    def _load_comm_plans(self, filepath: str) -> None:
        """Load comm plan from disk."""
        data = torch.load(filepath, map_location="cpu", weights_only=False)
        self.comm_plan = data["comm_plan"]

    def get_NCCL_comm_plans(self) -> List[NCCLGraphCommPlan]:
        """Return the list of NCCLGraphCommPlans for this dataset.

        For the homogeneous OGB graph there is a single plan covering the
        source-vertex gather direction.
        """
        return [self.comm_plan]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> NCCLGraphData:
        """Return graph data with pre-computed NCCLGraphCommPlan.

        Returns:
            NCCLGraphData containing local node features, global edge indices,
            local labels, rank mappings, the comm plan, and train/val/test masks.
        """
        rank = self._rank

        local_node_features = self.graph_obj.get_local_node_features(rank=rank)
        local_labels = self.graph_obj.get_local_labels(rank=rank)
        edge_index = self.graph_obj.get_global_edge_indices()
        rank_mappings = self.graph_obj.get_global_rank_mappings()

        train_mask = self.graph_obj.get_local_mask("train", rank)
        val_mask = self.graph_obj.get_local_mask("val", rank)
        test_mask = self.graph_obj.get_local_mask("test", rank)

        nodes_per_rank = self.graph_obj.get_nodes_per_rank()
        num_local_nodes = nodes_per_rank[rank].item()

        edges_per_rank = self.graph_obj.get_edges_per_rank()
        num_local_edges = edges_per_rank[rank].item() if rank < len(edges_per_rank) else 0

        return NCCLGraphData(
            node_features=local_node_features,
            edge_index=edge_index,
            labels=local_labels,
            rank_mappings=rank_mappings,
            comm_plan=self.comm_plan,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_local_nodes=num_local_nodes,
            num_local_edges=num_local_edges,
        )

    def get_graph_stats(self) -> Dict[str, Any]:
        """Return statistics about the graph and partitioning."""
        return {
            "dataset": self.dname,
            "num_classes": self.num_classes,
            "num_nodes": self.graph_obj.num_nodes,
            "num_edges": self.graph_obj.num_edges,
            "world_size": self._world_size,
            "nodes_per_rank": self.graph_obj.get_nodes_per_rank().tolist(),
            "edges_per_rank": self.graph_obj.get_edges_per_rank().tolist(),
            "max_nodes_per_rank": self.graph_obj.get_max_node_per_rank(),
            "max_edges_per_rank": self.graph_obj.get_max_edge_per_rank(),
        }


def create_nccl_ogb_dataset(
    dname: str,
    comm_object: CommunicatorBase,
    dir_name: Optional[str] = None,
    node_rank_placement: Optional[torch.Tensor] = None,
    force_reprocess: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[NCCLGraphData, int]:
    """Convenience function to create dataset and return graph data on device.

    Args:
        dname: Dataset name
        comm_object: Initialized communicator
        dir_name: Cache directory
        node_rank_placement: Optional node-to-rank mapping
        force_reprocess: Force recomputation
        device: Target device (defaults to current CUDA device)

    Returns:
        Tuple of (NCCLGraphData, num_classes)
    """
    dataset = NCCLOGBDataset(
        dname=dname,
        comm_object=comm_object,
        dir_name=dir_name,
        node_rank_placement=node_rank_placement,
        force_reprocess=force_reprocess,
    )

    graph_data = dataset[0]

    if device is None:
        device = torch.device(f"cuda:{comm_object.get_rank() % torch.cuda.device_count()}")

    graph_data = graph_data.to(device)

    return graph_data, dataset.num_classes
