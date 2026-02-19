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
import torch
import torch.distributed as dist
from typing import List, Optional, Tuple, Literal

from DGraph.distributed.nccl import (
    NCCLEdgeConditionedGraphCommPlan,
    COO_to_NCCLEdgeConditionedCommPlan,
)


def get_rank_mappings(num_vertices: int, world_size: int, rank: int):
    vertices_per_rank = (num_vertices + world_size - 1) // world_size
    rank_mappings = torch.zeros(num_vertices, dtype=torch.long)
    vertices_cur_rank = 0
    for r in range(world_size):
        start = r * vertices_per_rank
        end = (r + 1) * vertices_per_rank if r != world_size - 1 else num_vertices
        rank_mappings[start:end] = r
        if r == rank:
            vertices_cur_rank = end - start
    return rank_mappings, vertices_cur_rank


def get_vertex_offsets(num_vertices, world_size):

    vertices_per_rank = (num_vertices + world_size - 1) // world_size
    offsets = torch.arange(world_size + 1, dtype=torch.long) * vertices_per_rank
    vertex_offsets = torch.clamp(offsets, max=num_vertices)
    return vertex_offsets


def edge_mapping_from_vertex_mapping(edge_index, src_rank_mappings, dst_rank_mappings):
    # directed edges, so edge_index[0] -> edge_index[1]
    src_indices = edge_index[0]
    dest_indices = edge_index[1]
    # We put the edge on the rank where the destination vertex is located
    # Since heterogeneous graphs have different rank mappings for different
    # vertex types.
    src_data_mappings = src_rank_mappings[src_indices]
    dest_data_mappings = dst_rank_mappings[dest_indices]
    return (src_data_mappings, dest_data_mappings)


class DistributedHeteroGraphDataset:
    """Class to handle distributed heterogeneous graph datasets for OGB-LSC MAG graphs.


    This class initializes and manages the distributed graph dataset,
    including loading data, partitioning, and providing access to data
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        num_features: int,
        num_classes: int,
        num_relations: int,
        paper_features: torch.Tensor,
        author_features: torch.Tensor,
        institution_features: torch.Tensor,
        paper_vertex_offset: torch.Tensor,
        author_vertex_offset: torch.Tensor,
        institution_vertex_offset: torch.Tensor,
        paper_labels: torch.Tensor,
        paper_2_paper_edges: torch.Tensor,
        author_2_paper_edges: torch.Tensor,
        author_2_institution_edges: torch.Tensor,
        comm_plan_only: Optional[bool] = False,
        paper_vertex_rank_mapping: Optional[torch.Tensor] = None,
        author_vertex_rank_mapping: Optional[torch.Tensor] = None,
        institution_vertex_rank_mapping: Optional[torch.Tensor] = None,
    ):
        """Initialize the DistributedHeteroGraphDataset class."""
        self.rank = rank
        self.world_size = world_size

        self._num_features = num_features
        self._num_classes = num_classes
        self._num_relations = num_relations

        self.paper_features = paper_features
        self.author_features = author_features
        self.institution_features = institution_features

        assert (
            paper_vertex_offset.size(0) == world_size + 1
        ), "paper_vertex_offset size must match world_size + 1"
        assert (
            author_vertex_offset.size(0) == world_size + 1
        ), "author_vertex_offset size must match world_size + 1"
        assert (
            institution_vertex_offset.size(0) == world_size + 1
        ), "institution_vertex_offset size must match world_size + 1"

        self.paper_vertex_offset = paper_vertex_offset
        self.author_vertex_offset = author_vertex_offset
        self.institution_vertex_offset = institution_vertex_offset

        self.paper_2_paper_edges = paper_2_paper_edges
        self.author_2_paper_edges = author_2_paper_edges
        self.author_2_institution_edges = author_2_institution_edges

        self.comm_plan_only = comm_plan_only
        if not comm_plan_only:
            assert (
                paper_vertex_rank_mapping is not None
            ), "paper_vertex_rank_mapping must be provided if comm_plan_only is False"
            assert (
                author_vertex_rank_mapping is not None
            ), "author_vertex_rank_mapping must be provided if comm_plan_only is False"
            assert (
                institution_vertex_rank_mapping is not None
            ), "institution_vertex_rank_mapping must be provided if comm_plan_only is False"

            (
                paper_2_paper_src_data_mappings,
                paper_2_paper_dest_data_mappings,
            ) = edge_mapping_from_vertex_mapping(
                edge_index=paper_2_paper_edges,
                src_rank_mappings=paper_vertex_rank_mapping,
                dst_rank_mappings=paper_vertex_rank_mapping,
            )

            self.paper_src_data_mappings = paper_2_paper_src_data_mappings
            self.paper_dest_data_mappings = paper_2_paper_dest_data_mappings

            (
                author_2_paper_src_data_mappings,
                author_2_paper_dest_data_mappings,
            ) = edge_mapping_from_vertex_mapping(
                edge_index=author_2_paper_edges,
                src_rank_mappings=author_vertex_rank_mapping,
                dst_rank_mappings=paper_vertex_rank_mapping,
            )

            self.author_2_paper_src_data_mappings = author_2_paper_src_data_mappings
            self.author_2_paper_dest_data_mappings = author_2_paper_dest_data_mappings

            (
                author_2_institution_src_data_mappings,
                author_2_institution_dest_data_mappings,
            ) = edge_mapping_from_vertex_mapping(
                edge_index=author_2_institution_edges,
                src_rank_mappings=author_vertex_rank_mapping,
                dst_rank_mappings=institution_vertex_rank_mapping,
            )

            self.author_2_institution_src_data_mappings = (
                author_2_institution_src_data_mappings
            )
            self.author_2_institution_dest_data_mappings = (
                author_2_institution_dest_data_mappings
            )

            self.local_paper_edges = torch.nonzero(
                self.paper_src_data_mappings == self.rank
            ).squeeze()
            self.local_author_2_paper_edges = torch.nonzero(
                self.author_2_paper_src_data_mappings == self.rank
            ).squeeze()
            self.local_author_2_institution_edges = torch.nonzero(
                self.author_2_institution_src_data_mappings == self.rank
            ).squeeze()

        else:
            # assume conotiguous vertex mapping
            local_paper_vertex_start = self.paper_vertex_offset[self.rank]
            local_paper_vertex_end = self.paper_vertex_offset[self.rank + 1]

            local_edges = paper_2_paper_edges[0] >= local_paper_vertex_start
            local_edges &= paper_2_paper_edges[0] < local_paper_vertex_end

            self.local_paper_edges = torch.nonzero(
                local_edges, as_tuple=False
            ).squeeze()

            local_author_vertex_start = self.author_vertex_offset[self.rank]
            local_author_vertex_end = self.author_vertex_offset[self.rank + 1]

            local_edges = author_2_paper_edges[0] >= local_author_vertex_start
            local_edges &= author_2_paper_edges[0] < local_author_vertex_end
            self.local_author_2_paper_edges = torch.nonzero(
                local_edges, as_tuple=False
            ).squeeze()

            local_edges = author_2_paper_edges[1] >= local_paper_vertex_start
            local_edges &= author_2_paper_edges[1] < local_paper_vertex_end

            self.local_paper_2_author_edges = torch.nonzero(
                local_edges, as_tuple=False
            ).squeeze()

            local_instituion_vertex_start = self.institution_vertex_offset[self.rank]
            local_instituion_vertex_end = self.institution_vertex_offset[self.rank + 1]

            local_edges = author_2_institution_edges[0] >= local_author_vertex_start
            local_edges &= author_2_institution_edges[0] < local_author_vertex_end
            self.local_author_2_institution_edges = torch.nonzero(
                local_edges, as_tuple=False
            ).squeeze()

            local_edges = author_2_institution_edges[1] >= local_instituion_vertex_start
            local_edges &= author_2_institution_edges[1] < local_instituion_vertex_end

            self.local_institution_2_author_edges = torch.nonzero(
                local_edges, as_tuple=False
            ).squeeze()

        self.y = paper_labels

    @property
    def num_features(self) -> int:
        return self._num_features

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def num_relations(self) -> int:
        return self._num_relations

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        edge_index = [
            self.paper_2_paper_edges,
            self.author_2_paper_edges,
            self.author_2_paper_edges.flip(self.author_2_paper_edges.dim() - 2),
            self.author_2_institution_edges,
            self.author_2_institution_edges.flip(
                self.author_2_institution_edges.dim() - 2
            ),
        ]

        if not self.comm_plan_only:

            # Locations of the edges
            rank_mappings = [
                [self.paper_src_data_mappings, self.paper_dest_data_mappings],
                [
                    self.author_2_paper_src_data_mappings,
                    self.author_2_paper_dest_data_mappings,
                ],
                [
                    self.author_2_paper_dest_data_mappings,
                    self.author_2_paper_src_data_mappings,
                ],
                [
                    self.author_2_institution_src_data_mappings,
                    self.author_2_institution_dest_data_mappings,
                ],
                [
                    self.author_2_institution_dest_data_mappings,
                    self.author_2_institution_src_data_mappings,
                ],
            ]
        else:
            rank_mappings = None
        edge_type = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)]
        features = [
            self.paper_features,
            self.author_features,
            self.institution_features,
        ]
        return (features, edge_index, edge_type, rank_mappings)

    def add_batch_dimension(self):
        """Add a batch dimension to all tensors. This is particularly useful
        because we only have one graph and DGraph is built to handle batches of graphs.
        We want to do this here because this allows us to avoid copying the data
        and requiring a data loader.
        """
        self.paper_features = self.paper_features.unsqueeze(0)
        self.author_features = self.author_features.unsqueeze(0)
        self.institution_features = self.institution_features.unsqueeze(0)
        self.y = self.y.unsqueeze(0)
        self.train_mask = self.train_mask.unsqueeze(0)
        self.val_mask = self.val_mask.unsqueeze(0)
        self.test_mask = self.test_mask.unsqueeze(0)
        self.paper_2_paper_edges = self.paper_2_paper_edges.unsqueeze(0)
        self.author_2_paper_edges = self.author_2_paper_edges.unsqueeze(0)
        self.author_2_institution_edges = self.author_2_institution_edges.unsqueeze(0)

        return self

    def to(self, device):
        """Move the dataset tensors to the specified device.
        We want to do this here because this allows us to avoid
        copying the data when the different individual tensors are
        accessed.

        Args:
            device: The device to move the tensors to.
        Returns:
            self: The dataset with tensors moved to the specified device.
        """
        self.paper_features = self.paper_features.to(device)
        self.author_features = self.author_features.to(device)
        self.institution_features = self.institution_features.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.paper_2_paper_edges = self.paper_2_paper_edges.to(device)
        self.author_2_paper_edges = self.author_2_paper_edges.to(device)
        self.author_2_institution_edges = self.author_2_institution_edges.to(device)
        self.local_paper_edges = self.local_paper_edges.to(device)
        self.local_author_2_paper_edges = self.local_author_2_paper_edges.to(device)
        self.local_author_2_institution_edges = (
            self.local_author_2_institution_edges.to(device)
        )

        return self

    def _get_vertex_rank_mask(
        self, mask_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_type == "train":
            global_int_mask = self.train_mask
        elif mask_type == "val":
            global_int_mask = self.val_mask
        elif mask_type == "test":
            global_int_mask = self.test_mask
        else:
            raise ValueError(f"Invalid mask type: {mask_type}")

        # global_int_mask is a set of integers
        global_int_mask.squeeze_(0)

        local_rank_start = self.paper_vertex_offset[self.rank]
        local_rank_end = self.paper_vertex_offset[self.rank + 1]

        # vertex_ranks is location of the vertices in the global_int_mask
        vertex_ranks_mask = torch.nonzero(
            (global_int_mask >= local_rank_start) & (global_int_mask < local_rank_end)
        ).squeeze()

        return global_int_mask, vertex_ranks_mask

    def get_mask(self, mask_type: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Given a set type (train, val, test), return the indices of the
        vertices that are in the set.

        Args:
            mask_type: The set type (train, val, test).
        Returns:
            local_int_mask: The indices of the vertices that are in the set.
        """

        global_int_mask, local_vertices = self._get_vertex_rank_mask(mask_type)
        local_int_mask = global_int_mask[local_vertices]

        num_local_vertices = (
            self.paper_vertex_offset[self.rank + 1]
            - self.paper_vertex_offset[self.rank]
        )

        local_int_mask = local_int_mask - self.paper_vertex_offset[self.rank]
        local_int_mask = local_int_mask % num_local_vertices

        return local_int_mask

    def get_target(self, mask_type: Literal["train", "val", "test"]) -> torch.Tensor:
        """
        Given a set type (train, val, test), return the targets of the
        vertices that are in the set.

        Args:
            mask_type: The set type (train, val, test).
        Returns:
            local_training_targets: The targets of the vertices that are in the set.

        """
        global_int_mask, local_vertices = self._get_vertex_rank_mask(mask_type)

        global_training_targets = self.y[:, global_int_mask.squeeze(0)]
        local_training_targets = global_training_targets[:, local_vertices]

        return local_training_targets

    def _save_comm_plans(self, filepath: str):

        torch.save(
            {
                "paper_2_paper_comm_plan": self.paper_2_paper_comm_plan,
                "paper_2_author_comm_plan": self.paper_2_author_comm_plan,
                "author_2_paper_comm_plan": self.author_2_paper_comm_plan,
                "author_2_institution_comm_plan": self.author_2_institution_comm_plan,
                "institution_2_author_comm_plan": self.institution_2_author_comm_plan,
            },
            filepath,
        )

    def _load_comm_plans(self, filepath: str):
        comm_plans = torch.load(filepath, map_location="cpu", weights_only=False)
        self.paper_2_paper_comm_plan = comm_plans["paper_2_paper_comm_plan"]
        self.paper_2_author_comm_plan = comm_plans["paper_2_author_comm_plan"]
        self.author_2_institution_comm_plan = comm_plans[
            "author_2_institution_comm_plan"
        ]
        self.author_2_paper_comm_plan = comm_plans["author_2_paper_comm_plan"]
        self.institution_2_author_comm_plan = comm_plans[
            "institution_2_author_comm_plan"
        ]

    def _generate_comm_plans(self, fname: str):

        self.paper_2_paper_comm_plan = COO_to_NCCLEdgeConditionedCommPlan(
            rank=self.rank,
            world_size=self.world_size,
            global_edges_src=self.paper_2_paper_edges[0],
            global_edges_dst=self.paper_2_paper_edges[1],
            local_edge_list=self.local_paper_edges,
            src_offset=self.paper_vertex_offset,
            dest_offset=self.paper_vertex_offset,
        )

        self.paper_2_author_comm_plan = COO_to_NCCLEdgeConditionedCommPlan(
            rank=self.rank,
            world_size=self.world_size,
            global_edges_src=self.author_2_paper_edges[1],
            global_edges_dst=self.author_2_paper_edges[0],
            local_edge_list=self.local_paper_2_author_edges,
            src_offset=self.paper_vertex_offset,
            dest_offset=self.author_vertex_offset,
        )
        self.author_2_paper_comm_plan = COO_to_NCCLEdgeConditionedCommPlan(
            rank=self.rank,
            world_size=self.world_size,
            global_edges_src=self.author_2_paper_edges[0],
            global_edges_dst=self.author_2_paper_edges[1],
            local_edge_list=self.local_author_2_paper_edges,
            src_offset=self.author_vertex_offset,
            dest_offset=self.paper_vertex_offset,
        )

        self.author_2_institution_comm_plan = COO_to_NCCLEdgeConditionedCommPlan(
            rank=self.rank,
            world_size=self.world_size,
            global_edges_src=self.author_2_institution_edges[0],
            global_edges_dst=self.author_2_institution_edges[1],
            local_edge_list=self.local_author_2_institution_edges,
            src_offset=self.author_vertex_offset,
            dest_offset=self.institution_vertex_offset,
        )
        self.institution_2_author_comm_plan = COO_to_NCCLEdgeConditionedCommPlan(
            rank=self.rank,
            world_size=self.world_size,
            global_edges_src=self.author_2_institution_edges[1],
            global_edges_dst=self.author_2_institution_edges[0],
            local_edge_list=self.local_institution_2_author_edges,
            src_offset=self.institution_vertex_offset,
            dest_offset=self.author_vertex_offset,
        )
        self._save_comm_plans(fname)

    def get_NCCL_comm_plans(self) -> List[NCCLEdgeConditionedGraphCommPlan]:

        comm_plans = []
        # paper -> paper
        comm_plans.append(self.paper_2_paper_comm_plan)
        # paper -> author
        comm_plans.append(self.paper_2_author_comm_plan)
        # author -> paper
        comm_plans.append(self.author_2_paper_comm_plan)
        # author -> institution
        comm_plans.append(self.author_2_institution_comm_plan)
        # institution -> author
        comm_plans.append(self.institution_2_author_comm_plan)

        return comm_plans
