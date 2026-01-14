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
from DGraph.Communicator import Communicator
import torch
from typing import List, Optional, Tuple

from DGraph.distributed.nccl import NCCLGraphCommPlan
from distributed_graph_dataset import (
    get_rank_mappings,
    edge_mapping_from_vertex_mapping,
    DistributedHeteroGraphDataset,
)

torch.random.manual_seed(0)


def _generate_paper_2_paper_edges(num_papers):
    # Average degree of a paper is ~11
    num_edges = num_papers * 11
    coo_list = torch.randint(
        low=0, high=num_papers, size=(2, num_edges), dtype=torch.long
    )
    coo_list = torch.unique(coo_list, dim=1)
    transpose = coo_list.flip(0)
    coo_list = torch.cat([coo_list, transpose], dim=1)
    coo_list = torch.sort(coo_list, dim=1).values
    return coo_list


def _generate_author_2_paper_edges(num_authors, num_papers):
    # Average number of authors per paper is ~3.5
    num_edges = int(num_authors * 3.5)
    dest_papers = torch.randint(
        low=0, high=num_papers, size=(1, num_edges), dtype=torch.long
    )
    src_authors = torch.randint(
        low=0, high=num_authors, size=(1, num_edges), dtype=torch.long
    )
    coo_list = torch.cat([src_authors, dest_papers], dim=0)
    coo_list = torch.unique(coo_list, dim=1)
    return coo_list


def _generate_author_2_institution_edges(num_authors, num_institutions):
    # Average number of institutions per author is ~0.35
    num_edges = int(num_authors * 0.35)
    dest_num_institutions = torch.randint(
        low=0, high=num_institutions, size=(1, num_edges), dtype=torch.long
    )
    src_authors = torch.randint(
        low=0, high=num_authors, size=(1, num_edges), dtype=torch.long
    )
    coo_list = torch.cat([src_authors, dest_num_institutions], dim=0)
    coo_list = torch.unique(coo_list, dim=1)
    return coo_list


class HeterogeneousDataset(DistributedHeteroGraphDataset):
    def __init__(
        self,
        num_papers,
        num_authors,
        num_institutions,
        num_features,
        num_classes,
        comm: Communicator,
        cached_comm_plans: Optional[str] = None,
    ):
        self.num_papers = num_papers
        self.num_authors = num_authors
        self.num_institutions = num_institutions
        self._num_classes = num_classes
        self._num_features = num_features
        self._num_relations = 5
        self.comm = comm
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()
        self.paper_vertex_rank_mapping, self.num_paper_vertices = get_rank_mappings(
            num_vertices=num_papers, world_size=self.world_size, rank=self.rank
        )
        self.author_vertex_rank_mapping, self.num_author_vertices = get_rank_mappings(
            num_vertices=num_authors, world_size=self.world_size, rank=self.rank
        )
        self.institution_vertex_rank_mapping, self.num_institution_vertices = (
            get_rank_mappings(
                num_vertices=num_institutions,
                world_size=self.world_size,
                rank=self.rank,
            )
        )
        _vertices = torch.randperm(num_papers)
        self.train_mask = _vertices[: int(0.7 * num_papers)]
        self.val_mask = _vertices[int(0.7 * num_papers) : int(0.85 * num_papers)]
        self.test_mask = _vertices[int(0.85 * num_papers) :]
        self.y = torch.randint(
            low=0, high=self.num_classes, size=(num_papers,), dtype=torch.long
        )

        self.paper_2_paper_edges = _generate_paper_2_paper_edges(num_papers)

        (
            paper_2_paper_src_data_mappings,
            paper_2_paper_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.paper_2_paper_edges,
            src_rank_mappings=self.paper_vertex_rank_mapping,
            dst_rank_mappings=self.paper_vertex_rank_mapping,
        )

        self.paper_src_data_mappings = paper_2_paper_src_data_mappings
        self.paper_dest_data_mappings = paper_2_paper_dest_data_mappings

        self.author_2_paper_edges = _generate_author_2_paper_edges(
            num_authors, num_papers
        )

        (
            author_2_paper_src_data_mappings,
            author_2_paper_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.author_2_paper_edges,
            src_rank_mappings=self.author_vertex_rank_mapping,
            dst_rank_mappings=self.paper_vertex_rank_mapping,
        )
        self.author_2_paper_src_data_mappings = author_2_paper_src_data_mappings
        self.author_2_paper_dest_data_mappings = author_2_paper_dest_data_mappings

        self.author_2_institution_edges = _generate_author_2_institution_edges(
            num_authors, num_institutions
        )

        (
            author_2_institution_src_data_mappings,
            author_2_institution_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.author_2_institution_edges,
            src_rank_mappings=self.author_vertex_rank_mapping,
            dst_rank_mappings=self.institution_vertex_rank_mapping,
        )

        self.author_2_institution_src_data_mappings = (
            author_2_institution_src_data_mappings
        )
        self.author_2_institution_dest_data_mappings = (
            author_2_institution_dest_data_mappings
        )

        paper_vertices_cur_rank = int(
            (self.paper_vertex_rank_mapping == self.rank).sum()
        )
        author_vertices_cur_rank = int(
            (self.author_vertex_rank_mapping == self.rank).sum()
        )
        institution_vertices_cur_rank = int(
            (self.institution_vertex_rank_mapping == self.rank).sum()
        )
        self.paper_vertices_cur_rank = paper_vertices_cur_rank

        self.paper_features = torch.randn(
            (paper_vertices_cur_rank, num_features), dtype=torch.float32
        )
        self.author_features = torch.randn(
            (author_vertices_cur_rank, num_features), dtype=torch.float32
        )
        self.institution_features = torch.randn(
            (institution_vertices_cur_rank, num_features), dtype=torch.float32
        )

        if cached_comm_plans is not None:
            comm_plans = torch.load(cached_comm_plans)
            self.paper_2_paper_comm_plan = comm_plans["paper_2_paper_comm_plan"]
            self.paper_2_author_comm_plan = comm_plans["paper_2_author_comm_plan"]
            self.author_2_institution_comm_plan = comm_plans[
                "author_2_institution_comm_plan"
            ]
            self.institution_2_author_comm_plan = comm_plans[
                "institution_2_author_comm_plan"
            ]
            self.author_2_paper_comm_plan = comm_plans["author_2_paper_comm_plan"]

        else:
            self._generate_comm_plans()

    def get_vertex_rank_mask(self, mask_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_type == "train":
            global_int_mask = self.train_mask
        elif mask_type == "val":
            global_int_mask = self.val_mask
        elif mask_type == "test":
            global_int_mask = self.test_mask
        else:
            raise ValueError(f"Invalid mask type: {mask_type}")

        # Get the ranks of the vertices
        # paper_vertex_rank_mapping -> vector of size num_papers,
        # where each entry is the location / rank of the vertex
        paper_vertex_rank_mapping = self.paper_vertex_rank_mapping.to(
            global_int_mask.device
        )
        vertex_ranks = paper_vertex_rank_mapping[global_int_mask]
        # vertex_ranks is location of the vertices in the global_int_mask
        vertex_ranks_mask = vertex_ranks == self.rank
        return global_int_mask, vertex_ranks_mask

    def get_mask(self, mask_type: str) -> torch.Tensor:

        global_int_mask, vertex_ranks_mask = self.get_vertex_rank_mask(mask_type)
        local_int_mask = global_int_mask[vertex_ranks_mask]
        local_int_mask = local_int_mask % self.paper_vertices_cur_rank
        return local_int_mask

    def get_target(self, _type: str) -> torch.Tensor:
        global_int_mask, vertex_ranks_mask = self.get_vertex_rank_mask(_type)

        global_training_targets = self.y[:, global_int_mask.squeeze(0)]
        local_training_targets = global_training_targets[vertex_ranks_mask]

        return local_training_targets

    def _generate_comm_plans(self):
        self.paper_2_paper_comm_plan = NCCLGraphCommPlan.generate_from_edge_index(
            edge_index=self.paper_2_paper_edges,
            src_rank_mapping=self.paper_vertex_rank_mapping,
            dst_rank_mapping=self.paper_vertex_rank_mapping,
            comm=self.comm,
        )

    def get_NCCL_comm_plans(self) -> List[NCCLGraphCommPlan]:

        comm_plans = []
        # paper -> paper
        comm_plans.append(self.paper_2_paper_comm_plan)

        return comm_plans


if __name__ == "__main__":
    rank = 0
    world_size = 16
    COMM = type(
        "dummy_comm",
        (object,),
        {"get_rank": lambda self: rank, "get_world_size": lambda self: world_size},
    )
    comm = COMM()

    dataset = HeterogeneousDataset(
        num_papers=512,
        num_authors=128,
        num_institutions=32,
        num_features=16,
        num_classes=4,
        comm=comm,
    )
    print(dataset[0])
