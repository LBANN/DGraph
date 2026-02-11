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
from typing import Optional

from .distributed_graph_dataset import (
    get_vertex_offsets,
    DistributedHeteroGraphDataset,
)

import os.path as osp


import hashlib


def generate_config_hash(numbers):
    # Convert tuple to string, encode to bytes, hash
    payload = str(tuple(numbers)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


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


class SyntheticHeterogeneousDataset(DistributedHeteroGraphDataset):
    def __init__(
        self,
        synthetic_config,
        comm: Communicator,
        cached_comm_plans: Optional[str] = None,
    ):
        """Synthetic heterogeneous graph dataset for OGB-LSC experiments built to
        mimic the MAG240M dataset.

        Args:
            synthetic_config: Configuration object for synthetic dataset. Must have `num_papers`,
                              `num_authors`, `num_institutions`, `num_features`, and `num_classes` attributes.
            comm: DGraph communicator object.
            cached_comm_plans: Optional path to cached communication plans.

        """
        num_papers = synthetic_config.num_papers
        num_authors = synthetic_config.num_authors
        num_institutions = synthetic_config.num_institutions
        num_features = synthetic_config.num_features
        num_classes = synthetic_config.num_classes
        self._num_relations = 5
        self.comm = comm
        rank = comm.get_rank()
        world_size = comm.get_world_size()

        # Set up synthetic data for papers, authors, and institutions and call superclass init

        _vertices = torch.randperm(num_papers)

        self.train_mask = _vertices[: int(0.7 * num_papers)]
        self.val_mask = _vertices[int(0.7 * num_papers) : int(0.85 * num_papers)]
        self.test_mask = _vertices[int(0.85 * num_papers) :]

        labels = torch.randint(
            low=0, high=num_classes, size=(num_papers,), dtype=torch.long
        )

        # Generate edges
        paper_2_paper_edges = _generate_paper_2_paper_edges(num_papers)
        author_2_paper_edges = _generate_author_2_paper_edges(num_authors, num_papers)
        author_2_institution_edges = _generate_author_2_institution_edges(
            num_authors, num_institutions
        )

        paper_vertex_offsets = get_vertex_offsets(
            num_vertices=num_papers, world_size=comm.get_world_size()
        )
        author_vertex_offsets = get_vertex_offsets(
            num_vertices=num_authors, world_size=comm.get_world_size()
        )
        institution_vertex_offsets = get_vertex_offsets(
            num_vertices=num_institutions, world_size=comm.get_world_size()
        )

        num_paper_vertices_cur_rank = int(
            paper_vertex_offsets[rank + 1] - paper_vertex_offsets[rank]
        )
        num_author_vertices_cur_rank = int(
            author_vertex_offsets[rank + 1] - author_vertex_offsets[rank]
        )
        num_institution_vertices_cur_rank = int(
            institution_vertex_offsets[rank + 1] - institution_vertex_offsets[rank]
        )

        # Generate random feature data for vertices
        paper_features = torch.randn(
            (num_paper_vertices_cur_rank, num_features), dtype=torch.float32
        )
        author_features = torch.randn(
            (num_author_vertices_cur_rank, num_features), dtype=torch.float32
        )
        institution_features = torch.randn(
            (num_institution_vertices_cur_rank, num_features), dtype=torch.float32
        )

        super().__init__(
            rank=rank,
            world_size=comm.get_world_size(),
            num_features=num_features,
            num_classes=num_classes,
            num_relations=5,
            paper_features=paper_features,
            author_features=author_features,
            institution_features=institution_features,
            paper_vertex_offset=paper_vertex_offsets,
            author_vertex_offset=author_vertex_offsets,
            institution_vertex_offset=institution_vertex_offsets,
            paper_labels=labels,
            paper_2_paper_edges=paper_2_paper_edges,
            author_2_paper_edges=author_2_paper_edges,
            author_2_institution_edges=author_2_institution_edges,
            comm_plan_only=True,
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
            dataset_hash = generate_config_hash(
                [num_papers, num_authors, num_institutions, num_features, num_classes]
            )
            cur_dir = osp.dirname(osp.abspath(__file__))
            f_name = f"synthetic_dataset_{dataset_hash}_rank_{self.rank}_of_{self.world_size}_comm_plans.pt"
            f_path = osp.join(cur_dir, f_name)
            if osp.exists(f_path):
                print(f"Loading comm plans from {f_path}")
                self._load_comm_plans(f_path)
            else:
                print(f"Generating comm plans and saving to {f_path}")
                self._generate_comm_plans(f_path)
