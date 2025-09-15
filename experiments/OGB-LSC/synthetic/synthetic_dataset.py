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


def _generate_paper_2_paper_edges(num_papers):
    # Average degree of a paper is ~11
    num_edges = num_papers * 11
    coo_list = torch.randint(
        low=0, high=num_papers, size=(2, num_edges), dtype=torch.long
    )
    coo_list = torch.unique(coo_list, dim=1)
    return coo_list


def _generate_paper_2_author_edges(num_papers, num_authors):
    # Average number of authors per paper is ~3.5
    num_edges = int(num_papers * 3.5)
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


def _get_rank_mappings(num_vertices, world_size, rank):
    vertices_per_rank = num_vertices // world_size
    rank_mappings = torch.zeros(num_vertices, dtype=torch.uint8)
    vertices_cur_rank = 0
    for r in range(world_size):
        start = r * vertices_per_rank
        end = (r + 1) * vertices_per_rank if r != world_size - 1 else num_vertices
        rank_mappings[start:end] = r
        if r == rank:
            vertices_cur_rank = end - start
    return rank_mappings, vertices_cur_rank


def edge_mapping_from_vertex_mapping(edge_index, rank_mappings):
    # directed edges, so edge_index[0] -> edge_index[1]
    src_indices = edge_index[0]
    dest_indices = edge_index[1]
    # We put the edge on the rank where the destination vertex is located
    edge_placement = rank_mappings[dest_indices]
    src_data_mappings = rank_mappings[src_indices]
    dest_data_mappings = rank_mappings[dest_indices]
    return (edge_placement, src_data_mappings, dest_data_mappings)


class HeterogeneousDataset:
    def __init__(
        self,
        num_papers,
        num_authors,
        num_institutions,
        num_features,
        num_classes,
        comm: Communicator,
    ):
        self.num_papers = num_papers
        self.num_authors = num_authors
        self.num_institutions = num_institutions
        self.num_classes = num_classes
        self.num_features = num_features
        self.comm = comm
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()
        self.paper_vertex_rank_mapping, self.num_paper_vertices = _get_rank_mappings(
            num_vertices=num_papers, world_size=self.world_size, rank=self.rank
        )
        self.author_vertex_rank_mapping, self.num_author_vertices = _get_rank_mappings(
            num_vertices=num_authors, world_size=self.world_size, rank=self.rank
        )
        self.institution_vertex_rank_mapping, self.num_institution_vertices = (
            _get_rank_mappings(
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
            paper_2_paper_edge_location,
            paper_2_paper_src_data_mappings,
            paper_2_paper_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.paper_2_paper_edges,
            rank_mappings=self.paper_vertex_rank_mapping,
        )

        self.paper_edge_locations = paper_2_paper_edge_location
        self.paper_src_data_mappings = paper_2_paper_src_data_mappings
        self.paper_dest_data_mappings = paper_2_paper_dest_data_mappings

        self.paper_2_author_edges = _generate_paper_2_author_edges(
            num_papers, num_authors
        )

        (
            paper_2_author_edge_location,
            paper_2_author_src_data_mappings,
            paper_2_author_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.paper_2_author_edges,
            rank_mappings=self.author_vertex_rank_mapping,
        )
        self.paper_2_author_edge_locations = paper_2_author_edge_location
        self.paper_2_author_src_data_mappings = paper_2_author_src_data_mappings
        self.paper_2_author_dest_data_mappings = paper_2_author_dest_data_mappings

        self.author_2_institution_edges = _generate_author_2_institution_edges(
            num_authors, num_institutions
        )

        (
            author_2_institution_edge_location,
            author_2_institution_src_data_mappings,
            author_2_institution_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.author_2_institution_edges,
            rank_mappings=self.institution_vertex_rank_mapping,
        )
        self.author_2_institution_edge_locations = author_2_institution_edge_location
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

        self.paper_features = torch.randn(
            (self.num_papers, paper_vertices_cur_rank), dtype=torch.float32
        )
        self.author_features = torch.randn(
            (self.num_authors, author_vertices_cur_rank), dtype=torch.float32
        )
        self.institution_features = torch.randn(
            (self.num_institutions, institution_vertices_cur_rank), dtype=torch.float32
        )

    def get_validation_mask(self):
        # Only papers are classified
        validation_vertices_mappings = self.paper_vertex_rank_mapping[self.val_mask]
        num_validation_vertices = (validation_vertices_mappings == self.rank).sum()
        if num_validation_vertices > 0:
            return self.val_mask[validation_vertices_mappings == self.rank]
        else:
            return torch.tensor([], dtype=torch.long)

    def get_test_mask(self):
        # Only papers are classified
        paper_vertices = self.paper_vertex_rank_mapping == self.rank
        num_test_vertices = (paper_vertices[self.test_mask] == self.rank).sum()
        if num_test_vertices > 0:
            return self.test_mask[paper_vertices[self.test_mask] == self.rank]
        else:
            return torch.tensor([], dtype=torch.long)

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        # There are 5 relations:
        # paper -> paper
        # paper -> author
        # author -> paper
        # author -> institution
        # institution -> author
        edge_index = [
            self.paper_2_paper_edges,
            self.paper_2_author_edges,
            self.paper_2_author_edges.flip(0),
            self.author_2_institution_edges,
            self.author_2_institution_edges.flip(0),
        ]
        # Locations of the edges
        rank_mappings = [
            [self.paper_edge_locations, self.paper_dest_data_mappings],
            [self.paper_2_author_edge_locations, self.paper_2_author_src_data_mappings],
            [
                self.paper_2_author_edge_locations,
                self.paper_2_author_dest_data_mappings,
            ],
            [
                self.author_2_institution_edge_locations,
                self.author_2_institution_src_data_mappings,
            ],
            [
                self.author_2_institution_dest_data_mappings,
                self.author_2_institution_src_data_mappings,
            ],
        ]
        edge_type = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)]
        features = [
            self.paper_features,
            self.author_features,
            self.institution_features,
        ]
        return (features, edge_index, edge_type, rank_mappings)
