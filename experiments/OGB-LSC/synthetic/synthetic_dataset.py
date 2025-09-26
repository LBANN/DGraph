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


def _get_rank_mappings(num_vertices, world_size, rank):
    vertices_per_rank = num_vertices // world_size
    rank_mappings = torch.zeros(num_vertices, dtype=torch.long)
    vertices_cur_rank = 0
    for r in range(world_size):
        start = r * vertices_per_rank
        end = (r + 1) * vertices_per_rank if r != world_size - 1 else num_vertices
        rank_mappings[start:end] = r
        if r == rank:
            vertices_cur_rank = end - start
    return rank_mappings, vertices_cur_rank


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
        self.num_relations = 5
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

        self.paper_features = torch.randn(
            (paper_vertices_cur_rank, num_features), dtype=torch.float32
        )
        self.author_features = torch.randn(
            (author_vertices_cur_rank, num_features), dtype=torch.float32
        )
        self.institution_features = torch.randn(
            (institution_vertices_cur_rank, num_features), dtype=torch.float32
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
        return self

    def __getitem__(self, idx):
        # There are 5 relations:
        # paper -> paper
        # paper -> author
        # author -> paper
        # author -> institution
        # institution -> author

        edge_index = [
            self.paper_2_paper_edges,
            self.author_2_paper_edges,
            self.author_2_paper_edges.flip(self.author_2_paper_edges.dim() - 2),
            self.author_2_institution_edges,
            self.author_2_institution_edges.flip(
                self.author_2_institution_edges.dim() - 2
            ),
        ]
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
        edge_type = [(0, 0), (1, 0), (0, 1), (1, 2), (2, 1)]
        features = [
            self.paper_features,
            self.author_features,
            self.institution_features,
        ]
        return (features, edge_index, edge_type, rank_mappings)


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
