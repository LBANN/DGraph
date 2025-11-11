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
from ogb.lsc import MAG240MDataset
import torch
from typing import Optional
from torch_sparse import SparseTensor
import numpy as np
from tqdm import tqdm
import os.path as osp
from DGraph.Communicator import Communicator


def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    """Obtained from:
    https://github.com/snap-stanford/ogb/blob/master/examples/lsc/mag240m/rgnn.py
    """
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)


def save_col_slice(
    x_src, x_dst, start_row_idx, end_row_idx, start_col_idx, end_col_idx
):
    """Obtained from:
    https://github.com/snap-stanford/ogb/blob/master/examples/lsc/mag240m/rgnn.py
    """
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i : offset + j, start_col_idx:end_col_idx] = x_src[i:j]


def get_rank_mappings(num_nodes, world_size, rank):
    nodes_per_rank = num_nodes // world_size
    print(f"Rank {rank}: nodes_per_rank = {nodes_per_rank}")
    # Don't use uint8 if world_size > 256
    # Doing this to save memory
    if world_size > 256:
        raise ValueError("world_size > 256 not supported yet")
    rank_mappings = torch.zeros(num_nodes, dtype=torch.uint8)
    for r in range(world_size):
        start = r * nodes_per_rank
        end = (r + 1) * nodes_per_rank if r != world_size - 1 else num_nodes
        rank_mappings[start:end] = r
    return rank_mappings

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

def get_edge_mappings(src_indices, dst_indices, rank_mappings):
    edge_mappings = torch.zeros_like(src_indices)
    # The edges are mapped to the rank of the destination node
    # Because that is the accumulation rank
    edge_mappings = rank_mappings[dst_indices]
    return edge_mappings


def _generate_features_from_paper_features(
    out: np.memmap,
    num_nodes: int,
    num_papers: int,
    paper_feat: np.ndarray,
    edge_index: np.ndarray,
    num_features: int,
):

    row, col = torch.from_numpy(edge_index)
    adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_papers), is_sorted=True
    )

    dim_chunk_size = 64

    for i in tqdm(range(0, num_features, dim_chunk_size)):
        j = min(i + dim_chunk_size, num_features)
        inputs = get_col_slice(
            paper_feat,
            start_row_idx=0,
            end_row_idx=num_papers,
            start_col_idx=i,
            end_col_idx=j,
        )
        inputs = torch.from_numpy(inputs)
        out_ = adj.matmul(inputs, reduce="mean").numpy()  # type: ignore
        del inputs
        save_col_slice(
            x_src=out_,
            x_dst=out,
            start_row_idx=0,
            end_row_idx=num_nodes,
            start_col_idx=i,
            end_col_idx=j,
        )
        del out_
    out.flush()


class DGraph_MAG240M:

    # data_dir must be the location where all ranks can access
    def __init__(
        self,
        comm: Communicator,
        data_dir: str = "data/MAG240M",
        paper_rank_mappings: Optional[torch.Tensor] = None,
        author_rank_mappings: Optional[torch.Tensor] = None,
        institution_rank_mappings: Optional[torch.Tensor] = None,
    ):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.comm = comm
        self.dataset = MAG240MDataset(root=data_dir)
        self.num_papers = self.dataset.num_papers
        self.num_authors = self.dataset.num_authors
        self.num_institutions = self.dataset.num_institutions
        # self.num_classes = self.dataset.num_classes
        self.paper_rank_mappings = (
            paper_rank_mappings
            if paper_rank_mappings is not None
            else get_rank_mappings(self.num_papers, self.world_size, self.rank)
        )
        self.author_rank_mappings = (
            author_rank_mappings
            if author_rank_mappings is not None
            else get_rank_mappings(self.num_authors, self.world_size, self.rank)
        )
        self.institution_rank_mappings = (
            institution_rank_mappings
            if institution_rank_mappings is not None
            else get_rank_mappings(self.num_institutions, self.world_size, self.rank)
        )

        # authors -> paper
        self.write_mappings = get_edge_mappings(
            torch.from_numpy(self.dataset.edge_index("author", "paper")[0]),
            torch.from_numpy(self.dataset.edge_index("author", "paper")[1]),
            self.paper_rank_mappings,
        )

        # author -> institution
        self.write_mappings_author_institution = get_edge_mappings(
            torch.from_numpy(self.dataset.edge_index("author", "institution")[0]),
            torch.from_numpy(self.dataset.edge_index("author", "institution")[1]),
            self.institution_rank_mappings,
        )

        self.train_mask = self.dataset.get_idx_split('train')
        self.val_mask = self.dataset.get_idx_split('valid')
        self.test_mask = self.dataset.get_idx_split('test-dev')

        local_papers_mask = self.paper_rank_mappings == self.rank
        local_authors_mask = self.author_rank_mappings == self.rank
        local_institutions_mask = self.institution_rank_mappings == self.rank
        self.num_local_papers  = int(
            local_papers_mask.sum()
        )

        self.generate_feature_data()

        self.paper_features = torch.from_numpy(self.dataset.paper_feat[local_papers_mask])
        path = self.dataset.dir
        self.author_features = torch.from_numpy(np.memmap(
                    filename=path + "/author_feat.npy",
                    mode="r",
                    dtype=np.float16,
                    shape=(self.num_authors, self.num_features),
                )[local_authors_mask])
        self.institution_features = torch.from_numpy(np.memmap(
                    filename=path + "/institution_feat.npy",
                    mode="r",
                    dtype=np.float16,
                    shape=(self.num_institutions, self.num_features),
                )[local_institutions_mask])
        self.y = torch.from_numpy(self.dataset.paper_label)

        self.paper_2_paper_edges = torch.from_numpy(self.dataset.edge_index('paper', 'cites', 'paper'))
        (
            paper_2_paper_src_data_mappings,
            paper_2_paper_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.paper_2_paper_edges,
            src_rank_mappings=self.paper_rank_mappings,
            dst_rank_mappings=self.paper_rank_mappings,
        )
        self.paper_src_data_mappings = paper_2_paper_src_data_mappings
        self.paper_dest_data_mappings = paper_2_paper_dest_data_mappings

        self.author_2_paper_edges = torch.from_numpy(self.dataset.edge_index('author', 'writes', 'paper'))
        (
            author_2_paper_src_data_mappings,
            author_2_paper_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.author_2_paper_edges,
            src_rank_mappings=self.author_rank_mappings,
            dst_rank_mappings=self.paper_rank_mappings,
        )
        self.author_2_paper_src_data_mappings = author_2_paper_src_data_mappings
        self.author_2_paper_dest_data_mappings = author_2_paper_dest_data_mappings

        self.author_2_institution_edges = torch.from_numpy(self.dataset.edge_index('author', 'institution'))
        (
            author_2_institution_src_data_mappings,
            author_2_institution_dest_data_mappings,
        ) = edge_mapping_from_vertex_mapping(
            edge_index=self.author_2_institution_edges,
            src_rank_mappings=self.author_rank_mappings,
            dst_rank_mappings=self.institution_rank_mappings,
        )

        self.author_2_institution_src_data_mappings = (
            author_2_institution_src_data_mappings
        )
        self.author_2_institution_dest_data_mappings = (
            author_2_institution_dest_data_mappings
        )

    @property
    def num_features(self) -> int:
        # 768
        return self.dataset.num_paper_features

    @property
    def num_classes(self) -> int:
        # 153
        return self.dataset.num_classes

    @property
    def num_relations(self) -> int:
        # paper -> paper
        # paper -> author
        # author -> paper
        # author -> institution
        # institution -> author
        return 5

    def generate_feature_data(self):
        dataset = self.dataset
        # This function emulates the author and institute features generation steps here
        # https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/lsc/mag240m/rgnn.py#L82

        # Generate author features
        # Mag240M author features are generated from paper features
        num_authors = dataset.num_authors
        num_papers = dataset.num_papers
        path = dataset.dir
        paper_feat = dataset.paper_feat

        # Only one rank must do this work
        if self.rank == 0:
            if not osp.exists(path + "/author_feat.npy"):
                print("Generating author features")
                author_feat = np.memmap(
                    filename=path + "/author_feat.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(num_authors, self.num_features),
                )
                _generate_features_from_paper_features(
                    out=author_feat,
                    num_nodes=num_authors,
                    num_papers=num_papers,
                    paper_feat=paper_feat,
                    edge_index=dataset.edge_index("author", "paper"),
                    num_features=self.num_features,
                )

            if not osp.exists(path + "/institution_feat.npy"):
                print("Generating institution features")
                # Generate institution features
                num_institutions = dataset.num_institutions
                institution_feat = np.memmap(
                    filename=path + "/institution_feat.npy",
                    mode="w+",
                    dtype=np.float16,
                    shape=(num_institutions, self.num_features),
                )
                _generate_features_from_paper_features(
                    out=institution_feat,
                    num_nodes=num_authors,
                    num_papers=num_institutions,
                    paper_feat=paper_feat,
                    edge_index=dataset.edge_index("author", "institution"),
                    num_features=self.num_features,
                )
        self.comm.barrier()

        # Make sure all ranks can see the generated files
        if not osp.exists(path + "/author_feat.npy"):
            raise FileNotFoundError("author_feat.npy not found")
        if not osp.exists(path + "/institution_feat.npy"):
            raise FileNotFoundError("institution_feat.npy not found")
        self.comm.barrier()

        print("Data processing complete")

    # Same as synthetic?
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
        paper_rank_mappings = self.paper_rank_mappings.to(
            global_int_mask.device
        )
        vertex_ranks = paper_rank_mappings[global_int_mask]
        # vertex_ranks is location of the vertices in the global_int_mask
        vertex_ranks_mask = vertex_ranks == self.rank
        return global_int_mask, vertex_ranks_mask

    # Same as synthetic?
    def get_mask(self, mask_type: str) -> torch.Tensor:

        global_int_mask, vertex_ranks_mask = self.get_vertex_rank_mask(mask_type)
        local_int_mask = global_int_mask[vertex_ranks_mask]
        local_int_mask = local_int_mask % self.num_local_papers
        return local_int_mask

    # Same as synthetic?
    def get_target(self, _type: str) -> torch.Tensor:
        global_int_mask, vertex_ranks_mask = self.get_vertex_rank_mask(_type)

        global_training_targets = self.y[:, global_int_mask.squeeze(0)]
        local_training_targets = global_training_targets[vertex_ranks_mask]

        return local_training_targets

    def __len__(self):
        return 0

    # Same as synthetic?
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

    # Same as synthetic?
    def to(self, device):
        """Move the dataset tensors to the specified device.
        We want to do this here because this allows us to avoid
        copying the data when the different individual tensors are
        accessed.
        """
        self.paper_features = self.paper_features.to(device, dtype=torch.float32)
        self.author_features = self.author_features.to(device, dtype=torch.float32)
        self.institution_features = self.institution_features.to(device, dtype=torch.float32)
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
    import fire

    def main(data_dir: str = "data/MAG240M"):
        rank = 0
        world_size = 64
        # Python is so weird haha
        COMM = type(
            "dummy_comm",
            (object,),
            {"get_rank": lambda self: rank, "get_world_size": lambda self: world_size},
        )
        comm = COMM()
        dgraph = DGraph_MAG240M(comm, data_dir=data_dir)

    fire.Fire(main)
