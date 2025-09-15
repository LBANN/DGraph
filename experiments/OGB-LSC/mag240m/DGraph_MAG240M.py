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
    def __init__(
        self,
        comm,
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
        self.num_classes = self.dataset.num_classes
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
            self.dataset.edge_index("author", "paper")[0],
            self.dataset.edge_index("author", "paper")[1],
            self.paper_rank_mappings,
        )

        # author -> institution
        self.write_mappings_author_institution = get_edge_mappings(
            self.dataset.edge_index("author", "institution")[0],
            self.dataset.edge_index("author", "institution")[1],
            self.institution_rank_mappings,
        )
        self.num_features = 768
        # paper -> paper
        self.process_feature_data()

    def process_feature_data(self):
        dataset = self.dataset
        # This function emulates the data processing step here:
        # https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/lsc/mag240m/rgnn.py#L82

        # The above function converts the heterogenous graph to a homogeneous graph
        # So we will do the same here

        # Generate author features
        # Mag240M author features are generated from paper features
        num_authors = dataset.num_authors
        num_papers = dataset.num_papers
        path = dataset.dir
        paper_feat = dataset.paper_feat

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
            print("Generating institution features")
            _generate_features_from_paper_features(
                out=institution_feat,
                num_nodes=num_institutions,
                num_papers=num_papers,
                paper_feat=paper_feat,
                edge_index=dataset.edge_index("author", "institution"),
                num_features=self.num_features,
            )
        print("Data processing complete")


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
