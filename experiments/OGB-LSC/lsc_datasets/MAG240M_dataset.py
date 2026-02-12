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
from typing import Optional, Tuple

from torch_sparse import SparseTensor
import numpy as np
from tqdm import tqdm
import os.path as osp
from DGraph.Communicator import Communicator
from DGraph.distributed.nccl import NCCLGraphCommPlan, COO_to_NCCLCommPlan
from .distributed_graph_dataset import (
    get_rank_mappings,
    get_vertex_offsets,
    DistributedHeteroGraphDataset,
)


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


def load_or_generate_vertex_rank_mask(
    rank_mapping: Optional[torch.Tensor], num_vertices: int, world_size: int, rank: int
):
    if rank_mapping is None:
        rank_mapping, vertices_cur_rank = get_rank_mappings(
            num_vertices, world_size, rank
        )
    rank_mask = rank_mapping == rank
    return rank_mask, vertices_cur_rank


class DGraph_MAG240M_Dataset(DistributedHeteroGraphDataset):

    # data_dir must be the location where all ranks can access
    def __init__(
        self,
        comm: Communicator,
        data_dir: str = "lsc_datasets/data/MAG240M",
        comm_plan_only: bool = True,
        paper_rank_mappings: Optional[torch.Tensor] = None,
        author_rank_mappings: Optional[torch.Tensor] = None,
        institution_rank_mappings: Optional[torch.Tensor] = None,
        cached_comm_plans: Optional[str] = None,
    ):
        rank = comm.get_rank()
        world_size = comm.get_world_size()
        self.comm = comm
        self.dataset = MAG240MDataset(root=data_dir)

        num_papers = self.dataset.num_papers
        num_authors = self.dataset.num_authors
        num_institutions = self.dataset.num_institutions
        # num_features = self.dataset.num_paper_features
        num_features = 1
        num_classes = self.dataset.num_classes

        self.train_mask = torch.from_numpy(self.dataset.get_idx_split("train")).long()
        self.val_mask = torch.from_numpy(self.dataset.get_idx_split("valid")).long()
        self.test_mask = torch.from_numpy(self.dataset.get_idx_split("test-dev")).long()

        local_papers_mask, num_local_papers = load_or_generate_vertex_rank_mask(
            paper_rank_mappings, num_papers, world_size, rank
        )

        local_authors_mask, num_local_authors = load_or_generate_vertex_rank_mask(
            author_rank_mappings, num_authors, world_size, rank
        )

        local_institutions_mask, num_local_institutions = (
            load_or_generate_vertex_rank_mask(
                institution_rank_mappings, num_institutions, world_size, rank
            )
        )

        self.num_local_papers = num_local_papers
        self.num_local_authors = num_local_authors
        self.num_local_institutions = num_local_institutions

        self.generate_feature_data()

        # paper_features = torch.from_numpy(self.dataset.paper_feat[local_papers_mask])
        paper_features = torch.randn((num_local_papers, 1), dtype=torch.float32)

        path = self.dataset.dir

        # author_features = torch.from_numpy(
        #     np.memmap(
        #         filename=path + "/author_feat.npy",
        #         mode="r",
        #         dtype=np.float16,
        #         shape=(num_authors, num_features),
        #     )[local_authors_mask]
        # )
        # institution_features = torch.from_numpy(
        #     np.memmap(
        #         filename=path + "/institution_feat.npy",
        #         mode="r",
        #         dtype=np.float16,
        #         shape=(num_institutions, num_features),
        #     )[local_institutions_mask]
        # )

        author_features = torch.randn((num_local_authors, 1), dtype=torch.float32)
        institution_features = torch.randn(
            (num_local_institutions, 1), dtype=torch.float32
        )

        labels = torch.from_numpy(self.dataset.paper_label).long()

        paper_2_paper_edges = torch.from_numpy(
            self.dataset.edge_index("paper", "cites", "paper")
        )
        author_2_paper_edges = torch.from_numpy(
            self.dataset.edge_index("author", "writes", "paper")
        )
        author_2_institution_edges = torch.from_numpy(
            self.dataset.edge_index("author", "institution")
        )

        paper_vertex_offsets = get_vertex_offsets(
            num_vertices=num_papers, world_size=world_size
        )
        author_vertex_offsets = get_vertex_offsets(
            num_vertices=num_authors, world_size=world_size
        )

        institution_vertex_offsets = get_vertex_offsets(
            num_vertices=num_institutions, world_size=world_size
        )

        super().__init__(
            rank=rank,
            world_size=world_size,
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
            comm_plan_only=comm_plan_only,
            paper_vertex_rank_mapping=paper_rank_mappings,
            author_vertex_rank_mapping=author_rank_mappings,
            institution_vertex_rank_mapping=institution_rank_mappings,
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
            f_name = (
                f"MAG240M_dataset_rank_{self.rank}_of_{self.world_size}_comm_plans.pt"
            )
            f_name = osp.join(data_dir, f_name)
            if osp.exists(f_name):
                print(
                    f"Rank {rank} loading cached comm plans from {f_name}", flush=True
                )
                self._load_comm_plans(f_name)
            else:
                self._generate_comm_plans(f_name)

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
        rank = self.comm.get_rank()
        # Only one rank must do this work
        if rank == 0:
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


if __name__ == "__main__":
    import DGraph.Communicator as Comm

    class DummyCommunicator(Comm):
        def __init__(self, rank, world_size):
            self._rank = rank
            self._world_size = world_size

        def get_rank(self):
            return self._rank

        def get_world_size(self):
            return self._world_size

        def barrier(self):
            pass

    dataset = DGraph_MAG240M_Dataset(comm=DummyCommunicator(0, 1))
