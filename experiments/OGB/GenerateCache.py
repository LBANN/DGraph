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

from DGraph.data.ogbn_datasets import process_homogenous_data
from ogb.nodeproppred import NodePropPredDataset
from fire import Fire
import os
import torch
from DGraph.distributed.nccl._nccl_cache import (
    NCCLGatherCacheGenerator,
    NCCLScatterCacheGenerator,
)

from DGraph.distributed.nccl import COO_to_NCCLEdgeConditionedCommPlan

from time import perf_counter
from tqdm import tqdm
from multiprocessing import get_context


cache_prefix = {
    "ogbn-arxiv": "arxiv",
    "ogbn-products": "products",
    "ogbn-papers100M": "papers100M",
}


def generate_comm_plan(
    coo_list,
    offsets,
    rank,
    world_size,
    dest_offsets=None,
):
    # Source edges belonging to this rank should be where the source
    # vertex falls within the rank's offset range.
    src_start = offsets[rank].item()
    src_end = offsets[rank + 1].item()
    local_edges = torch.nonzero(
        (coo_list[0] >= src_start) & (coo_list[0] < src_end), as_tuple=True
    )[0]

    comm_plan = COO_to_NCCLEdgeConditionedCommPlan(
        rank,
        world_size,
        coo_list[0],
        coo_list[1],
        local_edges,
        offsets,
        dest_offset=dest_offsets,
    )
    return comm_plan


def main(dset: str, world_size: int, node_rank_placement_file: str):
    assert dset in ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]

    assert world_size > 0
    dataset = NodePropPredDataset(
        dset,
    )

    split_index = dataset.get_idx_split()
    assert split_index is not None, "Split index is None."

    graph, labels = dataset[0]
    num_edges = graph["edge_index"].shape
    print(num_edges)
    num_nodes = graph["num_nodes"]


if __name__ == "__main__":
    Fire(main)
