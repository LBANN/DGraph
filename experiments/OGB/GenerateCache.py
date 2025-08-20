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
from time import perf_counter
from tqdm import tqdm
from multiprocessing import get_context


cache_prefix = {
    "ogbn-arxiv": "arxiv",
    "ogbn-products": "products",
    "ogbn-papers100M": "papers100M",
}


def generate_cache_file(
    dist_graph,
    src_indices,
    dst_indices,
    edge_placement,
    edge_src_placement,
    edge_dest_placement,
    cache_prefix_str: str,
    rank: int,
    world_size: int,
):
    print(f"Generating cache for rank {rank}...")
    local_node_features = dist_graph.get_local_node_features(rank).unsqueeze(0)
    num_input_rows = local_node_features.size(1)

    print(
        f"Rank {rank} has {num_input_rows} input rows with shape {local_node_features.shape}"
    )
    gather_cache = NCCLGatherCacheGenerator(
        dst_indices,
        edge_placement,
        edge_dest_placement,
        num_input_rows,
        rank,
        world_size,
    )

    nodes_per_rank = dist_graph.get_nodes_per_rank()
    nodes_per_rank = int(nodes_per_rank[rank].item())

    scatter_cache = NCCLScatterCacheGenerator(
        src_indices,
        edge_placement,
        edge_src_placement,
        nodes_per_rank,
        rank,
        world_size,
    )
    print(f"Rank {rank}  completed cache generation")
    with open(
        f"{cache_prefix_str}_gather_cache_rank_{world_size}_{rank}.pt", "wb"
    ) as f:
        torch.save(gather_cache, f)

    with open(
        f"{cache_prefix_str}_scatter_cache_rank_{world_size}_{rank}.pt", "wb"
    ) as f:
        torch.save(scatter_cache, f)
    return 0


def main(dset: str, world_size: int, node_rank_placement_file: str):
    assert dset in ["ogbn-arxiv", "ogbn-products", "ogbn-papers100M"]

    assert world_size > 0
    assert os.path.exists(
        node_rank_placement_file
    ), "Node rank placement file does not exist."

    node_rank_placement = torch.load(node_rank_placement_file)

    dataset = NodePropPredDataset(
        dset,
    )

    split_index = dataset.get_idx_split()
    assert split_index is not None, "Split index is None."

    graph, labels = dataset[0]

    num_edges = graph["edge_index"].shape
    print(num_edges)

    dist_graph = process_homogenous_data(
        graph_data=graph,
        labels=labels,
        world_Size=world_size,
        split_idx=split_index,
        node_rank_placement=node_rank_placement,
        rank=0,
    )

    edge_indices = dist_graph.get_global_edge_indices()
    rank_mappings = dist_graph.get_global_rank_mappings()

    print("Edge indices shape:", edge_indices.shape)
    print("Rank mappings shape:", rank_mappings.shape)

    edge_indices = edge_indices.unsqueeze(0)
    src_indices = edge_indices[:, 0, :]
    dst_indices = edge_indices[:, 1, :]

    edge_placement = rank_mappings[0]
    edge_src_placement = rank_mappings[0]
    edge_dest_placement = rank_mappings[1]

    start_time = perf_counter()
    cache_prefix_str = f"cache/{cache_prefix[dset]}"
    with get_context("spawn").Pool(min(world_size, 8)) as pool:
        args = [
            (
                dist_graph,
                src_indices,
                dst_indices,
                edge_placement,
                edge_src_placement,
                edge_dest_placement,
                cache_prefix_str,
                rank,
                world_size,
            )
            for rank in range(world_size)
        ]

        out = pool.starmap(generate_cache_file, args)

    end_time = perf_counter()
    print(f"Cache generation time: {end_time - start_time:.4f} seconds")
    print("Cache files generated successfully.")
    print(
        f"Gather cache file: {cache_prefix_str}_gather_cache_rank_{world_size}_<rank>.pt"
    )
    print(
        f"Scatter cache file: {cache_prefix_str}_scatter_cache_rank_{world_size}_<rank>.pt"
    )


if __name__ == "__main__":
    Fire(main)
