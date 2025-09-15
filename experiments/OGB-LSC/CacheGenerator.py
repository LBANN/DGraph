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

import os.path as osp
from DGraph.distributed.nccl._nccl_cache import (
    NCCLGatherCacheGenerator,
    NCCLScatterCacheGenerator,
)


def get_cache(
    src_gather_cache,
    dest_gather_cache,
    dest_scatter_cache,
    src_gather_cache_file,
    dest_gather_cache_file,
    dest_scatter_cache_file,
    rank,
    world_size,
    src_indices,
    dest_indices,
    edge_location,
    src_data_mappings,
    dest_data_mappings,
    num_input_rows,
    num_output_rows,
):
    if src_gather_cache is None:

        _src_gather_cache = NCCLGatherCacheGenerator(
            indices=src_indices,
            edge_placement=edge_location,
            edge_dest_ranks=src_data_mappings,
            num_input_rows=num_input_rows,
            rank=rank,
            world_size=world_size,
        )

        torch.save(_src_gather_cache, src_gather_cache_file)
    else:
        _src_gather_cache = src_gather_cache

    if dest_scatter_cache is None:
        _dest_scatter_cache = NCCLScatterCacheGenerator(
            indices=dest_indices,
            edge_placement=edge_location,
            edge_dest_ranks=dest_data_mappings,
            num_output_rows=num_output_rows,
            rank=rank,
            world_size=world_size,
        )

        torch.save(_dest_scatter_cache, dest_scatter_cache_file)
    else:
        _dest_scatter_cache = dest_scatter_cache

    if dest_gather_cache is None:
        _dest_gather_cache = NCCLGatherCacheGenerator(
            indices=dest_indices,
            edge_placement=edge_location,
            edge_dest_ranks=dest_data_mappings,
            num_input_rows=num_input_rows,
            rank=rank,
            world_size=world_size,
        )

        torch.save(_dest_gather_cache, dest_gather_cache_file)
    else:
        _dest_gather_cache = dest_gather_cache

    return _src_gather_cache, _dest_scatter_cache, _dest_gather_cache


if __name__ == "__main__":
    from fire import Fire
    from functools import partial
    from config import SyntheticDatasetConfig

    # Use this script to generate the caches prior to running the main training script
    # This is useful because cache generation can take a long time and could cause issues
    # with timeouts on some systems.

    def main(dataset):
        assert dataset in ["synthetic", "mag240m"]
        if dataset == "synthetic":
            from synthetic.synthetic_dataset import HeterogeneousDataset as Dataset

            synthetic_config = SyntheticDatasetConfig()
            graph_dataset = partial(
                Dataset,
                num_papers=synthetic_config.num_papers,
                num_authors=synthetic_config.num_authors,
                num_institutions=synthetic_config.num_institutions,
                num_features=synthetic_config.num_features,
                num_classes=synthetic_config.num_classes,
            )
        elif dataset == "mag240m":
            from mag240m.DGraph_MAG240M import DGraph_MAG240M as Dataset

            graph_dataset = partial(Dataset, data_dir="data/MAG240M")

        rank = 0
        world_size = 16
        COMM = type(
            "dummy_comm",
            (object,),
            {"get_rank": lambda self: rank, "get_world_size": lambda self: world_size},
        )
        comm = COMM()

        dataset = graph_dataset(
            comm=comm,
        )

        dataset = dataset.add_batch_dimension()
        dataset = dataset.to("cpu")
        xs, edge_index, edge_type, rank_mapping = dataset[0]
        print("Dataset loaded")

        breakpoint()
        # get_cache(
        #     src_gather_cache=None,
        #     dest_gather_cache=None,
        #     dest_scatter_cache=None,
        #     src_gather_cache_file="paper_src_gather_cache.pt",
        #     dest_gather_cache_file="paper_dest_gather_cache.pt",
        #     dest_scatter_cache_file="paper_dest_scatter_cache.pt",
        #     rank=rank,
        #     world_size=world_size,
        #     src_indices=edge_index[0][0][0],
        #     dest_indices=edge_index[0][0][1],
        #     edge_location=rank_mapping[0],
        #     src_data_mappings=rank_mapping[2],
        #     dest_data_mappings=rank_mapping[3],)

    Fire(main)
