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
