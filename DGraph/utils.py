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
import torch.distributed as dist


def largest_split(global_size, world_size):
    return (global_size + world_size) // world_size


def split_per_rank(global_size, current_rank, world_size):
    _split = largest_split(global_size, world_size)
    if current_rank != world_size - 1:
        return _split
    else:
        return global_size - (current_rank * _split)


def try_barrier():
    """Attempt a barrier but ignore any exceptions"""
    try:
        dist.barrier()
    except:
        pass
