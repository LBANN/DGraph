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
"""
This file contains the implementation of the RankLocalOps.
"""

import torch


def RankLocalMaskedGather(
    _src: torch.Tensor, indices: torch.Tensor, rank_mapping: torch.Tensor, rank: int
) -> torch.Tensor:
    """
    This function gathers the indices from the source rank to the destination rank.
    """
    local_indices = indices[rank_mapping == rank]
    num_features = _src.shape[-1]
    local_indices = local_indices.view(-1, 1).expand(-1, num_features)
    local_gathered_data = torch.gather(_src, 0, local_indices)
    return local_gathered_data


def OutOfPlaceRankLocalMaskedGather(
    _src: torch.Tensor, indices: torch.Tensor, rank_mapping: torch.Tensor, rank: int
) -> torch.Tensor:
    """
    This function gathers the indices from the source rank to the destination rank.
    """
    local_indices = indices[rank_mapping == rank]
    local_gathered_data = torch.gather(_src, 0, local_indices)
    return local_gathered_data


def RankLocalMaskedScatter(
    _src: torch.Tensor,
    _output: torch.Tensor,
    indices: torch.Tensor,
    rank_mapping: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """
    This function scatters the data from the source rank to the destination rank.
    """
    local_indices = indices[rank_mapping == rank]  # Masked local indices
    _output.scatter_(0, local_indices, _src)
    return _output


def RankLocalReNumbering(_indices):
    """
    This function removes duplicates from the indices tensor.
    """
    unique_indices = torch.unique(_indices)
    renumbered_indices = torch.zeros_like(_indices)
    # TODO: Optimize this code
    for i, idx in enumerate(unique_indices):
        renumbered_indices[_indices == idx] = i

    return renumbered_indices, unique_indices


def RankLocalGather(
    _src: torch.Tensor, indices: torch.Tensor, rank_mapping: torch.Tensor, rank: int
) -> torch.Tensor:
    """
    This function gathers the indices from the source rank to the destination rank.
    """
    local_indices = indices[rank_mapping == rank]
    local_gathered_data = torch.gather(_src, 0, local_indices)
    return local_gathered_data
