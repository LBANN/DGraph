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

try:
    from DGraph.torch_local import local_masked_gather, local_masked_scatter

    _LOCAL_OPT_KERNELS_AVAILABLE = True
except ImportError:
    _LOCAL_OPT_KERNELS_AVAILABLE = False
import warnings


def RankLocalMaskedGather(
    _src: torch.Tensor, indices: torch.Tensor, rank_mapping: torch.Tensor, rank: int
) -> torch.Tensor:
    """
    This function gathers the indices from the source rank to the destination rank.
    """
    local_indices = indices[rank_mapping == rank]
    num_features = _src.shape[-1]
    local_indices = local_indices.view(1, -1, 1).expand(1, -1, num_features)
    local_gathered_data = torch.gather(_src, 1, local_indices)
    return local_gathered_data


def __Local_Gather_impl(_src_tensor, local_indices):
    num_features = _src_tensor.shape[-1]
    bs = _src_tensor.shape[0]
    local_indices = local_indices.view(bs, -1, 1).expand(bs, -1, num_features)
    local_gathered_data = torch.gather(_src_tensor, 1, local_indices)
    return local_gathered_data


def OptimizedRankLocalMaskedGather(
    src: torch.Tensor,
    indices: torch.Tensor,
    rank_mapping: torch.Tensor,
    output: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """
    This function gathers the indices from the source rank to the destination rank.
    """
    if not _LOCAL_OPT_KERNELS_AVAILABLE:
        warnings.warn(
            "Optimized local kernels are not available. Falling back to the default implementation."
        )
        return RankLocalMaskedGather(src, indices, rank_mapping, rank)
    bs = src.shape[0]
    indices = indices.view(bs, -1, 1)
    num_output_rows = indices.shape[1]
    num_src_rows = src.shape[1]
    num_features = src.shape[-1]
    local_masked_gather(
        src,
        indices.cuda(),
        rank_mapping.cuda(),
        output,
        bs,
        num_src_rows,
        num_features,
        num_output_rows,
        rank,
    )
    return output


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
    local_indices_slice: torch.Tensor,
    local_dest_ranks: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """
    This function scatters the data from the source rank to the destination rank.
    """
    local_comm_mask = local_dest_ranks == rank
    num_features = _src.shape[-1]
    num_local_output_rows = _output.shape[1]

    if torch.any(local_comm_mask):
        local_scatter_indices = local_indices_slice[local_comm_mask]
        local_scatter_indices = local_scatter_indices.view(1, -1, 1).expand(
            1, -1, num_features
        )

        _output.scatter_add_(
            1,
            local_scatter_indices % num_local_output_rows,
            _src[:, local_comm_mask, :],
        )
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


def RankLocalRenumberingWithMapping(_indices, rank_mapping):
    """
    This function removes duplicates from the indices tensor.
    """
    unique_indices, inverse_indices = torch.unique(_indices, return_inverse=True)
    rank_mapping = rank_mapping.to(_indices.device)
    renumbered_indices = inverse_indices
    unique_rank_mapping = torch.zeros_like(unique_indices, dtype=rank_mapping.dtype, device=rank_mapping.device)
    unique_rank_mapping.scatter_(0, inverse_indices, rank_mapping)

    return renumbered_indices, unique_indices, unique_rank_mapping


def RankLocalGather(
    _src: torch.Tensor, indices: torch.Tensor, rank_mapping: torch.Tensor, rank: int
) -> torch.Tensor:
    """
    This function gathers the indices from the source rank to the destination rank.
    """
    local_indices = indices[rank_mapping == rank]
    local_gathered_data = torch.gather(_src, 0, local_indices)
    return local_gathered_data


def LocalAggregateWithRemapping(
    global_data: torch.Tensor,
    global_indices: torch.Tensor,
    global_mapping: torch.Tensor,
    num_features: int,
    device: torch.device,
):
    """
    This function aggregates the global_data such that rows with the same
    global_indices are aggregated. The global_mapping is used to map the
    global_indices to the local indices. Mathematically,

    local_aggregated_data[local_mapping[global_indices[i]] += global_data[i]

    Also returns the new_mapping tensor which maps the local indices to the
    appropriate global rank. Mathematically,

    new_mapping[k] = global_mapping[torch.argwhere(local_mapping == k)[0]]

    Args:
        global_data (torch.Tensor): The global data tensor
        global_indices (torch.Tensor): The global indices tensor
        global_mapping (torch.Tensor): The global mapping tensor
        num_features (int): The number of features in the global_data
        device (torch.device): The device to use
    returns:
        local_aggregated_data (torch.Tensor): The locally aggregated data tensor
        new_mapping (torch.Tensor): The new mapping tensor
    """
    renumbered_indices, unique_indices, new_mapping = RankLocalRenumberingWithMapping(
        global_indices, global_mapping
    )

    num_unique_indices = unique_indices.shape[0]

    local_aggregated_data = torch.zeros(1, num_unique_indices, num_features).to(device)

    renumbered_indices = renumbered_indices.view(1, -1, 1).expand(1, -1, num_features)
    local_aggregated_data.scatter_add_(1, renumbered_indices, global_data)

    return local_aggregated_data, new_mapping
