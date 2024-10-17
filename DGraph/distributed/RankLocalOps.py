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

    # assert torch.all(renumbered_indices[unique_indices] == torch.arange(0, len(unique_indices)))
    return renumbered_indices, unique_indices
