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
import pytest

import torch


def test_local_kernel_submodule_import():
    try:
        import torch_local
    except ImportError as e:
        pytest.fail(f"Failed to import torch_local: {e}")


def test_optimized_local_gather():
    try:
        from torch_local import local_masked_gather
    except ImportError as e:
        pytest.fail(f"Failed to import local_masked_gather: {e}")

    num_src_rows = 4
    num_out_rows = 8
    num_features = 16
    bs = 1
    src_tensor = torch.randn(bs, num_src_rows, num_features)
    indices = torch.tensor([0, 0, 2, 1, 1, 2, 3, 3])
    rank_mapping = torch.tensor([0, 0, 1, 0, 0, 1, 1, 1])
    rank = 0

    out_tensor_gt = torch.zeros(bs, num_out_rows, num_features)

    for i in range(bs):
        for j in range(num_out_rows):
            if rank_mapping[j] == rank:
                out_tensor_gt[i, j] = src_tensor[i, indices[j]]

    out_tensor_gt = out_tensor_gt.view(bs, num_out_rows, num_features)

    out_tensor = torch.zeros_like(out_tensor_gt)
    local_masked_gather(
        src_tensor,
        indices.long(),
        rank_mapping.long(),
        out_tensor,
        bs,
        num_src_rows,
        num_features,
        num_out_rows,
        rank,
    )

    assert torch.allclose(out_tensor, out_tensor_gt), "Optimized local gather failed"
