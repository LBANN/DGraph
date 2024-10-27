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
import torch.distributed as dist

import DGraph.Communicator as Comm
import torch


@pytest.fixture(scope="module")
def init_nccl_backend():
    dist.init_process_group(backend="nccl")

    comm = Comm.Communicator.init_process_group("nccl")
    return comm


@pytest.fixture(scope="module")
def setup_gather_data():
    all_rank_input_data = torch.randn(1, 4, 64)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])
    rank_mappings = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]])

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        for i in range(8):
            all_rank_output[k][i] = all_rank_input_data[:, all_edge_coo[k, i]]

    return all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


@pytest.fixture(scope="module")
def setup_scatter_data():
    all_rank_input_data = torch.randn(1, 4, 64)
    all_rank_indices = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2]])
    rank_mappings = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    all_rank_output = torch.tensor([2.0, 4.0, 1.0, 3.0]).view(1, -1, 1)
    return all_rank_input_data, all_rank_indices, rank_mappings, all_rank_output


def test_nccl_backend_init(init_nccl_backend):
    comm = init_nccl_backend
    rank = comm.get_rank()
    print(f"Rank: {rank}")


def test_nccl_backend_gather(init_nccl_backend, setup_gather_data):
    comm: Comm.Communicator = init_nccl_backend
    all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output = (
        setup_gather_data
    )

    local_input_data = comm.get_local_rank_slice(all_rank_input_data)

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    # print(rank, world_size)
    start_index = (all_rank_input_data.shape[1] // world_size) * rank
    end_index = (all_rank_input_data.shape[1] // world_size) * (rank + 1)
    print(start_index, end_index, all_rank_input_data.shape, local_input_data.shape)
    local_input_data_gt = all_rank_input_data[:, start_index:end_index]
    # assert False
    assert local_input_data.shape == (1, 2, 64)
    assert torch.allclose(local_input_data, local_input_data_gt)


def test_nccl_backend_scatter(init_nccl_backend, setup_scatter_data):
    comm: Comm.Communicator = init_nccl_backend
    all_rank_input_data, all_rank_indices, rank_mappings, all_rank_output = (
        setup_scatter_data
    )

    local_input_data = comm.get_local_rank_slice(all_rank_input_data)
    local_indices = comm.get_local_rank_slice(all_rank_indices)
    local_rank_mappings = comm.get_local_rank_slice(rank_mappings)

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    local_size = all_rank_input_data.shape[1] // world_size
    start_index = local_size * rank
    end_index = local_size * (rank + 1)

    local_output = torch.zeros(1, local_size, 64)
    for i in range(local_size):
        local_output[:, i] = local_input_data[:, local_indices[0, i]]

    assert local_output.shape == (1, 2, 64)
    assert torch.allclose(local_output, all_rank_output[:, start_index:end_index])
