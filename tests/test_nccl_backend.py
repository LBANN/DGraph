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
def setup_gather_data(init_nccl_backend):
    comm = init_nccl_backend
    # Must make sure all the ranks generate the same random data
    torch.manual_seed(0)
    torch.cuda.set_device(comm.get_rank())
    all_rank_input_data = torch.randn(1, 4, 64)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])

    rank_mappings = torch.tensor(
        [
            [[0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]],
            [[0, 1, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1]],
        ]
    )

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        for i in range(8):
            all_rank_output[k][i] = all_rank_input_data[:, all_edge_coo[k, i]]

    return comm, all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


@pytest.fixture(scope="module")
def setup_scatter_data():
    num_features = 4
    all_rank_input_data = torch.randn(
        1, 8, num_features, requires_grad=True, device="cuda"
    )
    all_rank_indices = torch.tensor(
        [[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]]
    )
    rank_mappings = torch.tensor(
        [
            [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1]],
            [[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]],
        ]
    )

    all_rank_output_0 = torch.zeros(1, 4, num_features)
    all_rank_output_0.scatter_add_(
        1,
        all_rank_indices[0].view(1, -1, 1).expand(1, -1, num_features),
        all_rank_input_data,
    )

    all_rank_output_1 = torch.zeros(1, 4, num_features)
    all_rank_output_1.scatter_add_(
        1,
        all_rank_indices[1].view(1, -1, 1).expand(1, -1, num_features),
        all_rank_input_data,
    )

    all_rank_output = torch.cat([all_rank_output_0, all_rank_output_1], dim=0)

    return all_rank_input_data, all_rank_indices, rank_mappings, all_rank_output


def test_nccl_backend_init(init_nccl_backend):
    comm = init_nccl_backend
    rank = comm.get_rank()
    print(f"Rank: {rank}")


def test_nccl_backend_gather(setup_gather_data):

    comm, all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output = (
        setup_gather_data
    )

    local_input_data = comm.get_local_rank_slice(all_rank_input_data)

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    # print(rank, world_size)
    start_index = (all_rank_input_data.shape[1] // world_size) * rank
    end_index = (all_rank_input_data.shape[1] // world_size) * (rank + 1)
    local_input_data_gt = all_rank_input_data[:, start_index:end_index]

    output_start_index = (all_rank_output.shape[1] // world_size) * rank
    output_end_index = (all_rank_output.shape[1] // world_size) * (rank + 1)

    local_output_data_gt = all_rank_output[:, output_start_index:output_end_index]

    assert local_input_data.shape == (1, 2, 64)
    assert torch.allclose(local_input_data, local_input_data_gt)
    assert rank_mappings.shape == (2, 2, 8)
    assert all_rank_output.shape == (2, 8, 64)
    assert local_output_data_gt.shape == (2, 4, 64)
    assert all_edge_coo.shape == (2, 8)

    dgraph_output_tensor_0 = comm.gather(
        local_input_data.cuda(), all_edge_coo[[0]].cuda(), rank_mappings[0]
    )
    assert dgraph_output_tensor_0.shape == (1, 4, 64)
    assert torch.allclose(dgraph_output_tensor_0.cpu(), local_output_data_gt[0])

    dgraph_output_tensor_1 = comm.gather(
        local_input_data.cuda(), all_edge_coo[[1]].cuda(), rank_mappings[1]
    )

    assert dgraph_output_tensor_1.shape == (1, 4, 64)
    assert torch.allclose(dgraph_output_tensor_1.cpu(), local_output_data_gt[1])


def test_nccl_backend_scatter(init_nccl_backend, setup_scatter_data):
    comm: Comm.Communicator = init_nccl_backend
    all_rank_input_data, all_rank_indices, rank_mappings, all_rank_output = (
        setup_scatter_data
    )

    local_input_data = comm.get_local_rank_slice(all_rank_input_data)

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # local_output = torch.zeros(1, local_size, 64)
    local_output_size = all_rank_output.shape[1] // world_size
    output_start_index = local_output_size * rank
    output_end_index = local_output_size * (rank + 1)

    for i in range(all_rank_indices.shape[0]):
        local_output_gt = all_rank_output[[i], output_start_index:output_end_index]
        dgraph_output_tensor = comm.scatter(
            local_input_data.cuda(), all_rank_indices[i].cuda(), rank_mappings[i], 2
        )
        assert local_output_gt.shape == (1, 2, 4)
        assert dgraph_output_tensor.shape == (1, 2, 4)
        assert torch.allclose(dgraph_output_tensor.cpu(), local_output_gt)
