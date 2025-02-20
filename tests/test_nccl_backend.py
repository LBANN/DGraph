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
    print(f"Rank: {comm.get_rank()}")
    torch.cuda.set_device(comm.get_rank())
    all_rank_input_data = torch.randn(1, 4, 64)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])

    rank_mappings = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]],
    )

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        for i in range(8):
            all_rank_output[k][i] = all_rank_input_data[:, all_edge_coo[k, i]]

    return comm, all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


@pytest.fixture(scope="module")
def setup_unbalanced_gather_data(init_nccl_backend):
    # Set up the data for the gather operation where
    # there are uneven number of edges per rank
    comm = init_nccl_backend
    torch.manual_seed(0)
    torch.cuda.set_device(comm.get_rank())
    num_global_rows = 16
    num_features = 64
    all_rank_input_data = torch.randn(1, num_global_rows, num_features)
    node_rank_placement = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    # Graph viz:
    # 0 - 2     4  - 6
    # | X |  X  |    |
    # 1 - 3     5  - 7
    # rank 0    rank 1

    all_edge_coo = torch.tensor(
        [  # Rank mappings in comments
            [0, 1],  # (0, 0)
            [0, 2],  # (0, 0)
            [0, 3],  # (0, 0)
            [1, 0],  # (0, 0)
            [1, 2],  # (0, 0)
            [1, 3],  # (0, 0)
            [2, 0],  # (0, 0)
            [2, 1],  # (0, 0)
            [2, 3],  # (0, 0)
            [2, 5],  # (0, 1)
            [3, 0],  # (0, 0)
            [3, 1],  # (0, 0)
            [3, 2],  # (0, 0)
            [3, 4],  # (0, 1)
            [4, 3],  # (1, 0)
            [4, 5],  # (1, 1)
            [4, 6],  # (1, 1)
            [5, 2],  # (0, 1)
            [5, 4],  # (1, 1)
            [5, 7],  # (1, 1)
            [6, 4],  # (1, 1)
            [6, 7],  # (1, 1)
            [7, 5],  # (1, 1)
            [7, 6],  # (1, 1)
        ]
    ).T

    rank_mappings = torch.zeros_like(all_edge_coo)
    rank_1_indices = all_edge_coo > 3
    rank_mappings[rank_1_indices] = 1

    num_edges = all_edge_coo.shape[1]
    all_rank_output = torch.zeros(2, num_edges, num_features)

    for k in range(2):
        for i in range(num_edges):
            all_rank_output[k][i] = all_rank_input_data[:, all_edge_coo[k, i]]

    return (
        comm,
        all_rank_input_data,
        all_edge_coo,
        node_rank_placement,
        rank_mappings,
        all_rank_output,
    )


@pytest.fixture(scope="module")
def setup_scatter_data(init_nccl_backend):
    comm = init_nccl_backend
    torch.manual_seed(0)
    torch.cuda.set_device(comm.get_rank())
    num_features = 4
    all_rank_input_data = torch.randn(1, 8, num_features, requires_grad=True)
    all_rank_indices = torch.tensor(
        [[0, 0, 0, 1, 2, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]]
    )
    rank_mappings = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]],
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


@pytest.fixture(scope="module")
def setup_unbalanced_scatter_data(init_nccl_backend):
    num_features = 64
    comm = init_nccl_backend
    torch.manual_seed(0)
    torch.cuda.set_device(comm.get_rank())
    num_global_rows = 12
    all_rank_input_data = torch.randn(1, num_global_rows, num_features)
    node_rank_placement = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    # Graph viz:
    # 0 - 2     4  - 6
    # | X |  X  |    |
    # 1 - 3     5  - 7
    # rank 0    rank 1

    all_rank_indices = torch.tensor(
        [  # Rank mappings in comments
            [0, 1],  # (0, 0)
            [0, 2],  # (0, 0)
            [0, 3],  # (0, 0)
            [1, 0],  # (0, 0)
            [1, 2],  # (0, 0)
            [1, 3],  # (0, 0)
            [2, 0],  # (0, 0)
            [2, 1],  # (0, 0)
            [2, 3],  # (0, 0)
            [2, 5],  # (0, 1)
            [3, 0],  # (0, 0)
            [3, 1],  # (0, 0)
            [3, 2],  # (0, 0)
            [3, 4],  # (0, 1)
            [4, 3],  # (1, 0)
            [4, 5],  # (1, 1)
            [4, 6],  # (1, 1)
            [5, 2],  # (0, 1)
            [5, 4],  # (1, 1)
            [5, 7],  # (1, 1)
            [6, 4],  # (1, 1)
            [6, 7],  # (1, 1)
            [7, 5],  # (1, 1)
            [7, 6],  # (1, 1)
        ]
    ).T

    rank_mappings = torch.zeros_like(all_rank_indices)
    rank_1_indices = all_rank_indices > 3
    rank_mappings[rank_1_indices] = 1

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

    return (
        comm,
        all_rank_input_data,
        all_rank_indices,
        rank_mappings,
        node_rank_placement,
        all_rank_output,
    )


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
    assert rank_mappings.shape == (2, 8)
    assert all_rank_output.shape == (2, 8, 64)
    assert local_output_data_gt.shape == (2, 4, 64)
    assert all_edge_coo.shape == (2, 8)

    dgraph_output_tensor_0 = comm.gather(
        local_input_data.cuda(), all_edge_coo[[0]].cuda(), rank_mappings
    )
    assert dgraph_output_tensor_0.shape == (1, 4, 64)
    assert torch.allclose(dgraph_output_tensor_0.cpu(), local_output_data_gt[0])

    dgraph_output_tensor_1 = comm.gather(
        local_input_data.cuda(), all_edge_coo[[1]].cuda(), rank_mappings
    )

    assert dgraph_output_tensor_1.shape == (1, 4, 64)
    assert torch.allclose(dgraph_output_tensor_1.cpu(), local_output_data_gt[1])


def test_nccl_backend_gather_assymetric(setup_unbalanced_gather_data):

    (
        comm,
        all_rank_input_data,
        all_edge_coo,
        node_rank_placement,
        rank_mappings,
        all_rank_output,
    ) = setup_unbalanced_gather_data

    local_input_data = comm.get_local_rank_slice(all_rank_input_data)

    rank = comm.get_rank()

    local_input_rows = node_rank_placement == rank
    local_input_data_gt = all_rank_input_data[:, local_input_rows, :]

    assert local_input_data.shape == (1, 2, 64)
    assert torch.allclose(local_input_data, local_input_data_gt)
    assert rank_mappings.shape == (2, 8)
    assert all_rank_output.shape == (2, 8, 64)
    assert all_edge_coo.shape == (2, 8)

    for i in range(2):
        edge_rank_placement = node_rank_placement[i]
        local_output_data_gt = all_rank_output[
            i, node_rank_placement == edge_rank_placement, :
        ]
        dgraph_output_tensor = comm.gather(
            local_input_data.cuda(), all_edge_coo[[i]].cuda(), rank_mappings
        )
        assert torch.allclose(dgraph_output_tensor.cpu(), local_output_data_gt[0])


def test_nccl_backend_scatter(init_nccl_backend, setup_scatter_data):
    comm: Comm.Communicator = init_nccl_backend
    all_rank_input_data, all_rank_indices, rank_mappings, all_rank_output = (
        setup_scatter_data
    )
    local_edge_placement = rank_mappings[0]
    local_input_data = comm.get_local_tensor(
        all_rank_input_data, local_edge_placement, dim=1
    )

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    # local_output = torch.zeros(1, local_size, 64)
    local_output_size = all_rank_output.shape[1] // world_size
    output_start_index = local_output_size * rank
    output_end_index = local_output_size * (rank + 1)

    for i in range(all_rank_indices.shape[0]):
        local_output_gt = all_rank_output[[i], output_start_index:output_end_index]
        _mappings = torch.cat(
            [local_edge_placement.unsqueeze(0), rank_mappings[[i]]], dim=0
        )
        dgraph_output_tensor = comm.scatter(
            local_input_data.cuda(), all_rank_indices[i].cuda(), _mappings, 2
        )
        assert local_output_gt.shape == (1, 2, 4)
        assert dgraph_output_tensor.shape == (1, 2, 4)
        assert torch.allclose(dgraph_output_tensor.cpu(), local_output_gt)
