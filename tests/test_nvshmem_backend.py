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
def init_nvshmem_backend():
    dist.init_process_group(backend="nvshmem")

    comm = Comm.Communicator.init_process_group("nvshmem")
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
    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])
    rank_mappings = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]])

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        for i in range(4):
            all_rank_output[k][all_edge_coo[k, i]] += all_rank_input_data[:, i]

    return all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


def test_nvshmem_backend_init(init_nvshmem_backend):
    comm = init_nvshmem_backend
    rank = comm.get_rank()
    print(f"Rank: {rank}")


def test_nvshmem_backend_gather(init_nvshmem_backend, setup_gather_data):
    comm: Comm.Communicator = init_nvshmem_backend
    all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output = (
        setup_gather_data
    )

    local_input_data = comm.get_local_rank_slice(all_rank_input_data)
    local_index = comm.get_local_rank_slice(all_edge_coo)
    local_rank_mappings = comm.get_local_rank_slice(rank_mappings)

    local_output = comm.gather(local_input_data, local_index, local_rank_mappings)

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    input_start_index = (all_rank_input_data.shape[1] // world_size) * rank
    input_end_index = (all_rank_input_data.shape[1] // world_size) * (rank + 1)
    local_input_data_gt = all_rank_input_data[:, input_start_index:input_end_index]
    local_index_gt = all_edge_coo[:, input_start_index:input_end_index]
    local_rank_mappings_gt = rank_mappings[:, input_start_index:input_end_index]

    # Check if the local slicing is correct
    assert torch.allclose(local_input_data, local_input_data_gt)
    assert torch.allclose(local_index.squeeze(0), local_index_gt)
    assert torch.allclose(local_rank_mappings.squeeze(0), local_rank_mappings_gt)
    assert torch.allclose(local_output, all_rank_output)

    # Check if the gather is correct
    output_start_index = (all_rank_output.shape[1] // world_size) * rank
    output_end_index = (all_rank_output.shape[1] // world_size) * (rank + 1)

    local_output_gt = all_rank_output[:, output_start_index:output_end_index]

    assert torch.allclose(local_output, local_output_gt)


def test_nvshmem_backend_scatter(init_mpi_backend, setup_scatter_data):
    comm = init_mpi_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output = (
        setup_scatter_data
    )

    all_edge_coo = all_edge_coo.T
    rank_mappings = rank_mappings.T

    local_input_data = comm.get_local_rank_slice(all_rank_input_data)
    local_index = comm.get_local_rank_slice(all_edge_coo.unsqueeze(0))
    local_rank_mappings = comm.get_local_rank_slice(rank_mappings.unsqueeze(0))
    local_output = comm.scatter(local_input_data, local_index, local_rank_mappings)

    input_start_index = (all_rank_input_data.shape[1] // world_size) * rank
    input_end_index = (all_rank_input_data.shape[1] // world_size) * (rank + 1)

    local_input_data_gt = all_rank_input_data[:, input_start_index:input_end_index]
    local_index_gt = all_edge_coo[:, input_start_index:input_end_index]
    local_rank_mappings_gt = rank_mappings[:, input_start_index:input_end_index]

    # Check if the local slicing is correct
    assert torch.allclose(local_input_data, local_input_data_gt)
    assert torch.allclose(local_index.squeeze(0), local_index_gt)
    assert torch.allclose(local_rank_mappings.squeeze(0), local_rank_mappings_gt)

    # Check if the scatter is correct
    output_start_index = (all_rank_output.shape[1] // world_size) * rank
    output_end_index = (all_rank_output.shape[1] // world_size) * (rank + 1)

    local_output_gt = all_rank_output[:, output_start_index:output_end_index]

    assert torch.allclose(local_output, local_output_gt)
