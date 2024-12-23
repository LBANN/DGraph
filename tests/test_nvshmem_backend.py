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

    comm = Comm.Communicator.init_process_group("nvshmem")
    return comm


@pytest.fixture(scope="module")
def setup_gather_data(init_nvshmem_backend):
    comm = init_nvshmem_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.manual_seed(0)

    num_features = 64
    all_rank_input_data = torch.randn(1, 4, 64)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])
    rank_mappings = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]])

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        _indices = all_edge_coo[k].view(1, -1, 1).expand(1, -1, num_features)
        output_data = torch.gather(all_rank_input_data, 1, _indices)
        all_rank_output[k] = output_data.squeeze(0)

    return all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


@pytest.fixture(scope="module")
def setup_scatter_data(init_nvshmem_backend):

    comm = init_nvshmem_backend
    rank = comm.get_rank()

    torch.cuda.set_device(rank % torch.cuda.device_count())

    torch.manual_seed(0)

    num_features = 8
    all_rank_input_data = torch.randn(1, 8, num_features)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])
    rank_mappings = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]])

    all_rank_output = torch.zeros(2, 4, num_features)

    for k in range(2):
        _indices = all_edge_coo[k].view(1, -1, 1).expand(1, -1, num_features)
        output_data = torch.zeros_like(all_rank_output[[k]])
        output_data.scatter_add_(1, _indices, all_rank_input_data)
        all_rank_output[k] = output_data

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

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    input_slice_start = (all_rank_input_data.shape[1] // world_size) * rank
    input_slice_end = (all_rank_input_data.shape[1] // world_size) * (rank + 1)

    edge_slice_start = (all_edge_coo.shape[1] // world_size) * rank
    edge_slice_end = (all_edge_coo.shape[1] // world_size) * (rank + 1)

    local_input_data_gt = all_rank_input_data[:, input_slice_start:input_slice_end, :]
    local_index_gt = all_edge_coo[:, edge_slice_start:edge_slice_end]
    local_rank_mappings_gt = rank_mappings[:, edge_slice_start:edge_slice_end]

    # Check if the gather is correct
    output_start_index = (all_rank_output.shape[1] // world_size) * rank
    output_end_index = (all_rank_output.shape[1] // world_size) * (rank + 1)

    local_output_gt = all_rank_output[:, output_start_index:output_end_index]

    for i in range(2):
        local_indices_gt = local_index_gt[[i], :]
        local_indices = comm.get_local_rank_slice(all_edge_coo[[i]], dim=1)
        assert torch.allclose(local_indices, local_indices_gt)

        local_rank_mapping_gt = local_rank_mappings_gt[[i], :]
        local_rank_mapping = comm.get_local_rank_slice(rank_mappings[[i]], dim=1)
        assert torch.allclose(local_rank_mapping, local_rank_mapping_gt)

        local_input_data = comm.get_local_rank_slice(all_rank_input_data, dim=1)
        assert torch.allclose(local_input_data, local_input_data_gt)

        gathered_tensor = comm.gather(
            local_input_data.cuda(), local_indices.cuda(), local_rank_mapping.cuda()
        )

        assert torch.allclose(gathered_tensor, local_output_gt[[i]].cuda())


def test_nvshmem_backend_scatter(init_nvshmem_backend, setup_scatter_data):
    comm = init_nvshmem_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output = (
        setup_scatter_data
    )

    all_edge_coo = all_edge_coo.T
    rank_mappings = rank_mappings.T

    input_slice_start = (all_rank_input_data.shape[1] // world_size) * rank
    input_slice_end = (all_rank_input_data.shape[1] // world_size) * (rank + 1)

    edge_slice_start = (all_edge_coo.shape[1] // world_size) * rank
    edge_slice_end = (all_edge_coo.shape[1] // world_size) * (rank + 1)

    local_input_data_gt = all_rank_input_data[:, input_slice_start:input_slice_end, :]
    local_edge_coo = all_edge_coo[:, edge_slice_start:edge_slice_end]
    local_rank_mappings_gt = rank_mappings[:, edge_slice_start:edge_slice_end]
    local_output_data_gt = all_rank_output[:, edge_slice_start:edge_slice_end, :]

    for i in range(2):
        local_indices_gt = local_edge_coo[[i], :]
        local_indices = comm.get_local_rank_slice(all_edge_coo[[i]], dim=1)

        assert torch.allclose(local_indices, local_indices_gt)

        local_rank_mapping_gt = local_rank_mappings_gt[[i], :]
        local_rank_mapping = comm.get_local_rank_slice(rank_mappings[[i]], dim=1)
        assert torch.allclose(local_rank_mapping, local_rank_mapping_gt)

        local_input_data = comm.get_local_rank_slice(all_rank_input_data, dim=1)
        assert torch.allclose(local_input_data, local_input_data_gt)

        scattered_tensor = comm.scatter(
            local_input_data.cuda(), local_indices.cuda(), local_rank_mapping.cuda()
        )
        assert torch.allclose(scattered_tensor, local_output_data_gt[[i]].cuda())
