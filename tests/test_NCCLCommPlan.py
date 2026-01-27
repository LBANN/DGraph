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
from DGraph.distributed.nccl import (
    NCCLGraphCommPlan,
    COO_to_NCCLCommPlan,
    COO_to_NCCLEdgeConditionedCommPlan,
)
from DGraph.Communicator import Communicator
import torch.distributed as dist
import torch


@pytest.fixture(scope="module")
def init_nccl_backend_communicator():
    comm = Communicator.init_process_group("nccl")

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    devices_per_rank = torch.cuda.device_count()

    device = torch.device(f"cuda:{rank % devices_per_rank}")
    torch.cuda.set_device(device)

    return comm


def setup_coo_matrix(world_size):
    torch.manual_seed(0)
    num_nodes = 32 * world_size

    # generate num_nodes x num_nodes adjacency matrix
    adj_matrix = torch.rand(num_nodes, num_nodes)
    adj_matrix = (adj_matrix + adj_matrix.t()) / 2
    adj_matrix[adj_matrix < 0.8] = 0.0  # sparsify
    adj_matrix[adj_matrix >= 0.8] = 1.0
    adj_matrix.fill_diagonal_(0)
    coo_matrix = adj_matrix.nonzero(as_tuple=False).t().contiguous()
    return num_nodes, coo_matrix


@pytest.fixture(scope="module")
def setup_graph_data(init_nccl_backend_communicator):
    comm = init_nccl_backend_communicator
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    num_nodes, coo_matrix = setup_coo_matrix(world_size)
    coo_matrix = coo_matrix.to(device)
    nodes_per_rank = (num_nodes + world_size - 1) // world_size
    offset = torch.arange(world_size + 1, device=device) * nodes_per_rank
    offset[-1] = max(offset[-1], num_nodes)
    num_edges = coo_matrix.shape[-1]

    torch.random.manual_seed(0)
    num_features = 16
    global_X = torch.rand(num_nodes, num_features).cuda()
    global_E = torch.zeros(num_edges, num_features).cuda()

    return coo_matrix, offset, num_nodes, num_edges, global_X, global_E


def get_local_indices(rank, offset, src):
    my_start = offset[rank]
    my_end = offset[rank + 1]
    is_local_edge = (src >= my_start) & (src < my_end)
    local_edge_indices = torch.nonzero(is_local_edge, as_tuple=True)[0]
    return local_edge_indices


@pytest.fixture(scope="module")
def setup_gather_ground_truth(setup_graph_data):
    comm = init_nccl_backend_communicator
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    global_coo_matrix, offset, num_nodes, num_edges, global_X, global_E = (
        setup_graph_data
    )

    dst = global_coo_matrix[1]
    src = global_coo_matrix[0]
    for i in range(num_edges):
        global_E[i, :] = global_X[dst[i], :]

    my_start = offset[rank]
    my_end = offset[rank + 1]

    local_edge_indices = get_local_indices(rank, offset, src)

    local_E = global_E[local_edge_indices, :]

    local_X = global_X[my_start:my_end, :]

    return local_E, local_X, dst, offset, local_edge_indices


@pytest.fixture(scope="module")
def setup_scatter_ground_truth(setup_graph_data):
    comm = init_nccl_backend_communicator
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    global_coo_matrix, offset, num_nodes, num_edges, global_X, global_E = (
        setup_graph_data
    )

    dst = global_coo_matrix[1]
    src = global_coo_matrix[0]

    global_Y = torch.zeros(num_nodes, global_X.shape[1]).to(device=global_X.device)

    for i in range(num_edges):
        global_Y[dst[i], :] += global_E[i, :]

    my_start = offset[rank]
    my_end = offset[rank + 1]
    local_Y = global_Y[my_start:my_end, :]

    local_indices = get_local_indices(rank, offset, src)
    local_E = global_E[local_indices, :]

    return local_E, local_Y, dst, offset, local_indices


def test_coo_to_nccl_comm_plan(init_nccl_backend_communicator, setup_graph_data):
    comm = init_nccl_backend_communicator

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    coo_matrix, offset, num_nodes, num_edges, global_X, global_E = setup_graph_data

    my_start = offset[rank]
    my_end = offset[rank + 1]

    src = coo_matrix[0]
    dst = coo_matrix[1]

    is_local_edge = (src >= my_start) & (src < my_end)
    local_edge_indices = torch.nonzero(is_local_edge, as_tuple=True)[0]

    plan = COO_to_NCCLCommPlan(
        rank=rank,
        world_size=world_size,
        global_edges_dst=dst,
        local_edge_list=local_edge_indices,
        offset=offset,
    )

    # 1. Check internal vs boundary edges
    my_dst = dst[local_edge_indices]
    is_internal_gt = (my_dst >= my_start) & (my_dst < my_end)
    internal_indices_gt = torch.nonzero(is_internal_gt, as_tuple=True)[0]

    assert torch.equal(
        plan.local_edge_idx.sort()[0], internal_indices_gt.sort()[0]
    ), f"Rank {rank}: Local edge indices mismatch"

    internal_dst_gt = my_dst[internal_indices_gt]
    local_vertex_idx_gt = internal_dst_gt - my_start

    assert torch.equal(
        plan.local_vertex_idx.sort()[0], local_vertex_idx_gt.sort()[0]
    ), f"Rank {rank}: Local vertex indices mismatch"

    # 2. Check boundary edges
    boundary_indices_gt = torch.nonzero(~is_internal_gt, as_tuple=True)[0]
    assert torch.equal(
        plan.boundary_edge_idx.sort()[0], boundary_indices_gt.sort()[0]
    ), f"Rank {rank}: Boundary edge indices mismatch"

    # 3. Check boundary vertices (received from other ranks)
    expected_recv_vertices_unique_per_rank = []
    for r in range(world_size):
        if r == rank:
            continue
        r_start = offset[r]
        r_end = offset[r + 1]
        is_r_edge = (src >= r_start) & (src < r_end)
        r_dst = dst[is_r_edge]
        is_to_me = (r_dst >= my_start) & (r_dst < my_end)
        dst_to_me = r_dst[is_to_me]
        unique_dst_to_me = torch.unique(dst_to_me)
        expected_recv_vertices_unique_per_rank.append(unique_dst_to_me)

    if len(expected_recv_vertices_unique_per_rank) > 0:
        expected_recv_stream = torch.cat(expected_recv_vertices_unique_per_rank)
    else:
        expected_recv_stream = torch.tensor([], device=device, dtype=torch.long)

    expected_local_stream = expected_recv_stream - my_start

    assert torch.equal(
        plan.boundary_vertex_idx.sort()[0], expected_local_stream.sort()[0]
    ), f"Rank {rank}: Boundary vertex indices mismatch"


def test_edge_conditioned_comm_plan(init_nccl_backend_communicator):
    comm = init_nccl_backend_communicator
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    num_nodes, coo_matrix = setup_coo_matrix(world_size)
    coo_matrix = coo_matrix.to(device)

    nodes_per_rank = num_nodes // world_size
    offset = torch.arange(world_size + 1, device=device) * nodes_per_rank

    my_start = offset[rank]
    my_end = offset[rank + 1]

    src = coo_matrix[0]
    dst = coo_matrix[1]
    is_local_edge = (src >= my_start) & (src < my_end)
    local_edge_indices = torch.nonzero(is_local_edge, as_tuple=True)[0]

    ec_plan = COO_to_NCCLEdgeConditionedCommPlan(
        rank=rank,
        world_size=world_size,
        global_edges_src=src,
        global_edges_dst=dst,
        local_edge_list=local_edge_indices,
        src_offset=offset,
        dest_offset=offset,
    )

    assert ec_plan.source_graph_plan is not None
    assert ec_plan.dest_graph_plan is not None

    assert ec_plan.source_graph_plan.boundary_edge_idx.numel() == 0
    assert (
        ec_plan.source_graph_plan.local_edge_idx.numel() == local_edge_indices.numel()
    )
    assert ec_plan.dest_graph_plan.num_local_edges == local_edge_indices.numel()


def test_comm_plan_gather(init_nccl_backend_communicator, setup_gather_ground_truth):
    comm = init_nccl_backend_communicator

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    local_E, local_X, dst, offset, local_edge_indices = setup_gather_ground_truth

    plan = COO_to_NCCLCommPlan(
        rank=rank,
        world_size=world_size,
        global_edges_dst=dst,
        local_edge_list=local_edge_indices,
        offset=offset,
    )
    comm_local_E = comm.gather(local_X.unsqueeze_(0), comm_plan=plan)
    comm_local_E.squeeze_(0)
    torch.allclose(local_E, comm_local_E)


def test_comm_plan_scatter_sum(
    init_nccl_backend_communicator, setup_scatter_ground_truth
):
    comm = init_nccl_backend_communicator

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    local_E, local_Y, dst, offset, local_indices = setup_scatter_ground_truth

    plan = COO_to_NCCLCommPlan(
        rank=rank,
        world_size=world_size,
        global_edges_dst=dst,
        local_edge_list=local_indices,
        offset=offset,
    )
    comm_local_Y = comm.scatter(local_E.unsqueeze_(0), comm_plan=plan)
    comm_local_Y.squeeze_(0)
    torch.allclose(local_Y, comm_local_Y)
