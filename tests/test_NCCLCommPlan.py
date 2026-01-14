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
    dist.init_process_group(backend="nccl")

    comm = Communicator.init_process_group("nccl")

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


def test_coo_to_nccl_comm_plan(init_nccl_backend_communicator):
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
