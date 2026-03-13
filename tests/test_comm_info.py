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
Tests for DGraph.distributed.commInfo

Single-process tests (no dist required):
    Run with: python -m pytest tests/test_comm_info.py

Distributed tests (require 2 GPUs):
    Run with: torchrun --nnodes 1 --nproc-per-node 2 -m pytest tests/test_comm_info.py

Test graphs
-----------
Homogeneous (4 vertices, 2 ranks):

    Vertices  : 0, 1, 2, 3
    Partitioning: [0, 0, 1, 1]  (rank 0 → {0,1}, rank 1 → {2,3})
    Edges (undirected → stored as directed pairs):
        0↔1  local on rank 0
        0↔2  cross rank
        1↔3  cross rank
        2↔3  local on rank 1

    Rank 0 expected:
        local verts : [0, 1]
        halo verts  : [2, 3]
        local edges : (0,1),(1,0),(0,2),(1,3)  in local numbering
        send_local_idx : [0, 1]  (verts 0 and 1 sent to rank 1)
        send_offset    : [0, 0, 2]
        recv_offset    : [0, 0, 2]

    Rank 1 expected:
        local verts : [2, 3]
        halo verts  : [0, 1]
        local edges : (0,1),(1,0),(0,2),(1,3)  in local numbering (2→0,3→1,0→2,1→3)
        send_local_idx : [0, 1]  (local indices of verts 2 and 3, sent to rank 0)
        send_offset    : [0, 2, 2]
        recv_offset    : [0, 2, 2]

    comm_map = [[0, 2],
                [2, 0]]

Heterogeneous (V_src=3, V_dst=4, 2 ranks):

    src_partitioning = [0, 0, 1]   (rank 0 → src{0,1}, rank 1 → src{2})
    dst_partitioning = [0, 0, 1, 1] (rank 0 → dst{0,1}, rank 1 → dst{2,3})
    Edges (src_class → dst_class):
        (0,0), (0,2), (1,1), (1,3), (2,0), (2,2)

    Rank 0:
        halo dst verts  : [2, 3]  (cross edges (0,2),(1,3))
        boundary src verts: {0,1} → send_local_idx=[0,1], send_offset=[0,0,2]
    Rank 1:
        halo dst verts  : [0]     (cross edge (2,0))
        boundary src verts: {2}   → send_local_idx=[0],   send_offset=[0,1,1]
"""

import pytest
import torch
import torch.distributed as dist

from DGraph.distributed.commInfo import (
    CommunicationPattern,
    compute_local_vertices,
    compute_halo_vertices,
    compute_local_edge_list,
    compute_boundary_vertices,
    compute_comm_map,
    compute_recv_offsets,
    build_communication_pattern,
)

# ---------------------------------------------------------------------------
# Shared graph tensors (CPU, used by single-process tests)
# ---------------------------------------------------------------------------

# fmt: off
HOMO_EDGE_LIST = torch.tensor([
    [0, 1], [1, 0],   # local on rank 0
    [0, 2], [2, 0],   # cross
    [1, 3], [3, 1],   # cross
    [2, 3], [3, 2],   # local on rank 1
], dtype=torch.long)

HOMO_PARTITIONING = torch.tensor([0, 0, 1, 1], dtype=torch.long)

HETERO_EDGE_LIST = torch.tensor([
    [0, 0], [0, 2],
    [1, 1], [1, 3],
    [2, 0], [2, 2],
], dtype=torch.long)

HETERO_SRC_PARTITIONING = torch.tensor([0, 0, 1], dtype=torch.long)
HETERO_DST_PARTITIONING = torch.tensor([0, 0, 1, 1], dtype=torch.long)

# comm_map for the homogeneous graph (known analytically)
HOMO_COMM_MAP = torch.tensor([[0., 2.], [2., 0.]])
# fmt: on


# ===========================================================================
# compute_local_vertices
# ===========================================================================


@pytest.mark.parametrize(
    "rank, expected",
    [
        (0, torch.tensor([0, 1])),
        (1, torch.tensor([2, 3])),
    ],
)
def test_compute_local_vertices_correct_ids(rank, expected):
    result = compute_local_vertices(HOMO_PARTITIONING, rank)
    assert torch.equal(result, expected)


def test_compute_local_vertices_is_1d():
    result = compute_local_vertices(HOMO_PARTITIONING, rank=0)
    assert result.ndim == 1, "Result must be a 1-D tensor"


def test_compute_local_vertices_covers_all_ranks():
    all_local = torch.cat(
        [compute_local_vertices(HOMO_PARTITIONING, r) for r in range(2)]
    ).sort()[0]
    all_verts = torch.arange(HOMO_PARTITIONING.size(0))
    assert torch.equal(all_local, all_verts), "Union of local verts must cover all vertices"


# ===========================================================================
# compute_halo_vertices — homogeneous
# ===========================================================================


@pytest.mark.parametrize(
    "rank, expected_halo",
    [
        (0, torch.tensor([2, 3])),
        (1, torch.tensor([0, 1])),
    ],
)
def test_compute_halo_vertices_homogeneous(rank, expected_halo):
    result = compute_halo_vertices(HOMO_EDGE_LIST, HOMO_PARTITIONING, rank)
    assert result.ndim == 1
    assert torch.equal(result, expected_halo)


def test_compute_halo_vertices_no_cross_edges_returns_empty():
    edge_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    partitioning = torch.tensor([0, 0], dtype=torch.long)
    result = compute_halo_vertices(edge_list, partitioning, rank=0)
    assert result.numel() == 0


def test_compute_halo_vertices_unique():
    # Multiple edges to the same remote vertex should deduplicate
    edge_list = torch.tensor([[0, 2], [0, 2], [1, 2]], dtype=torch.long)
    partitioning = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    result = compute_halo_vertices(edge_list, partitioning, rank=0)
    assert result.tolist() == [2], f"Expected [2], got {result.tolist()}"


def test_compute_halo_vertices_dst_none_equals_src():
    """Passing dst_partitioning=None must be identical to passing src_partitioning."""
    r0_implicit = compute_halo_vertices(HOMO_EDGE_LIST, HOMO_PARTITIONING, rank=0)
    r0_explicit = compute_halo_vertices(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, rank=0, dst_partitioning=HOMO_PARTITIONING
    )
    assert torch.equal(r0_implicit, r0_explicit)


# ===========================================================================
# compute_halo_vertices — heterogeneous
# ===========================================================================


@pytest.mark.parametrize(
    "rank, expected_halo",
    [
        (0, torch.tensor([2, 3])),  # cross edges (0,2),(1,3)
        (1, torch.tensor([0])),     # cross edge  (2,0)
    ],
)
def test_compute_halo_vertices_heterogeneous(rank, expected_halo):
    result = compute_halo_vertices(
        HETERO_EDGE_LIST,
        HETERO_SRC_PARTITIONING,
        rank,
        dst_partitioning=HETERO_DST_PARTITIONING,
    )
    assert result.ndim == 1
    assert torch.equal(result, expected_halo)


def test_compute_halo_vertices_hetero_disjoint_from_local_dst():
    """Halo dst vertices must not be owned by this rank."""
    for rank in [0, 1]:
        halo = compute_halo_vertices(
            HETERO_EDGE_LIST,
            HETERO_SRC_PARTITIONING,
            rank,
            dst_partitioning=HETERO_DST_PARTITIONING,
        )
        halo_ranks = HETERO_DST_PARTITIONING[halo]
        assert (halo_ranks != rank).all(), (
            f"Rank {rank}: halo contains a locally-owned dst vertex"
        )


# ===========================================================================
# compute_local_edge_list
# ===========================================================================


@pytest.mark.parametrize(
    "rank, expected_edges",
    [
        # Rank 0: edges with src ∈ {0,1}; g2l: 0→0,1→1,2→2,3→3
        (0, torch.tensor([[0, 1], [1, 0], [0, 2], [1, 3]])),
        # Rank 1: edges with src ∈ {2,3}; g2l: 2→0,3→1,0→2,1→3
        (1, torch.tensor([[0, 2], [0, 1], [1, 3], [1, 0]])),
    ],
)
def test_compute_local_edge_list_correct_remapping(rank, expected_edges):
    local_verts = compute_local_vertices(HOMO_PARTITIONING, rank)
    halo_verts = compute_halo_vertices(HOMO_EDGE_LIST, HOMO_PARTITIONING, rank)
    result = compute_local_edge_list(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, halo_verts, rank
    )
    # Order of rows may differ; compare as sets of edge tuples
    result_set = set(map(tuple, result.tolist()))
    expected_set = set(map(tuple, expected_edges.tolist()))
    assert result_set == expected_set, (
        f"Rank {rank}: edge sets differ.\nGot: {result_set}\nExpected: {expected_set}"
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_compute_local_edge_list_source_always_local(rank):
    local_verts = compute_local_vertices(HOMO_PARTITIONING, rank)
    halo_verts = compute_halo_vertices(HOMO_EDGE_LIST, HOMO_PARTITIONING, rank)
    result = compute_local_edge_list(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, halo_verts, rank
    )
    num_local = local_verts.size(0)
    assert (result[:, 0] < num_local).all(), "All source indices must be in [0, num_local)"


@pytest.mark.parametrize("rank", [0, 1])
def test_compute_local_edge_list_all_indices_in_bounds(rank):
    local_verts = compute_local_vertices(HOMO_PARTITIONING, rank)
    halo_verts = compute_halo_vertices(HOMO_EDGE_LIST, HOMO_PARTITIONING, rank)
    result = compute_local_edge_list(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, halo_verts, rank
    )
    total = local_verts.size(0) + halo_verts.size(0)
    assert (result >= 0).all()
    assert (result < total).all()


# ===========================================================================
# compute_boundary_vertices — homogeneous
# ===========================================================================


@pytest.mark.parametrize(
    "rank, expected_send_offset",
    [
        (0, torch.tensor([0, 0, 2])),  # nothing to rank 0 (self), 2 to rank 1
        (1, torch.tensor([0, 2, 2])),  # 2 to rank 0, nothing to rank 1 (self)
    ],
)
def test_compute_boundary_vertices_send_offset_homogeneous(rank, expected_send_offset):
    local_verts = compute_local_vertices(HOMO_PARTITIONING, rank)
    _, send_offset = compute_boundary_vertices(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, rank, num_ranks=2
    )
    assert torch.equal(send_offset, expected_send_offset), (
        f"Rank {rank}: send_offset {send_offset.tolist()} != {expected_send_offset.tolist()}"
    )


@pytest.mark.parametrize("rank", [0, 1])
def test_compute_boundary_vertices_send_idx_are_local_indices(rank):
    local_verts = compute_local_vertices(HOMO_PARTITIONING, rank)
    send_local_idx, _ = compute_boundary_vertices(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, rank, num_ranks=2
    )
    num_local = local_verts.size(0)
    assert (send_local_idx >= 0).all()
    assert (send_local_idx < num_local).all()


@pytest.mark.parametrize("rank", [0, 1])
def test_compute_boundary_vertices_unique_per_dest_rank(rank):
    local_verts = compute_local_vertices(HOMO_PARTITIONING, rank)
    send_local_idx, send_offset = compute_boundary_vertices(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, rank, num_ranks=2
    )
    for r in range(2):
        segment = send_local_idx[send_offset[r] : send_offset[r + 1]]
        assert segment.unique().size(0) == segment.size(0), (
            f"Rank {rank}: duplicate send indices for dest rank {r}"
        )


def test_compute_boundary_vertices_self_send_is_zero():
    """The segment for this rank's own index in send_offset must always be empty."""
    for rank in [0, 1]:
        local_verts = compute_local_vertices(HOMO_PARTITIONING, rank)
        _, send_offset = compute_boundary_vertices(
            HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, rank, num_ranks=2
        )
        assert send_offset[rank + 1] == send_offset[rank], (
            f"Rank {rank}: non-zero self-send segment"
        )


def test_compute_boundary_vertices_no_cross_edges_empty():
    edge_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    partitioning = torch.tensor([0, 0], dtype=torch.long)
    local_verts = torch.tensor([0, 1])
    send_local_idx, send_offset = compute_boundary_vertices(
        edge_list, partitioning, local_verts, rank=0, num_ranks=2
    )
    assert send_local_idx.numel() == 0
    assert torch.equal(send_offset, torch.zeros(3, dtype=torch.long))


def test_compute_boundary_vertices_duplicate_edges_deduplicated():
    """A vertex connected by multiple edges to the same remote rank is sent only once."""
    edge_list = torch.tensor([[0, 2], [0, 2], [0, 3]], dtype=torch.long)
    partitioning = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    local_verts = torch.tensor([0, 1])
    send_local_idx, send_offset = compute_boundary_vertices(
        edge_list, partitioning, local_verts, rank=0, num_ranks=2
    )
    segment = send_local_idx[send_offset[1] : send_offset[2]]
    assert segment.unique().size(0) == segment.size(0)
    # vertex 0 should appear exactly once for rank 1
    assert (segment == 0).sum().item() == 1


def test_compute_boundary_vertices_dst_none_equals_src():
    local_verts = compute_local_vertices(HOMO_PARTITIONING, rank=0)
    idx_implicit, off_implicit = compute_boundary_vertices(
        HOMO_EDGE_LIST, HOMO_PARTITIONING, local_verts, rank=0, num_ranks=2
    )
    idx_explicit, off_explicit = compute_boundary_vertices(
        HOMO_EDGE_LIST,
        HOMO_PARTITIONING,
        local_verts,
        rank=0,
        num_ranks=2,
        dst_partitioning=HOMO_PARTITIONING,
    )
    assert torch.equal(idx_implicit, idx_explicit)
    assert torch.equal(off_implicit, off_explicit)


# ===========================================================================
# compute_boundary_vertices — heterogeneous
# ===========================================================================


@pytest.mark.parametrize(
    "rank, expected_send_offset, expected_num_sends",
    [
        (0, torch.tensor([0, 0, 2]), 2),  # send src{0,1} to rank 1
        (1, torch.tensor([0, 1, 1]), 1),  # send src{2} to rank 0
    ],
)
def test_compute_boundary_vertices_heterogeneous(
    rank, expected_send_offset, expected_num_sends
):
    local_verts = compute_local_vertices(HETERO_SRC_PARTITIONING, rank)
    send_local_idx, send_offset = compute_boundary_vertices(
        HETERO_EDGE_LIST,
        HETERO_SRC_PARTITIONING,
        local_verts,
        rank,
        num_ranks=2,
        dst_partitioning=HETERO_DST_PARTITIONING,
    )
    assert torch.equal(send_offset, expected_send_offset)
    assert send_local_idx.numel() == expected_num_sends
    num_local = local_verts.size(0)
    assert (send_local_idx >= 0).all()
    assert (send_local_idx < num_local).all()


# ===========================================================================
# compute_recv_offsets  (no dist required — pure tensor arithmetic)
# ===========================================================================


@pytest.mark.parametrize(
    "rank, expected_recv_offset, expected_recv_bwd",
    [
        (0, torch.tensor([0, 0, 2]), torch.tensor([0., 0.])),
        (1, torch.tensor([0, 2, 2]), torch.tensor([0., 2.])),
    ],
)
def test_compute_recv_offsets(rank, expected_recv_offset, expected_recv_bwd):
    recv_offset, recv_bwd = compute_recv_offsets(HOMO_COMM_MAP, rank)
    assert torch.equal(recv_offset, expected_recv_offset)
    assert torch.equal(recv_bwd, expected_recv_bwd)


@pytest.mark.parametrize("rank", [0, 1])
def test_compute_recv_offsets_total_matches_comm_map_col(rank):
    recv_offset, _ = compute_recv_offsets(HOMO_COMM_MAP, rank)
    expected_total = int(HOMO_COMM_MAP[:, rank].sum().item())
    assert recv_offset[-1].item() == expected_total


def test_compute_recv_offsets_is_non_decreasing():
    for rank in [0, 1]:
        recv_offset, _ = compute_recv_offsets(HOMO_COMM_MAP, rank)
        assert (recv_offset[1:] >= recv_offset[:-1]).all()


# ===========================================================================
# Distributed fixture
# (tests below require: torchrun --nnodes 1 --nproc-per-node 2)
# ===========================================================================


@pytest.fixture(scope="module")
def dist_setup():
    """Initialize NCCL process group and set the per-rank CUDA device."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    yield rank, world_size, device


def _homo_tensors(device):
    return HOMO_EDGE_LIST.to(device), HOMO_PARTITIONING.to(device)


def _hetero_tensors(device):
    return (
        HETERO_EDGE_LIST.to(device),
        HETERO_SRC_PARTITIONING.to(device),
        HETERO_DST_PARTITIONING.to(device),
    )


# ===========================================================================
# compute_comm_map (distributed)
# ===========================================================================


def test_compute_comm_map_correct_values(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)

    local_verts = compute_local_vertices(partitioning, rank)
    _, send_off = compute_boundary_vertices(
        edge_list, partitioning, local_verts, rank, world_size
    )
    comm_map = compute_comm_map(send_off, world_size)

    expected = torch.tensor([[0., 2.], [2., 0.]], device=device)
    assert torch.equal(comm_map, expected), (
        f"Rank {rank}: comm_map {comm_map.tolist()} != {expected.tolist()}"
    )


def test_compute_comm_map_row_matches_local_send_counts(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)

    local_verts = compute_local_vertices(partitioning, rank)
    _, send_off = compute_boundary_vertices(
        edge_list, partitioning, local_verts, rank, world_size
    )
    comm_map = compute_comm_map(send_off, world_size)

    send_counts = (send_off[1:] - send_off[:-1]).float()
    assert torch.equal(comm_map[rank], send_counts), (
        f"Rank {rank}: comm_map row {comm_map[rank].tolist()} != send_counts {send_counts.tolist()}"
    )


def test_compute_comm_map_diagonal_zero(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)

    local_verts = compute_local_vertices(partitioning, rank)
    _, send_off = compute_boundary_vertices(
        edge_list, partitioning, local_verts, rank, world_size
    )
    comm_map = compute_comm_map(send_off, world_size)

    assert comm_map[rank, rank].item() == 0.0, "Self-send entry must be zero"


# ===========================================================================
# build_communication_pattern (distributed, homogeneous)
# ===========================================================================


def test_build_communication_pattern_vertex_counts(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)
    cp = build_communication_pattern(edge_list, partitioning, rank, world_size)

    assert cp.rank == rank
    assert cp.world_size == world_size
    assert cp.num_local_vertices == 2, f"Rank {rank}: expected 2 local verts"
    assert cp.num_halo_vertices == 2, f"Rank {rank}: expected 2 halo verts"


def test_build_communication_pattern_send_offset(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)
    cp = build_communication_pattern(edge_list, partitioning, rank, world_size)

    expected = {0: torch.tensor([0, 0, 2]), 1: torch.tensor([0, 2, 2])}
    assert torch.equal(cp.send_offset.cpu(), expected[rank])


def test_build_communication_pattern_recv_offset(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)
    cp = build_communication_pattern(edge_list, partitioning, rank, world_size)

    expected = {0: torch.tensor([0, 0, 2]), 1: torch.tensor([0, 2, 2])}
    assert torch.equal(cp.recv_offset.cpu(), expected[rank])


def test_build_communication_pattern_put_forward_remote_offset(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)
    cp = build_communication_pattern(edge_list, partitioning, rank, world_size)

    # put_forward_remote_offset[i] = number of vertices lower-ranked ranks
    # collectively send to rank i — tells this rank where to write in rank i's
    # recv buffer.
    expected = cp.comm_map[:rank, :].sum(0)
    assert torch.equal(cp.put_forward_remote_offset, expected)


def test_build_communication_pattern_put_backward_remote_offset(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)
    cp = build_communication_pattern(edge_list, partitioning, rank, world_size)

    expected = cp.comm_map[:, :rank].sum(1)
    assert torch.equal(cp.put_backward_remote_offset, expected)


# ===========================================================================
# Full invariants (from CommInfoDesignDocument.md)
# ===========================================================================


def test_communication_pattern_invariants(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)
    cp = build_communication_pattern(edge_list, partitioning, rank, world_size)

    # Local edge list bounds
    assert (cp.local_edge_list >= 0).all()
    assert (cp.local_edge_list < cp.num_local_vertices + cp.num_halo_vertices).all()

    # Source vertices are always local
    assert (cp.local_edge_list[:, 0] < cp.num_local_vertices).all()

    # Send indices are in the local range
    assert (cp.send_local_idx >= 0).all()
    assert (cp.send_local_idx < cp.num_local_vertices).all()

    # send_offset is non-decreasing
    assert (cp.send_offset[1:] >= cp.send_offset[:-1]).all()

    # comm_map row sum == total sends
    assert cp.comm_map[rank].long().sum().item() == cp.send_offset[-1].item(), (
        "comm_map row sum must equal total sends"
    )

    # comm_map column sum == total recvs
    assert cp.comm_map[:, rank].long().sum().item() == cp.recv_offset[-1].item(), (
        "comm_map col sum must equal total recvs"
    )

    # Self-send is zero
    assert cp.comm_map[rank, rank].item() == 0.0

    # Uniqueness within each destination rank segment
    for r in range(world_size):
        segment = cp.send_local_idx[cp.send_offset[r] : cp.send_offset[r + 1]]
        assert segment.unique().size(0) == segment.size(0), (
            f"Rank {rank}: duplicate send indices in segment for dest rank {r}"
        )

    # put offset derivations
    assert torch.equal(cp.put_forward_remote_offset, cp.comm_map[:rank, :].sum(0))
    assert torch.equal(cp.put_backward_remote_offset, cp.comm_map[:, :rank].sum(1))


def test_communication_pattern_send_recv_symmetry(dist_setup):
    """
    For any pair (A, B): comm_map[A, B] == comm_map[B, A] holds for undirected
    graphs (our test graph is undirected).  Verify via reconstructing comm_map
    from a fresh all-gather and comparing it to the stored one.
    """
    rank, world_size, device = dist_setup
    edge_list, partitioning = _homo_tensors(device)
    cp = build_communication_pattern(edge_list, partitioning, rank, world_size)

    send_counts = (cp.send_offset[1:] - cp.send_offset[:-1]).float().to(device)
    gathered = [torch.zeros(world_size, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, send_counts)
    reconstructed = torch.stack(gathered)

    assert torch.equal(reconstructed, cp.comm_map), (
        f"Rank {rank}: reconstructed comm_map differs from stored comm_map"
    )


# ===========================================================================
# Heterogeneous (distributed) — test compute_halo_vertices +
# compute_boundary_vertices with separate partitionings, then build a
# CommunicationPattern manually.
# ===========================================================================


def test_hetero_halo_and_boundary_distributed(dist_setup):
    rank, world_size, device = dist_setup
    edge_list, src_part, dst_part = _hetero_tensors(device)

    src_local_verts = compute_local_vertices(src_part, rank)
    halo_verts = compute_halo_vertices(
        edge_list, src_part, rank, dst_partitioning=dst_part
    )
    send_local_idx, send_off = compute_boundary_vertices(
        edge_list, src_part, src_local_verts, rank, world_size,
        dst_partitioning=dst_part,
    )
    comm = compute_comm_map(send_off, world_size)
    recv_off, _ = compute_recv_offsets(comm, rank)

    expected_local_verts = {0: 2, 1: 1}
    expected_halo_verts = {0: 2, 1: 1}
    expected_send_offset = {
        0: torch.tensor([0, 0, 2]),
        1: torch.tensor([0, 1, 1]),
    }
    expected_recv_offset = {
        0: torch.tensor([0, 0, 1]),   # rank 0 receives 1 vert from rank 1
        1: torch.tensor([0, 2, 2]),   # rank 1 receives 2 verts from rank 0
    }
    expected_comm_map = torch.tensor([[0., 2.], [1., 0.]], device=device)

    assert src_local_verts.size(0) == expected_local_verts[rank]
    assert halo_verts.size(0) == expected_halo_verts[rank]
    assert torch.equal(send_off.cpu(), expected_send_offset[rank])
    assert torch.equal(recv_off.cpu(), expected_recv_offset[rank])
    assert torch.equal(comm, expected_comm_map), (
        f"Rank {rank}: comm_map {comm.tolist()} != {expected_comm_map.tolist()}"
    )

    # Halo vertices must be owned by a remote rank
    halo_ranks = dst_part[halo_verts]
    assert (halo_ranks != rank).all()

    # send indices must be in [0, num_local_src)
    assert (send_local_idx >= 0).all()
    assert (send_local_idx < src_local_verts.size(0)).all()

    # No self-send
    assert comm[rank, rank].item() == 0.0
