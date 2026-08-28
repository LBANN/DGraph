"""
Distributed regression test for GraphCast's comm-pattern construction.

This test must be launched with >1 process so it actually exercises cross-rank
edges (world_size=1 has no halo vertices and would trivially pass even with a
broken bipartite partitioning). Run with, e.g.:

    torchrun --standalone --nproc_per_node 2 -m pytest \
        experiments/GraphCast/tests/test_distributed_comm_patterns.py

It asserts the core invariant CommunicationPattern guarantees (see
DGraph/distributed/commInfo.py): for every edge type's local_edge_list,
column 1 (the aggregation target / central vertex) is always a locally-owned
vertex, i.e. `local_edge_list[:, 1] < num_local_vertices`. This is exactly the
invariant that GraphCast's data_utils/graphcast_graph.py violated before it was
fixed to match commInfo.py's (col0=neighbor/src, col1=central/dst) convention,
and would have caught that regression immediately.
"""

import os
import sys

import pytest
import torch

_CUR_DIR = os.path.dirname(__file__)
_GRAPHCAST_DIR = os.path.dirname(_CUR_DIR)
sys.path.append(_GRAPHCAST_DIR)

from data_utils.graphcast_graph import DistributedGraphCastGraphGenerator  # noqa: E402


def _requires_multi_rank():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to build GraphCast comm patterns (commInfo.py's "
                    "compute_comm_map hardcodes .cuda() collectives).")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        pytest.skip(
            "torch.distributed process group is not initialized. Launch this test "
            "via torchrun --nproc_per_node 2 -m pytest ... (see module docstring)."
        )
    if torch.distributed.get_world_size() < 2:
        pytest.skip(
            "This test needs world_size >= 2 to exercise cross-rank edges; "
            "run via torchrun --nproc_per_node 2."
        )


def test_local_edge_list_dst_column_is_always_local():
    _requires_multi_rank()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)

    mesh_level = 2  # small mesh for a fast test (162 mesh vertices)

    mesh_vertex_placement = DistributedGraphCastGraphGenerator.get_mesh_graph_partition(
        mesh_level=mesh_level, world_size=world_size
    )

    latitudes = torch.linspace(-90, 90, steps=21)
    longitudes = torch.linspace(-180, 180, steps=41)[1:]
    lat_lon_grid = torch.stack(
        torch.meshgrid(latitudes, longitudes, indexing="ij"), dim=-1
    )

    graph = DistributedGraphCastGraphGenerator(
        lat_lon_grid,
        mesh_level=mesh_level,
        ranks_per_graph=world_size,
        rank=rank,
        world_size=world_size,
    ).get_graphcast_graph(mesh_vertex_rank_placement=mesh_vertex_placement)

    comm_patterns = graph.distributed_comm_patterns
    for name, cp in (
        ("grid2mesh", comm_patterns.grid2mesh),
        ("mesh", comm_patterns.mesh),
        ("mesh2grid", comm_patterns.mesh2grid),
    ):
        dst_col = cp.local_edge_list[:, 1]
        assert torch.all(dst_col < cp.num_local_vertices), (
            f"{name}: local_edge_list[:, 1] must always index a locally-owned "
            f"vertex (< num_local_vertices={cp.num_local_vertices}), got max "
            f"{dst_col.max().item() if dst_col.numel() else 'N/A'}"
        )
        assert cp.num_local_vertices > 0, f"{name}: no local vertices on rank {rank}"

    torch.distributed.barrier()
