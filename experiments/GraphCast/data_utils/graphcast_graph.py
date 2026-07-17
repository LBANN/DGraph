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
import numpy as np
import torch
from torch import Tensor
from experiments.GraphCast.data_utils.spatial_utils import max_edge_length
from .icosahedral_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
    faces_to_edges,
    merge_meshes,
)
from .utils import (
    create_graph,
    create_grid2mesh_graph,
    create_mesh2grid_graph,
    pad_indices,
)
from .preprocess import graphcast_graph_to_nxgraph, partition_graph
from dataclasses import dataclass
from DGraph.distributed import CommunicationPattern, build_communication_pattern


@dataclass
class GraphCastCommPatterns:
    grid2mesh: CommunicationPattern
    mesh: CommunicationPattern
    mesh2grid: CommunicationPattern


@dataclass
class GraphCastTopology:
    rank: int
    world_size: int
    ranks_per_graph: int
    mesh_rank_placement: Tensor
    grid_rank_placement: Tensor

    mesh_graph_src_indices: Tensor
    mesh_graph_dst_indices: Tensor

    mesh2grid_graph_src_indices: Tensor
    mesh2grid_graph_dst_indices: Tensor
    grid2mesh_graph_src_indices: Tensor
    grid2mesh_graph_dst_indices: Tensor


@dataclass
class DistributedGraphCastGraph:
    # Distributed environment info
    rank: int
    world_size: int
    ranks_per_graph: int

    # Graph metadata
    mesh_level: int
    lat_lon_grid: Tensor

    # Mesh vertex features
    mesh_graph_node_features: Tensor
    mesh_graph_edge_features: Tensor

    # Grid vertex features
    mesh2grid_graph_node_features: Tensor
    grid2mesh_graph_node_features: Tensor

    # Mesh <--> Grid edge features
    mesh2grid_graph_edge_features: Tensor
    grid2mesh_graph_edge_features: Tensor

    # Distributed graph info
    distributed_comm_patterns: GraphCastCommPatterns


def build_graphcast_comm_patterns(graph: GraphCastTopology) -> GraphCastCommPatterns:
    """
    Build CommunicationPatterns for all three GraphCast edge types.

    The graph's *_src_indices / *_dst_indices use the message-flow convention:
      src = vertex originating the message (neighbor)
      dst = vertex aggregating the message (central)

    DGraph.distributed.commInfo's CommunicationPattern contract guarantees
    col1 of local_edge_list is always the locally-owned aggregation target
    (dst/central); col0 is the neighbor (src), which may be a halo vertex.
    We therefore keep the edge list in [src (neighbor), dst (central)] =
    [col0, col1] order, i.e. unchanged message-flow order, and pass the two
    endpoints' partitionings as (dst partitioning, src partitioning) for the
    bipartite grid2mesh/mesh2grid relations.
    """
    rank = graph.rank
    world_size = graph.ranks_per_graph
    mesh_part = graph.mesh_rank_placement
    grid_part = graph.grid_rank_placement

    # --- grid2mesh ---
    # Message flow: grid → mesh.  Central/dst = mesh, neighbor/src = grid.
    grid2mesh_edges = torch.stack(
        [
            graph.grid2mesh_graph_src_indices,  # grid (neighbor, col 0)
            graph.grid2mesh_graph_dst_indices,
        ],  # mesh (central, col 1)
        dim=1,
    )
    grid2mesh_cp = build_communication_pattern(
        global_edge_list=grid2mesh_edges,
        partitioning=mesh_part,
        src_partitioning=grid_part,
        rank=rank,
        world_size=world_size,
    )

    # --- mesh ↔ mesh ---
    # Homogeneous, undirected.  Central/dst = mesh, neighbor/src = mesh.
    mesh_edges = torch.stack(
        [
            graph.mesh_graph_src_indices,  # mesh (neighbor, col 0)
            graph.mesh_graph_dst_indices,
        ],  # mesh (central, col 1)
        dim=1,
    )
    mesh_cp = build_communication_pattern(
        global_edge_list=mesh_edges,
        partitioning=mesh_part,
        rank=rank,
        world_size=world_size,
    )

    # --- mesh2grid ---
    # Message flow: mesh → grid.  Central/dst = grid, neighbor/src = mesh.
    mesh2grid_edges = torch.stack(
        [
            graph.mesh2grid_graph_src_indices,  # mesh (neighbor, col 0)
            graph.mesh2grid_graph_dst_indices,
        ],  # grid (central, col 1)
        dim=1,
    )
    mesh2grid_cp = build_communication_pattern(
        global_edge_list=mesh2grid_edges,
        partitioning=grid_part,
        src_partitioning=mesh_part,
        rank=rank,
        world_size=world_size,
    )

    return GraphCastCommPatterns(
        grid2mesh=grid2mesh_cp,
        mesh=mesh_cp,
        mesh2grid=mesh2grid_cp,
    )


class DistributedGraphCastGraphGenerator:
    """Graph class for creating graphcast graphs that support distributed graphs.
    Based on the GraphCast implementation in NVIDIA's Modulus.
    See: https://github.com/NVIDIA/modulus/blob/main/modulus/utils/graphcast/graph.py#L43

    """

    def __init__(
        self,
        lat_lon_grid: Tensor,
        mesh_level: int = 6,
        ranks_per_graph: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.lat_lon_grid = lat_lon_grid
        self.mesh_level = mesh_level
        self.ranks_per_graph = ranks_per_graph
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % ranks_per_graph

        # create the multi-mesh
        _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
        finest_mesh = _meshes[-1]  # get the last one in the list of meshes
        self.finest_mesh_src, self.finest_mesh_dst = faces_to_edges(finest_mesh.faces)
        self.finest_mesh_vertices = np.array(finest_mesh.vertices).astype(np.float64)

        mesh = merge_meshes(_meshes)
        self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)  # type: ignore
        self.mesh_src: Tensor = torch.tensor(self.mesh_src, dtype=torch.int32)
        self.mesh_dst: Tensor = torch.tensor(self.mesh_dst, dtype=torch.int32)
        self.mesh_vertices = np.array(mesh.vertices)
        self.mesh_faces = mesh.faces

    @staticmethod
    def get_mesh_graph_partition(mesh_level: int, world_size: int):
        """Generate the partitioning of the mesh graph."""
        # Only rank 1 should generate the partitioning
        _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
        finest_mesh = _meshes[-1]
        finest_mesh_src, finest_mesh_dst = faces_to_edges(finest_mesh.faces)
        mesh = merge_meshes(_meshes)
        mesh_src, mesh_dst = faces_to_edges(mesh.faces)  # type: ignore
        mesh_src: Tensor = torch.tensor(mesh_src, dtype=torch.int32)
        mesh_dst: Tensor = torch.tensor(mesh_dst, dtype=torch.int32)
        mesh_vertices = np.array(mesh.vertices)  # No op but just in case
        mesh_pos = torch.tensor(mesh_vertices, dtype=torch.float32)

        mesh_graph = create_graph(
            mesh_src,
            mesh_dst,
            mesh_pos,
            to_bidirected=True,
        )

        nx_graph = graphcast_graph_to_nxgraph(mesh_graph)
        mesh_vertex_rank_placement = partition_graph(nx_graph, world_size)
        mesh_vertex_rank_placement = torch.tensor(mesh_vertex_rank_placement)
        return mesh_vertex_rank_placement

    @staticmethod
    def get_grid_vertex_partition(
        num_grid: int,
        mesh_vertex_rank_placement: torch.Tensor,
        grid2mesh_grid_src_indices: torch.Tensor,
        grid2mesh_mesh_dst_indices: torch.Tensor,
        mesh2grid_mesh_src_indices: torch.Tensor,
        mesh2grid_grid_dst_indices: torch.Tensor,
        world_size: int,
    ) -> torch.Tensor:
        """Generate the partitioning of grid vertices to minimize cross-rank edges.

        For each grid vertex, counts how many of its connected mesh vertices
        (via both grid2mesh and mesh2grid edges) live on each rank, then assigns
        the grid vertex to the rank with the plurality of connections.

        All indices/placements here are in the *original* (pre rank-sort) mesh
        and grid vertex numbering.
        """
        votes = torch.zeros(num_grid, world_size, dtype=torch.long)

        # --- grid2mesh contribution: grid vertex is src, mesh vertex is dst ---
        g2m_ranks = mesh_vertex_rank_placement[grid2mesh_mesh_dst_indices.long()]
        # Flatten (grid_vertex, rank) into a 1D index for scatter_add_
        g2m_flat_idx = grid2mesh_grid_src_indices.long() * world_size + g2m_ranks
        votes.view(-1).scatter_add_(0, g2m_flat_idx, torch.ones_like(g2m_flat_idx))

        # --- mesh2grid contribution: mesh vertex is src, grid vertex is dst ---
        m2g_ranks = mesh_vertex_rank_placement[mesh2grid_mesh_src_indices.long()]
        m2g_flat_idx = mesh2grid_grid_dst_indices.long() * world_size + m2g_ranks
        votes.view(-1).scatter_add_(0, m2g_flat_idx, torch.ones_like(m2g_flat_idx))

        # Assign each grid vertex to the rank with the most connections
        grid_partitioning = votes.argmax(dim=1)

        return grid_partitioning

    @staticmethod
    def _renumber_by_rank(rank_placement: torch.Tensor):
        """Sort vertices by rank so each rank owns a contiguous numbering range.

        Returns (sorted_ranks, forward_perm, inverse_perm): forward_perm maps
        new (sorted) position -> original vertex id, and is used to reorder
        per-vertex feature tensors. inverse_perm maps original vertex id ->
        new position, and is used to remap edge-index tensors that still
        reference original vertex ids into the new numbering.
        """
        sorted_ranks, forward_perm = torch.sort(rank_placement)
        inverse_perm = torch.empty_like(forward_perm)
        inverse_perm[forward_perm] = torch.arange(forward_perm.numel())
        return sorted_ranks, forward_perm, inverse_perm

    def get_mesh_graph(self, mesh_vertex_rank_placement: torch.Tensor):
        """Get the graph for the distributed graphcast graph."""

        mesh_pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)

        mesh_graph = create_graph(
            self.mesh_src,
            self.mesh_dst,
            mesh_pos,
            to_bidirected=True,
        )

        node_features, edge_features, src_indices, dst_indices = mesh_graph

        num_nodes = node_features.size(0)

        assert num_nodes == mesh_vertex_rank_placement.size(0)

        contiguous_rank_mapping, renumbered_nodes, reverse_renumbered_nodes = (
            self._renumber_by_rank(mesh_vertex_rank_placement)
        )

        # renumber the nodes
        node_features = node_features[renumbered_nodes]

        # renumber the edges
        new_src_indices = reverse_renumbered_nodes[src_indices]
        new_dst_indices = reverse_renumbered_nodes[dst_indices]

        # Base the edge placements on the source indices
        edge_placement_tensor = contiguous_rank_mapping[new_src_indices]
        dst_indices_rank_placement = contiguous_rank_mapping[new_dst_indices]

        # TODO: Check if this is correct

        contigous_edge_mapping, renumbered_edges = torch.sort(edge_placement_tensor)

        # Rearrange the indices
        src_indices = new_src_indices[renumbered_edges]
        dst_indices = new_dst_indices[renumbered_edges]

        edge_features = edge_features[renumbered_edges]
        src_indices_rank_placement = contigous_edge_mapping
        dst_indices_rank_placement = dst_indices_rank_placement[renumbered_edges]

        mesh_graph_dict = {
            "node_features": node_features,
            "edge_features": edge_features,
            "src_indices": src_indices,
            "dst_indices": dst_indices,
            "node_rank_placement": contiguous_rank_mapping,
            "edge_rank_placement": contigous_edge_mapping,
            "src_rank_placement": src_indices_rank_placement,
            "dst_rank_placement": dst_indices_rank_placement,
            "original_rank_placement": mesh_vertex_rank_placement,
            "mesh_vertex_renumbering": renumbered_nodes,
            "renumbered_vertices": renumbered_nodes,
            "inverse_vertex_renumbering": reverse_renumbered_nodes,
        }

        return mesh_graph_dict

    def _get_raw_mesh2grid_edges(self):
        """mesh2grid edge_features/src(mesh)/dst(grid), all in original vertex ids."""
        lat_lon_grid_flat = self.lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)
        return create_mesh2grid_graph(
            lat_lon_grid_flat, self.mesh_vertices, self.mesh_faces
        )

    def get_grid2mesh_graph(self, mesh_graph_dict: dict, mesh2grid_raw: tuple = None):
        """Get the grid2mesh bipartite graph, renumbered into per-rank-contiguous order.

        Each grid vertex is assigned to the rank holding the plurality of its
        connected mesh vertices, counting both grid2mesh and mesh2grid edges
        (see get_grid_vertex_partition) -- not just the grid2mesh direction.

        mesh2grid_raw lets callers that already computed the mesh2grid edges
        (get_graphcast_graph) pass them in so the mesh2grid nearest-neighbor
        search isn't repeated; if omitted (e.g. external callers that only need
        the grid2mesh graph) it is computed here.
        """
        inverse_vertex_renumbering = mesh_graph_dict["inverse_vertex_renumbering"]
        original_mesh_rank_placement = mesh_graph_dict["original_rank_placement"]

        max_edge_len = max_edge_length(
            self.finest_mesh_vertices, self.finest_mesh_src, self.finest_mesh_dst
        )

        # create the grid2mesh bipartite graph
        lat_lon_grid_flat = self.lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

        g2m_graph = create_grid2mesh_graph(
            max_edge_len, lat_lon_grid_flat, self.mesh_vertices
        )
        edge_features, src_grid_indices, dst_mesh_indices = g2m_graph

        if mesh2grid_raw is None:
            mesh2grid_raw = self._get_raw_mesh2grid_edges()
        _, m2g_src_mesh_indices, m2g_dst_grid_indices = mesh2grid_raw

        num_grid = lat_lon_grid_flat.shape[0]
        grid_vertex_rank_placement = self.get_grid_vertex_partition(
            num_grid=num_grid,
            mesh_vertex_rank_placement=original_mesh_rank_placement,
            grid2mesh_grid_src_indices=src_grid_indices,
            grid2mesh_mesh_dst_indices=dst_mesh_indices,
            mesh2grid_mesh_src_indices=m2g_src_mesh_indices,
            mesh2grid_grid_dst_indices=m2g_dst_grid_indices,
            world_size=self.world_size,
        )
        continuous_grid_mapping, renumbered_grid, inverse_grid_renumbering = (
            self._renumber_by_rank(grid_vertex_rank_placement)
        )

        # Remap edge endpoints from original ids into the new, rank-contiguous
        # numbering: grid side via the grid renumbering, mesh side via the
        # mesh renumbering computed in get_mesh_graph.
        src_grid_indices = inverse_grid_renumbering[src_grid_indices.long()]
        dst_mesh_indices = inverse_vertex_renumbering[dst_mesh_indices.long()]

        grid2mesh_graph_dict = {
            "node_features": torch.tensor([]),
            "edge_features": edge_features,
            "src_indices": src_grid_indices,
            "dst_indices": dst_mesh_indices,
            "grid_vertex_rank_placement": continuous_grid_mapping,
            "renumbered_grid": renumbered_grid,
            "inverse_grid_renumbering": inverse_grid_renumbering,
        }
        return grid2mesh_graph_dict

    def get_mesh2grid_edges(
        self,
        inverse_vertex_renumbering,
        inverse_grid_renumbering,
        mesh2grid_raw: tuple = None,
    ):
        if mesh2grid_raw is None:
            mesh2grid_raw = self._get_raw_mesh2grid_edges()
        edge_features, src_mesh_indices, dst_grid_indices = mesh2grid_raw

        src_mesh_indices = inverse_vertex_renumbering[src_mesh_indices.long()]
        dst_grid_indices = inverse_grid_renumbering[dst_grid_indices.long()]

        mesh2grid_graph_dict = {
            "edge_features": edge_features,
            "src_indices": src_mesh_indices,
            "dst_indices": dst_grid_indices,
        }

        return mesh2grid_graph_dict

    def get_graphcast_graph(
        self, mesh_vertex_rank_placement: torch.Tensor
    ) -> DistributedGraphCastGraph:
        """Get the distributed graphcast graph.

        Args:
            mesh_vertex_rank_placement (torch.Tensor): The rank placement of the mesh vertices.

        Returns:
            (DistributedGraphCastGraph): The distributed graphcast graph object.
        """

        assert (
            self.rank <= mesh_vertex_rank_placement.max().item()
        ), "Mesh vertex placement does not include current rank"

        assert (
            self.rank >= mesh_vertex_rank_placement.min().item()
        ), "Mesh vertex placement does not include current rank"

        assert (
            self.world_size == mesh_vertex_rank_placement.max().item() + 1
        ), "World size does not match the number of partitions in mesh rank placement"

        mesh_vertex_rank_placement = mesh_vertex_rank_placement.view(-1)

        mesh_graph = self.get_mesh_graph(mesh_vertex_rank_placement)
        mesh_vertex_rank_placement = mesh_graph["node_rank_placement"]
        inverse_vertex_renumbering = mesh_graph["inverse_vertex_renumbering"]

        # Computed once and shared between get_grid2mesh_graph (needs it for the
        # grid rank vote) and get_mesh2grid_edges (needs it for the edges
        # themselves), so the mesh2grid nearest-neighbor search isn't repeated.
        mesh2grid_raw = self._get_raw_mesh2grid_edges()

        grid2mesh_graph = self.get_grid2mesh_graph(mesh_graph, mesh2grid_raw=mesh2grid_raw)
        grid_vertex_rank_placement = grid2mesh_graph["grid_vertex_rank_placement"]
        inverse_grid_renumbering = grid2mesh_graph["inverse_grid_renumbering"]

        mesh2grid_graph = self.get_mesh2grid_edges(
            inverse_vertex_renumbering,
            inverse_grid_renumbering,
            mesh2grid_raw=mesh2grid_raw,
        )

        topology = GraphCastTopology(
            rank=self.local_rank,
            world_size=self.world_size,
            ranks_per_graph=self.ranks_per_graph,
            mesh_rank_placement=mesh_vertex_rank_placement,
            grid_rank_placement=grid_vertex_rank_placement,
            mesh_graph_src_indices=mesh_graph["src_indices"],
            mesh_graph_dst_indices=mesh_graph["dst_indices"],
            mesh2grid_graph_src_indices=mesh2grid_graph["src_indices"],
            mesh2grid_graph_dst_indices=mesh2grid_graph["dst_indices"],
            grid2mesh_graph_src_indices=grid2mesh_graph["src_indices"],
            grid2mesh_graph_dst_indices=grid2mesh_graph["dst_indices"],
        )

        comm_patterns = build_graphcast_comm_patterns(topology)

        # --- Slice node/edge feature tensors down to this rank's local partition ---
        # HaloExchange (DGraph/distributed/haloExchange.py) expects x_local of shape
        # [num_local, F]: only this rank's locally-owned vertices/edges, row-aligned
        # with comm_patterns.*.local_edge_list's local numbering. mesh_graph/
        # grid2mesh_graph/mesh2grid_graph's raw feature tensors are the FULL
        # (renumbered-by-rank, but unfiltered) global set, so they must be filtered
        # here with the exact same masks build_communication_pattern used internally
        # (vertex: placement==rank; edge: dst-owned-by-rank) so rows stay aligned.
        rank = topology.rank

        mesh_node_local_mask = mesh_vertex_rank_placement == rank
        local_mesh_node_features = mesh_graph["node_features"][mesh_node_local_mask]

        mesh_edge_local_mask = mesh_vertex_rank_placement[mesh_graph["dst_indices"]] == rank
        local_mesh_edge_features = mesh_graph["edge_features"][mesh_edge_local_mask]

        grid2mesh_edge_local_mask = (
            mesh_vertex_rank_placement[grid2mesh_graph["dst_indices"]] == rank
        )
        local_grid2mesh_edge_features = grid2mesh_graph["edge_features"][
            grid2mesh_edge_local_mask
        ]

        mesh2grid_edge_local_mask = (
            grid_vertex_rank_placement[mesh2grid_graph["dst_indices"]] == rank
        )
        local_mesh2grid_edge_features = mesh2grid_graph["edge_features"][
            mesh2grid_edge_local_mask
        ]

        return DistributedGraphCastGraph(
            rank=self.rank,
            world_size=self.world_size,
            ranks_per_graph=self.ranks_per_graph,
            mesh_level=self.mesh_level,
            lat_lon_grid=self.lat_lon_grid,
            mesh_graph_node_features=local_mesh_node_features,
            mesh_graph_edge_features=local_mesh_edge_features,
            mesh2grid_graph_node_features=torch.tensor([]),
            grid2mesh_graph_node_features=grid2mesh_graph["node_features"],
            mesh2grid_graph_edge_features=local_mesh2grid_edge_features,
            grid2mesh_graph_edge_features=local_grid2mesh_edge_features,
            distributed_comm_patterns=comm_patterns,
        )
