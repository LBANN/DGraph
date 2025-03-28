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


@dataclass
class DistributedGraphCastGraph:
    rank: int
    world_size: int
    ranks_per_graph: int
    mesh_level: int
    lat_lon_grid: Tensor
    mesh_graph_node_features: Tensor
    mesh_graph_edge_features: Tensor
    mesh_graph_node_rank_placement: Tensor
    mesh_graph_edge_rank_placement: Tensor
    mesh_graph_src_indices: Tensor
    mesh_graph_dst_indices: Tensor
    mesh_graph_src_rank_placement: Tensor
    mesh_graph_dst_rank_placement: Tensor
    grid_rank_placement: Tensor
    mesh2grid_graph_node_features: Tensor
    mesh2grid_graph_edge_features: Tensor
    mesh2grid_graph_edge_rank_placement: Tensor
    mesh2grid_graph_src_indices: Tensor
    mesh2grid_graph_dst_indices: Tensor
    grid2mesh_graph_node_features: Tensor
    grid2mesh_graph_edge_features: Tensor
    grid2mesh_graph_edge_rank_placement: Tensor
    grid2mesh_graph_src_indices: Tensor
    grid2mesh_graph_dst_indices: Tensor


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

        contiguous_rank_mapping, renumbered_nodes = torch.sort(
            mesh_vertex_rank_placement
        )

        # renumber the nodes
        node_features = node_features[renumbered_nodes]

        # renumber the edges
        src_indices = renumbered_nodes[src_indices]
        dst_indices = renumbered_nodes[dst_indices]

        # Base the edge placements on the source indices
        edge_placement_tensor = mesh_vertex_rank_placement[src_indices]

        contigous_edge_mapping, renumbered_edges = torch.sort(edge_placement_tensor)

        src_indices_rank_placement = contiguous_rank_mapping

        # Rearrange the indices
        src_indices = src_indices[renumbered_edges]
        dst_indices = dst_indices[renumbered_edges]

        dst_indices_rank_placement = mesh_vertex_rank_placement[dst_indices]
        edge_features = edge_features[renumbered_edges]

        mesh_graph_dict = {
            "node_features": node_features,
            "edge_features": edge_features,
            "src_indices": src_indices,
            "dst_indices": dst_indices,
            "node_rank_placement": contiguous_rank_mapping,
            "edge_rank_placement": contigous_edge_mapping,
            "src_rank_placement": edge_placement_tensor,
            "dst_rank_placement": dst_indices_rank_placement,
            "mesh_vertex_renumbering": renumbered_nodes,
            "renumbered_vertices": renumbered_nodes,
        }

        return mesh_graph_dict

    def get_grid_placement(
        self, mesh_vertex_rank_placement, grid2mesh_mesh_dst_indices
    ):
        meshtogrid_edge_placement = mesh_vertex_rank_placement[
            grid2mesh_mesh_dst_indices
        ]

        return meshtogrid_edge_placement

    def get_grid2mesh_graph(self, mesh_graph_dict: dict):

        mesh_vertex_rank_placement = mesh_graph_dict["mesh_vertex_renumbering"]
        max_edge_len = max_edge_length(
            self.finest_mesh_vertices, self.finest_mesh_src, self.finest_mesh_dst
        )

        renumbered_vertices = mesh_graph_dict["node_rank_placement"]

        # create the grid2mesh bipartite graph
        lat_lon_grid_flat = self.lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

        g2m_graph = create_grid2mesh_graph(
            max_edge_len, lat_lon_grid_flat, self.mesh_vertices
        )
        edge_features, src_grid_indices, dst_mesh_indices = g2m_graph

        meshtogrid_edge_placement = self.get_grid_placement(
            mesh_vertex_rank_placement, dst_mesh_indices
        )
        dst_mesh_indices = renumbered_vertices[dst_mesh_indices]

        contigous_edge_mapping, renumbered_edges = torch.sort(meshtogrid_edge_placement)

        src_grid_indices = src_grid_indices[renumbered_edges]
        grid_vertex_rank_placement = torch.zeros_like(lat_lon_grid_flat)
        for i, rank in enumerate(meshtogrid_edge_placement):
            loc = src_grid_indices[i]
            grid_vertex_rank_placement[loc] = rank

        continuous_grid_mapping, renumbered_grid = torch.sort(
            grid_vertex_rank_placement
        )

        # TODO: Consider we can have it to so grid2mesh edges don't require a
        # backpropagation. (If encoder is after the gather / scatter)
        grid2mesh_graph_dict = {
            "node_features": torch.tensor([]),
            "edge_features": edge_features,
            "src_indices": src_grid_indices,
            "dst_indices": dst_mesh_indices,
            "grid2mesh_edge_rank_placement": contigous_edge_mapping,
            "grid_vertex_rank_placement": continuous_grid_mapping,
            "renumbered_grid": renumbered_grid,
        }
        return grid2mesh_graph_dict

    def get_mesh2grid_graph(
        self,
        grid_vertex_rank_placement,
        renumbered_vertices,
        renumbered_grid,
    ):
        lat_lon_grid_flat = self.lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

        m2g_graph = create_mesh2grid_graph(
            lat_lon_grid_flat, self.mesh_vertices, self.mesh_faces
        )

        edge_features, src_mesh_indices, dst_grid_indices = m2g_graph
        src_mesh_indices = renumbered_vertices[src_mesh_indices]
        dst_grid_indices = renumbered_grid[dst_grid_indices]

        mesh2grid_edge_rank_placement = grid_vertex_rank_placement[dst_grid_indices]

        mesh2grid_graph_dict = {
            "node_features": torch.tensor([]),
            "edge_features": edge_features,
            "src_indices": src_mesh_indices,
            "dst_indices": dst_grid_indices,
            "mesh2grid_edge_rank_placement": mesh2grid_edge_rank_placement,
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
        renumbered_vertices = mesh_graph["renumbered_vertices"]
        grid2mesh_graph = self.get_grid2mesh_graph(mesh_graph)
        grid_vertex_rank_placement = grid2mesh_graph["grid_vertex_rank_placement"]
        renumbered_grid = grid2mesh_graph["renumbered_grid"]

        mesh2grid_graph = self.get_mesh2grid_graph(
            grid_vertex_rank_placement, renumbered_vertices, renumbered_grid
        )

        return DistributedGraphCastGraph(
            rank=self.rank,
            world_size=self.world_size,
            ranks_per_graph=self.ranks_per_graph,
            mesh_level=self.mesh_level,
            lat_lon_grid=self.lat_lon_grid,
            mesh_graph_node_features=mesh_graph["node_features"],
            mesh_graph_edge_features=mesh_graph["edge_features"],
            mesh_graph_node_rank_placement=mesh_graph["node_rank_placement"],
            mesh_graph_edge_rank_placement=mesh_graph["edge_rank_placement"],
            mesh_graph_src_indices=mesh_graph["src_indices"],
            mesh_graph_dst_indices=mesh_graph["dst_indices"],
            mesh_graph_src_rank_placement=mesh_graph["src_rank_placement"],
            mesh_graph_dst_rank_placement=mesh_graph["dst_rank_placement"],
            grid_rank_placement=grid2mesh_graph["grid_vertex_rank_placement"],
            mesh2grid_graph_node_features=mesh2grid_graph["node_features"],
            mesh2grid_graph_edge_features=mesh2grid_graph["edge_features"],
            mesh2grid_graph_edge_rank_placement=mesh2grid_graph[
                "mesh2grid_edge_rank_placement"
            ],
            mesh2grid_graph_src_indices=mesh2grid_graph["src_indices"],
            mesh2grid_graph_dst_indices=mesh2grid_graph["dst_indices"],
            grid2mesh_graph_node_features=grid2mesh_graph["node_features"],
            grid2mesh_graph_edge_features=grid2mesh_graph["edge_features"],
            grid2mesh_graph_edge_rank_placement=grid2mesh_graph[
                "grid2mesh_edge_rank_placement"
            ],
            grid2mesh_graph_src_indices=grid2mesh_graph["src_indices"],
            grid2mesh_graph_dst_indices=grid2mesh_graph["dst_indices"],
        )
