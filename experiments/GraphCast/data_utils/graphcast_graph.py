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
    mesh_graph_src_indices: Tensor
    mesh_graph_dst_indices: Tensor
    mesh2grid_graph_node_features: Tensor
    mesh2grid_graph_edge_features: Tensor
    mesh2grid_graph_src_indices: Tensor
    mesh2grid_graph_dst_indices: Tensor
    grid2mesh_graph_node_features: Tensor
    grid2mesh_graph_edge_features: Tensor
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
        self.finest_mesh_vertices = np.array(finest_mesh.vertices)

        mesh = merge_meshes(_meshes)
        self.mesh_src, self.mesh_dst = faces_to_edges(mesh.faces)  # type: ignore
        self.mesh_src: Tensor = torch.tensor(self.mesh_src, dtype=torch.int32)
        self.mesh_dst: Tensor = torch.tensor(self.mesh_dst, dtype=torch.int32)
        self.mesh_vertices = np.array(mesh.vertices)
        self.mesh_faces = mesh.faces

    def get_mesh_graph(self):
        """Get the graph for the distributed graphcast graph."""

        mesh_pos = torch.tensor(self.mesh_vertices, dtype=torch.float32)

        mesh_graph = create_graph(
            self.mesh_src,
            self.mesh_dst,
            mesh_pos,
            to_bidirected=True,
        )

        node_features, edge_features, src_indices, dst_indices = mesh_graph

        src_indices = pad_indices(src_indices, self.ranks_per_graph)
        dst_indices = pad_indices(src_indices, self.ranks_per_graph)

        num_edges = src_indices.shape[0]
        start_index = self.local_rank * (num_edges // self.ranks_per_graph)
        end_index = start_index + (num_edges // self.ranks_per_graph)
        src_indices = src_indices[start_index:end_index]
        dst_indices = dst_indices[start_index:end_index]
        edge_features = edge_features[start_index:end_index]

        num_nodes = node_features.shape[0]
        start_index = self.local_rank * (num_nodes // self.ranks_per_graph)
        end_index = start_index + (num_nodes // self.ranks_per_graph)
        node_features = node_features[start_index:end_index]

        return node_features, edge_features, src_indices, dst_indices

    def get_mesh2grid_graph(self):
        max_edge_len = max_edge_length(
            self.finest_mesh_vertices, self.finest_mesh_src, self.finest_mesh_dst
        )

        # create the grid2mesh bipartite graph
        lat_lon_grid_flat = self.lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

        g2m_graph = create_grid2mesh_graph(
            max_edge_len, lat_lon_grid_flat, self.mesh_vertices
        )
        edge_features, src_mesh_indices, dst_grid_indices = g2m_graph
        return torch.Tensor([]), edge_features, src_mesh_indices, dst_grid_indices

    def get_grid2mesh_graph(self):
        lat_lon_grid_flat = self.lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

        m2g_graph = create_mesh2grid_graph(
            lat_lon_grid_flat, self.mesh_vertices, self.mesh_faces
        )
        edge_features, src_grid_indices, dst_mesh_indices = m2g_graph
        return torch.Tensor([]), edge_features, src_grid_indices, dst_mesh_indices

    def get_graphcast_graph(self) -> DistributedGraphCastGraph:
        """Get the distributed graphcast graph."""
        mesh_graph = self.get_mesh_graph()
        mesh2grid_graph = self.get_mesh2grid_graph()
        grid2mesh_graph = self.get_grid2mesh_graph()

        return DistributedGraphCastGraph(
            rank=self.rank,
            world_size=self.world_size,
            ranks_per_graph=self.ranks_per_graph,
            mesh_level=self.mesh_level,
            lat_lon_grid=self.lat_lon_grid,
            mesh_graph_node_features=mesh_graph[0],
            mesh_graph_edge_features=mesh_graph[1],
            mesh_graph_src_indices=mesh_graph[2],
            mesh_graph_dst_indices=mesh_graph[3],
            mesh2grid_graph_node_features=mesh2grid_graph[0],
            mesh2grid_graph_edge_features=mesh2grid_graph[1],
            mesh2grid_graph_src_indices=mesh2grid_graph[2],
            mesh2grid_graph_dst_indices=mesh2grid_graph[3],
            grid2mesh_graph_node_features=grid2mesh_graph[0],
            grid2mesh_graph_edge_features=grid2mesh_graph[1],
            grid2mesh_graph_src_indices=grid2mesh_graph[2],
            grid2mesh_graph_dst_indices=grid2mesh_graph[3],
        )
