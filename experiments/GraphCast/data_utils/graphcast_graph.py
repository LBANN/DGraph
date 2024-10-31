import numpy as np
import torch
from torch import Tensor
from icosahedral_mesh import (
    get_hierarchy_of_triangular_meshes_for_sphere,
    faces_to_edges,
    merge_meshes,
)

from utils import create_graph, pad_indices


class DistributedGraphCastGraph:
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

    def get_graph(self):
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
