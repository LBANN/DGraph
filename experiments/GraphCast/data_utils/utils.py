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

from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
from typing import List, Tuple
from torch import Tensor
from .spatial_utils import (
    get_face_centroids,
    latlon2xyz,
    polar_angle,
    azimuthal_angle,
    geospatial_rotation,
    xyz2latlon,
)


def padded_size(x: int, y: int) -> int:
    """
    Returns the next multiple of y that is greater than x.
    """
    return (x + y - 1) // y * y


def pad_indices(_indices: Tensor, ranks_per_graph: int) -> Tensor:

    num_indices = _indices.shape[0]
    pad_to = padded_size(num_indices, ranks_per_graph)
    padded_indices = torch.zeros(pad_to, dtype=torch.int32) - 1
    padded_indices[:num_indices] = _indices
    return padded_indices


def generate_node_features(node_positions: Tensor):
    """
    Generate node features for the graphs using the spatial information.

    Adds cosine of lattiude, sine and cosine of longitude as node features.
    """
    latlon = xyz2latlon(node_positions)
    lat, lon = latlon[:, 0], latlon[:, 1]
    node_features = torch.stack(
        (torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1
    )
    return node_features


def generate_edge_features(src_pos, dst_pos):
    """
    Generate edge features for the graphs using the spatial information.
    """

    dst_latlon = xyz2latlon(dst_pos, unit="rad")
    dst_lat, dst_lon = dst_latlon[:, 0], dst_latlon[:, 1]

    # azimuthal & polar rotation
    theta_azimuthal = azimuthal_angle(dst_lon)
    theta_polar = polar_angle(dst_lat)
    src_pos = geospatial_rotation(src_pos, theta=theta_azimuthal, axis="z", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_azimuthal, axis="z", unit="rad")

    src_pos = geospatial_rotation(src_pos, theta=theta_polar, axis="y", unit="rad")
    dst_pos = geospatial_rotation(dst_pos, theta=theta_polar, axis="y", unit="rad")

    disp = src_pos - dst_pos
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)

    max_disp_norm = torch.max(disp_norm)
    edge_features = torch.cat((disp / max_disp_norm, disp_norm / max_disp_norm), dim=-1)
    return edge_features


def create_graph(
    src_indices: Tensor,
    dst_indices: Tensor,
    pos: Tensor,
    to_bidirected: bool = True,
):
    if to_bidirected:
        _src_indices = torch.cat([src_indices, dst_indices])
        _dst_indices = torch.cat([dst_indices, src_indices])
        src_indices = _src_indices
        dst_indices = _dst_indices

    src_pos = pos[src_indices.long()]
    dst_pos = pos[dst_indices.long()]

    edge_features = generate_edge_features(src_pos, dst_pos)
    node_features = generate_node_features(pos)

    return node_features, edge_features, src_indices, dst_indices


def create_mesh2grid_graph(
    lat_lon_grid_flat: Tensor,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
):
    """Creates a bipartite graph between the mesh and the grid."""
    cartesian_grid = latlon2xyz(lat_lon_grid_flat)
    n_nbrs = 1
    face_centroids = get_face_centroids(mesh_vertices, mesh_faces)
    neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(face_centroids)
    _, indices = neighbors.kneighbors(cartesian_grid.numpy())

    src = [p for i in indices for p in mesh_faces[i].flatten()]
    dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]

    src_mesh_indices = torch.tensor(src, dtype=torch.int32)
    dst_grid_indices = torch.tensor(dst, dtype=torch.int32)

    src_mesh_node_positions = torch.from_numpy(mesh_vertices)
    dst_grid_node_positions = cartesian_grid

    src_pos = src_mesh_node_positions[src_mesh_indices.long()]
    dst_pos = dst_grid_node_positions[dst_grid_indices.long()]

    edge_features = generate_edge_features(src_pos, dst_pos)
    return edge_features, src_mesh_indices, dst_grid_indices


def create_grid2mesh_graph(
    max_edge_len: float, lat_lon_grid_flat: Tensor, mesh_vertices: np.ndarray
):

    cartesian_grid = latlon2xyz(lat_lon_grid_flat)
    n_nbrs = 4
    neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(mesh_vertices)
    distances, indices = neighbors.kneighbors(cartesian_grid.numpy())

    src, dst = [], []
    for i in range(len(cartesian_grid)):
        for j in range(n_nbrs):
            if distances[i][j] <= 0.6 * max_edge_len:
                src.append(i)
                dst.append(indices[i][j])
                # NOTE this gives 1,618,820 edges, in the paper it is 1,618,746

    # Note that this is a uni-directional bipartite graph
    # The reverse direction is not added because grid nodes are not
    # receivers of any edges
    src_grid_node_positions = torch.Tensor(cartesian_grid)
    dst_mesh_node_positions = torch.Tensor(mesh_vertices)

    src_grid_indices = torch.tensor(src, dtype=torch.int32)
    dst_mesh_indices = torch.tensor(dst, dtype=torch.int32)

    src_pos = src_grid_node_positions[src_grid_indices.long()]
    dst_pos = dst_mesh_node_positions[dst_mesh_indices.long()]

    edge_features = generate_edge_features(src_pos, dst_pos)
    return edge_features, src_grid_indices, dst_mesh_indices
