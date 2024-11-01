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
import torch
import numpy as np
from torch import Tensor
from typing import List, Tuple
from numpy import ndarray


def max_edge_length(
    vertices: ndarray, source_nodes: ndarray, destination_nodes: ndarray
) -> float:
    """
    Compute the maximum edge length in a graph.

    Parameters:
    vertices (List[List[float]]): A list of tuples representing the coordinates of the vertices.
    source_nodes (List[int]): A list of indices representing the source nodes of the edges.
    destination_nodes (List[int]): A list of indices representing the destination nodes of the edges.

    Returns:
    The maximum edge length in the graph (float).
    """
    vertices_np = np.array(vertices)
    source_coords = vertices_np[source_nodes]
    dest_coords = vertices_np[destination_nodes]

    # Compute the squared distances for all edges
    squared_differences = np.sum((source_coords - dest_coords) ** 2, axis=1)

    # Compute the maximum edge length
    max_length = np.sqrt(np.max(squared_differences))

    return max_length


def polar_angle(lat: Tensor) -> Tensor:
    """
    Gives the polar angle of a point on the sphere

    Parameters
    ----------
    lat : Tensor
        Tensor of shape (N, ) containing the latitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the polar angle
    """
    angle = torch.where(lat >= 0.0, lat, 2 * np.pi + lat)
    return angle


def azimuthal_angle(lon: Tensor) -> Tensor:
    """
    Gives the azimuthal angle of a point on the sphere

    Parameters
    ----------
    lon : Tensor
        Tensor of shape (N, ) containing the longitude of the point

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the azimuthal angle
    """
    angle = torch.where(lon >= 0.0, 2 * np.pi - lon, -lon)
    return angle


def geospatial_rotation(
    invar: Tensor, theta: Tensor, axis: str, unit: str = "rad"
) -> Tensor:
    """Rotation using right hand rule

    Parameters
    ----------
    invar : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    theta : Tensor
        Tensor of shape (N, ) containing the rotation angle
    axis : str
        Axis of rotation
    unit : str, optional
        Unit of the theta, by default "rad"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing the rotated x, y, z coordinates
    """

    # get the right unit
    if unit == "deg":
        invar = rad2deg(invar)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")

    invar = torch.unsqueeze(invar, -1)
    rotation = torch.zeros((theta.size(0), 3, 3))
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    if axis == "x":
        rotation[:, 0, 0] += 1.0
        rotation[:, 1, 1] += cos
        rotation[:, 1, 2] -= sin
        rotation[:, 2, 1] += sin
        rotation[:, 2, 2] += cos
    elif axis == "y":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 2] += sin
        rotation[:, 1, 1] += 1.0
        rotation[:, 2, 0] -= sin
        rotation[:, 2, 2] += cos
    elif axis == "z":
        rotation[:, 0, 0] += cos
        rotation[:, 0, 1] -= sin
        rotation[:, 1, 0] += sin
        rotation[:, 1, 1] += cos
        rotation[:, 2, 2] += 1.0
    else:
        raise ValueError("Invalid axis")

    outvar = torch.matmul(rotation, invar)
    outvar = outvar.squeeze()
    return outvar


def xyz2latlon(xyz: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts xyz to latlon in degrees
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    xyz : Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    """
    lat = torch.arcsin(xyz[:, 2] / radius)
    lon = torch.arctan2(xyz[:, 1], xyz[:, 0])
    if unit == "deg":
        return torch.stack((rad2deg(lat), rad2deg(lon)), dim=1)
    elif unit == "rad":
        return torch.stack((lat, lon), dim=1)
    else:
        raise ValueError("Not a valid unit")


def latlon2xyz(latlon: Tensor, radius: float = 1, unit: str = "deg") -> Tensor:
    """
    Converts latlon in degrees to xyz
    Based on: https://stackoverflow.com/questions/1185408
    - The x-axis goes through long,lat (0,0);
    - The y-axis goes through (0,90);
    - The z-axis goes through the poles.

    Parameters
    ----------
    latlon : Tensor
        Tensor of shape (N, 2) containing latitudes and longitudes
    radius : float, optional
        Radius of the sphere, by default 1
    unit : str, optional
        Unit of the latlon, by default "deg"

    Returns
    -------
    Tensor
        Tensor of shape (N, 3) containing x, y, z coordinates
    """
    if unit == "deg":
        latlon = deg2rad(latlon)
    elif unit == "rad":
        pass
    else:
        raise ValueError("Not a valid unit")
    lat, lon = latlon[:, 0], latlon[:, 1]
    x = radius * torch.cos(lat) * torch.cos(lon)
    y = radius * torch.cos(lat) * torch.sin(lon)
    z = radius * torch.sin(lat)
    return torch.stack((x, y, z), dim=1)


def deg2rad(deg: Tensor) -> Tensor:
    """Converts degrees to radians

    Parameters
    ----------
    deg :
        Tensor of shape (N, ) containing the degrees

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the radians
    """
    return deg * np.pi / 180


def rad2deg(rad):
    """Converts radians to degrees

    Parameters
    ----------
    rad :
        Tensor of shape (N, ) containing the radians

    Returns
    -------
    Tensor
        Tensor of shape (N, ) containing the degrees
    """
    return rad * 180 / np.pi


def get_face_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute the centroids of triangular faces in a graph.

    Parameters:
    vertices (np.ndarray): A list of tuples representing the coordinates of the vertices.
    faces (np.ndarray): A list of lists, where each inner list contains three indices representing a triangular face.

    Returns:
    np.ndarray: A list of tuples representing the centroids of the faces.
    """
    centroids = []

    for face in faces:
        # Extract the coordinates of the vertices for the current face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Compute the centroid of the triangle
        centroid = (
            (v0[0] + v1[0] + v2[0]) / 3,
            (v0[1] + v1[1] + v2[1]) / 3,
            (v0[2] + v1[2] + v2[2]) / 3,
        )

        centroids.append(centroid)

    centroids = np.array(centroids)
    return centroids
