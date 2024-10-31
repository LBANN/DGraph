import torch
import numpy as np
from typing import List, Tuple
from torch import Tensor
from spatial_utils import polar_angle, azimuthal_angle, geospatial_rotation, xyz2latlon


def padded_size(x, y):
    return (x + y - 1) // y * y


def pad_indices(_indices: Tensor, ranks_per_graph: int) -> Tensor:

    num_indices = _indices.shape[0]
    pad_to = padded_size(num_indices, ranks_per_graph)
    padded_indices = torch.zeros(pad_to, dtype=torch.int32)
    padded_indices[:num_indices] = _indices
    return padded_indices


def create_graph(
    src_indices: Tensor,
    dst_indices: Tensor,
    pos: Tensor,
    to_bidirected: bool = True,
    ranks_per_graph: int = 1,
):
    if to_bidirected:
        src_indices = torch.cat([src_indices, dst_indices])
        dst_indices = torch.cat([dst_indices, src_indices])

    src_pos = pos[src_indices]
    dst_pos = pos[dst_indices]

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

    latlon = xyz2latlon(pos)
    lat, lon = latlon[:, 0], latlon[:, 1]

    node_features = torch.stack(
        (torch.cos(lat), torch.sin(lon), torch.cos(lon)), dim=-1
    )

    return node_features, edge_features, src_indices, dst_indices
