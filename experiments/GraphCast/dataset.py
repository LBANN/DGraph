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
import time
from typing import Any, Dict, List, Tuple, Union
from torch.utils.data import Dataset
from data_utils.graphcast_graph import DistributedGraphCastGraphGenerator
from data_utils.utils import padded_size
from torch.nn.functional import pad


class SyntheticWeatherDataset(Dataset):
    """
    A dataset for generating synthetic temperature data on a latitude-longitude grid for multiple atmospheric layers.

    Args:
        channels (list): List of channels representing different atmospheric layers.
        num_samples_per_year (int): Total number of days to simulate per year.
        num_steps (int): Number of consecutive days in each training sample.
        grid_size (tuple): Latitude by longitude dimensions of the temperature grid.
        base_temp (float): Base temperature around which variations are simulated.
        amplitude (float): Amplitude of the sinusoidal temperature variation.
        noise_level (float): Standard deviation of the noise added to temperature data.
        **kwargs: Additional keyword arguments for advanced configurations.
    """

    def __init__(
        self,
        channels: List[int],
        num_samples_per_year: int,
        num_steps: int,
        mesh_vertex_placement: torch.Tensor,
        device: Union[str, torch.device] = "cuda",
        grid_size: Tuple[int, int] = (721, 1440),
        base_temp: float = 15,
        amplitude: float = 10,
        noise_level: float = 2,
        mesh_level: int = 6,
        rank: int = 0,
        world_size: int = 1,
        ranks_per_graph: int = 1,
        **kwargs: Any,
    ):
        self.num_days: int = num_samples_per_year
        self.num_steps: int = num_steps
        self.num_channels: int = len(channels)
        self.device = device
        self.grid_size: Tuple[int, int] = grid_size
        self.mesh_level: int = mesh_level
        self.rank: int = rank
        self.world_size: int = world_size
        self.ranks_per_graph: int = ranks_per_graph
        start_time = time.time()
        self.temperatures: np.ndarray = self.generate_data(
            self.num_days,
            self.num_channels,
            self.grid_size,
            base_temp,
            amplitude,
            noise_level,
        )
        print(
            f"Generated synthetic temperature data in {time.time() - start_time:.2f} seconds."
        )

        # Generate static graph structure used for all time steps
        # This could be generated once and saved to disk for future use
        # For simplicity, all ranks generate the same graph, but this can be modified
        # so that a single rank generates the graph and broadcasts it to all other ranks
        # or each rank generates it's own partition of the graph

        start_time = time.time()
        self.latitudes = torch.linspace(-90, 90, steps=grid_size[0])
        self.longitudes = torch.linspace(-180, 180, steps=grid_size[1] + 1)[1:]
        self.lat_lon_grid = torch.stack(
            torch.meshgrid(self.latitudes, self.longitudes, indexing="ij"), dim=-1
        )
        self.graph_cast_graph = DistributedGraphCastGraphGenerator(
            self.lat_lon_grid,
            mesh_level=self.mesh_level,
            ranks_per_graph=self.ranks_per_graph,
            rank=self.rank,
            world_size=self.world_size,
        ).get_graphcast_graph(mesh_vertex_rank_placement=mesh_vertex_placement)
        print(f"Generated static graph in {time.time() - start_time:.2f} seconds.")
        self.extra_args: Dict[str, Any] = kwargs

    def generate_data(
        self,
        num_days: int,
        num_channels: int,
        grid_size: Tuple[int, int],
        base_temp: float,
        amplitude: float,
        noise_level: float,
    ) -> np.ndarray:
        """
        Generates synthetic temperature data over a specified number of days for multiple atmospheric layers.

        Args:
            num_days (int): Number of days to generate data for.
            num_channels (int): Number of channels representing different layers.
            grid_size (tuple): Grid size (latitude, longitude).
            base_temp (float): Base mean temperature for the data.
            amplitude (float): Amplitude of temperature variations.
            noise_level (float): Noise level to add stochasticity to the temperature.

        Returns:
            numpy.ndarray: A 4D array of temperature values across days, channels, latitudes, and longitudes.
        """
        days = np.arange(num_days)
        latitudes, longitudes = grid_size

        # Create altitude effect and reshape
        altitude_effect = np.arange(num_channels) * -0.5
        altitude_effect = altitude_effect[
            :, np.newaxis, np.newaxis
        ]  # Shape: (num_channels, 1, 1)
        altitude_effect = np.tile(
            altitude_effect, (1, latitudes, longitudes)
        )  # Shape: (num_channels, latitudes, longitudes)
        altitude_effect = altitude_effect[
            np.newaxis, :, :, :
        ]  # Shape: (1, num_channels, latitudes, longitudes)
        altitude_effect = np.tile(
            altitude_effect, (num_days, 1, 1, 1)
        )  # Shape: (num_days, num_channels, latitudes, longitudes)

        # Create latitude variation and reshape
        lat_variation = np.linspace(-amplitude, amplitude, latitudes)
        lat_variation = lat_variation[:, np.newaxis]  # Shape: (latitudes, 1)
        lat_variation = np.tile(
            lat_variation, (1, longitudes)
        )  # Shape: (latitudes, longitudes)
        lat_variation = lat_variation[
            np.newaxis, np.newaxis, :, :
        ]  # Shape: (1, 1, latitudes, longitudes)
        lat_variation = np.tile(
            lat_variation, (num_days, num_channels, 1, 1)
        )  # Shape: (num_days, num_channels, latitudes, longitudes)

        # Create time effect and reshape
        time_effect = np.sin(2 * np.pi * days / 365)
        time_effect = time_effect[
            :, np.newaxis, np.newaxis, np.newaxis
        ]  # Shape: (num_days, 1, 1, 1)
        time_effect = np.tile(
            time_effect, (1, num_channels, latitudes, longitudes)
        )  # Shape: (num_days, num_channels, latitudes, longitudes)

        # Generate noise
        noise = np.random.normal(
            scale=noise_level, size=(num_days, num_channels, latitudes, longitudes)
        )

        # Calculate daily temperatures
        daily_temps = base_temp + altitude_effect + lat_variation + time_effect + noise

        return daily_temps

    def __len__(self) -> int:
        """
        Returns the number of samples available in the dataset.
        """
        return self.num_days - self.num_steps

    def get_static_graph(self):
        """
        Returns the static graph structure used for all time steps. Use this when
        minimizing host memory usage
        """
        return self.graph_cast_graph

    def __getitem__(self, idx: int):
        """
        Retrieves a sample from the dataset at the specified index.
        """
        in_var = (
            torch.tensor(self.temperatures[idx], dtype=torch.float32)
            .permute(1, 2, 0)
            .reshape(-1, self.num_channels)
        )

        out_var = (
            torch.tensor(
                self.temperatures[idx + 1 : idx + self.num_steps + 2],
                dtype=torch.float32,
            )
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, self.num_channels)
        )

        if self.world_size > 1:
            # Get oartitioned inputs instead of the full graph
            num_grid_nodes = in_var.shape[0]
            padded_num_grid_nodes = padded_size(num_grid_nodes, self.ranks_per_graph)

            num_nodes_per_rank = padded_num_grid_nodes // self.ranks_per_graph
            in_var = pad(in_var, (padded_num_grid_nodes - num_grid_nodes, 0), value=-0)
            out_var = pad(
                out_var, (padded_num_grid_nodes - num_grid_nodes, 0), value=-0
            )

            start_index = self.rank * num_nodes_per_rank
            end_index = start_index + num_nodes_per_rank

            in_var = in_var[start_index:end_index]
            out_var = out_var[start_index:end_index]

        return {
            "invar": in_var.to(self.device),
            "outvar": out_var.to(self.device),
        }


def test_synthetic_weather_dataset(num_days, batch_size=1):
    latlon_res = (721, 1440)
    num_samples_per_year_train = num_days
    num_workers = 8
    num_channels_climate = 73
    num_history = 0
    dt = 6.0
    start_year = 1980
    use_time_of_year_index = True
    channels_list = [i for i in range(num_channels_climate)]

    cos_zenith_args = {
        "dt": dt,
        "start_year": start_year,
    }

    mesh_vertex_placement = torch.ones(40962)
    test_dataset = SyntheticWeatherDataset(
        channels=channels_list,
        num_samples_per_year=num_samples_per_year_train,
        num_steps=1,
        grid_size=latlon_res,
        cos_zenith_args=cos_zenith_args,
        batch_size=batch_size,
        num_workers=num_workers,
        num_history=num_history,
        use_time_of_year_index=use_time_of_year_index,
        mesh_vertex_placement=mesh_vertex_placement,
    )
    print(len(test_dataset))
    print("=" * 80)
    static_graph = test_dataset.get_static_graph()
    print("Static graph:")
    print("Mesh label:\t", static_graph.mesh_level)
    print("Mesh Node features:\t", static_graph.mesh_graph_node_features.shape)
    print("Mesh Edge features:\t", static_graph.mesh_graph_edge_features.shape)
    print("Mesh src indices:\t", static_graph.mesh_graph_src_indices.shape)
    print("Mesh dst indices:\t", static_graph.mesh_graph_dst_indices.shape)
    print("=" * 80)
    print(
        "mesh2grid edge features:\t", static_graph.mesh2grid_graph_edge_features.shape
    )
    print("mesh2grid src indices:\t", static_graph.mesh2grid_graph_src_indices.shape)
    print("mesh2grid dst indices:\t", static_graph.mesh2grid_graph_dst_indices.shape)
    print("=" * 80)
    print(
        "grid2mesh edge features:\t", static_graph.grid2mesh_graph_edge_features.shape
    )
    print("grid2mesh src indices:\t", static_graph.grid2mesh_graph_src_indices.shape)
    print("grid2mesh dst indices:\t", static_graph.grid2mesh_graph_dst_indices.shape)
    print("=" * 80)


if __name__ == "__main__":
    from fire import Fire

    Fire(test_synthetic_weather_dataset)
