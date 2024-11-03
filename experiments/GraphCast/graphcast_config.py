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

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    lr_step3 = 3e-7
    num_iters_step1: int = 1000
    num_iters_step2: int = 299000
    num_iters_step3: int = 11000
    step_change_freq: int = 1000
    save_freq: int = 1
    grad_clip_norm: float = 32.0
    val_freq: int = 5


@dataclass
class DataConfig:
    # Resolution of the latitude-longitude grid.
    # If smaller than the native resolution, bilinear interpolation is applied
    latlon_res: tuple[int, int] = (721, 1440)
    # Number of samples per year for training.
    num_samples_per_year_train: int = 1408
    # Number of climate channels.
    num_channels_climate: int = 73
    # Number of static channels
    # (e.g., land-sea mask, geopotential, cosine of latitudes,
    # sine and cosine of longitudes).
    num_channels_static: int = 5
    # Number of historical (previous time steps) to use.
    # With history=1, the model uses t-1 and t to predict t+1.
    num_history: int = 0
    # If true, uses cosine zenith angle as additional channel(s).
    # It can replace the total incident solar radiation.
    use_cos_zenith: bool = True
    # Time in hours between each timestep in the dataset.
    # A dt of 6.0 means four timesteps per day.
    dt: float = 6.0
    # Start year of the dataset, used in computing the cosine zenith angle.
    start_year: int = 1980
    # If true, the dataloader also gives the
    # index of the sample for calculating the time of day and year progress.
    use_time_of_year_index: bool = True
    # Number of steps between input and output variables. For example,
    # if data is every 6 hours, stride 1 = 6 hour delta t,
    # and stride 2 = 12 hours delta t.
    stride: int = 1


@dataclass
class ModelConfig:
    # Number of processor layers in the model
    processor_layers: int = 16
    hidden_dim: int = 512
    mesh_level: int = 6
    multimesh: bool = True
    # Input dimension of the grid node features
    input_grid_dim: int = 474
    # Input dimension of the mesh node features
    input_mesh_dim: int = 3
    # Input dimension of the mesh edge features
    input_edge_dim: int = 4
    # Predicted variables for each grid node
    output_grid_dim: int = 227


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
