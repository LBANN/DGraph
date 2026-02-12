# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
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

from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_channels: int = 2
    dropout: float = 0.5
    num_layers: int = 2
    heads: int = 4
    use_cache: bool = True
    # Those numbers are available in the dataset classes (synthetic or mag240m)
    # num_features: int = 768
    # num_relations: int = 5
    # num_classes: int = 153


@dataclass
class TrainingConfig:
    epochs: int = 100
    lr: float = 0.0001
    lr_step_size: int = 25
    lr_gamma: float = 0.25


@dataclass
class SyntheticDatasetConfig:
    num_papers: int = 2048 * 16
    num_authors: int = 1024 * 16
    num_institutions: int = 16 * 16
    num_features: int = 16
    num_classes: int = 153
