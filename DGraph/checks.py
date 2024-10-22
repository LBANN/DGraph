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
import torch.distributed as dist


def check_dist_initialized():
    if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
    if not dist.is_initialized():
        raise RuntimeError("Requires distributed package to be initialized")


def check_nccl_availability():
    if not dist.is_nccl_available():
        raise RuntimeError("Requires NCCL backend")
