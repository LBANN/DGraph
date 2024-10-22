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
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
import torch_nvshmem_p2p as nvshmem


class NVSHMEMBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        # check if already initialized
        self._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not self._initialized:
            dist.init_process_group(backend="nccl", *args, **kwargs)
            nvshmem.init()
            self._initialized = True

    def get_rank(self) -> int:
        return nvshmem.get_rank()

    def get_world_size(self) -> int:
        return nvshmem.get_world_size()
