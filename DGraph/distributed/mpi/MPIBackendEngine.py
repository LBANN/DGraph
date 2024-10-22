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
from mpi4py import MPI
from torch.utils.dlpack import to_dlpack  # type: ignore
from torch.utils.dlpack import from_dlpack
import cupy as cp
import warnings


class MPIBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        self._iniitalized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not self._initialized:
            # We want both NCCL and MPI to be initialized
            dist.init_process_group(backend="nccl", *args, **kwargs)
            # Dist initialization is done by the user and handles
            # the collective operations need for SGD

            MPI.Init()
            self._initialized = True
            self._comm = MPI.COMM_WORLD

    def finalize(self):
        if self._initialized:
            MPI.Finalize()

            self._initialized = False

    def Malloc(self, size: int) -> torch.Tensor:
        """Allocates memory on the GPU that is accessible by MPI one-sided communication"""

        cupy_tensor = cp.empty(size, dtype=cp.float32)
        MPI.Win.Attach(cupy_tensor.data.ptr)  # type: ignore
        torch_tensor = from_dlpack(to_dlpack(cupy_tensor))  # Zero copy
        return torch_tensor

    def get_rank(self) -> int:
        return self._comm.Get_rank()

    def get_world_size(self) -> int:
        return self._comm.Get_size()

    def scatter(self, *args, **kwargs) -> torch.Tensor:
        input_tensor: torch.Tensor = args[0]
        indices: torch.Tensor = args[1]
        local_size: int = args[2]
        batch_size: int = input_tensor.shape[0]
        feature_size: int = input_tensor.shape[2]

        output_tensor: torch.Tensor = self.Malloc(
            batch_size * local_size * feature_size
        ).reshape(batch_size, local_size, feature_size)

        if indices.device != torch.device("cpu"):
            indices = indices.cpu()
            warnings.warn(
                "Scatter indices not on CPU, moving to CPU."
                + "MPI requires indices to be on CPU."
            )
        raise NotImplementedError

    def gather(self, *args, **kwargs) -> torch.Tensor:
        input_tensor: torch.Tensor = args[0]
        indices: torch.Tensor = args[1]
        indices_shape = indices.shape
        b_size = indices_shape[0]
        n = indices_shape[1]
        feature_size = input_tensor.shape[2]
        output_tensor = self.Malloc(b_size * n * feature_size).reshape(
            b_size, n, feature_size
        )

        if indices.device != torch.device("cpu"):
            indices = indices.cpu()
            warnings.warn(
                "Gather indices not on CPU, moving to CPU."
                + "MPI requires indices to be on CPU."
            )

        raise NotImplementedError
