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
from torch.autograd import Function


class MPIGatherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        indices: torch.LongTensor,
    ):
        ctx.save_for_backward(send_tensor, indices)
        bs = send_tensor.shape[0]
        send_size = send_tensor.shape[2]
        num_indices = indices.shape[1]

        recv_tensor = MPIBackendEngine.Malloc(bs * num_indices * send_size).reshape(
            bs, num_indices, send_size
        )
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        send_tensor, indices = ctx.saved_tensors
        size = send_tensor.numel()
        bs = send_tensor.shape[0]
        num_nodes = send_tensor.shape[1]
        feat_size = send_tensor.shape[2]
        grad_input = MPIBackendEngine.Malloc(size).reshape(bs, num_nodes, feat_size)
        return grad_input, None


class MPIBackendEngine(BackendEngine):
    _is_initialized = False
    _rank = -1
    _world_size = -1

    def __init__(self, *args, **kwargs):
        # self._iniitalized = dist.is_initialized()
        pass

    def init_process_group(self, *args, **kwargs):
        if not MPIBackendEngine._is_initialized:
            # We want both NCCL and MPI to be initialized
            dist.init_process_group(backend="nccl", *args, **kwargs)
            # Dist initialization is done by the user and handles
            # the collective operations need for SGD
            MPI.Init()
            MPIBackendEngine._is_initialized = True

            self._comm = MPI.COMM_WORLD
            MPIBackendEngine._rank = self._comm.Get_rank()
            MPIBackendEngine._world_size = self._comm.Get_size()

    def finalize(self):
        if self._initialized:
            MPI.Finalize()

            self._initialized = False

    @staticmethod
    def Malloc(size: int) -> torch.Tensor:
        """Allocates memory on the GPU that is accessible by MPI one-sided communication"""

        cupy_tensor = cp.empty(size, dtype=cp.float32)
        MPI.Win.Attach(cupy_tensor.data.ptr)  # type: ignore
        torch_tensor = from_dlpack(to_dlpack(cupy_tensor))  # Zero copy
        return torch_tensor

    def get_local_rank_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        rank = self.get_rank()
        world_size = self.get_world_size()
        tensor_shape = tensor.shape
        tensor_size = tensor_shape[1]
        local_size = tensor_size // world_size
        start_index = rank * local_size
        end_index = start_index + local_size
        return tensor[:, start_index:end_index]

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
