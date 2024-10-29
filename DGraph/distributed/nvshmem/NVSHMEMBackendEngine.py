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
from torch.utils.dlpack import from_dlpack
import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
import torch_nvshmem_p2p as nvshmem
import cupy as cp
import warnings
from torch.autograd import Function

def _nvshmmem_gather(send_tensor, indices, gathered_tensor):
    # Gather the tensors
    gathered_tensors = [torch.zeros_like(send_tensor) for _ in range(nvshmem.get_world_size())]
    gathered_tensors[nvshmem.get_rank()] = send_tensor
    dist.all_gather(gathered_tensors, send_tensor)

    # Gather the indices
    gathered_indices = [torch.zeros_like(indices) for _ in range(nvshmem.get_world_size())]
    gathered_indices[nvshmem.get_rank()] = indices
    dist.all_gather(gathered_indices, indices)

    return gathered_tensors

def _nvshmem_scatter(input_tensor, indices, scattered_tensor):
    # Scatter the tensors
    scattered_tensors = [torch.zeros_like(input_tensor) for _ in range(nvshmem.get_world_size())]
    scattered_tensors[nvshmem.get_rank()] = input_tensor
    dist.all_gather(scattered_tensors, input_tensor)

    # Scatter the indices
    scattered_indices = [torch.zeros_like(indices) for _ in range(nvshmem.get_world_size())]
    scattered_indices[nvshmem.get_rank()] = indices
    dist.all_gather(scattered_indices, indices)

    return scattered_tensors


class NVSHMEMGatherFunction(Function):
    @staticmethod
    def forward(ctx, send_tensor, indices, gathered_tensor):
        gathered_tensors = _nvshmmem_gather(send_tensor, indices, gathered_tensor)
        return gathered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None
    
class NVSHMEMScatterFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, indices, scattered_tensor):
        scattered_tensors = _nvshmem_scatter(input_tensor, indices, scattered_tensor)
        return scattered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class NVSHMEMBackendEngine(BackendEngine):
    _is_initialized = False
    _rank = -1
    _world_size = -1
    _ranks_per_graph = -1
    
    def __init__(self, *args, **kwargs):
        # check if already initialized
        self._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not self._initialized:
            nvshmem.init()

            dist.init_process_group(backend="nccl", *args, **kwargs)
            
            self._initialized = True

    def get_rank(self) -> int:
        return nvshmem.get_rank()

    def get_world_size(self) -> int:
        return nvshmem.get_world_size()
