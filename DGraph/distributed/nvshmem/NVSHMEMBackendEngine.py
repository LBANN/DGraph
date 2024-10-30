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
import warnings
from torch.autograd import Function


def _nvshmmem_gather(send_tensor, indices, gathered_tensor):
    # Gather the tensors
    gathered_tensors = [
        torch.zeros_like(send_tensor) for _ in range(nvshmem.get_world_size())
    ]
    gathered_tensors[nvshmem.get_rank()] = send_tensor
    dist.all_gather(gathered_tensors, send_tensor)

    # Gather the indices
    gathered_indices = [
        torch.zeros_like(indices) for _ in range(nvshmem.get_world_size())
    ]
    gathered_indices[nvshmem.get_rank()] = indices
    dist.all_gather(gathered_indices, indices)

    return gathered_tensors


def _nvshmem_scatter(input_tensor, indices, scattered_tensor):
    # Scatter the tensors
    scattered_tensors = [
        torch.zeros_like(input_tensor) for _ in range(nvshmem.get_world_size())
    ]
    scattered_tensors[nvshmem.get_rank()] = input_tensor
    dist.all_gather(scattered_tensors, input_tensor)

    # Scatter the indices
    scattered_indices = [
        torch.zeros_like(indices) for _ in range(nvshmem.get_world_size())
    ]
    scattered_indices[nvshmem.get_rank()] = indices
    dist.all_gather(scattered_indices, indices)

    return scattered_tensors


class NVSHMEMGatherFunction(Function):
    @staticmethod
    def forward(ctx, send_tensor, indices):
        # Register the send tensor
        ctx.save_for_backward(indices, torch.tensor(send_tensor.shape))
        nvshmem.register_memory(send_tensor)
        bs = send_tensor.shape[0]
        num_rows = indices.shape[1]
        num_features = send_tensor.shape[2]

        gathered_tensor = torch.zeros((bs, num_rows, num_features)).to(
            send_tensor.device
        )
        gathered_tensors = _nvshmmem_gather(send_tensor, indices, gathered_tensor)
        nvshmem.deregister_memory(send_tensor)
        return gathered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        indices, shape = ctx.saved_tensors

        num_elements = torch.cumprod(shape, dim=0)[-1].item()

        scattered_grad_tensor = nvshmem.AllocateSymmetricMemory(num_elements).reshape(
            shape
        )

        _nvshmem_scatter(grad_output, indices, scattered_grad_tensor)

        return scattered_grad_tensor, None


class NVSHMEMScatterFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, indices):
        # Allocate buffer
        ctx.save_for_backward(indices)
        _size = input_tensor.numel()
        _put_buffer = nvshmem.AllocateSymmetricMemory(_size).reshape(input_tensor.shape)

        scattered_tensors = _nvshmem_scatter(input_tensor, indices, _put_buffer)
        return scattered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        nvshmem.register_memory(grad_output)
        indices = ctx.saved_tensors[0]
        scattered_tensor = torch.zeros_like(grad_output)
        nvshmem.deregister_memory(grad_output)
        return grad_output, None


class NVSHMEMBackendEngine(BackendEngine):
    _is_initialized = False
    _rank = -1
    _world_size = -1
    _ranks_per_graph = -1
    _nvshmem_p2p_obj = None

    def __init__(self, *args, **kwargs):
        # check if already initialized
        NVSHMEMBackendEngine._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not self._initialized:
            nvshmem.init()

            dist.init_process_group(backend="nccl", *args, **kwargs)

            NVSHMEMBackendEngine._rank = nvshmem.get_rank()
            NVSHMEMBackendEngine._world_size = nvshmem.get_world_size()
            NVSHMEMBackendEngine._ranks_per_graph = NVSHMEMBackendEngine._world_size

            dist.init_process_group(
                backend="nccl",
                rank=NVSHMEMBackendEngine._rank,
                world_size=NVSHMEMBackendEngine._world_size,
                *args,
                **kwargs,
            )
            NVSHMEMBackendEngine._initialized = True
            NVSHMEMBackendEngine._nvshmem_p2p_obj = nvshmem.NVSHMEMP2P()

    def get_rank(self) -> int:
        return nvshmem.get_rank()

    def get_world_size(self) -> int:
        return nvshmem.get_world_size()

    def gather(self, input_tensor, indices, rank_mappings):
        gathered_tensors = NVSHMEMGatherFunction.apply(input_tensor, indices)
        return gathered_tensors

    def scatter(self, input_tensor, indices, rank_mappings):
        scattered_tensors = NVSHMEMScatterFunction.apply(input_tensor, indices)
        return scattered_tensors
