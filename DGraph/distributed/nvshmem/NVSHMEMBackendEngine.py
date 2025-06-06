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
import DGraph.torch_nvshmem_p2p as nvshmem
import warnings
from torch.autograd import Function


def _nvshmmem_gather(send_tensor, indices, rank_mappings):

    bs = send_tensor.shape[0]
    num_input_rows = send_tensor.shape[1]
    num_output_rows = indices.shape[1]
    num_features = send_tensor.shape[2]

    num_elem = bs * num_output_rows * num_features
    gathered_tensor = nvshmem.NVSHMEMP2P.allocate_symmetric_memory(
        num_elem, send_tensor.device.index
    )
    gathered_tensor.fill_(0).float()
    gathered_tensor = gathered_tensor.reshape((bs, num_output_rows, num_features))

    # Gather the tensors

    # TODO: Add an option to cache the max value
    # max_num_input_rows = nvshmem.NVSHMEMP2P.get_max(num_input_rows)

    # if num_input_rows != max_num_input_rows:
    #     # Pad the tensor because NVSHMEM requires memory to be symmetric on all ranks
    #     # Slice the tensor to the max number of input rows

    #     # TODO: Look into a better way to do this, possibly a seperate memory
    #     # manager for NVSHMEM tensors and workspace tensors
    #     nvshmem_send_tensor = nvshmem.NVSHMEMP2P.padded_clone_tensor(
    #         send_tensor, max_num_input_rows * num_features
    #     )[: num_input_rows * num_features]
    # else:
    nvshmem_send_tensor = nvshmem.NVSHMEMP2P.clone_tensor(send_tensor)

    nvshmem.NVSHMEMP2P.dist_get(
        nvshmem_send_tensor,
        gathered_tensor,
        indices,
        rank_mappings,
        bs,
        num_input_rows,
        num_features,
        num_output_rows,
    )
    return gathered_tensor


def _nvshmem_scatter(input_tensor, indices, rank_mappings, num_output_rows):
    # Scatter the tensors
    bs = input_tensor.shape[0]
    num_input_rows = input_tensor.shape[1]
    num_features = input_tensor.shape[2]
    device = input_tensor.device

    # max_output_rows = nvshmem.NVSHMEMP2P.get_max(num_output_rows)
    # if num_output_rows != max_output_rows:
    #     num_elem = max_output_rows * num_features
    # else:
    num_elem = num_output_rows * num_features

    # TODO: Look into using calloc here to avoid zeroing out the tensor
    scattered_tensor = nvshmem.NVSHMEMP2P.allocate_symmetric_memory(
        num_elem, device.index
    )[: num_output_rows * num_features].reshape((bs, num_output_rows, num_features))
    scattered_tensor.zero_()

    cur_rank = nvshmem.NVSHMEMP2P.get_rank()
    indices = indices % num_output_rows
    local_send_tensor = input_tensor[rank_mappings == cur_rank].unsqueeze(0)
    local_indices = (
        indices[rank_mappings == cur_rank]
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(1, -1, num_features)
    )
    scattered_tensor.scatter_add_(1, local_indices, local_send_tensor)
    nvshmem.NVSHMEMP2P.dist_put(
        input_tensor,
        scattered_tensor,
        indices,
        rank_mappings,
        bs,
        num_input_rows,
        num_features,
        num_output_rows,
    )

    torch.cuda.synchronize()

    return scattered_tensor


class NVSHMEMGatherFunction(Function):
    @staticmethod
    def forward(ctx, send_tensor, indices, rank_mappings):
        # Register the send tensor
        ctx.save_for_backward(indices, rank_mappings)
        num_rows = indices.shape[1]
        ctx.num_rows = num_rows
        gathered_tensors = _nvshmmem_gather(send_tensor, indices, rank_mappings)
        return gathered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        indices, rank_mappings = ctx.saved_tensors

        num_output_rows = ctx.num_rows

        input_grad = _nvshmem_scatter(
            grad_output, indices, rank_mappings, num_output_rows
        )

        indices_grad = None
        rank_mappings_grad = None
        return input_grad, indices_grad, rank_mappings_grad


class NVSHMEMScatterFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, indices, rank_mappings, num_output_rows):
        # Allocate buffer
        ctx.save_for_backward(indices, rank_mappings)
        scattered_tensors = _nvshmem_scatter(
            input_tensor,
            indices,
            rank_mappings,
            num_output_rows,
        )
        return scattered_tensors

    @staticmethod
    def backward(ctx, grad_output):

        # nvshmem_grad_output =
        nvshmem.register_memory(grad_output)
        indices, rank_mappings = ctx.saved_tensors
        input_grad = _nvshmmem_gather(grad_output, indices, rank_mappings)
        nvshmem.deregister_memory(grad_output)
        indices_grad = None
        rank_mappings_grad = None
        num_output_rows_grad = None
        return input_grad, indices_grad, rank_mappings_grad, num_output_rows_grad


class NVSHMEMBackendEngine(BackendEngine):
    _is_initialized = False
    _rank = -1
    _world_size = -1
    _ranks_per_graph = -1
    _nvshmem_p2p_obj = None
    _partition_size = -1
    _local_rank = -1
    _partition_num = -1

    def __init__(self, ranks_per_graph=-1, *args, **kwargs):
        # check if already initialized
        if not NVSHMEMBackendEngine._is_initialized:
            self.init_process_group(ranks_per_graph=-1, *args, **kwargs)

    def init_process_group(self, ranks_per_graph=-1, *args, **kwargs):
        if not NVSHMEMBackendEngine._is_initialized:
            nvshmem.NVSHMEMP2P.init()
            NVSHMEMBackendEngine._is_initialized = True
            NVSHMEMBackendEngine._rank = nvshmem.NVSHMEMP2P.get_rank()
            NVSHMEMBackendEngine._world_size = nvshmem.NVSHMEMP2P.get_world_size()
            NVSHMEMBackendEngine._ranks_per_graph = NVSHMEMBackendEngine._world_size

            NVSHMEMBackendEngine._nvshmem_p2p_obj = nvshmem.NVSHMEMP2P

            if ranks_per_graph > 0:
                assert (
                    NVSHMEMBackendEngine._world_size % ranks_per_graph == 0
                ), "World size not divisible by ranks per graph"
                NVSHMEMBackendEngine._ranks_per_graph = ranks_per_graph
                NVSHMEMBackendEngine._partition_size = (
                    NVSHMEMBackendEngine._world_size // ranks_per_graph
                )
                NVSHMEMBackendEngine._partition_num = (
                    NVSHMEMBackendEngine._rank // NVSHMEMBackendEngine._partition_size
                )
            else:
                NVSHMEMBackendEngine._partition_size = NVSHMEMBackendEngine._world_size
                NVSHMEMBackendEngine._ranks_per_graph = NVSHMEMBackendEngine._world_size
                NVSHMEMBackendEngine._partition_num = 0

    @staticmethod
    def get_rank() -> int:
        return NVSHMEMBackendEngine._rank

    @staticmethod
    def get_world_size() -> int:
        return NVSHMEMBackendEngine._world_size

    @staticmethod
    def get_local_rank() -> int:
        return NVSHMEMBackendEngine._local_rank

    @staticmethod
    def get_partition_num() -> int:
        return NVSHMEMBackendEngine._partition_num

    @staticmethod
    def get_partition_size() -> int:
        return NVSHMEMBackendEngine._partition_size

    def gather(self, input_tensor, indices, rank_mappings):
        assert (
            len(input_tensor.shape) == 3
        ), "Input tensor must be 3D of shape (bs, N, F)"
        assert len(indices.shape) == 2, "Indices tensor must be 2D of shape (bs, E)"

        assert (
            input_tensor.shape[0] == indices.shape[0]
        ), "Batch size of input tensor and indices tensor must match"

        bs = input_tensor.shape[0]
        assert bs == 1, "Batch size must be 1"
        assert rank_mappings.shape == indices.shape, "Rank mappings shape mismatch"

        # Check if on CUDA
        assert input_tensor.is_cuda, "Input tensor must be on CUDA"
        assert indices.is_cuda, "Indices tensor must be on CUDA"
        assert rank_mappings.is_cuda, "Rank mappings tensor must be on CUDA"

        # Convert the rank mappings to global rank mappings
        rank_mappings = (
            NVSHMEMBackendEngine._ranks_per_graph * NVSHMEMBackendEngine._partition_num
            + rank_mappings
        )

        num_input_rows = input_tensor.shape[1]
        num_output_rows = indices.shape[1]

        gathered_tensors = NVSHMEMGatherFunction.apply(
            input_tensor, indices, rank_mappings
        )
        return gathered_tensors

    def scatter(self, input_tensor, indices, rank_mappings, num_output_rows):
        assert (
            len(input_tensor.shape) == 3
        ), "Input tensor must be 3D of shape (bs, N, F)"
        assert len(indices.shape) == 2, "Indices tensor must be 2D of shape (bs, E)"
        bs = input_tensor.shape[0]
        assert bs == 1, "Batch size must be 1"
        assert (
            input_tensor.shape[0] == indices.shape[0]
        ), "Batch size of input tensor and indices tensor must match"
        assert rank_mappings.shape == indices.shape, "Rank mappings shape mismatch"

        rank_mappings = (
            NVSHMEMBackendEngine._ranks_per_graph * NVSHMEMBackendEngine._partition_num
            + rank_mappings
        )

        scattered_tensors = NVSHMEMScatterFunction.apply(
            input_tensor, indices, rank_mappings, num_output_rows
        )
        return scattered_tensors

    def get_max(self, val) -> int:
        assert dist.is_initialized(), "Distributed not initialized"
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
        return val

    def barrier(self):
        assert NVSHMEMBackendEngine._is_initialized, "NVSHMEM not initialized"
        assert NVSHMEMBackendEngine._nvshmem_p2p_obj is not None, "NVSHMEM P2P obj None"
        NVSHMEMBackendEngine._nvshmem_p2p_obj.barrier()
        return

    def destroy(self):
        assert NVSHMEMBackendEngine._is_initialized, "NVSHMEM not initialized"
        assert NVSHMEMBackendEngine._nvshmem_p2p_obj is not None, "NVSHMEM P2P obj None"
        NVSHMEMBackendEngine._nvshmem_p2p_obj.finalize()
        NVSHMEMBackendEngine._is_initialized = False
        NVSHMEMBackendEngine._rank = -1
        NVSHMEMBackendEngine._world_size = -1
        NVSHMEMBackendEngine._ranks_per_graph = -1
        NVSHMEMBackendEngine._nvshmem_p2p_obj = None
        return

    def get_local_rank_slice(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert NVSHMEMBackendEngine._is_initialized, "NVSHMEM not initialized"
        assert NVSHMEMBackendEngine._nvshmem_p2p_obj is not None, "NVSHMEM P2P obj None"
        rank = NVSHMEMBackendEngine._rank
        world_size = NVSHMEMBackendEngine._world_size

        tensor_shape = tensor.shape
        tensor_size = tensor_shape[dim]
        assert tensor_size % world_size == 0, "Tensor size not divisible by world size"
        local_size = tensor_size // world_size
        start_index = rank * local_size
        end_index = start_index + local_size
        length = end_index - start_index
        slice_tensor = tensor.narrow(dim, start_index, length)
        return slice_tensor
