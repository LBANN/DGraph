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
from DGraph.distributed.nccl.gather_op_impl import _nccl_gather_op
from DGraph.distributed.nccl.scatter_op_impl import _nccl_scatter_op
from DGraph.distributed.RankLocalOps import RankLocalMaskedGather
from torch.autograd import Function


class GatherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        indices: torch.LongTensor,
        global_rank_mapping: torch.LongTensor,
        local_indices: torch.LongTensor,
        local_rank_mapping: torch.LongTensor,
        rank: int,
        world_size: int,
    ):
        ctx.save_for_backward(
            send_tensor,
            recv_tensor,
            indices,
            torch.tensor(rank),
            torch.tensor(world_size),
        )

        # do local gather
        recv_tensor[local_rank_mapping == rank] = RankLocalMaskedGather(
            send_tensor, local_indices, local_rank_mapping, rank
        )

        # do global gather
        _nccl_gather_op(send_tensor, recv_tensor, indices, rank, world_size)
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        send_tensor, recv_tensor, indices, rank, world_size = ctx.saved_tensors

        _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output, None, None


class ScatterFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        recv_tensor: torch.Tensor,
        indices: torch.Tensor,
        global_rank_mapping: torch.Tensor,
        local_indices: torch.Tensor,
        local_rank_mapping: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            send_tensor,
            recv_tensor,
            indices,
            torch.tensor(rank),
            torch.tensor(world_size),
        )
        recv_tensor.scatter_add_(0, local_indices, send_tensor)
        _nccl_scatter_op(send_tensor, recv_tensor, indices, rank, world_size)
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        send_tensor, recv_tensor, indices = ctx.saved_tensors
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output, None, None


def scatter(
    send_tensor: torch.Tensor, recv_tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return ScatterFunction.apply(send_tensor, recv_tensor, indices)  # type: ignore


def gather(send_tensor, recv_tensor, indices) -> torch.Tensor:
    return GatherFunction.apply(send_tensor, recv_tensor, indices)  # type: ignore


class NCCLBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        # check if already initialized
        self._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", *args, **kwargs)

        self._initialized = True

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def get_local_rank_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        rank = self.get_rank()
        world_size = self.get_world_size()
        tensor_shape = tensor.shape
        tensor_size = tensor_shape[1]
        local_size = tensor_size // world_size
        start_index = rank * local_size
        end_index = start_index + local_size
        return tensor[:, start_index:end_index]

    def scatter(self, *args, **kwargs) -> torch.Tensor:
        input_tensor: torch.Tensor = args[0]
        indices: torch.Tensor = args[1]
        local_size: int = args[2]
        batch_size: int = input_tensor.shape[0]
        feature_size: int = input_tensor.shape[2]

        output_tensor: torch.Tensor = torch.zeros(
            (
                batch_size,
                local_size,
                feature_size,
            )
        )
        return scatter(input_tensor, output_tensor, indices)

    def gather(self, *args, **kwargs) -> torch.Tensor:
        input_tensor = args[0]
        indices = args[1]
        rank_mappings = args[2]
        indices_shape = indices.shape
        b_size = indices_shape[0]

        assert b_size == 1, "Multi-batch gather disabled for testing"
        n = indices_shape[1]
        feature_size = input_tensor.shape[2]
        output_tensor = torch.zeros(b_size, n, feature_size, device=input_tensor.device)

        # do local gather

        local_indices = self.get_local_rank_slice(indices)
        local_rank_mapping = self.get_local_rank_slice(rank_mappings)
        return output_tensor

    def destroy(self) -> None:
        if self._initialized:
            # dist.destroy_process_group()
            self._initialized = False

    def finalize(self) -> None:
        if self._initialized:
            dist.barrier()
