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
import sys
from typing import Optional
import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
from DGraph.distributed.nccl._NCCLCommPlan import NCCLGraphCommPlan
from DGraph.distributed.nccl._torch_func_impl import (
    GatherFunction,
    ScatterFunction,
    CommPlan_ScatterFunction,
    CommPlan_GatherFunction,
)

from torch.autograd import Function
from DGraph.utils import largest_split
from typing import overload, List


TIMINGS = {"Gather_Index_Forward": [], "Gather_Forward_Local": []}


class NCCLBackendEngine(BackendEngine):
    _is_initialized = False
    _rank = -1
    _world_size = -1
    _ranks_per_partition = -1
    _partition_rank = -1
    _partition_id = -1

    def __init__(self, ranks_per_graph=-1, *args, **kwargs):
        # check if already initialized
        # self._initialized = dist.is_initialized()
        if not NCCLBackendEngine._is_initialized:
            self.init_process_group(ranks_per_graph)

    def init_process_group(self, ranks_per_graph=-1, *args, **kwargs):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", *args, **kwargs)

        NCCLBackendEngine._is_initialized = True
        NCCLBackendEngine._rank = dist.get_rank()
        NCCLBackendEngine._world_size = dist.get_world_size()
        if ranks_per_graph == -1:
            NCCLBackendEngine._ranks_per_partition = NCCLBackendEngine._world_size
        else:
            assert (
                NCCLBackendEngine._world_size % ranks_per_graph == 0
            ), "Invalid ranks per partition"
            NCCLBackendEngine._ranks_per_partition = ranks_per_graph
        NCCLBackendEngine._partition_rank = NCCLBackendEngine._rank % ranks_per_graph
        NCCLBackendEngine._partition_id = NCCLBackendEngine._rank // ranks_per_graph

    @staticmethod
    def get_rank() -> int:
        return dist.get_rank()

    @staticmethod
    def get_local_rank() -> int:
        return NCCLBackendEngine._partition_rank

    @staticmethod
    def get_partition_size() -> int:
        return NCCLBackendEngine._ranks_per_partition

    @staticmethod
    def get_partition_id() -> int:
        return NCCLBackendEngine._partition_id

    @staticmethod
    def get_world_size() -> int:
        return dist.get_world_size()

    def get_local_rank_slice(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        rank = self.get_rank()
        world_size = self.get_world_size()
        tensor_shape = tensor.shape
        tensor_size = tensor_shape[1]
        local_size = tensor_size // world_size
        start_index = rank * local_size
        end_index = start_index + local_size
        return tensor[:, start_index:end_index]

    @overload
    def scatter(
        self,
        local_send_tensor: torch.Tensor,
        indices: torch.Tensor,
        rank_mappings: torch.Tensor,
        output_size: int,
    ) -> torch.Tensor: ...

    @overload
    def scatter(
        self,
        local_send_tensor: torch.Tensor,
        *,
        comm_plan: NCCLGraphCommPlan,
    ) -> torch.Tensor: ...

    def scatter(
        self,
        local_send_tensor: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        rank_mappings: Optional[torch.Tensor] = None,
        output_size: Optional[int] = None,
        comm_plan: Optional[NCCLGraphCommPlan] = None,
    ) -> torch.Tensor:

        if comm_plan is not None:
            return CommPlan_ScatterFunction.apply(local_send_tensor, comm_plan)  # type: ignore
        else:
            if indices is None or rank_mappings is None or output_size is None:
                raise ValueError(
                    "Indices, rank mappings, and output size must be provided for NCCL backend"
                )

            send_tensor_shape = local_send_tensor.shape
            b_size = send_tensor_shape[0]

            world_size = self.get_world_size()
            rank = self.get_rank()
            assert b_size == 1, "Multi-batch gather disabled for testing"
            assert len(send_tensor_shape) == 3, "Currently only support 3D tensors"
            assert indices.shape[-1] == rank_mappings.shape[-1], (
                f"Indices shape: {indices.shape} and rank mappings shape: "
                + f" {rank_mappings.shape} must match"
            )
            assert rank_mappings.shape[0] == 2, (
                "Rank mappings shape[0] expected to be 2, "
                + f"but got {rank_mappings.shape[0]}"
            )
            assert (
                local_send_tensor.device.type == "cuda"
            ), f"Device: {local_send_tensor.device.type} expected cuda"
            assert output_size > 0, "Output size must be greater than 0"

            src_ranks = rank_mappings[0]
            dest_ranks = rank_mappings[1]

            output_tensor = ScatterFunction.apply(
                local_send_tensor,
                indices,
                src_ranks,
                dest_ranks,
                output_size,
                rank,
                world_size,
            )

        return output_tensor  # type: ignore

    @overload
    def gather(
        self,
        local_send_tensor: torch.Tensor,
        indices: torch.Tensor,
        rank_mappings: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor: ...

    @overload
    def gather(
        self,
        local_send_tensor: torch.Tensor,
        *,
        comm_plan: NCCLGraphCommPlan,
        **kwargs,
    ) -> torch.Tensor: ...

    def gather(
        self,
        local_send_tensor: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        rank_mappings: Optional[torch.Tensor] = None,
        comm_plan: Optional[NCCLGraphCommPlan] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Gather the distributed tensor across all ranks according to the indices

        Performs the operation:

        output_tensor[i] = local_send_tensor[indices[i]]

        if rank_mappings[indices[i]] == RankOf(output_tensor[i]). Otherwise, communication
        is needed such that, send_rank = rank_mappings[indices[i]] and
        recv_rank = RankOf(output_tensor[i]), and

        # on send_rank
        Send(local_send_tensor[indices[i]]) to recv_rank

        # on recv_rank
        output_tensor[i] = Recv(local_send_tensor[indices[i]]) from send_rank

        Args:
            local_send_tensor (torch.Tensor): The local slice of the tensor to
                be gathered by all ranks
            indices (torch.Tensor): The indices for the gather operation
            rank_mappings (torch.Tensor): The rank mappings for the gather operation
        """

        if comm_plan is not None:
            return CommPlan_GatherFunction.apply(local_send_tensor, comm_plan)  # type: ignore

        send_tensor_shape = local_send_tensor.shape
        b_size = send_tensor_shape[0]
        world_size = self.get_world_size()
        rank = self.get_rank()
        assert b_size == 1, "Multi-batch gather disabled for testing"
        assert len(send_tensor_shape) == 3, "Currently only support 3D tensors"

        if rank_mappings is None:
            raise ValueError("Rank mappings cannot be None for NCCL backend")

        assert (
            len(rank_mappings.shape) == 2
        ), f"Rank mappings shape: {rank_mappings.shape} expected 2-D."
        assert (
            rank_mappings.shape[0] == 2
        ), f"Rank mappings shape[0]: {rank_mappings.shape[0]} is expected be 2."

        assert indices.shape[-1] == rank_mappings.shape[-1]
        assert local_send_tensor.device.type == "cuda"

        send_rank = rank_mappings[0]
        recv_rank = rank_mappings[1]

        output_tensor = GatherFunction.apply(
            local_send_tensor,
            indices,
            send_rank,
            recv_rank,
            rank,
            world_size,
        )

        dist.barrier()
        return output_tensor  # type: ignore

    def destroy(self) -> None:
        if NCCLBackendEngine._is_initialized:
            # dist.destroy_process_group()
            NCCLBackendEngine._is_initialized = False

    def _get_splits(self, send_offset, recv_offset) -> tuple[List[int], List[int]]:
        """
        Return (send_splits, recv_splits) as plain Python lists of ints.

        send_splits[i] = number of *vertices* this rank sends to rank i.
        recv_splits[i] = number of *vertices* this rank receives from rank i.

        These are in vertex units; the caller must multiply by feature_dim
        before passing to all_to_all_single.
        """

        send_off = send_offset
        recv_off = recv_offset

        send_splits = (send_off[1:] - send_off[:-1]).tolist()
        recv_splits = (recv_off[1:] - recv_off[:-1]).tolist()

        return (send_splits, recv_splits)

    @staticmethod
    def _scale_splits(splits: List[int], factor: int) -> List[int]:
        """Multiply each split count by a scalar (feature dimension)."""
        return [s * factor for s in splits]

    def put(
        self,
        send_buffer: torch.Tensor,
        recv_buffer: torch.Tensor,
        send_offsets: torch.Tensor,
        recv_offsets: torch.Tensor,
        remote_offsets: torch.Tensor | None = None,
    ) -> None:
        _ = remote_offsets  #  remote_offsets not needed in 2-sided semantices

        send_splits, recv_splits = self._get_splits(
            send_offset=send_offsets, recv_offset=recv_offsets
        )
        feature_dim = send_buffer.shape[1] if send_buffer.ndim == 2 else 1

        # all_to_all_single operates on flat views; split sizes are in
        # element counts, so scale vertex counts by feature_dim.
        send_flat = send_buffer.contiguous().view(-1)
        recv_flat = recv_buffer.contiguous().view(-1)

        dist.all_to_all_single(
            output=recv_flat,
            input=send_flat,
            output_split_sizes=self._scale_splits(recv_splits, feature_dim),
            input_split_sizes=self._scale_splits(send_splits, feature_dim),
        )

    def finalize(self) -> None:
        if NCCLBackendEngine._is_initialized:
            dist.barrier()

    def barrier(self) -> None:
        if NCCLBackendEngine._is_initialized:
            dist.barrier()
        else:
            raise RuntimeError(
                "NCCLBackendEngine is not initialized, cannot call barrier"
            )
