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
from DGraph.distributed.nccl._indices_utils import (
    _generate_local_rank_mapping,
    _get_local_unique_recv_placement,
)
from DGraph.distributed.nccl._nccl_cache import NCCLGatherCache, NCCLScatterCache
from DGraph.distributed.nccl.alltoallv_impl import (
    _nccl_alltoall_v,
    _nccl_alltoallv_with_dict,
)
from DGraph.distributed.RankLocalOps import (
    RankLocalMaskedGather,
    RankLocalMaskedScatter,
    RankLocalRenumberingWithMapping,
    OptimizedRankLocalMaskedGather,
)
from torch.autograd import Function
from DGraph.utils import largest_split


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

    def scatter(
        self,
        local_send_tensor: torch.Tensor,
        indices: torch.Tensor,
        rank_mappings: torch.Tensor,
        output_size: int,
        cache: Optional[NCCLScatterCache] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
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

        use_cache = cache is not None

        if use_cache:
            assert type(cache) == NCCLScatterCache
            scatter_cache = cache
        else:
            scatter_cache = None

        output_tensor = ScatterFunction.apply(
            local_send_tensor,
            indices,
            src_ranks,
            dest_ranks,
            output_size,
            rank,
            world_size,
            scatter_cache,
        )

        return output_tensor  # type: ignore

    def gather(
        self,
        local_send_tensor: torch.Tensor,
        indices: torch.Tensor,
        rank_mappings: torch.Tensor,
        cache: Optional[NCCLGatherCache] = None,
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

        use_cache = cache is not None

        if use_cache:
            assert type(cache) == NCCLGatherCache, f"Invalid cache type {type(cache)}"
            gather_cache = cache
        else:
            gather_cache = None

        output_tensor = GatherFunction.apply(
            local_send_tensor,
            indices,
            send_rank,
            recv_rank,
            rank,
            world_size,
            gather_cache,
        )

        dist.barrier()
        return output_tensor  # type: ignore

    def destroy(self) -> None:
        if NCCLBackendEngine._is_initialized:
            # dist.destroy_process_group()
            NCCLBackendEngine._is_initialized = False

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
