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


class GatherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        local_send_tensor: torch.Tensor,
        indices: torch.LongTensor,
        # vertex_ranks: torch.Tensor,
        edge_rank_loc: torch.Tensor,
        edge_dest_ranks: torch.Tensor,
        rank: int,
        world_size: int,
        cache: Optional[NCCLGatherCache] = None,
    ):
        num_local_input_rows = local_send_tensor.shape[1]

        if cache is not None:
            # We have a cache, use it, don't need to save anything
            ctx.has_cache = True
            ctx.cache = cache
            # TODO: Should we cash the indices as well? - S.Z
        else:
            ctx.has_cache = False

        ctx.save_for_backward(
            indices,
            edge_rank_loc,
            edge_dest_ranks,
            torch.tensor(num_local_input_rows),
            torch.tensor(rank),
            torch.tensor(world_size),
        )

        # Since NCCL is two-sided, we need to push from local rank and pull from
        # remote rank to get the global gather

        # TODO: One possible optmization is cache all these calculations
        # and only do the gather when the cache is invalidated. Essentially
        # if we are working with static graphs, the indices and distribution pattern
        # will not change and we can cache the communication pattern. - S.Z

        # We can also pre-compute this on the data ingestion side. Might
        # be worth looking to some kind of cached communication pattern store
        # that can be passed to the communicator. - S.Z

        batch_size = 1
        num_features = local_send_tensor.shape[2]

        if cache is not None:
            local_indices = cache.gather_local_indices % local_send_tensor.shape[1]
            local_gather_mask = cache.gather_local_comm_mask
            needs_comm = cache.gather_needs_comm
            local_output_rows = cache.gather_num_output_rows
            local_rank_mapping = cache.gather_local_remapped_ranks
            recv_tensor = torch.zeros(batch_size, local_output_rows, num_features).to(
                local_send_tensor.device
            )
            local_recv_tensor = cache.gather_local_recv_mapping
        else:
            # Get the edges that are local to the rank

            local_slice_mask = edge_rank_loc == rank

            num_local_output_rows = int(local_slice_mask.sum().item())

            recv_tensor = torch.zeros(
                batch_size, num_local_output_rows, num_features
            ).to(local_send_tensor.device)

            local_indices_slice = indices[local_slice_mask.unsqueeze(0)]
            local_rank_mapping = edge_rank_loc[local_slice_mask]
            local_recv_tensor = edge_dest_ranks[local_slice_mask]

            # assert torch.all(local_recv_tensor == rank), local_recv_tensor

            local_indices = local_indices_slice % local_send_tensor.shape[1]

            needs_comm = (local_recv_tensor != rank).any()

        recv_tensor = OptimizedRankLocalMaskedGather(
            local_send_tensor,
            local_indices,
            local_rank_mapping,
            recv_tensor,
            rank,
        )

        if needs_comm:

            recv_tensor = _nccl_alltoall_v(
                local_send_tensor=local_send_tensor,
                local_recv_tensor=recv_tensor,
                indices=indices,
                local_rank_mapping=local_recv_tensor,
                edge_rank_loc=edge_rank_loc,
                src_rank_loc=edge_dest_ranks,
                rank=rank,
                world_size=world_size,
                cache=cache,
            )

        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # We need to switch the send and recv ranks
        (
            indices,
            recv_ranks,
            send_ranks,
            # vertices_per_rank,
            num_local_input_rows,
            rank,
            world_size,
        ) = ctx.saved_tensors

        if ctx.has_cache:
            cache: Optional[NCCLGatherCache] = ctx.cache
        else:
            cache = None

        num_local_output_rows = num_local_input_rows.item()
        rank = rank.item()
        world_size = world_size.item()
        send_tensor = grad_output

        # Now it's a scatter operation
        num_features = send_tensor.shape[-1]
        device = send_tensor.device
        local_rank_output = torch.zeros(1, num_local_output_rows, num_features).to(
            device
        )

        indices = indices.view(-1)
        local_slice_mask = recv_ranks == rank
        local_indices_slice = indices[local_slice_mask]
        local_dest_ranks = send_ranks[local_slice_mask]

        local_rank_output = RankLocalMaskedScatter(
            send_tensor,
            local_rank_output,
            local_indices_slice,
            local_dest_ranks,
            rank,
        )

        if cache is not None:
            local_comm_mask = cache.scatter_local_comm_mask
        else:
            local_comm_mask = local_dest_ranks != rank

        send_buffer_dict = {}
        if torch.any(local_comm_mask):
            # These rows need to be sent to other ranks
            # First aggregate these into a single buffer

            if cache is not None:
                num_remote_rows = cache.scatter_num_remote_rows
                remapped_ranks = cache.scatter_local_remapped_ranks
                renumbered_indices = cache.scatter_renumbered_indices
                receiving_ranks = cache.scatter_remote_send_to_ranks

            else:

                local_comm_indices = local_indices_slice[local_comm_mask]
                local_remote_dest_mappings = local_dest_ranks[local_comm_mask]

                renumbered_indices, unique_indices, remapped_ranks = (
                    RankLocalRenumberingWithMapping(
                        local_comm_indices, local_remote_dest_mappings
                    )
                )
                receiving_ranks = torch.unique(local_dest_ranks[local_comm_mask])
                num_remote_rows = len(unique_indices)

            buffer = torch.zeros(1, num_remote_rows, num_features).to(device)
            buffer.scatter_add_(
                1,
                renumbered_indices.view(1, -1, 1).expand(1, -1, num_features),
                send_tensor[:, local_comm_mask, :],
            )

            for _recv_rank in receiving_ranks:
                _recv_indices = remapped_ranks == _recv_rank
                send_buffer_dict[_recv_rank.item()] = buffer[:, _recv_indices, :]

        # Now we need to receive the data from the remote ranks

        recv_buffer_dict = {}

        recv_placement = {}

        if cache is not None:
            recv_placement = cache.scatter_recv_local_placement

            # Allocate the receive buffers for the communication based on the
            # size of the recv_placement indices.
            for key, unique_send_indices in recv_placement.items():
                num_elements = unique_send_indices.shape[0]
                recv_buffer_dict[key] = torch.zeros(1, num_elements, num_features).to(
                    device
                )
        else:
            send_to_rank = send_ranks  # Pedantic variable name change
            all_comm_mask = send_to_rank != recv_ranks
            reciever_mask = send_to_rank == rank
            receive_from_remote = all_comm_mask & reciever_mask

            if torch.any(receive_from_remote):
                receive_from_ranks = recv_ranks[receive_from_remote]

                for _sender in range(world_size):
                    if _sender == rank:
                        continue
                    if torch.any(receive_from_ranks == _sender):
                        _send_mask = (recv_ranks == _sender) & receive_from_remote
                        _send_indices = indices[_send_mask] % num_local_output_rows
                        # TODO: This is brittle, look into a better way to do this - S.Z

                        unique_send_indices = torch.unique(_send_indices)
                        num_elements = unique_send_indices.shape[0]
                        recv_buffer_dict[_sender] = torch.zeros(
                            1, num_elements, num_features
                        ).cuda()
                        recv_placement[_sender] = unique_send_indices

        recv_buffer_dict = _nccl_alltoallv_with_dict(
            send_buffer_dict, recv_buffer_dict, rank, world_size
        )
        for key, recv_buffer in recv_buffer_dict.items():
            local_rank_output.scatter_add_(
                1,
                recv_placement[key].view(1, -1, 1).expand(1, -1, num_features),
                recv_buffer,
            )

        send_tensor_grad = local_rank_output
        indices_grad = None
        send_ranks_grad = None
        recv_ranks_grad = None
        rank_grad = None
        world_size_grad = None
        cache_grad = None

        return (
            send_tensor_grad,
            indices_grad,
            send_ranks_grad,
            recv_ranks_grad,
            rank_grad,
            world_size_grad,
            cache_grad,
        )


class ScatterFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        indices: torch.Tensor,
        edge_src_ranks: torch.Tensor,
        edge_dest_ranks: torch.Tensor,
        num_local_output_rows: int,
        rank: int,
        world_size: int,
        scatter_cache: Optional[NCCLScatterCache] = None,
    ) -> torch.Tensor:

        ctx.save_for_backward(
            indices,
            edge_src_ranks,
            edge_dest_ranks,
            torch.tensor(num_local_output_rows),
            torch.tensor(rank),
            torch.tensor(world_size),
        )
        use_cache = scatter_cache is not None
        if use_cache:
            ctx.scatter_cache = scatter_cache
            ctx.has_cache = True
        else:
            ctx.has_cache = False

        num_features = send_tensor.shape[-1]
        device = send_tensor.device

        local_rank_output = torch.zeros(1, num_local_output_rows, num_features).to(
            device
        )

        indices = indices.view(-1)

        local_edge_mask = edge_src_ranks == rank

        local_indices_slice = indices[local_edge_mask]
        local_dest_ranks = edge_dest_ranks[local_edge_mask]

        local_rank_output = RankLocalMaskedScatter(
            send_tensor,
            local_rank_output,
            local_indices_slice,
            local_dest_ranks,
            rank,
        )

        if use_cache:
            local_comm_mask = scatter_cache.scatter_local_comm_mask
        else:
            local_comm_mask = local_dest_ranks != rank

            all_comm_mask = edge_src_ranks != edge_dest_ranks
            reciever_mask = edge_dest_ranks == rank
            receive_from_remote_mask = all_comm_mask & reciever_mask

        send_buffer_dict = {}

        if torch.any(local_comm_mask):

            if use_cache:
                num_remote_rows = scatter_cache.scatter_num_remote_rows
                remapped_ranks = scatter_cache.scatter_local_remapped_ranks
                renumbered_indices = scatter_cache.scatter_local_renumbered_indices
                receving_ranks = scatter_cache.scatter_remote_send_to_ranks

            else:
                # These rows need to be sent to other ranks
                # First aggregate these into a single buffer
                local_comm_indices = local_indices_slice[local_comm_mask]
                local_remote_dest_mappings = local_dest_ranks[local_comm_mask]
                # TODO: This is very slow, look into a better way to do this - S.Z
                # Uncached is slow, should look into augmenting torch functions
                # to speed this up - S.Z
                renumbered_indices, unique_indices, remapped_ranks = (
                    RankLocalRenumberingWithMapping(
                        local_comm_indices, local_remote_dest_mappings
                    )
                )
                num_remote_rows = len(unique_indices)
                receving_ranks = torch.unique(local_dest_ranks[local_comm_mask])

            buffer = torch.zeros(1, num_remote_rows, num_features).to(device)
            buffer.scatter_add_(
                1,
                renumbered_indices.view(1, -1, 1).expand(1, -1, num_features),
                send_tensor[:, local_comm_mask, :],
            )

            for _recv_rank in receving_ranks:
                _recv_indices = remapped_ranks == _recv_rank
                send_buffer_dict[_recv_rank.item()] = buffer[:, _recv_indices, :]

        recv_buffer_dict = {}
        recv_placement = {}
        if use_cache:
            recv_placement = scatter_cache.scatter_recv_local_placement
        else:
            recv_placement = _get_local_unique_recv_placement(
                indices,
                edge_src_ranks,
                receive_from_remote_mask,
                num_local_output_rows,
                rank,
                world_size,
            )

        # Allocate the receive buffers for the communication based on the
        # size of the recv_placement indices.
        for key, unique_send_indices in recv_placement.items():
            num_elements = unique_send_indices.shape[0]
            recv_buffer_dict[key] = torch.zeros(1, num_elements, num_features).to(
                device
            )
        recv_buffer_dict = _nccl_alltoallv_with_dict(
            send_buffer_dict, recv_buffer_dict, rank, world_size
        )
        for key, recv_buffer in recv_buffer_dict.items():
            local_rank_output.scatter_add_(
                1,
                recv_placement[key].view(1, -1, 1).expand(1, -1, num_features),
                recv_buffer,
            )
        return local_rank_output

    @staticmethod
    def backward(ctx, grad_output):
        # We need to switch the send and recv ranks
        indices, recv_ranks, send_ranks, num_input_rows, rank, world_size = (
            ctx.saved_tensors
        )

        local_mask = recv_ranks == rank
        if ctx.has_cache:
            cache: NCCLScatterCache = ctx.scatter_cache
            num_local_output_rows = cache.gather_num_output_rows

        else:
            rank = int(rank.item())
            world_size = int(world_size.item())

            indices = indices.view(1, -1)

            # Now it's a gather operation

            num_local_output_rows = int(local_mask.sum().item())

        batch_size = 1
        num_features = grad_output.shape[2]

        recv_tensor = torch.zeros(batch_size, num_local_output_rows, num_features).to(
            grad_output.device
        )

        local_indices_slice = indices[0][local_mask]
        local_rank_mapping = send_ranks[local_mask]

        local_indices = local_indices_slice % grad_output.shape[1]

        if len(local_indices_slice) > 0:

            recv_tensor[:, local_rank_mapping == rank, :] = RankLocalMaskedGather(
                grad_output, local_indices, local_rank_mapping, rank
            )

        recv_tensor = _nccl_alltoall_v(
            local_send_tensor=grad_output,
            local_recv_tensor=recv_tensor,
            indices=indices,
            local_rank_mapping=local_rank_mapping,
            edge_rank_loc=send_ranks,
            src_rank_loc=recv_ranks,
            rank=rank,
            world_size=world_size,
        )

        # if rank == 0:
        #     breakpoint()
        # dist.barrier()
        # NOTE: even if the inputs are non-tensors, the number of backward outputs
        # must be the same as the number of inputs.
        send_tensor_grad = recv_tensor
        indices_grad = None
        send_ranks_grad = None
        recv_ranks_grad = None
        num_local_output_rows_grad = None
        rank_grad = None
        world_size_grad = None
        scatter_cache_grad = None

        return (
            send_tensor_grad,
            indices_grad,
            send_ranks_grad,
            recv_ranks_grad,
            num_local_output_rows_grad,
            rank_grad,
            world_size_grad,
            scatter_cache_grad,
        )


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
