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
from DGraph.distributed.nccl.alltoallv_impl import (
    _nccl_alltoall_v,
    _nccl_alltoallv_with_dict,
)
from DGraph.distributed.RankLocalOps import (
    RankLocalMaskedGather,
    RankLocalMaskedScatter,
    RankLocalReNumbering,
)
from torch.autograd import Function
from DGraph.utils import largest_split


class GatherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        local_send_tensor: torch.Tensor,
        indices: torch.LongTensor,
        send_ranks: torch.Tensor,
        recv_ranks: torch.Tensor,
        rank: int,
        world_size: int,
    ):
        num_local_input_rows = local_send_tensor.shape[1]

        ctx.save_for_backward(
            indices,
            send_ranks,
            recv_ranks,
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

        num_local_output_rows = largest_split(indices.shape[1], world_size)

        batch_size = 1
        num_features = local_send_tensor.shape[2]

        recv_tensor = torch.zeros(batch_size, num_local_output_rows, num_features).to(
            local_send_tensor.device
        )
        _start_index = num_local_output_rows * rank
        _end_index = num_local_output_rows * (rank + 1)
        local_indices_slice = indices[0][_start_index:_end_index]

        local_rank_mapping = send_ranks[_start_index:_end_index]

        local_indices = local_indices_slice % local_send_tensor.shape[1]

        if len(local_indices_slice) > 0:

            recv_tensor[:, local_rank_mapping == rank, :] = RankLocalMaskedGather(
                local_send_tensor, local_indices, local_rank_mapping, rank
            )

        recv_tensor = _nccl_alltoall_v(
            local_send_tensor=local_send_tensor,
            local_recv_tensor=recv_tensor,
            indices=indices,
            local_rank_mapping=local_rank_mapping,
            src_ranks=send_ranks,
            dest_ranks=recv_ranks,
            rank=rank,
            world_size=world_size,
        )

        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        indices, send_ranks, recv_ranks, num_local_input_rows, rank, world_size = (
            ctx.saved_tensors
        )

        # We need to switch the send and recv ranks
        _send_ranks = recv_ranks
        recv_ranks = send_ranks
        send_ranks = _send_ranks
        num_local_output_rows = num_local_input_rows.item()
        rank = rank.item()
        world_size = world_size.item()
        send_tensor = grad_output

        # Now it's a scatter operation
        num_features = send_tensor.shape[1]
        device = send_tensor.device
        local_rank_output = torch.zeros(1, num_local_output_rows, num_features).to(
            device
        )

        # Start local only scatter. Maybe a better way to do this - S.Z
        _start_index = (indices.shape[0] // world_size) * rank
        _end_index = (indices.shape[0] // world_size) * (rank + 1)

        local_indices_slice = indices[_start_index:_end_index]
        local_dest_ranks = recv_ranks[_start_index:_end_index]
        local_rank_output = RankLocalMaskedScatter(
            send_tensor,
            local_rank_output,
            local_indices_slice,
            local_dest_ranks,
            rank,
        )

        local_non_comm_mask = local_dest_ranks != rank

        send_buffer_dict = {}
        send_buffer_dict = {}
        if torch.any(local_non_comm_mask):
            # These rows need to be sent to other ranks
            # First aggregate these into a single buffer

            local_non_comm_indices = local_indices_slice[local_non_comm_mask]
            local_remote_dest_mappings = local_dest_ranks[local_non_comm_mask]
            renumbered_indices, unique_indices = RankLocalReNumbering(
                local_non_comm_indices
            )
            num_remote_rows = len(unique_indices)
            buffer = torch.zeros(1, num_remote_rows, num_features).to(device)
            buffer.scatter_add_(
                1,
                renumbered_indices.view(1, -1, 1).expand(1, -1, num_features),
                send_tensor[:, local_non_comm_mask, :],
            )
            receving_ranks = torch.unique(local_dest_ranks[local_non_comm_mask])
            for _recv_rank in receving_ranks:
                _recv_mask = local_remote_dest_mappings == _recv_rank
                _recv_indices = renumbered_indices[_recv_mask]
                send_buffer_dict[_recv_rank.item()] = buffer[:, _recv_indices, :]

        all_comm_mask = send_ranks != recv_ranks
        reciever_mask = recv_ranks == rank
        receive_from_remote = all_comm_mask & reciever_mask

        recv_buffer_dict = {}
        recv_placement = {}
        if torch.any(receive_from_remote):
            receive_from_ranks = send_ranks[receive_from_remote]

            for _sender in range(world_size):
                if torch.any(receive_from_ranks == _sender):
                    _send_mask = (send_ranks == _sender) & receive_from_remote
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

        return (
            send_tensor_grad,
            indices_grad,
            send_ranks_grad,
            recv_ranks_grad,
            rank_grad,
            world_size_grad,
        )


class ScatterFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        indices: torch.Tensor,
        send_ranks: torch.Tensor,
        recv_ranks: torch.Tensor,
        num_local_output_rows: int,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            indices,
            send_ranks,
            recv_ranks,
            torch.tensor(num_local_output_rows),
            torch.tensor(rank),
            torch.tensor(world_size),
        )
        num_features = send_tensor.shape[1]
        device = send_tensor.device
        local_rank_output = torch.zeros(1, num_local_output_rows, num_features).to(
            device
        )

        # Start local only scatter. Maybe a better way to do this - S.Z
        _start_index = (indices.shape[0] // world_size) * rank
        _end_index = (indices.shape[0] // world_size) * (rank + 1)

        local_indices_slice = indices[_start_index:_end_index]
        local_dest_ranks = recv_ranks[_start_index:_end_index]

        local_rank_output = RankLocalMaskedScatter(
            send_tensor,
            local_rank_output,
            local_indices_slice,
            local_dest_ranks,
            rank,
        )

        local_non_comm_mask = local_dest_ranks != rank

        send_buffer_dict = {}
        send_buffer_dict = {}
        if torch.any(local_non_comm_mask):
            # These rows need to be sent to other ranks
            # First aggregate these into a single buffer

            local_non_comm_indices = local_indices_slice[local_non_comm_mask]
            local_remote_dest_mappings = local_dest_ranks[local_non_comm_mask]
            renumbered_indices, unique_indices = RankLocalReNumbering(
                local_non_comm_indices
            )
            num_remote_rows = len(unique_indices)
            buffer = torch.zeros(1, num_remote_rows, num_features).to(device)
            buffer.scatter_add_(
                1,
                renumbered_indices.view(1, -1, 1).expand(1, -1, num_features),
                send_tensor[:, local_non_comm_mask, :],
            )
            receving_ranks = torch.unique(local_dest_ranks[local_non_comm_mask])
            for _recv_rank in receving_ranks:
                _recv_mask = local_remote_dest_mappings == _recv_rank
                _recv_indices = renumbered_indices[_recv_mask]
                send_buffer_dict[_recv_rank.item()] = buffer[:, _recv_indices, :]

        all_comm_mask = send_ranks != recv_ranks
        reciever_mask = recv_ranks == rank
        receive_from_remote = all_comm_mask & reciever_mask

        recv_buffer_dict = {}
        recv_placement = {}
        if torch.any(receive_from_remote):
            receive_from_ranks = send_ranks[receive_from_remote]

            for _sender in range(world_size):
                if torch.any(receive_from_ranks == _sender):
                    _send_mask = (send_ranks == _sender) & receive_from_remote
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
        return local_rank_output

    @staticmethod
    def backward(ctx, grad_output):
        indices, send_ranks, recv_ranks, _, rank, world_size = ctx.saved_tensors
        # We need to switch the send and recv ranks
        print(indices.shape)
        _send_ranks = recv_ranks
        recv_ranks = send_ranks
        send_ranks = _send_ranks
        rank = rank.item()
        world_size = world_size.item()
        send_tensor = grad_output

        indices = indices.unsqueeze(0)
        # Now it's a gather operation
        num_local_output_rows = largest_split(indices.shape[1], world_size)

        batch_size = 1
        num_features = grad_output.shape[2]

        recv_tensor = torch.zeros(batch_size, num_local_output_rows, num_features).to(
            grad_output.device
        )
        _start_index = num_local_output_rows * rank
        _end_index = num_local_output_rows * (rank + 1)
        local_indices_slice = indices[0][_start_index:_end_index]

        local_rank_mapping = send_ranks[_start_index:_end_index]

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
            src_ranks=send_ranks,
            dest_ranks=recv_ranks,
            rank=rank,
            world_size=world_size,
        )
        # NOTE: even if the inputs are non-tensors, the number of backward outputs
        # must be the same as the number of inputs.
        send_tensor_grad = recv_tensor
        indices_grad = None
        send_ranks_grad = None
        recv_ranks_grad = None
        num_local_output_rows_grad = None
        rank_grad = None
        world_size_grad = None

        return (
            send_tensor_grad,
            indices_grad,
            send_ranks_grad,
            recv_ranks_grad,
            num_local_output_rows_grad,
            rank_grad,
            world_size_grad,
        )


class NCCLBackendEngine(BackendEngine):
    _is_initialized = False
    _rank = -1
    _world_size = -1
    _ranks_per_partition = -1
    _partition_rank = -1
    _partition_id = -1

    def __init__(self, ranks_per_partition=-1, *args, **kwargs):
        # check if already initialized
        # self._initialized = dist.is_initialized()
        if not NCCLBackendEngine._is_initialized:
            self.init_process_group(ranks_per_partition)

    def init_process_group(self, ranks_per_partition=-1, *args, **kwargs):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", *args, **kwargs)

        NCCLBackendEngine._is_initialized = True
        NCCLBackendEngine._rank = dist.get_rank()
        NCCLBackendEngine._world_size = dist.get_world_size()
        if ranks_per_partition == -1:
            NCCLBackendEngine._ranks_per_partition = NCCLBackendEngine._world_size
        else:
            assert (
                NCCLBackendEngine._world_size % ranks_per_partition == 0
            ), "Invalid ranks per partition"
            NCCLBackendEngine._ranks_per_partition = ranks_per_partition
        NCCLBackendEngine._partition_rank = (
            NCCLBackendEngine._rank % ranks_per_partition
        )
        NCCLBackendEngine._partition_id = NCCLBackendEngine._rank // ranks_per_partition

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

    def scatter(
        self, local_send_tensor, indices, rank_mappings, output_size, *args, **kwargs
    ) -> torch.Tensor:
        send_tensor_shape = local_send_tensor.shape
        b_size = send_tensor_shape[0]
        assert b_size == 1, "Multi-batch gather disabled for testing"
        assert len(send_tensor_shape) == 3, "Currently only support 3D tensors"
        assert len(rank_mappings.shape) == 2
        assert rank_mappings.shape[0] == 2
        assert indices.shape[-1] == rank_mappings.shape[-1]
        assert local_send_tensor.device.type == "cuda"

        world_size = self.get_world_size()
        rank = self.get_rank()
        send_rank = rank_mappings[0]
        recv_rank = rank_mappings[1]
        output_tensor = ScatterFunction.apply(
            local_send_tensor,
            indices,
            send_rank,
            recv_rank,
            output_size,
            rank,
            world_size,
        )

        return output_tensor

    def gather(
        self, local_send_tensor, indices, rank_mappings, **kwargs
    ) -> torch.Tensor:

        send_tensor_shape = local_send_tensor.shape
        b_size = send_tensor_shape[0]
        assert b_size == 1, "Multi-batch gather disabled for testing"
        assert len(send_tensor_shape) == 3, "Currently only support 3D tensors"
        assert len(rank_mappings.shape) == 2
        assert rank_mappings.shape[0] == 2
        assert indices.shape[-1] == rank_mappings.shape[-1]
        assert local_send_tensor.device.type == "cuda"

        world_size = self.get_world_size()
        rank = self.get_rank()
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

        return output_tensor  # type: ignore

    def destroy(self) -> None:
        if self._initialized:
            # dist.destroy_process_group()
            self._initialized = False

    def finalize(self) -> None:
        if self._initialized:
            dist.barrier()
