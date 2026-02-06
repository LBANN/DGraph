import torch
from typing import Optional
from torch.autograd import Function
import torch.distributed as dist
from dataclasses import dataclass
from DGraph.distributed.nccl._nccl_cache import NCCLGatherCache, NCCLScatterCache
from DGraph.distributed.RankLocalOps import (
    OptimizedRankLocalMaskedGather,
    OptimizedLocalScatterGather,
    OptimizedLocalScatterSumGather,
)
from DGraph.distributed.nccl._NCCLCommPlan import NCCLGraphCommPlan


class CommPlan_GatherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        local_send_tensor: torch.Tensor,
        comm_plan: NCCLGraphCommPlan,
    ) -> torch.Tensor:
        """
        Forward pass for distributed gather using the common plan to effectively perform:
            y[i] = x[indices[i]]

        The process is as follows:
            1) Perform local gather from local vertices to local edges
            2) Gather

        Args:
            ctx (torch.autograd.FunctionContext): Context object
            local_send_tensor (torch.Tensor): Local send tensor
            comm_plan (GatherCommPlan): Communication plan
        """
        assert len(local_send_tensor.shape) == 3, (
            "Local send tensor must be of shape (batch_size, num_rows, num_features)"
            + f"but received {local_send_tensor.shape}"
        )
        ctx.comm_plan = comm_plan

        num_features = local_send_tensor.shape[-1]
        num_batches = local_send_tensor.shape[0]

        output_tensor = torch.zeros(
            num_batches, comm_plan.num_local_edges, num_features
        ).to(local_send_tensor.device)

        # Local vertex to edge gather
        output_tensor = OptimizedLocalScatterGather(
            src=local_send_tensor,
            src_indices=comm_plan.local_vertex_idx,
            dst_indices=comm_plan.local_edge_idx,
            output=output_tensor,
        )

        # To do: Combine this with the local gather above to reduce kernel launches
        total_send = sum(comm_plan.boundary_vertex_splits)
        if total_send > 0:

            send_buf = local_send_tensor[:, comm_plan.boundary_vertex_idx, :]
        else:
            send_buf = torch.empty(0, 0, num_features).to(local_send_tensor.device)

        total_recv = sum(comm_plan.boundary_edge_splits)

        # If no messages, the size at dim 0 must be 0
        _effective_batch_size = 1 if total_recv > 0 else 0

        recv_buffer = torch.empty(_effective_batch_size, total_recv, num_features).to(
            local_send_tensor.device
        )

        # For now assume single graph per GPU
        send_buf = send_buf.contiguous().squeeze() if total_send > 0 else send_buf
        recv_buffer = (
            recv_buffer.contiguous().squeeze() if total_recv > 0 else recv_buffer
        )
        dist.all_to_all_single(
            recv_buffer,
            send_buf,
            output_split_sizes=comm_plan.boundary_edge_splits,
            input_split_sizes=comm_plan.boundary_vertex_splits,
        )

        if total_recv > 0:
            # recv_buffer = recv_buffer.unsqueeze(0)
            recv_buffer = recv_buffer.reshape(_effective_batch_size, -1, num_features)

            output_tensor = OptimizedLocalScatterGather(
                src=recv_buffer,
                src_indices=comm_plan.boundary_edge_buffer_map,
                dst_indices=comm_plan.boundary_edge_idx,
                output=output_tensor,
            )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for distributed gather

        Args:
            ctx (torch.autograd.FunctionContext): Context object
            grad_output (torch.Tensor): Gradient of the output tensor.
                Shape: (batch_size, num_local_edges, num_features)
        """
        comm_plan = ctx.comm_plan
        num_features = grad_output.shape[-1]
        num_batches = grad_output.shape[0]
        device = grad_output.device

        grad_input = torch.zeros(
            num_batches, comm_plan.num_local_vertices, num_features, device=device
        )

        grad_input = OptimizedLocalScatterSumGather(
            src=grad_output,
            output=grad_input,
            src_indices=comm_plan.local_edge_idx,
            dst_indices=comm_plan.local_vertex_idx,
        )

        total_send = len(comm_plan.boundary_vertex_idx)
        if total_send > 0:
            send_buf = torch.zeros(num_batches, total_send, num_features, device=device)

            send_buf = grad_output[:, comm_plan.boundary_vertex_idx, :]
        else:
            send_buf = torch.empty(0, 0, num_features).to(device)
        total_recv = sum(comm_plan.boundary_vertex_splits)

        _effective_batch_size = 1 if total_recv > 0 else 0
        recv_buffer = torch.empty(_effective_batch_size, total_recv, num_features).to(
            device
        )

        send_buf = send_buf.contiguous().squeeze() if total_send > 0 else send_buf
        recv_buffer = (
            recv_buffer.contiguous().squeeze() if total_recv > 0 else recv_buffer
        )

        dist.all_to_all_single(
            recv_buffer,
            send_buf,
            output_split_sizes=comm_plan.boundary_vertex_splits,
            input_split_sizes=comm_plan.boundary_edge_splits,
        )
        if total_recv > 0:
            # recv_buffer = recv_buffer.unsqueeze(0)
            recv_buffer = recv_buffer.reshape(_effective_batch_size, -1, num_features)

            grad_input = OptimizedLocalScatterSumGather(
                src=recv_buffer,
                output=grad_input,
                src_indices=comm_plan.boundary_edge_buffer_map,
                dst_indices=comm_plan.boundary_vertex_idx,
            )

        return grad_input, None


class CommPlan_ScatterFunction(Function):
    @staticmethod
    def forward(
        ctx,
        local_send_tensor: torch.Tensor,
        comm_plan: NCCLGraphCommPlan,
    ) -> torch.Tensor:
        """
        Forward pass for distributed scatter

        Args:
            ctx (torch.autograd.FunctionContext): Context object
            local_send_tensor (torch.Tensor): Local send tensor
            comm_plan (NCCLGraphCommPlan): Communication plan
        """
        local_device = local_send_tensor.device
        assert (
            len(local_send_tensor.shape) == 3
        ), "Local send tensor must be of shape (batch_size, num_rows, num_features)"
        ctx.comm_plan = comm_plan

        num_features = local_send_tensor.shape[-1]
        num_batches = local_send_tensor.shape[0]

        output_tensor = torch.zeros(
            num_batches, comm_plan.num_local_vertices, num_features
        ).to(device=local_device)

        output_tensor = OptimizedLocalScatterSumGather(
            src=local_send_tensor,
            output=output_tensor,
            src_indices=comm_plan.local_edge_idx,
            dst_indices=comm_plan.local_vertex_idx,
        )

        total_send = sum(comm_plan.boundary_edge_splits)

        if total_send > 0:
            send_buf = torch.zeros(
                num_batches,
                total_send,
                num_features,
                device=local_device,
            )

            send_buf = OptimizedLocalScatterSumGather(
                src=local_send_tensor,
                output=send_buf,
                src_indices=comm_plan.boundary_edge_idx,
                dst_indices=comm_plan.boundary_edge_buffer_map,
            )
        else:
            send_buf = torch.empty(0, 0, num_features, device=local_device)

        total_recv = sum(comm_plan.boundary_vertex_splits)

        _effective_batch_size = 1 if total_recv > 0 else 0
        recv_buffer = torch.empty(
            _effective_batch_size,
            total_recv,
            num_features,
            device=local_device,
        )

        send_buf = send_buf.contiguous().squeeze() if total_send > 0 else send_buf
        recv_buffer = (
            recv_buffer.contiguous().squeeze() if total_recv > 0 else recv_buffer
        )
        dist.all_to_all_single(
            recv_buffer,
            send_buf,
            output_split_sizes=comm_plan.boundary_vertex_splits,
            input_split_sizes=comm_plan.boundary_edge_splits,
        )
        if total_recv > 0:
            # recv_buffer = recv_buffer.unsqueeze(0)
            recv_buffer = recv_buffer.reshape(_effective_batch_size, -1, num_features)

            output_tensor = OptimizedLocalScatterSumGather(
                src=recv_buffer,
                output=output_tensor,
                src_indices=torch.arange(total_recv, device=local_device),
                dst_indices=comm_plan.boundary_vertex_idx,
            )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for distributed scatter

        Args:
            ctx (torch.autograd.FunctionContext): Context object
            grad_output (torch.Tensor): Gradient of the output tensor
        """
        comm_plan = ctx.comm_plan
        num_features = grad_output.shape[-1]
        num_batches = grad_output.shape[0]
        device = grad_output.device
        num_output_rows = comm_plan.num_local_edges

        grad_input = torch.zeros(
            num_batches, num_output_rows, num_features, device=device
        )

        grad_input = OptimizedLocalScatterGather(
            src=grad_output,
            src_indices=comm_plan.local_vertex_idx,
            dst_indices=comm_plan.local_edge_idx,
            output=grad_input,
        )

        total_send = len(comm_plan.boundary_vertex_splits)
        if total_send > 0:
            send_buf_locs = torch.arange(total_send, device=device)
            send_buf = torch.zeros(num_batches, total_send, num_features, device=device)
            send_buf = OptimizedLocalScatterGather(
                src=grad_output,
                src_indices=comm_plan.boundary_vertex_idx,
                dst_indices=send_buf_locs,
                output=send_buf,
            )
        else:
            send_buf = torch.empty(0, 0, num_features, device=device)

        total_recv = sum(comm_plan.boundary_edge_splits)

        _effective_batch_size = 1 if total_recv > 0 else 0
        recv_buffer = torch.empty(
            _effective_batch_size, total_recv, num_features, device=device
        )
        dist.all_to_all_single(
            recv_buffer,
            send_buf,
            output_split_sizes=comm_plan.boundary_edge_splits,
            input_split_sizes=comm_plan.boundary_vertex_splits,
        )

        send_buf = send_buf.contiguous().squeeze() if total_send > 0 else send_buf
        recv_buffer = (
            recv_buffer.contiguous().squeeze() if total_recv > 0 else recv_buffer
        )

        if total_recv > 0:
            # recv_buffer = recv_buffer.unsqueeze(0)
            recv_buffer = recv_buffer.reshape(_effective_batch_size, -1, num_features)

            grad_input = OptimizedLocalScatterGather(
                src=recv_buffer,
                src_indices=comm_plan.boundary_edge_idx,
                dst_indices=comm_plan.boundary_edge_buffer_map,
                output=grad_input,
            )

        return grad_input, None


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
    ):
        num_local_input_rows = local_send_tensor.shape[1]

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

        local_slice_mask = edge_rank_loc == rank

        num_local_output_rows = int(local_slice_mask.sum().item())

        recv_tensor = torch.zeros(batch_size, num_local_output_rows, num_features).to(
            local_send_tensor.device
        )

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
            cache=cache,
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
