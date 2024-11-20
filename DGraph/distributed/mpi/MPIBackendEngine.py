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
from typing import Optional
import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
from DGraph.distributed.RankLocalOps import (
    RankLocalMaskedGather,
    RankLocalMaskedScatter,
    RankLocalReNumbering,
    RankLocalReNumberingWithRankMapping,
)
from mpi4py import MPI
import warnings
from torch.autograd import Function
from functools import reduce


def _mpi_vector_get(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    indices: torch.Tensor,
    local_placement: torch.Tensor,
    rank_mapping: torch.Tensor,
    win: MPI.Win,
) -> torch.Tensor:
    num_local_send_rows = send_tensor.shape[1]
    num_features = send_tensor.shape[-1]
    rank_mapping = rank_mapping.view(-1)
    indices = indices.view(-1)
    local_placement = local_placement.view(-1)
    data_type = MPI.FLOAT

    for _index, remote_rank, local_index in zip(indices, rank_mapping, local_placement):
        displacement = (_index.item() % num_local_send_rows) * num_features
        count = num_features

        target_spec = (displacement, count, data_type)
        remote_rank = MPIBackendEngine.to_global_rank(int(remote_rank.item()))
        win.Get(
            [recv_tensor[0][local_index], MPI.FLOAT],
            target_rank=remote_rank,
            target=target_spec,
        )
    return recv_tensor


def _mpi_vector_accumulate(
    send_tensor: torch.Tensor,
    indices: torch.Tensor,
    rank_mapping: torch.Tensor,
    num_local_output_rows: int,
    win: MPI.Win,
) -> None:
    num_features = send_tensor.shape[-1]
    rank_mapping = rank_mapping.view(-1)
    indices = indices.view(-1)
    data_type = MPI.FLOAT

    for i, (_index, remote_rank) in enumerate(zip(indices, rank_mapping)):
        displacement = (_index.item() % num_local_output_rows) * num_features
        count = num_features

        target_spec = (displacement, count, data_type)
        remote_rank = MPIBackendEngine.to_global_rank(int(remote_rank.item()))
        win.Accumulate(
            [send_tensor[0][i], MPI.FLOAT],
            target_rank=remote_rank,
            target=target_spec,
            op=MPI.SUM,
        )


def _mpi_gather_impl(
    send_tensor: torch.Tensor, indices: torch.Tensor, src_rank_mapping: torch.Tensor
):
    bs = send_tensor.shape[0]
    num_input_rows = send_tensor.shape[1]
    num_features = send_tensor.shape[2]
    num_indices = indices.shape[1]
    device = send_tensor.device
    # src_rank_mapping = src_rank_mapping.view(num_indices)

    rank = MPIBackendEngine.get_local_rank()
    world_size = MPIBackendEngine.get_world_size()

    # Attach the send tensor to the window.
    # TODO: his can potentially be done in parallel with the local gather
    # as the local gather is read-only on the send_tensor. - S.Z
    win, attached_send_tensor = MPIBackendEngine.Attach(send_tensor)

    recv_tensor = torch.zeros(bs, num_indices, num_features, device=device)
    local_rank_src = src_rank_mapping == rank

    # First do local gather
    if local_rank_src.any():
        _local_indices = indices % num_input_rows
        recv_tensor[:, local_rank_src[0], :] = RankLocalMaskedGather(
            send_tensor,
            _local_indices,
            rank_mapping=src_rank_mapping,
            rank=rank,
        )

    non_local_rank = ~local_rank_src
    win.Fence()  # Start the Epoch for MPI RMA
    if non_local_rank.any():
        local_placement = torch.where(src_rank_mapping.squeeze(0) != rank)[0]

        remote_rank_mapping = src_rank_mapping[non_local_rank]
        remote_indices = indices[non_local_rank]
        recv_tensor = _mpi_vector_get(
            send_tensor=attached_send_tensor,
            recv_tensor=recv_tensor,
            indices=remote_indices,
            local_placement=local_placement.unsqueeze(0),
            rank_mapping=remote_rank_mapping,
            win=win,
        )
    win.Fence()  # End the Epoch for MPI RMA
    win.Free()  # Free the window
    return recv_tensor


def _mpi_scatter_add_impl(
    send_tensor: torch.Tensor,
    indices: torch.Tensor,
    num_output_rows: int,
    target_rank_mapping: torch.Tensor,
):
    bs = send_tensor.shape[0]

    num_features = send_tensor.shape[2]
    device = send_tensor.device
    rank = MPIBackendEngine.get_local_rank()
    local_message = target_rank_mapping == rank

    recv_tensor = torch.zeros(bs, num_output_rows, num_features, device=device)

    if local_message.any():
        _local_indices = indices % num_output_rows
        recv_tensor = RankLocalMaskedScatter(
            send_tensor,
            recv_tensor,
            _local_indices.view(-1),
            target_rank_mapping.view(-1),
            rank,
        )

    non_local_messages = ~local_message

    win, attached_recv_tensor = MPIBackendEngine.Attach(recv_tensor)

    win.Fence()  # Start the Epoch for MPI RMA
    if non_local_messages.any():
        comm_indices = indices[non_local_messages]
        comm_ranks = target_rank_mapping[non_local_messages]

        renumbered_indices, original_locs, original_rank_mapping = (
            RankLocalReNumberingWithRankMapping(comm_indices, comm_ranks)
        )

        num_remote_rows = len(original_locs)
        buffer = torch.zeros(1, num_remote_rows, num_features, device=device)
        buffer.scatter_add_(
            1,
            renumbered_indices.view(1, -1, 1).expand(1, -1, num_features),
            send_tensor[:, non_local_messages.view(-1), :],
        )
        _mpi_vector_accumulate(
            send_tensor=buffer,
            indices=original_locs,
            rank_mapping=original_rank_mapping,
            num_local_output_rows=num_output_rows,
            win=win,
        )
    win.Fence()  # End the Epoch for MPI RMA
    win.Free()  # Free the window
    return recv_tensor


class MPIGatherFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        indices: torch.Tensor,
        src_rank_mapping: torch.Tensor,
    ):
        num_input_rows = send_tensor.shape[1]
        ctx.save_for_backward(indices, src_rank_mapping)
        ctx.num_grad_rows = num_input_rows
        return _mpi_gather_impl(send_tensor, indices, src_rank_mapping)

    @staticmethod
    def backward(ctx, grad_output):
        indices, target_rank_mapping = ctx.saved_tensors
        num_output_rows = ctx.num_grad_rows
        # From here on it is just scatter
        input_grad = _mpi_scatter_add_impl(
            grad_output, indices, num_output_rows, target_rank_mapping
        )
        indices_grad = None
        src_rank_mapping_grad = None
        return input_grad, indices_grad, src_rank_mapping_grad


class MPIScatterFunction(Function):
    @staticmethod
    def forward(
        ctx,
        send_tensor: torch.Tensor,
        indices: torch.LongTensor,
        num_output_rows: int,
        target_rank_mapping: torch.LongTensor,
    ):

        ctx.save_for_backward(indices)
        ctx.send_tensor_shape = send_tensor.shape

        recv_tensor = _mpi_scatter_add_impl(
            send_tensor, indices, num_output_rows, target_rank_mapping
        )

        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        indices, src_rank_mapping = ctx.saved_tensors
        # From here on it is just gather
        input_grad = _mpi_gather_impl(grad_output, indices, src_rank_mapping)
        indices_grad = None
        num_output_grad = None
        rank_mapping_grad = None

        return input_grad, indices_grad, num_output_grad, rank_mapping_grad


class MPIBackendEngine(BackendEngine):
    _is_initialized = False
    _global_rank = -1
    _world_size = -1
    _comm = None
    _partition_size = None
    _local_rank = None
    _partition_num = None

    def __init__(self, *args, **kwargs):
        # self._iniitalized = dist.is_initialized()
        if MPIBackendEngine._is_initialized:
            return
        self.init_process_group(*args, **kwargs)

    def init_process_group(self, ranks_per_graph=None, *args, **kwargs):
        if not MPIBackendEngine._is_initialized:
            # We want both NCCL and MPI to be initialized

            # Dist initialization is done by the user and handles
            # the collective operations need for SGD
            if not MPI.Is_initialized():
                # MPI is not initialized
                ret_code = MPI.Init_thread()
                if ret_code != MPI.THREAD_MULTIPLE:
                    raise RuntimeError(
                        "MPI_THREAD_MULTIPLE not supported. "
                        + "Please recompile MPI with THREAD_MULTIPLE support "
                        + "or intialize MPI with MPI.Init() outside of DGraph."
                    )
            MPIBackendEngine._is_initialized = True

            self._comm = MPI.COMM_WORLD
            MPIBackendEngine._global_rank = self._comm.Get_rank()
            MPIBackendEngine._world_size = self._comm.Get_size()

            if ranks_per_graph is not None:
                assert (
                    MPIBackendEngine._world_size % ranks_per_graph == 0
                ), f"World size {MPIBackendEngine._world_size} not divisible by ranks per graph {ranks_per_graph}"
                MPIBackendEngine._partition_size = (
                    MPIBackendEngine._world_size // ranks_per_graph
                )
                MPIBackendEngine._local_rank = (
                    MPIBackendEngine._global_rank % ranks_per_graph
                )
                MPIBackendEngine._partition_num = (
                    MPIBackendEngine._global_rank // ranks_per_graph
                )
            else:
                MPIBackendEngine._partition_size = MPIBackendEngine._world_size
                MPIBackendEngine._local_rank = MPIBackendEngine._global_rank
                MPIBackendEngine._partition_num = 0
            # TODO: Might be worth it to require a specific init_method
            # for MPI to ensure that the processes are started correctly
            # In my experience, file-based rendezvous is the most reliable - S.Z

            if not dist.is_initialized():
                # Only initialize NCCL if it hasn't been initialized by the user
                dist.init_process_group(
                    *args,
                    backend="nccl",
                    rank=self._comm.Get_rank(),
                    world_size=self._comm.Get_size(),
                )
            else:
                warnings.warn(
                    "NCCL already initialized. Skipping initialization of NCCL."
                )

                if "SKIP_NCCL_ASSERT" not in kwargs:
                    if kwargs["SKIP_NCCL_ASSERT"] is not True:
                        assert dist.get_rank() == self._comm.Get_rank(), (
                            f"Rank mismatch between NCCL and MPI. NCCL Rank: {dist.get_rank()},"
                            + f"MPI Rank: {self._comm.Get_rank()}. This may cause undefined behavior."
                            + "Either use DGraph to initialize NCCL using the Keyword arguments or"
                            + "pass SKIP_NCCL_ASSERT to the DGraph communicator."
                        )

                        assert dist.get_world_size() == self._comm.Get_size(), (
                            f"World size mismatch between NCCL and MPI. NCCL World Size: {dist.get_world_size()}, "
                            f"MPI World Size: {self._comm.Get_size()}"
                        )

    def finalize(self):
        if self._initialized:
            MPI.Finalize()

            MPIBackendEngine._initialized = False

    @staticmethod
    def Malloc(size: int, device: torch.device):
        """Allocates memory on the GPU that is accessible by MPI one-sided communication"""

        torch_tensor = torch.zeros(size, dtype=torch.float32, device=device)

        win = MPI.Win.Create(
            torch_tensor, disp_unit=MPI.FLOAT.Get_size(), comm=MPI.COMM_WORLD
        )
        return win, torch_tensor

    @staticmethod
    def Attach(tensor: torch.Tensor):
        torch.cuda.synchronize()
        win = MPI.Win.Create(
            tensor.data, disp_unit=MPI.FLOAT.Get_size(), comm=MPI.COMM_WORLD
        )
        return win, tensor

    @staticmethod
    def Detach(win):
        win.Detach()

    @staticmethod
    def get_rank() -> int:
        return MPIBackendEngine._global_rank

    @staticmethod
    def get_world_size() -> int:
        return MPIBackendEngine._world_size

    @staticmethod
    def get_local_rank() -> int:
        assert (
            MPIBackendEngine._local_rank is not None
        ), "MPIBackendEngine not initialized"
        return MPIBackendEngine._local_rank

    @staticmethod
    def get_comm() -> MPI.Comm:
        assert MPIBackendEngine._comm is not None, "MPIBackendEngine not initialized"
        return MPIBackendEngine._comm

    @staticmethod
    def get_partition_size() -> int:
        assert (
            MPIBackendEngine._partition_size is not None
        ), "MPIBackendEngine not initialized"
        return MPIBackendEngine._partition_size

    @staticmethod
    def to_global_rank(local_rank: int) -> int:
        """Converts a local rank in the current partition to it's global rank"""
        assert (
            MPIBackendEngine._partition_size is not None
        ), "MPIBackendEngine not initialized"
        partition_offset = (
            MPIBackendEngine._partition_num * MPIBackendEngine._partition_size
        )
        return partition_offset + local_rank

    def get_local_rank_slice(self, tensor: torch.Tensor, dim=-1) -> torch.Tensor:
        """Returns a slice of the tensor that corresponds to the local rank."""
        rank = self.get_rank()
        world_size = self.get_world_size()
        tensor_shape = tensor.shape
        tensor_size = tensor_shape[dim]
        local_size = tensor_size // world_size
        start_index = rank * local_size
        end_index = start_index + local_size

        length = end_index - start_index
        return torch.narrow(tensor, dim, start_index, length)

    def scatter(
        self,
        send_tensor: torch.Tensor,
        indices: torch.Tensor,
        num_output_rows: int,
        rank_mapping: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Scatters and accumulates the input tensor to the indices provided.
        Returns output_tensor such that:

            output_tensor[:, indices[i], :] += send_tensor[:, i, :]

        The optional rank_mapping tensor specifies the rank location
        output_tensor[:, indices[i], :] in the partition group. If
        rank_mapping[i] != rank, communication is initiated to the remote rank.

        Args:
            send_tensor (torch.Tensor): The tensor to scatter. Shape (1, N, F)
            indices (torch.Tensor): The indices to scatter to. Shape (1, N)
            num_output_rows (int): The number of output rows.
            rank_mapping (Optional[torch.Tensor], optional): The rank mapping tensor.
                Defaults to None. Shape (1, N)

        Returns:
            torch.Tensor: The scattered tensor. Shape (1, num_output_rows, F)
        """

        assert (
            indices.dtype == torch.long or indices.dtype == torch.int
        ), f"Indices must be long or int, found {indices.type}"

        send_tensor_shape = send_tensor.shape
        indices_shape = indices.shape
        b_size = indices_shape[0]

        assert b_size == 1, (
            "Multi-batch scatter disabled for testing."
            + "Maximize DDP distribution first."
        )
        assert send_tensor_shape[0] == 1, (
            "Multi-batch scatter disabled for testing."
            + "Maximize DDP distribution first."
        )

        if indices.device != torch.device("cpu"):
            indices = indices.cpu()
            warnings.warn(
                "Scatter indices not on CPU, moving to CPU."
                + "MPI requires indices to be on CPU."
            )

        if rank_mapping is None:
            rank_mapping = indices // self.get_partition_size()
        assert indices.shape == rank_mapping.shape
        out = MPIScatterFunction.apply(
            send_tensor, indices, num_output_rows, rank_mapping
        )
        return out

    def gather(
        self,
        send_tensor: torch.Tensor,
        indices: torch.Tensor,
        rank_mapping: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Gathers the input tensor to the indices provided.
        Returns output_tensor such that:

            output_tensor[:, i, :] += send_tensor[:, indices[i], :]

        The optional rank_mapping tensor specifies the rank location
        send_tensor[:, indices[i], :] in the partition group. If
        rank_mapping[i] != rank, communication is initiated to the remote rank.

        Args:
            send_tensor (torch.Tensor): The tensor to scatter. Shape (1, N, F)
            indices (torch.Tensor): The indices to scatter to. Shape (1, E)
            rank_mapping (Optional[torch.Tensor], optional): The rank mapping tensor.
                Defaults to None. Shape (1, E)

        Returns:
            torch.Tensor: The scattered tensor. Shape (1, E, F)
        """
        assert (
            indices.dtype == torch.long or indices.dtype == torch.int
        ), f"Indices must be long or int, found {indices.type}"
        indices_shape = indices.shape
        b_size = indices_shape[0]

        if indices.device != torch.device("cpu"):
            indices = indices.cpu()
            warnings.warn(
                "Gather indices not on CPU, moving to CPU."
                + "MPI requires indices to be on CPU."
            )
        if rank_mapping is None:
            rank_mapping = indices // self.get_partition_size()

        assert b_size == 1, (
            "Multi-batch gather disabled for testing."
            + "Maximize DDP distribution first."
        )

        assert (
            indices.shape == rank_mapping.shape
        ), f"Expected {indices.shape} == {rank_mapping.shape}"

        out = MPIGatherFunction.apply(send_tensor, indices, rank_mapping)

        return out
