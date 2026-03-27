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
from typing import Optional, Union, Tuple


class BackendEngine(object):
    """The abstract DGraph communication engine used by the Communicator. The engine
    is responsible for initializing the communication backend library, and performing
    the necessary communication operations.

    The engine should be implemented by the backend-specific classes. We currently
    supports the following backends:
    - NCCL
    - MPI
    - NVSHMEM
    """

    def __init__(self):
        """Initialize the communication backend library."""
        pass

    def init_process_group(self, *args, **kwargs):
        """Initialize the communication backend library."""
        raise NotImplementedError

    def get_rank(self) -> int:
        """Get the rank of the current process."""
        raise NotImplementedError

    def get_world_size(self) -> int:
        """Get the total number of processes."""
        raise NotImplementedError

    def scatter(
        self,
        src_tensor: torch.Tensor,
        indices: Union[torch.Tensor, torch.LongTensor],
        output_size: int,
        rank_mappings: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def gather(
        self,
        src_tensor: torch.Tensor,
        indices: Union[torch.Tensor, torch.LongTensor],
        rank_mappings: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def put(
        self,
        send_buffer: torch.Tensor,
        recv_buffer: torch.Tensor,
        send_offsets: torch.Tensor,
        recv_offsets: torch.Tensor,
        remote_offsets: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Exchange data between all ranks.

        Chunks send_buffer by send_offsets, delivers each chunk to the
        corresponding rank's recv_buffer. Must be synchronous: when this
        method returns, recv_buffer is fully populated and safe to read.

        Two-sided backends ignore remote_offsets.
        One-sided backends use remote_offsets[i] as the write position
        into rank i's recv_buffer.
        """
        raise NotImplementedError

    def allocate_buffer(
        self,
        size: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Allocate a communication buffer.

        Default: torch.empty. One-sided backends override this to
        return symmetric / registered memory.
        """
        return torch.empty(size, dtype=dtype, device=device)

    def finalize(self) -> None:
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError
