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

    def scatter(self, *args, **kwargs) -> torch.Tensor:
        """Scatter the input tensor to all processes in the group."""
        raise NotImplementedError

    def gather(self, *args, **kwargs) -> torch.Tensor:
        """Gather tensors from all processes in the group."""
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError
