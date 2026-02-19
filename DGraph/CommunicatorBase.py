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
from abc import ABC, abstractmethod


class CommunicatorBase(ABC):
    _is_initialized: bool = False

    def __init__(self):
        self.backend = ""
        pass

    @abstractmethod
    def init_process_group(self, backend: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_rank(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_world_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def barrier(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def scatter(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def gather(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def destroy(self) -> None:
        raise NotImplementedError
