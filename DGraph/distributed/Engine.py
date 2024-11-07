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
from typing import Optional, Union


class BackendEngine(object):
    def __init__(self):
        pass

    def init_process_group(self, *args, **kwargs):
        raise NotImplementedError

    def get_rank(self) -> int:
        raise NotImplementedError

    def get_world_size(self) -> int:
        raise NotImplementedError

    def scatter(
        self,
        src_tensor: torch.Tensor,
        indices: Union[torch.Tensor, torch.LongTensor],
        output_size: int,
        rank_mappings: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def gather(
        self,
        src_tensor: torch.Tensor,
        indices: Union[torch.Tensor, torch.LongTensor],
        rank_mappings: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def finalize(self) -> None:
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError
