# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
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


class DGraphSparseTensor:
    def __init__(
        self,
        row,
        col,
        value=None,
        comm=None,
        rank_mapping=None,
        **kwargs,
    ):
        super(DGraphSparseTensor, self).__init__()
        assert comm is not None, "Comm object cannot be None"
        assert rank_mapping is not None, "rank_mapping cannot be None"
        self.comm = comm
        self.rank_mapping = rank_mapping
        self.world_size = comm.get_world_size()
        self.rank = comm.get_rank()
        self.row = row
        self.col = col

    def to(self, device):
        self.row = self.row.to(device)
        self.col = self.col.to(device)
        if self.rank_mapping is not None:
            self.rank_mapping = self.rank_mapping.to(device)

        if self.value is not None:
            self.value = self.value.to(device)
        return self
