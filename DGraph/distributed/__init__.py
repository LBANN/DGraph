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

"""DGraph Distributed Module

Modules exported by this package:
- `Engine`: The DGraph communication engine used by the Communicator.
- `BackendEngine`: The abstract DGraph communication engine used by the Communicator.
- `HaloExchange`: Halo exchange class for communicating remote vertices
- `CommunicationPattern`: Dataclass for holding communication pattern information
"""
from DGraph.distributed.haloExchange import HaloExchange
from DGraph.distributed.commInfo import (
    CommunicationPattern,
    build_communication_pattern,
)

__all__ = [
    "HaloExchange",
    "CommunicationPattern",
    "build_communication_pattern",
]
