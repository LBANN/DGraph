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
import os
import torch
import glob
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

disable_dgraph_nvshmem = False
if "DISABLE_DGRAPH_NVSHMEM" in os.environ:
    disable_dgraph_nvshmem = os.environ["DISABLE_DGRAPH_NVSHMEM"] == "1"

nvshmem_p2p_sources = [
    "DGraph/distributed/csrc/torch_nvshmem_p2p.cu",
    "DGraph/distributed/csrc/torch_nvshmem_p2p_bindings.cpp",
]

kwargs = {}
if not disable_dgraph_nvshmem:
    # Check if CUDA is available
    if not torch.cuda.is_available() and "CUDA_HOME" not in os.environ:
        raise EnvironmentError("CUDA is required to build DGraph")
    # There is no good way to check if Torch was built with CUDA support
    # so we just check if CUDA is available. This could cause an issue
    # if the user is trying to build on a headnode that doesn't have a GPU
    # because CUDA will not be available even if Torch was built with CUDA

    # Check if NVSHMEM_HOME is set
    # TODO: Try to add the ability to input this path as an argument
    if "NVSHMEM_HOME" not in os.environ:
        raise EnvironmentError("NVSHMEM_HOME must be set to build DGraph")

    # TODO: Try to add the ability to input this path as an argument
    if "MPI_HOME" not in os.environ:
        raise EnvironmentError("MPI_HOME must be set to build DGraph")

    nvshmem_home = os.environ["NVSHMEM_HOME"]
    # print(f"Found NVSHMEM_HOME: {nvshmem_home}")

    nvshmem_include = os.path.join(nvshmem_home, "include")
    nvshmem_lib = os.path.join(nvshmem_home, "lib")

    mpi_home = os.environ["MPI_HOME"]
    # print(f"Found MPI_HOME: {mpi_home}")
    mpi_include = os.path.join(mpi_home, "include")
    mpi_lib = os.path.join(mpi_home, "lib")

    cuda_home = os.environ["CUDA_HOME"]
    # print(f"Found CUDA_HOME: {cuda_home}")
    cuda_lib = os.path.join(cuda_home, "lib64")

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dgraph_include_dir = os.path.join(cur_dir, "DGraph", "distributed", "include")
    include_dirs = [mpi_include, nvshmem_include, dgraph_include_dir]
    library_dirs = [mpi_lib, nvshmem_lib, cuda_lib]
    library_flags = [
        "-Wl,--no-as-needed",
        "-lmpi",
        "-lnvshmem",
    ]

    extra_compile_args = {
        "nvcc": [
            "-O3",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-rdc=true",
        ]
    }

    nvshmem_module = CUDAExtension(
        name="torch_nvshmem_p2p",
        sources=nvshmem_p2p_sources,
        include_dirs=include_dirs,
        dlink=True,
        dlink_libraries=["nvshmem"],
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=library_flags,
    )

    kwargs["ext_modules"] = [nvshmem_module]


EXTRAS_REQUIRE = {
    "ogb": ["ogb", "fire"],
    "graphcast": ["fire"],
}
# Conditional extra dependencies
if sys.version_info < (3, 11):
    EXTRAS_REQUIRE["graphcast"].append("more_itertools")

setup(
    name="DGraph",
    py_modules=["DGraph"],
    # ext_modules=[nvshmem_module],
    install_requires=["torch", "numpy", "ninja", "mpi4py>=3.1.4"],
    extras_require=EXTRAS_REQUIRE,
    cmdclass={"build_ext": BuildExtension},
    **