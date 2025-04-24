cmake_minimum_required(VERSION 3.25.0)
project(
  DGraph
  VERSION 0.0.1
  DESCRIPTION "A deep learning library for training graph neural networks at scale"
  HOMEPAGE_URL "https://github.com/LBANN/DGraph"
  LANGUAGES CXX CUDA
)

option(DGRAPH_ENABLE_NVSHMEM
  "Use NVSHMEM in the build."
  ON
)

# Dependencies
list(APPEND
  CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
)

find_package(CUDAToolkit REQUIRED)
find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
find_package(Torch 2.6 REQUIRED CONFIG)

if (DGRAPH_ENABLE_NVSHMEM)
  find_package(NVSHMEM 2.5 REQUIRED MODULE)
endif ()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(torch_local
  MODULE
  WITH_SOABI
  DGraph/distributed/csrc/torch_local_bindings.cpp
  DGraph/distributed/csrc/torch_local_kernels.cu
)

target_link_libraries(torch_local
  PUBLIC
  MPI::MPI_CXX
  torch
  PRIVATE
  pybind11::headers
)

target_sources(torch_local
  PUBLIC
  FILE_SET HEADERS
  BASE_DIRS DGraph/distributed/csrc DGraph/distributed/include
  FILES
  DGraph/distributed/include/macros.hpp
  DGraph/distributed/include/torch_local.hpp
  DGraph/distributed/csrc/local_data_kernels.cuh
)

set_target_properties(torch_local
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF

  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  CUDA_EXTENSIONS OFF

  INSTALL_RPATH_USE_LINK_PATH ON
)

install(TARGETS torch_local
  LIBRARY DESTINATION .
)

if (DGRAPH_ENABLE_NVSHMEM)
  python_add_library(torch_nvshmem_p2p
    MODULE
    WITH_SOABI
    DGraph/distributed/csrc/torch_nvshmem_p2p.cu
    DGraph/distributed/csrc/torch_nvshmem_p2p_bindings.cpp
  )

  target_sources(torch_nvshmem_p2p
    PUBLIC
    FILE_SET HEADERS
    BASE_DIRS DGraph/distributed/csrc DGraph/distributed/include
    FILES
    DGraph/distributed/include/torch_nvshmem_p2p.hpp
    DGraph/distributed/csrc/local_data_kernels.cuh
    DGraph/distributed/csrc/nvshmem_comm_kernels.cuh
  )

  target_link_libraries(torch_nvshmem_p2p
    PUBLIC
    NVSHMEM::NVSHMEM
    MPI::MPI_CXX
    torch
    PRIVATE
    pybind11::headers
  )

  set_target_properties(torch_nvshmem_p2p
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF

    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_SEPARABLE_COMPILATION ON

    INSTALL_RPATH_USE_LINK_PATH ON
  )

  install(TARGETS torch_nvshmem_p2p
    LIBRARY DESTINATION .
  )
endif ()
