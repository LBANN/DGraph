cmake_minimum_required(VERSION 3.25.0)

# This is a bit messy, where it comes before project(), but it needs
# to come before CUDA is enabled (I could also move that out of
# project() into an explicit enable_language(), but this doesn't
# require the CXX compiler, so it _can_ go here).
if (NOT CMAKE_CUDA_ARCHITECTURES AND NOT TORCH_CUDA_ARCH_LIST)
  execute_process(
    COMMAND nvidia-smi --query-gpu "compute_cap" --format=csv,noheader
    RESULT_VARIABLE _nvidia_smi_retcode
    OUTPUT_VARIABLE _nvidia_smi_output
    ERROR_QUIET
  )

  if (_nvidia_smi_retcode EQUAL 0 AND _nvidia_smi_output)
    string(REPLACE "\n" ";" _nvidia_compute_caps "${_nvidia_smi_output}")
    list(REMOVE_ITEM _nvidia_compute_caps "")
    list(SORT _nvidia_compute_caps COMPARE NATURAL)
    list(REMOVE_DUPLICATES _nvidia_compute_caps)
    string(REPLACE "." "" _nvidia_archs "${_nvidia_compute_caps}")

    # TRB: I don't _think_ these need to go in the cache... But I'm
    # not opposed to it.
    set(CMAKE_CUDA_ARCHITECTURES ${_nvidia_archs})
    set(TORCH_CUDA_ARCH_LIST ${_nvidia_compute_caps})
  else ()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
    set(TORCH_CUDA_ARCH_LIST 7.0 8.0 9.0)
  endif ()
elseif (NOT CMAKE_CUDA_ARCHITECTURES)
  string(REPLACE "." "" _nvidia_archs "${TORCH_CUDA_ARCH_LIST}")
  set(CMAKE_CUDA_ARCHITECTURES ${_nvidia_archs})
elseif (NOT TORCH_CUDA_ARCH_LIST)
  # AFAIK, these are all of the form XY, so just put a '.' in there...
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES
    REPLACE "([0-9])([0-9])" "\\1.\\2"
    OUTPUT_VARIABLE TORCH_CUDA_ARCH_LIST
  )
endif ()

message(STATUS "Detected CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Detected TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")

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

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
find_package(Torch 2.6 REQUIRED CONFIG)

# Also, torch_python!
# We also don't care about the limited API nonsense, so we can use
# libtorch. Let's find it.
if (TORCH_LIBRARY)
  get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
endif ()
find_library(TORCH_PYTHON_LIBRARY
  torch_python
  HINTS
  ${TORCH_LIB_DIR}
  ${Python_SITELIB}/torch/lib64
  ${Python_SITELIB}/torch/lib
  NO_DEFAULT_PATH)
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED)

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
  ${TORCH_PYTHON_LIBRARY}
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
    ${TORCH_PYTHON_LIBRARY}
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
