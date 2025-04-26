################################################################################
## Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################

# Output variables
#
#   NVSHMEM_FOUND
#   NVSHMEM_LIBRARY
#   NVSHMEM_INCLUDE_DIR
#
# Also creates an imported target NVSHMEM::NVSHMEM

# Find the library
find_library(NVSHMEM_LIBRARY nvshmem
  HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
  ${NVSHMEM_PREFIX} $ENV{NVSHMEM_PREFIX}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
  DOC "The location of NVSHMEM library.")
find_library(NVSHMEM_LIBRARY nvshmem)

# Find the header
find_path(NVSHMEM_INCLUDE_DIRS nvshmem.h
  HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
  ${NVSHMEM_PREFIX} $ENV{NVSHMEM_PREFIX}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The location of NVSHMEM headers.")
find_path(NVSHMEM_INCLUDE_DIRS nvshmemx.h)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NVSHMEM DEFAULT_MSG NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIRS)

# Build the imported target
if (NOT TARGET NVSHMEM::NVSHMEM)
  add_library(NVSHMEM::NVSHMEM INTERFACE IMPORTED)
endif ()

target_link_libraries(NVSHMEM::NVSHMEM INTERFACE "${NVSHMEM_LIBRARY}")
target_include_directories(NVSHMEM::NVSHMEM INTERFACE "${NVSHMEM_INCLUDE_DIRS}")
target_compile_definitions(NVSHMEM::NVSHMEM INTERFACE "-DNVSHMEM_TARGET")
