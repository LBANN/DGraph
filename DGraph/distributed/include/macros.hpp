/**
 * Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the LBANN Research Team (B. Van Essen, et al.) listed in
 * the CONTRIBUTORS file. See the top-level LICENSE file for details.
 *
 * LLNL-CODE-697807.
 * All rights reserved.
 *
 * This file is part of LBANN: Livermore Big Artificial Neural Network
 * Toolkit. For details, see http://software.llnl.gov/LBANN or
 * https://github.com/LBANN and https://github.com/LLNL/LBANN.
 *
 * SPDX-License-Identifier: (Apache-2.0)
 */
#include "mpi.h"

#define MPICHECK(cmd)                          \
  do                                           \
  {                                            \
    int e = cmd;                               \
    if (e != MPI_SUCCESS)                      \
    {                                          \
      printf("Failed: MPI error %s:%d '%d'\n", \
             __FILE__, __LINE__, e);           \
      exit(EXIT_FAILURE);                      \
    }                                          \
  } while (0)


#define CUDACHECK(cmd)                         \
  do                                           \
  {                                            \
    cudaError_t e = cmd;                       \
    if (e != cudaSuccess)                      \
    {                                          \
      printf("Failed: Cuda error %s:%d '%s'\n",\
             __FILE__, __LINE__,               \
             cudaGetErrorString(e));           \
      exit(EXIT_FAILURE);                      \
    }                                          \
  } while (0)
  