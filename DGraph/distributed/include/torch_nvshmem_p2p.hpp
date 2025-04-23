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
#pragma once
#include <torch/extension.h>

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

class NVSHMEMP2P
{
public:
  NVSHMEMP2P() {};
  static void init();
  static void finalize();
  static void dist_put(torch::Tensor src,
                       torch::Tensor dst,
                       torch::Tensor indices,
                       torch::Tensor destination_ranks,
                       const int mini_batches,
                       const int num_input_rows,
                       const int cols,
                       const int num_output_rows);
  static void dist_get(torch::Tensor src,
                       torch::Tensor dst,
                       torch::Tensor indices,
                       torch::Tensor source_ranks,
                       const int mini_batches,
                       const int num_input_rows,
                       const int cols,
                       const int num_output_rows);

  static void barrier();
  static void barrier_stream(const int device_ordinal);
  static torch::Tensor AllocateSymmetricMemory(const int size,
                                               const int device_ordinal);
  static torch::Tensor clone_tensor(torch::Tensor tensor);
  static torch::Tensor padded_clone_tensor(torch::Tensor tensor,
                                           const int padded_size);
  static void register_memory(torch::Tensor tensor);
  static void deregister_memory(torch::Tensor tensor);
  static int get_rank();
  static int get_world_size();
  static void set_device(int device);

  static int m_rank;
  static int m_world_size;
  static bool m_initialized;
};
