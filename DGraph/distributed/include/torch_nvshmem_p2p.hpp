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

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

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
                       torch::Tensor src_ranks,
                       const int mini_batches,
                       const int num_input_rows,
                       const int cols,
                       const int num_output_rows);
  static torch::Tensor AllocateSymmetricMemory(const int size,
                                               const int device_ordinal);

  static void register_memory(torch::Tensor tensor);
  static void deregister_memory(torch::Tensor tensor);
  static int get_rank();
  static int get_world_size();
  static void set_device(int device);

  static int m_rank;
  static int m_world_size;
  static bool m_initialized;
};
