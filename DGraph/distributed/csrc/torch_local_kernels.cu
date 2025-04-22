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
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "torch_local.hpp"
#include "local_data_kernels.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor local_masked_gather(torch::Tensor input,
                                  torch::Tensor indices,
                                  torch::Tensor rank_local_placement,
                                  torch::Tensor output,
                                  const int num_batches,
                                  const int num_values_rows,
                                  const int num_cols,
                                  const int num_output_rows) 
{
  CHECK_INPUT(input);
  CHECK_INPUT(indices);
  CHECK_INPUT(rank_local_placement);
  CHECK_INPUT(output);

  const float *input_ptr = input.data_ptr<float>();
  const long *indices_ptr = indices.data_ptr<long>();
  const long *rank_local_placement_ptr = rank_local_placement.data_ptr<long>();
  float *output_ptr = output.data_ptr<float>();

}