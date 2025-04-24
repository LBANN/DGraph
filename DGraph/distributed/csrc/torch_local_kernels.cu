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
#include "macros.hpp"

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
                                  const int num_output_rows,
                                  const int local_rank)
{
  CHECK_INPUT(input);
  CHECK_INPUT(indices);
  CHECK_INPUT(rank_local_placement);
  CHECK_INPUT(output);

  const float *input_ptr = input.data_ptr<float>();
  const long *indices_ptr = indices.data_ptr<long>();
  const long *rank_local_placement_ptr = rank_local_placement.data_ptr<long>();
  float *output_ptr = output.data_ptr<float>();

  dim3 block_dims, grid_dims;
  block_dims.x = 32;
  block_dims.y = 32;
  block_dims.z = 1;

  const auto num_grids_needed = (num_output_rows + block_dims.y - 1) / block_dims.y;
  const auto num_col_grids_needed = (num_cols + block_dims.x - 1) / block_dims.x;
  grid_dims.x = num_col_grids_needed < 65535 ? num_col_grids_needed : 65535;
  grid_dims.y = num_grids_needed < 65535 ? num_grids_needed : 65535;
  grid_dims.z = 1;

  // Get the default stream for the current device
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream(input.device().index());
  Local::Rank_Local_Gather_Kernel<<<grid_dims, block_dims>>>(input_ptr,
                                                             indices_ptr,
                                                             rank_local_placement_ptr,
                                                             output_ptr,
                                                             num_batches,
                                                             num_values_rows,
                                                             num_cols,
                                                             num_output_rows,
                                                             local_rank);
  CUDACHECK(cudaGetLastError());
  return output;
}

torch::Tensor local_masked_scatter(torch::Tensor input,
                                   torch::Tensor indices,
                                   torch::Tensor rank_local_placement,
                                   torch::Tensor output,
                                   const int num_batches,
                                   const int num_values_rows,
                                   const int num_cols,
                                   const int num_output_rows,
                                   const int rank)
{
  CHECK_INPUT(input);
  CHECK_INPUT(indices);
  CHECK_INPUT(rank_local_placement);
  CHECK_INPUT(output);

  const float *input_ptr = input.data_ptr<float>();
  const long *indices_ptr = indices.data_ptr<long>();
  const long *rank_local_placement_ptr = rank_local_placement.data_ptr<long>();
  float *output_ptr = output.data_ptr<float>();

  dim3 block_dims, grid_dims;
  block_dims.x = 32;
  block_dims.y = 32;
  block_dims.z = 1;

  const auto num_grids_needed = (num_output_rows + block_dims.y - 1) / block_dims.y;
  const auto num_col_grids_needed = (num_cols + block_dims.x - 1) / block_dims.x;
  grid_dims.x = num_col_grids_needed < 65535 ? num_col_grids_needed : 65535;
  grid_dims.y = num_grids_needed < 65535 ? num_grids_needed : 65535;
  grid_dims.z = 1;
  // Get the default stream for the current device
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream(input.device().index());
  Local::Rank_Local_Scatter_Kernel<<<grid_dims, block_dims>>>(input_ptr,
                                                              indices_ptr,
                                                              rank_local_placement_ptr,
                                                              output_ptr,
                                                              num_batches,
                                                              num_values_rows,
                                                              num_cols,
                                                              num_output_rows,
                                                              rank);
  CUDACHECK(cudaGetLastError());
  return output;
}