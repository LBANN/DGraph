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
#include <cuda.h>

/**
 *
 * This file houses all the kernels that we use for local data communication.
 * Currently all the kernels are in the Local namespace and in the same file, but
 * we can split this up in the future if needed for better organization.
 *
 */
namespace Local
{

  __device__ __forceinline__ float Max(const float &x, const float &y)
  {
    return y > x ? y : x;
  }

  __global__ void Fused_ReLU_Scatter_Kernel(
      const float *__restrict__ values,
      const long *__restrict__ indices,
      float *__restrict__ output,
      const int mini_batch_size,
      const int num_values_rows,
      const int num_cols,
      const int num_output_rows)
  {

    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

    const size_t nthreadsx = gridDim.x * blockDim.x;
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsz = gridDim.z * blockDim.z;

    for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz)
    {
      const auto values_offset = mb_i * num_cols * num_values_rows;
      const auto output_offset = mb_i * num_cols * num_output_rows;
      const auto ind_offset = mb_i * num_values_rows;

      for (size_t row = gidy; row < num_values_rows; row += nthreadsy)
      {
        const int ind = indices[ind_offset + row];

        for (size_t i = gidx; i < num_cols; i += nthreadsx)
        {
          if (ind > -1 && ind < num_output_rows)
          {
            const auto val = values[values_offset + row * num_cols + i];
            atomicAdd(&output[output_offset + ind * num_cols + i], Max(val, 0.0));
          }
        }
      }
    }
  }

  __global__ void Fused_Sum_Norm_Scatter_Kernel(
      const float *__restrict__ values_1,
      const float *__restrict__ values_2,
      const float *__restrict__ means,
      const float *__restrict__ inv_var,
      const long *__restrict__ indices,
      float *__restrict__ output,
      const int mini_batch_size,
      const int num_values_rows,
      const int num_cols,
      const int num_output_rows)
  {

    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

    const size_t nthreadsx = gridDim.x * blockDim.x;
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsz = gridDim.z * blockDim.z;

    for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz)
    {
      const auto values_offset = mb_i * num_cols * num_values_rows;
      const auto output_offset = mb_i * num_cols * num_output_rows;
      const auto ind_offset = mb_i * num_values_rows;

      for (size_t row = gidy; row < num_values_rows; row += nthreadsy)
      {
        const int ind = indices[ind_offset + row];

        for (size_t i = gidx; i < num_cols; i += nthreadsx)
        {
          if (ind > -1 && ind < num_output_rows)
          {
            const auto val = values_1[values_offset + row * num_cols + i] + values_2[values_offset + row * num_cols + i];
            atomicAdd(&output[output_offset + ind * num_cols + i], Max(val, 0.0));
          }
        }
      }
    }
  }

  __global__ void Sparse_Scatter_Kernel(
      const float *__restrict__ values,
      const long *__restrict__ indices,
      float *__restrict__ output,
      const int mini_batch_size,
      const int num_values_rows,
      const int num_cols,
      const int num_output_rows)
  {

    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

    const size_t nthreadsx = gridDim.x * blockDim.x;
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsz = gridDim.z * blockDim.z;

    for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz)
    {
      const auto values_offset = mb_i * num_cols * num_values_rows;
      const auto output_offset = mb_i * num_cols * num_output_rows;
      const auto ind_offset = mb_i * num_values_rows;

      for (size_t row = gidy; row < num_values_rows; row += nthreadsy)
      {
        const int ind = indices[ind_offset + row];

        for (size_t i = gidx; i < num_cols; i += nthreadsx)
        {
          if (ind > -1 && ind < num_output_rows)
          {
            const auto val = values[values_offset + row * num_cols + i];
            if (val > 0.0)
            {
              atomicAdd(&output[output_offset + ind * num_cols + i], Max(val, 0.0));
            }
          }
        }
      }
    }
  }

  __global__ void Rank_Local_Gather_Kernel(
      const float *__restrict__ values,
      const long *__restrict__ indices,
      const long *__restrict__ rank_placement,
      float *__restrict__ output,
      const int mini_batch_size,
      const int num_values_rows,
      const int num_cols,
      const int num_output_rows,
      const int local_rank)
  {

    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

    const size_t nthreadsx = gridDim.x * blockDim.x;
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsz = gridDim.z * blockDim.z;

    for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz)
    {
      const auto values_offset = mb_i * num_cols * num_values_rows;
      const auto output_offset = mb_i * num_cols * num_output_rows;
      const auto ind_offset = mb_i * num_output_rows;
      const auto rank_placement_offset = mb_i * num_output_rows;

      for (size_t row = gidy; row < num_output_rows; row += nthreadsy)
      {
        const int ind = indices[ind_offset + row];
        const int row_rank = rank_placement[rank_placement_offset + row];
        // Only gather the values if the rank is the same as the local rank
        if (row_rank == local_rank)
        {
          // Probably not needed, but just in case
          if (ind > -1 && ind < num_values_rows)
          {
            for (size_t i = gidx; i < num_cols; i += nthreadsx)
            {
              const auto val = values[values_offset + ind * num_cols + i];
              output[output_offset + row * num_cols + i] = val;
            }
          }
        }
      }
    }
  }

  __global__ void Rank_Local_Scatter_Kernel(
      const float *__restrict__ values,
      const long *__restrict__ indices,
      const long *__restrict__ rank_placement,
      float *__restrict__ output,
      const int mini_batch_size,
      const int num_values_rows,
      const int num_cols,
      const int num_output_rows,
      const int local_rank)
  {
    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;

    const size_t nthreadsx = gridDim.x * blockDim.x;
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsz = gridDim.z * blockDim.z;

    for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz)
    {
      const auto values_offset = mb_i * num_cols * num_values_rows;
      const auto output_offset = mb_i * num_cols * num_output_rows;
      const auto ind_offset = mb_i * num_values_rows;
      const auto rank_placement_offset = mb_i * num_output_rows;

      for (size_t row = gidy; row < num_values_rows; row += nthreadsy)
      {
        const int ind = indices[ind_offset + row];
        const int row_rank = rank_placement[rank_placement_offset + row];
        // Only gather the values if the rank is the same as the local rank
        if (row_rank == local_rank)
        {
          // Probably not needed, but just in case
          if (ind > -1 && ind < num_output_rows)
          {
            for (size_t i = gidx; i < num_cols; i += nthreadsx)
            {
              const auto val = values[values_offset + row * num_cols + i];
              atomicAdd(&output[output_offset + ind * num_cols + i], Max(val, 0.0));
            }
          }
        }
      }
    }
  }
} // namespace Local