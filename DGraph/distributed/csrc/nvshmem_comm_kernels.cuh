#pragma once
#include <cuda.h>
#include "nvshmem.h"
#include "nvshmemx.h"

/**
 *
 * This file houses all the kernels that we use for NVSHMEM communication.
 * Currently all the kernels are in the NVSHMEM namespace and in the same file, but
 * we can split this up in the future if needed for better organization.
 *
 */

namespace NVSHMEM
{
  __device__ __forceinline__ float atomic_add(float *const __restrict__ address,
                                              const float val,
                                              const int pe)
  {

    int *address_as_int = (int *)address;
    int assumed;
    int old = nvshmem_int_g(address_as_int, pe);
    do
    {
      assumed = old;
      old = nvshmem_int_atomic_compare_swap(address_as_int, assumed,
                                            __float_as_int(val +
                                                           __int_as_float(assumed)),
                                            pe);
    } while (assumed != old);
    return __int_as_float(old);
  }

  __device__ __forceinline__ double atomic_add(double *const __restrict__ address,
                                               const double val,
                                               const int pe)
  {

    long long int *address_as_ll = (long long int *)address;
    long long int assumed;
    long long int old = nvshmem_longlong_g(address_as_ll, pe);
    do
    {
      assumed = old;
      old = nvshmem_longlong_atomic_compare_swap(address_as_ll, assumed,
                                                 __double_as_longlong(val +
                                                                      __longlong_as_double(assumed)),
                                                 pe);
    } while (assumed != old);
    return __longlong_as_double(old);
  }

  template <typename DataType>
  __global__ void Scatter_NVSHMEM_Kernel(
      const DataType *__restrict__ values,
      const long *__restrict__ indices,
      DataType *__restrict__ outputs,
      const int mini_batch_size,
      const int num_local_values_rows,
      const int num_cols,
      const int num_local_output_rows)
  {
    // Indices
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidz = threadIdx.z + blockIdx.z * blockDim.z;
    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t nthreadsx = gridDim.x * blockDim.x;
    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsz = gridDim.z * blockDim.z;

    for (size_t mb_i = gidz; mb_i < mini_batch_size; mb_i += nthreadsz)
    {
      const auto values_offset = mb_i * num_local_values_rows * num_cols;
      const auto output_offset = mb_i * num_local_output_rows * num_cols;
      const auto indices_offset = mb_i * num_local_values_rows;

      for (size_t row = gidy; row < num_local_values_rows; row += nthreadsy)
      {
        // Figure out which rank to send the vector
        const auto ind = __float2int_rd(indices[indices_offset + row]);
        if (ind > -1)
        {
          const int pe = (ind) / num_local_output_rows;
          const int local_ind = ind % num_local_output_rows;
          for (size_t i = gidx; i < num_cols; i += nthreadsx)
          {
            const auto val = values[values_offset + row * num_cols + i];
            atomic_add(outputs + output_offset + local_ind * num_cols + i, val, pe);
          }
        }
      }
    }
  }

  template <typename DataType>
  __global__ void Gather_NVSHMEM_Kernel(
      const DataType *__restrict__ values,
      const DataType *__restrict__ indices,
      DataType *__restrict__ shared_buffer,
      const int mini_batch_size,
      const int num_local_values_rows,
      const int num_local_cols,
      const int num_local_output_rows)
  {

    // Indice
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsx = gridDim.x * blockDim.x;

    const int n_pes = nvshmem_n_pes();

    for (size_t mb_i = gidy; mb_i < mini_batch_size; mb_i += nthreadsy)
    {
      // Figure out which rank to send the vector
      const auto mb_offset = mb_i * num_local_cols * num_local_output_rows;
      const auto values_offest = mb_i * num_local_cols * num_local_values_rows;
      const auto ind_offset = mb_i * num_local_output_rows;

      for (size_t row = gidx; row < num_local_output_rows; row += nthreadsx)
      {
        const auto ind = __float2int_rd(indices[ind_offset + row]);
        if (ind > -1)
        {
          const int pe = (ind) / num_local_values_rows;
          const int local_ind = ind % num_local_values_rows;
          nvshmem_getmem_nbi(shared_buffer + mb_offset + row * num_local_cols,
                             values + values_offest + local_ind * num_local_cols,
                             num_local_cols * sizeof(DataType),
                             pe);
        }
      }
    }
  }

  template <typename DataType>
  __global__ void Comm_Aware_Gather_NVSHMEM_Kernel(
      const DataType *__restrict__ values,
      const DataType *__restrict__ indices,
      DataType *__restrict__ shared_buffer,
      const int mini_batch_size,
      const int num_local_values_rows,
      const int num_local_cols,
      const int num_local_output_rows,
      const int rank)
  {

    // Indice
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsx = gridDim.x * blockDim.x;

    const int n_pes = nvshmem_n_pes();

    for (size_t mb_i = gidy; mb_i < mini_batch_size; mb_i += nthreadsy)
    {
      // Figure out which rank to send the vector
      const auto mb_offset = mb_i * num_local_cols * num_local_output_rows;
      const auto values_offest = mb_i * num_local_cols * num_local_values_rows;
      const auto ind_offset = mb_i * num_local_output_rows;

      for (size_t row = gidx; row < num_local_output_rows; row += nthreadsx)
      {
        const auto ind = __float2int_rd(indices[ind_offset + row]);
        if (ind > -1)
        {
          const int pe = (ind) / num_local_values_rows;
          if (pe != rank)
          {
            const int local_ind = ind % num_local_values_rows;
            nvshmem_getmem_nbi(shared_buffer + mb_offset + row * num_local_cols,
                               values + values_offest + local_ind * num_local_cols,
                               num_local_cols * sizeof(DataType),
                               pe);
          }
        }
      }
    }
  }

  template <typename DataType>
  __global__ void Gather_NVSHMEM_Kernel_Warp(
      const DataType *__restrict__ values,
      const long *__restrict__ indices,
      DataType *__restrict__ shared_buffer,
      const int mini_batch_size,
      const int num_local_values_rows,
      const int num_local_cols,
      const int num_local_output_rows)
  {

    constexpr int warp_size = 32;
    // Indice
    const size_t gidy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t gidx = threadIdx.x + blockIdx.x * blockDim.x;

    const size_t nthreadsy = gridDim.y * blockDim.y;
    const size_t nthreadsx = gridDim.x * blockDim.x;

    const int n_pes = nvshmem_n_pes();

    for (size_t mb_i = gidy; mb_i < mini_batch_size; mb_i += nthreadsy)
    {
      // Figure out which rank to send the vector
      const auto mb_offset = mb_i * num_local_cols * num_local_output_rows;
      const auto values_offest = mb_i * num_local_cols * num_local_values_rows;
      const auto ind_offset = mb_i * num_local_output_rows;

      for (size_t row = gidx; row < num_local_output_rows * warp_size; row += nthreadsx)
      {
        const auto ind = __float2int_rd(indices[ind_offset + row / warp_size]);
        if (ind > -1)
        {
          const int pe = (ind) / num_local_values_rows;
          const int local_ind = ind % num_local_values_rows;
          nvshmemx_getmem_nbi_warp(shared_buffer + mb_offset + (row / warp_size) * num_local_cols,
                                   values + values_offest + local_ind * num_local_cols,
                                   num_local_cols * sizeof(DataType),
                                   pe);
        }
      }
    }
  }

} // namespace NVSHMEM