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
#include "torch_nvshmem_p2p.hpp"
#include "mpi.h"
#include "macros.hpp"
#include "nvshmem.h"
#include "nvshmem_comm_kernels.cuh"
#include <functional>

// Don't know if making these statc is the right thing to do
// but going with it for now, will need to revisit - S.Z

bool NVSHMEMP2P::m_initialized = false;
int NVSHMEMP2P::m_rank = 0;
int NVSHMEMP2P::m_world_size = 0;

void NVSHMEMP2P::init()
{
  // Initialize NVSHMEM and MPI
  // No-op if already initialized
  if (!m_initialized)
  {
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    MPI_Init(NULL, NULL);
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int mpi_rank, mpi_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    m_rank = mpi_rank;
    m_world_size = mpi_size;
    m_initialized = true;
  }
}

void NVSHMEMP2P::finalize()
{
  if (!m_initialized)
  {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }
  nvshmem_finalize();
  MPICHECK(MPI_Finalize());
  m_initialized = false;
}

int NVSHMEMP2P::get_rank()
{
  if (!m_initialized)
  {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }
  return m_rank;
}

int NVSHMEMP2P::get_world_size()
{
  if (!m_initialized)
  {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }
  return m_world_size;
}

void NVSHMEMP2P::set_device(int device)
{
  if (!m_initialized)
  {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }
  CUDACHECK(cudaSetDevice(device));
}

void NVSHMEMP2P::dist_put(torch::Tensor input,
                          torch::Tensor output,
                          torch::Tensor indices,
                          torch::Tensor dst_ranks,
                          const int mini_batches,
                          const int num_input_rows,
                          const int num_cols,
                          const int num_output_rows)
{
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(indices);

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(indices.is_contiguous());
  TORCH_CHECK(dst_ranks.is_contiguous());
  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(output.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(dst_ranks.device().type() == at::DeviceType::CUDA);

  if (!m_initialized)
  {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }

  // Get the pointers to the data
  const float *input_ptr = input.data_ptr<float>();
  const long *indices_ptr = indices.data_ptr<long>();
  const long *dst_ranks_ptr = dst_ranks.data_ptr<long>();
  float *output_ptr = output.data_ptr<float>();

  dim3 block_dims, grid_dims;
  block_dims.x = 32;
  block_dims.y = 16;

  grid_dims.y = (num_output_rows + block_dims.y - 1) / block_dims.y;
  grid_dims.x = (num_cols + block_dims.x - 1) / block_dims.x;

  // Launch the kernel
  NVSHMEM::Scatter_NVSHMEM_Kernel<<<grid_dims, block_dims>>>(input_ptr,
                                                             indices_ptr,
                                                             dst_ranks_ptr,
                                                             output_ptr,
                                                             num_input_rows,
                                                             num_cols,
                                                             num_output_rows);
}

void NVSHMEMP2P::dist_get(torch::Tensor input,
                          torch::Tensor output,
                          torch::Tensor indices,
                          torch::Tensor src_ranks,
                          const int mini_batches,
                          const int num_input_rows,
                          const int num_cols,
                          const int num_output_rows)
{
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(indices);
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(indices.is_contiguous());
  TORCH_CHECK(src_ranks.is_contiguous());
  TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(output.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(src_ranks.device().type() == at::DeviceType::CUDA);
  if (!m_initialized)
  {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }

  // Get the pointers to the data
  const float *input_ptr = input.data_ptr<float>();
  const long *indices_ptr = indices.data_ptr<long>();
  float *output_ptr = output.data_ptr<float>();
  dim3 block_dims, grid_dims;

  block_dims.x = 32;
  block_dims.y = 16;

  grid_dims.y = (num_output_rows + block_dims.y - 1) / block_dims.y;
  grid_dims.x = (num_cols + block_dims.x - 1) / block_dims.x;

  // // Launch the kernel
  NVSHMEM::Gather_NVSHMEM_Kernel_Wrap_Rank<<<grid_dims, block_dims>>>(input_ptr,
                                                                      indices_ptr,
                                                                      output_ptr,
                                                                      mini_batches,
                                                                      num_input_rows,
                                                                      num_cols,
                                                                      num_output_rows);
}

torch::Tensor
NVSHMEMP2P::AllocateSymmetricMemory(const int size, const int device_ordinal)
{
  if (size <= 0)
  {
    throw std::runtime_error("Invalid size");
  }

  if (!m_initialized)
  {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }

  void *ptr = nvshmem_malloc((size_t)size);

  std::function<void(void *)> deleter = [](void *ptr)
  {
    nvshmem_free(ptr);
  };
  // See torch::from_blob for more details
  // https://pytorch.org/cppdocs/api/function_namespacetorch_1ad7fb2a7759ef8c9443b489ddde494787.html

  // The Torch device get / set functions are not great, the following
  // does not work.
  // device_int device = torch::cuda::current_device();

  auto device = torch::Device(torch::kCUDA, device_ordinal);

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(device);
  return torch::from_blob(ptr, {size}, {1}, deleter);
}

void NVSHMEMP2P::register_memory(torch::Tensor tensor)
{
  if (!tensor.is_contiguous())
  {
    throw std::runtime_error("Tensor is not contiguous");
  }

  if (tensor.device().type() != at::DeviceType::CUDA)
  {
    throw std::runtime_error("Tensor is not on CUDA device");
  }

  void *ptr = tensor.data_ptr();
  size_t size = tensor.numel() * tensor.element_size();
  // TODO: It would be nice to be able wrap the torch::Tensor so we
  // can this through that and store the state in the tensor
  nvshmemx_buffer_register(ptr, size);
}

void NVSHMEMP2P::deregister_memory(torch::Tensor tensor)
{
  if (!tensor.is_contiguous())
  {
    throw std::runtime_error("Tensor is not contiguous");
  }

  if (tensor.device().type() != at::DeviceType::CUDA)
  {
    throw std::runtime_error("Tensor is not on CUDA device");
  }

  nvshmemx_buffer_unregister(tensor.data_ptr());
}