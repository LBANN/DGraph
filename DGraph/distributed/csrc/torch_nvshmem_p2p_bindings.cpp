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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  py::class_<NVSHMEMP2P>(m, "NVSHMEMP2P")
      .def(py::init<>())
      .def("init", &NVSHMEMP2P::init)
      .def("finalize", &NVSHMEMP2P::finalize)
      .def("dist_put", &NVSHMEMP2P::dist_put)
      .def("allocate_symmetric_memory", &NVSHMEMP2P::AllocateSymmetricMemory)
      .def("clone_tensor", &NVSHMEMP2P::clone_tensor)
      .def("register_memory", &NVSHMEMP2P::register_memory)
      .def("deregister_memory", &NVSHMEMP2P::deregister_memory)
      .def("dist_get", &NVSHMEMP2P::dist_get)
      .def("get_rank", &NVSHMEMP2P::get_rank)
      .def("get_world_size", &NVSHMEMP2P::get_world_size)
      .def("barrier", &NVSHMEMP2P::barrier)
      .def("barrier_stream", &NVSHMEMP2P::barrier_stream);
}
