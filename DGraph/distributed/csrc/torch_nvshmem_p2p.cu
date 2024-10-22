#include <torch/extension.h>
#include "torch_nvshmem_p2p.hpp"
#include "mpi.h"
#include "macros.hpp"
#include "nvshmem.h"

// Don't know if making these statc is the right thing to do
// but going with it for now, will need to revisit - S.Z

bool NVSHMEMP2P::m_initialized = false;
int NVSHMEMP2P::m_rank = 0;
int NVSHMEMP2P::m_world_size = 0;

void NVSHMEMP2P::init(int rank, int world_size) {
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr;
  MPI_Init(NULL, NULL);
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int mpi_rank, mpi_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  assert(mpi_rank == rank);
  assert(mpi_size == world_size);

  m_rank = rank;
  m_world_size = world_size;
  m_initialized = true;
}

void NVSHMEMP2P::finalize() {
  if (!m_initialized) {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }
  nvshmem_finalize();
  MPICHECK(MPI_Finalize());
  m_initialized = false;
}

void NVSHMEMP2P::dist_put(torch::Tensor src, torch::Tensor dst, torch::Tensor indices) {
  CHECK_INPUT(src);
  CHECK_INPUT(dst);
  CHECK_INPUT(indices);
  if (!m_initialized) {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }
}

void NVSHMEMP2P::dist_get(torch::Tensor src, torch::Tensor dst, torch::Tensor indices) {
  CHECK_INPUT(src);
  CHECK_INPUT(dst);
  CHECK_INPUT(indices);
  if (!m_initialized) {
    throw std::runtime_error("NVSHMEMP2P is not initialized");
  }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   py::class_<NVSHMEMP2P>(m, "NVSHMEMP2P")
//     .def(py::init<>())
//     .def("init", &NVSHMEMP2P::init)
//     .def("finalize", &NVSHMEMP2P::finalize)
//     .def("dist_put", &NVSHMEMP2P::dist_put)
//     .def("dist_get", &NVSHMEMP2P::dist_get);
// }