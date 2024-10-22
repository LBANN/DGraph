#include <torch/extension.h>
#include "torch_nvshmem_p2p.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<NVSHMEMP2P>(m, "NVSHMEMP2P")
    .def(py::init<>())
    .def("init", &NVSHMEMP2P::init)
    .def("finalize", &NVSHMEMP2P::finalize)
    .def("dist_put", &NVSHMEMP2P::dist_put)
    .def("dist_get", &NVSHMEMP2P::dist_get);
}