import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
from mpi4py import MPI
from torch.utils.dlpack import to_dlpack  # type: ignore
from torch.utils.dlpack import from_dlpack
import cupy as cp


class MPIBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        self._iniitalized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not self._initialized:
            # We want both NCCL and MPI to be initialized
            dist.init_process_group(backend="nccl", *args, **kwargs)
            # Dist initialization is done by the user and handles
            # the collective operations need for SGD

            MPI.Init()
            self._initialized = True
            self._comm = MPI.COMM_WORLD

    def finalize(self):
        if self._initialized:
            MPI.Finalize()

            self._initialized = False

    def Malloc(self, size: int) -> torch.Tensor:
        """Allocates memory on the GPU that is accessible by MPI one-sided communication"""

        cupy_tensor = cp.empty(size, dtype=cp.float32)
        MPI.Win.Attach(cupy_tensor.data.ptr)  # type: ignore
        torch_tensor = from_dlpack(to_dlpack(cupy_tensor))  # Zero copy
        return torch_tensor

    def get_rank(self) -> int:
        return self._comm.Get_rank()

    def get_world_size(self) -> int:
        return self._comm.Get_size()

    def scatter(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def gather(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
