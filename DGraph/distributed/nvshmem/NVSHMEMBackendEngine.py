import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
import torch_nvshmem_p2p as nvshmem


class NVSHMEMBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        # check if already initialized
        self._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not self._initialized:
            dist.init_process_group(backend="nccl", *args, **kwargs)
            nvshmem.init()
            self._initialized = True

    def get_rank(self) -> int:
        return nvshmem.get_rank()

    def get_world_size(self) -> int:
        return nvshmem.get_world_size()
