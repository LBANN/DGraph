import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
from DGraph.distributed.nccl import scatter, gather


class NCCLBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        # check if already initialized
        self._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", *args, **kwargs)

            self._initialized = True

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def scatter(self, *args, **kwargs):
        input_tensor = args[0]
        indices = args[1]
        output_tensor = torch.zeros_like(input_tensor)
        scatter(input_tensor, output_tensor, indices)

    def gather(self, *args, **kwargs):
        input_tensor = args[0]
        indices = args[1]
        output_tensor = torch.zeros_like(input_tensor)
        gather(input_tensor, output_tensor, indices)
