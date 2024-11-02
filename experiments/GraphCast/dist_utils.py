import torch
from DGraph.CommunicatorBase import CommunicatorBase


class SingleProcessDummyCommunicator(CommunicatorBase):
    """Single process communicator for debugging purposes"""

    def __init__(self):
        super().__init__()
        self._rank = 0
        self._world_size = 1
        self._is_initialized = True
        self._ranks_per_graph = 1
        self.backend = "single"

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def scatter(
        self, tensor: torch.Tensor, src: torch.Tensor, rank_mappings, num_local_nodes
    ):
        # TODO: Wrap this in the datawrapper class
        src = src.unsqueeze(-1).expand(-1, tensor.shape[-1])

        out = torch.zeros(num_local_nodes, tensor.shape[1]).to(tensor.device)
        out.scatter_add(0, src, tensor)
        return out

    def gather(self, tensor, dst, rank_mappings):
        # TODO: Wrap this in the datawrapper class
        dst = dst.unsqueeze(-1).expand(-1, tensor.shape[-1])
        out = torch.gather(tensor, 0, dst)
        return out

    def __str__(self) -> str:
        return self.backend

    def rank_cuda_device(self):
        device = torch.cuda.current_device()
        return device
