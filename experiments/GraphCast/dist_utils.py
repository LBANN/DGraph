import torch
from torch.utils.data import Sampler
from DGraph.CommunicatorBase import CommunicatorBase
from typing import Iterator
import math


class SingleProcessDummyCommunicator(CommunicatorBase):
    """Single process communicator for debugging purposes"""

    def __init__(self):
        super().__init__()
        self._rank = 0
        self._world_size = 1
        self._is_initialized = True
        self._ranks_per_sample = 1
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


# TODO: Incorporate this into main DGraph Codebase
class CommAwareDistributedSampler(Sampler):
    """
    Distributed Sampler that is coupled with the DGraph Communicator.
    The sampler ensures that each partition group has the same samples at each timestep.
    """

    def __init__(
        self,
        dataset,
        comm,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset)
        self.comm = comm
        self.dataset = dataset
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.partitiion_group = self.world_size // self.comm._ranks_per_sample
        self.local_rank_in_partition = self.rank % self.comm._ranks_per_sample
        self.num_replicas = self.world_size // self.comm._ranks_per_sample
        self.ranks_per_sample = comm._ranks_per_sample
        self.num_samples = (
            len(self.dataset) * self.ranks_per_sample
        ) // self.world_size
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch, but each partition_group
        # must have the exact same iterator. This is to ensure that each partition
        # must request the same sample from the dataset.

        # The dataset
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.partitiion_group : self.total_size : self.num_replicas]

        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
