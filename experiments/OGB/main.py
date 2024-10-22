from DGraph.data.datasets import DistributedOGBWrapper
from DGraph.Communicator import CommunicatorBase, Communicator

import fire
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from GCN import CommAwareGCN as GCN


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def write_experiment_log(log: str, fname: str, rank: int):
    if rank == 0:
        with open(fname, "a") as f:
            f.write(log + "\n")


class SingleProcessDummyCommunicator(CommunicatorBase):
    def __init__(self):
        super().__init__()
        self._rank = 0
        self._world_size = 1
        self._is_initialized = True

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


def main(_communicator: str = "dummy", num_epochs: int = 10):
    assert _communicator.lower() in [
        "dummy",
        "nccl",
        "nvshmem",
        "mpi",
    ], "Invalid communicator"

    if _communicator.lower() == "dummy":
        # Dummy communicator for single process testing
        comm = SingleProcessDummyCommunicator()
    else:
        dist.init_process_group(backend="nccl")
        comm = Communicator.init_process_group(_communicator)
    dataset = DistributedOGBWrapper("ogbn-arxiv", comm)
    model = GCN(in_channels=128, hidden_dims=256, num_classes=10, comm=comm)
    rank = comm.get_rank()
    model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    stream = torch.cuda.Stream()

    start_time.record(stream)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        output = model(dataset[0])
        loss = output.mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end_time.record(stream)
    torch.cuda.synchronize()

    average_time = start_time.elapsed_time(end_time) / num_epochs
    log_str = f"Average time per epoch: {average_time:.4f} ms"
    write_experiment_log(log_str, f"{_communicator}_experiment.log", rank)
    cleanup()


if __name__ == "__main__":
    fire.Fire(main)
