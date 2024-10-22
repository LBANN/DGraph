import pytest
import torch.distributed as dist
import DGraph.Communicator as Comm
import torch


@pytest.fixture(scope="module")
def init_mpi_backend():
    dist.init_process_group(backend="nccl")
    comm = Comm.Communicator.init_process_group("mpi")
    return comm


def test_mpi_backend_init(init_mpi_backend):
    comm = init_mpi_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    print(f"Rank: {rank}")
    assert rank > -1
    assert world_size > 0
    assert rank < world_size
