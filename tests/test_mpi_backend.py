import pytest
import torch.distributed as dist
import DGraph.Communicator as Comm
import torch


@pytest.fixture(scope="module")
def init_mpi_backend():
    dist.init_process_group(backend="nccl")
    comm = Comm.Communicator.init_process_group("mpi")
    return comm


@pytest.fixture(scope="module")
def setup_gather_data():
    all_rank_input_data = torch.randn(1, 4, 64)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])
    rank_mappings = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]])

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        for i in range(8):
            all_rank_output[k][i] = all_rank_input_data[:, all_edge_coo[k, i]]

    return all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


def test_mpi_backend_init(init_mpi_backend):
    comm = init_mpi_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    print(f"Rank: {rank}")
    assert rank > -1
    assert world_size > 0
    assert rank < world_size
