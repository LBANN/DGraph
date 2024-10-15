import pytest
import torch.distributed as dist
import DGraph.Communicator as Comm
import torch


@pytest.fixture(scope="module")
def init_nccl_backend():
    dist.init_process_group(backend="nccl")

    comm = Comm.Communicator.init_process_group("nccl")

    return comm


@pytest.fixture(scope="module")
def setup_data():
    all_rank_input_data = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, -1, 1)
    all_rank_indices = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2]]).view(1, 4, 2)
    rank_mappings = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).view(1, 4, 2)
    all_rank_output = torch.tensor([2.0, 4.0, 1.0, 3.0]).view(1, -1, 1)
    return all_rank_input_data, all_rank_indices, rank_mappings, all_rank_output


def test_nccl_backend_scatter(init_nccl_backend, setup_data):
    comm = init_nccl_backend
    rank = comm.get_rank()
    print(f"Rank: {rank}")
    
