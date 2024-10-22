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
def setup_gather_data():
    all_rank_input_data = torch.randn(4, 64)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])
    rank_mappings = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1], [0, 1, 1, 0, 1, 0, 1, 0]])

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        for i in range(8):
            all_rank_output[k][i] = all_rank_input_data[all_edge_coo[k, i]]

    return all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


@pytest.fixture(scope="module")
def setup_scatter_data():
    all_rank_input_data = torch.randn(4, 64)
    all_rank_indices = torch.tensor([[0, 1, 2, 3], [1, 3, 0, 2]])
    rank_mappings = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    all_rank_output = torch.tensor([2.0, 4.0, 1.0, 3.0]).view(1, -1, 1)
    return all_rank_input_data, all_rank_indices, rank_mappings, all_rank_output


def test_nccl_backend_init(init_nccl_backend):
    comm = init_nccl_backend
    rank = comm.get_rank()
    print(f"Rank: {rank}")


def test_nccl_backend_gather(init_nccl_backend, setup_gather_data):
    comm = init_nccl_backend
    all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output = (
        setup_gather_data
    )
    local_input_data = comm.get_local_rank_slice(all_rank_input_data.unsqueeze(0))

    rank = comm.get_rank()
    world_size = comm.get_world_size()
    start_index = all_rank_input_data.shape[0] // world_size * rank
    end_index = all_rank_input_data.shape[0] // world_size * (rank + 1)
    local_input_data_gt = all_rank_input_data[start_index:end_index]

    assert local_input_data.shape == (2, 64)
    assert torch.allclose(local_input_data, local_input_data_gt)
