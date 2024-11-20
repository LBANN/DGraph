import pytest
import DGraph.Communicator as Comm
import torch


@pytest.fixture(scope="module")
def init_mpi_backend():
    import os

    # Once HPCLauncher is available, the following lines can be removed - S.Z
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  # Random port
    # We are only testing the MPI backend, the Torch Dist does not need to be
    # in sync with the MPI backend for this test. So we can skip the NCCL assert
    comm = Comm.Communicator.init_process_group("mpi", SKIP_NCCL_ASSERT=True)
    return comm


@pytest.fixture(scope="module")
def setup_gather_data():
    torch.random.manual_seed(0)
    num_features = 64
    all_rank_input_data = torch.randn(1, 4, num_features)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])

    rank_mappings = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0, 1, 0],
        ]
    )

    all_rank_output = torch.zeros(2, 8, 64)

    for k in range(2):
        for i in range(8):
            all_rank_output[k][i] = all_rank_input_data[:, all_edge_coo[k, i]]

    return all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output


@pytest.fixture(scope="module")
def setup_scatter_data():
    torch.random.manual_seed(0)
    num_features = 4
    all_rank_input_data = torch.randn(1, 8, num_features)

    all_edge_coo = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 3, 0, 3, 0]])

    rank_mappings = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0, 1, 0],
        ]
    )

    all_rank_output_gt = torch.zeros(2, 4, num_features)

    for i in range(2):
        all_rank_output_gt[[i]] = all_rank_output_gt[[i]].scatter_add(
            1,
            all_edge_coo[i].unsqueeze(-1).expand(1, -1, num_features),
            all_rank_input_data,
        )
    return all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output_gt


def test_mpi_backend_init(init_mpi_backend):
    comm = init_mpi_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    assert rank > -1
    assert world_size > 0
    assert rank < world_size


def test_mpi_backend_gather(init_mpi_backend, setup_gather_data):
    comm = init_mpi_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output = (
        setup_gather_data
    )

    local_input_data = comm.get_local_rank_slice(all_rank_input_data, dim=1)
    local_index = comm.get_local_rank_slice(all_edge_coo.unsqueeze(0))
    local_rank_mappings = comm.get_local_rank_slice(rank_mappings.unsqueeze(0))

    input_start_index = (all_rank_input_data.shape[1] // world_size) * rank
    input_end_index = (all_rank_input_data.shape[1] // world_size) * (rank + 1)

    local_input_data_gt = all_rank_input_data[:, input_start_index:input_end_index]

    indices_start_index = (all_edge_coo.shape[-1] // world_size) * rank
    indices_end_index = (all_edge_coo.shape[-1] // world_size) * (rank + 1)

    local_index_gt = all_edge_coo[:, indices_start_index:indices_end_index].unsqueeze(0)
    local_rank_mappings_gt = rank_mappings[
        :, indices_start_index:indices_end_index
    ].unsqueeze(0)
    # Check if the local slicing is correct
    assert torch.allclose(local_input_data, local_input_data_gt)
    assert torch.allclose(local_index, local_index_gt)
    assert torch.allclose(local_rank_mappings, local_rank_mappings_gt)

    # Check if the gather is correct
    output_start_index = (all_rank_output.shape[1] // world_size) * rank
    output_end_index = (all_rank_output.shape[1] // world_size) * (rank + 1)

    local_output_gt = all_rank_output[:, output_start_index:output_end_index]
    for i in range(2):
        local_output = comm.gather(
            local_input_data, local_index[:, i], local_rank_mappings[0][[i], :]
        )
        assert (
            local_output.shape == local_output_gt[[i]].shape
        ), f"{local_output.shape} != {local_output_gt[[i]].shape}"
        assert torch.allclose(local_output, local_output_gt[[i]])


def test_mpi_backend_scatter(init_mpi_backend, setup_scatter_data):
    comm = init_mpi_backend
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    all_rank_input_data, all_edge_coo, rank_mappings, all_rank_output_gt = (
        setup_scatter_data
    )

    _input_split_start_index = (all_rank_input_data.shape[1] // 2) * rank
    _input_split_end_index = (all_rank_input_data.shape[1] // 2) * (rank + 1)

    # Split the input tensor
    local_input_split_gt = all_rank_input_data[
        :, _input_split_start_index:_input_split_end_index
    ]

    # Split the indices tensor
    _indices_split_start_index = (all_edge_coo.shape[-1] // 2) * rank
    _indices_split_end_index = (all_edge_coo.shape[-1] // 2) * (rank + 1)

    local_indices_split_gt = all_edge_coo[
        :, _indices_split_start_index:_indices_split_end_index
    ]

    local_rank_mappings_split_gt = rank_mappings[
        :, _indices_split_start_index:_indices_split_end_index
    ]

    num_local_output_rows = all_rank_output_gt.shape[1] // 2

    # Split the expected output tensor
    local_rank_output_gt = all_rank_output_gt[
        :, num_local_output_rows * rank : num_local_output_rows * (rank + 1)
    ]

    # Test the splitting of the tensors by the communicator

    for test_inp in range(2):
        local_indices_split = comm.get_local_rank_slice(all_edge_coo[[test_inp]], dim=1)

        assert (
            local_indices_split_gt[[test_inp]].shape == local_indices_split.shape
        ), f"{local_indices_split_gt.shape} != {local_indices_split.shape}"
        assert torch.allclose(local_indices_split_gt[[test_inp]], local_indices_split)

        local_rank_mappings_split = comm.get_local_rank_slice(
            rank_mappings[[test_inp]], dim=1
        )

        assert (
            local_rank_mappings_split_gt[[test_inp]].shape
            == local_rank_mappings_split.shape
        ), f"{local_rank_mappings_split_gt.shape} != {local_rank_mappings_split.shape}"

        assert torch.allclose(
            local_rank_mappings_split_gt[[test_inp]], local_rank_mappings_split
        ), f"{local_rank_mappings_split_gt} != {local_rank_mappings_split}"

        local_input_split = comm.get_local_rank_slice(all_rank_input_data, dim=1)

        assert local_input_split_gt.shape == local_input_split.shape
        assert torch.allclose(local_input_split_gt, local_input_split)

        # Perform the scatter operation

        local_output = comm.scatter(
            local_input_split,
            local_indices_split,
            num_local_output_rows,
            local_rank_mappings_split,
        )

    assert local_rank_output_gt[[test_inp]].shape == local_output.shape
    assert torch.allclose(local_rank_output_gt[[test_inp]], local_output)
