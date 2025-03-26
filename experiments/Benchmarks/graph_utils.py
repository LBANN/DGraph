from dataclasses import dataclass
import torch
from typing import Optional
import os


@dataclass
class GatherGraphData:
    """Dataclass to store graph data."""

    vertex_data: torch.Tensor
    vertex_rank_mapping: torch.Tensor
    edge_rank_placement: torch.Tensor
    edge_src_rank: torch.Tensor
    edge_indices: torch.Tensor


@dataclass
class ScatterGraphData:
    """Dataclass to store graph data."""

    vertex_data: torch.Tensor
    data_rank_mapping: torch.Tensor
    edge_rank_placement: torch.Tensor
    edge_dest_rank: torch.Tensor
    edge_indices: torch.Tensor
    num_local_vertices: int


def get_nccl_gather_benchmark_data(
    sample_size,
    world_size,
    device,
    ranks_per_node: Optional[int] = None,
    intra_node_only: Optional[bool] = False,
):
    """Generates data for distributed gather benchmarking."""

    assert (
        ranks_per_node is not None if intra_node_only else True
    ), "ranks_per_node must be provided if intra_node_only is True"

    torch.cuda.manual_seed(0)

    # Generate random data
    data = torch.randn(world_size, sample_size).to(device)

    vertex_mapping = torch.arange(world_size).to(device)
    edge_placement = (
        torch.repeat_interleave(torch.arange(world_size), world_size - 1)
        .unsqueeze(0)
        .to(device)
    )

    edges = []
    for i in range(world_size):
        for j in range(world_size):
            if i == j:
                continue
            edges.append(j)
    edge_src_rank = torch.tensor(edges).to(device).unsqueeze(0)
    edge_indices = edge_src_rank.clone().to(device)

    return GatherGraphData(
        vertex_data=data,
        vertex_rank_mapping=vertex_mapping,
        edge_rank_placement=edge_placement,
        edge_src_rank=edge_src_rank,
        edge_indices=edge_indices,
    )


def get_nccl_scatter_benchmark_data(
    sample_size,
    world_size,
    device,
    ranks_per_node: Optional[int] = None,
    intra_node_only: Optional[bool] = False,
):
    """Generates data for distributed scatter benchmarking."""

    assert (
        ranks_per_node is not None if intra_node_only else True
    ), "ranks_per_node must be provided if intra_node_only is True"

    torch.cuda.manual_seed(0)

    # Generate random data
    data = torch.randn(1, world_size * (world_size - 1), sample_size).to(device)

    data_mapping = (
        (
            torch.zeros(world_size, world_size - 1)
            + torch.arange(world_size).unsqueeze(1)
        )
        .reshape(1, -1)
        .to(device)
    )

    edge_placement = data_mapping.clone().long()

    edges = []
    for i in range(world_size):
        for j in range(world_size):
            if i == j:
                continue
            edges.append(j)
    edge_dest_rank = torch.tensor(edges).to(device).unsqueeze(0).long()
    edge_indices = edge_dest_rank.clone()

    return ScatterGraphData(
        vertex_data=data,
        data_rank_mapping=data_mapping,
        edge_rank_placement=edge_placement,
        edge_dest_rank=edge_dest_rank,
        edge_indices=edge_indices,
        num_local_vertices=1,  # One vertex per rank
    )


def get_nvshmem_gather_benchmark_data(
    sample_size,
    rank,
    world_size,
    device,
    ranks_per_node: Optional[int] = None,
    intra_node_only: Optional[bool] = False,
):
    """Generates data for distributed gather benchmarking."""

    assert (
        ranks_per_node is not None if intra_node_only else True
    ), "ranks_per_node must be provided if intra_node_only is True"

    torch.cuda.manual_seed(0)

    # Generate random data
    data = torch.randn(world_size, sample_size).to(device)

    vertex_mapping = torch.arange(world_size).to(device)
    edge_placement = (
        torch.repeat_interleave(torch.arange(world_size), world_size - 1)
        .unsqueeze(0)
        .to(device)
    )

    edges = []
    for i in range(world_size):
        for j in range(world_size):
            if i == j:
                continue
            edges.append(j)
    edge_src_rank = torch.tensor(edges).to(device).unsqueeze(0)
    local_vertex_data = data[vertex_mapping == rank].view(1, -1, sample_size)
    local_edge_src_rank = edge_src_rank[edge_placement == rank].view(1, -1)

    edge_indices = edge_src_rank.clone().to(device)
    local_edge_indices = edge_indices[edge_placement == rank].view(1, -1)

    return GatherGraphData(
        vertex_data=local_vertex_data,
        vertex_rank_mapping=vertex_mapping,
        edge_rank_placement=edge_placement,
        edge_src_rank=local_edge_src_rank,
        edge_indices=local_edge_indices,
    )


def get_nvshmem_scatter_benchmark_data(
    sample_size,
    rank,
    world_size,
    device,
    ranks_per_node: Optional[int] = None,
    intra_node_only: Optional[bool] = False,
):
    """Generates data for distributed scatter benchmarking."""

    assert (
        ranks_per_node is not None if intra_node_only else True
    ), "ranks_per_node must be provided if intra_node_only is True"

    torch.cuda.manual_seed(0)

    # Generate random data
    data = torch.randn(1, world_size * (world_size - 1), sample_size).to(device)

    data_mapping = (
        (
            torch.zeros(world_size, world_size - 1)
            + torch.arange(world_size).unsqueeze(1)
        )
        .reshape(1, -1)
        .to(device)
    )

    edge_placement = data_mapping.clone()

    edges = []
    for i in range(world_size):
        for j in range(world_size):
            if i == j:
                continue
            edges.append(j)
    edge_dest_rank = torch.tensor(edges).to(device).unsqueeze(0)
    local_edge_dest_rank = edge_dest_rank[edge_placement == rank].view(1, -1)

    edge_indices = edge_dest_rank.clone()
    local_edge_indices = edge_indices[edge_placement == rank].view(1, -1)

    local_data = data[edge_placement == rank, :].unsqueeze(0)
    return ScatterGraphData(
        vertex_data=local_data,
        data_rank_mapping=data_mapping,
        edge_rank_placement=edge_placement,
        edge_dest_rank=local_edge_dest_rank,
        edge_indices=local_edge_indices,
        num_local_vertices=1,  # One vertex per rank
    )


def safe_create_dir(directory, rank):
    if rank == 0:
        if not os.path.exists(directory):
            os.makedirs(directory)


if __name__ == "__main__":
    gather_data = get_nvshmem_gather_benchmark_data(64, 0, 4, "cuda")
    scatter_data = get_nvshmem_scatter_benchmark_data(64, 0, 4, "cuda")
