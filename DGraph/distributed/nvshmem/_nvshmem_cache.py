from dataclasses import dataclass
import torch
import torch.distributed as dist


@dataclass
class NVSHMEMScatterCache:
    """This class caches the local scatter operators and index remappings
    for each rank to perform local accumulations on a workspace buffer.
    """

    num_output_rows: int
    local_indices_slice: torch.Tensor
    local_dest_ranks: torch.Tensor
    comm_indices: torch.Tensor
    rank_mapping: torch.Tensor
    index_offsets_per_rank: torch.Tensor
    min_workspace_size: int
    rank: int
    world_size: int


def NVSHMEMScatterCacheGenerator(
    num_output_rows: int,
    local_indices_slice: torch.Tensor,
    local_rank_mapping: torch.Tensor,
    vertices_per_rank: torch.Tensor,
    rank: int,
    world_size: int,
):
    """
    This function generates the NVSHMEM scatter cache for each rank. It does the
    following:

    1. Computes the index offsets per rank using the vertices per rank. This is used
        by the NVSHMEM scatter operation to place the data in the correct location
        in the symmetric workspace buffer which is indexed by local rank.
    2. Computes the minimum workspace size needed for the NVSHMEM scatter operation.
    3. Remaps the indices that need to be communicated to other ranks to an internal
        workspace index. This allows each rank to perform local accumulations of the
        communicated data before doing the communication step.
    """
    # Use the vertices per rank to compute the index offsets
    index_offsets_per_rank = torch.zeros(world_size, dtype=torch.long)
    index_offsets_per_rank[1:] = torch.cumsum(vertices_per_rank, dim=0)[:-1]

    # Find the number of unique messages from the local rank mapping
    comm_mask = local_rank_mapping != rank
    local_dest_ranks = local_rank_mapping[comm_mask]
    local_indices_slice = local_indices_slice[comm_mask]
    unique_dest_ranks, dest_indices = torch.unique(
        local_dest_ranks, return_inverse=True
    )

    # The amount of communication saved with this operation is equal to the
    # len(unique_dest_ranks) - len(local_dest_ranks)

    local_unique_messages = unique_dest_ranks.numel()
    num_unique_messages = torch.tensor([unique_dest_ranks.numel()])

    # This is the size of the NVSHMEM workspace buffer needed
    # We need to find the maximum number of unique messages across all ranks
    # because all ranks must allocate the same sized buffer
    dist.all_reduce(num_unique_messages, op=dist.ReduceOp.MAX)

    global_min_workspace_size = int(num_unique_messages[0].item())

    workspace_mapping = torch.zeros(global_min_workspace_size, dtype=torch.long) - 1
    workspace_mapping[0:local_unique_messages] = unique_dest_ranks
    updated_local_indices = local_indices_slice.clone()
    remap_comm_to_workspace = torch.zeros_like(local_dest_ranks)
    remap_comm_to_workspace.scatter_(
        0, dest_indices, torch.arange(local_unique_messages, device=dest_indices.device)
    )
    updated_local_indices[comm_mask] = remap_comm_to_workspace

    return NVSHMEMScatterCache(
        num_output_rows=num_output_rows,
        local_indices_slice=updated_local_indices,
        local_dest_ranks=local_dest_ranks,
        rank_mapping=local_rank_mapping,
        comm_indices=workspace_mapping,
        index_offsets_per_rank=index_offsets_per_rank,
        min_workspace_size=global_min_workspace_size,
        rank=rank,
        world_size=world_size,
    )
