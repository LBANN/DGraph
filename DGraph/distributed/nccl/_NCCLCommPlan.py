import torch
from dataclasses import dataclass
from typing import List, Optional
import torch.distributed as dist


@dataclass
class NCCLGraphCommPlan:
    """
    Class to store communication plan for distributed gather-scatter (vector addressing)

    Attributes:
        rank (int): Local rank
        world_size (int): World size
        local_edge_idx (torch.Tensor): IDs of edges that are local to the rank
        loca_vertex_idx (torch.Tensor): IDs of vertices that are local to the rank
        boundary_edge_idx (torch.Tensor): IDs of edges that are boundary edges (edges that have have vertices on other ranks)
                                        Information must be reecived from other ranks for these edges. Values are 0 to num_local_edges.
        boundary_edge_buffer_map (torch.Tensor): Buffer map to store boundary edge data to send to other ranks
                                                 Values are 0 to sum(boundary_edge_splits).
        boundary_edge_splits (List[int]): World_size element list. Number of boundary edges to send to each rank.
                                          Boundary_edge_splits[r] corresponds to the number of unique vertices in
                                          this rank that have neighbors on rank r.
        boundary_vertex_idx (torch.Tensor): IDs of boundary vertices (vertices that have edges on other ranks).
                                            Information must be sent to other ranks for these vertices.
        boundary_vertex_splits (List[int]): World_size element list. Number of
    """

    rank: int
    world_size: int

    # Allocation meta data
    num_local_vertices: int
    num_local_edges: int

    # Local edge-vertex mapping
    #
    # Used for:
    #   1) Local scatter-sum (edge -> vertex aggregation)
    #      y[local_vertex_idx] += x[local_edge_idx]
    #   2) Local gather (vertex -> edge gathering)
    #      y[local_edge_idx] = x[local_vertex_idx]

    local_edge_idx: torch.Tensor
    local_vertex_idx: torch.Tensor

    # Boundary edges (data must be sent/received to/from other ranks for gather/scatter)

    boundary_edge_idx: torch.Tensor
    boundary_edge_buffer_map: torch.Tensor
    boundary_edge_splits: List[int]

    # Boundary vertices (vertices that have edges on other ranks)
    boundary_vertex_idx: torch.Tensor
    boundary_vertex_splits: List[int]

    def to(self, device: torch.device):
        self.local_edge_idx = self.local_edge_idx.to(device)
        self.local_vertex_idx = self.local_vertex_idx.to(device)
        self.boundary_edge_idx = self.boundary_edge_idx.to(device)
        self.boundary_edge_buffer_map = self.boundary_edge_buffer_map.to(device)
        self.boundary_vertex_idx = self.boundary_vertex_idx.to(device)
        return self


@dataclass
class NCCLEdgeConditionedGraphCommPlan:
    """
    Class to store communication plan for distributed gather-scatter for edge-conditioned
    graphs where both source and destination vertices are needed.

    Attributes:
        rank (int): Local rank
        world_size (int): World size

        source_graph_plan (NCCLGraphCommPlan): Communication plan for source vertices
        dest_graph_plan (NCCLGraphCommPlan): Communication plan for destination vertices
    """

    rank: int
    world_size: int

    source_graph_plan: NCCLGraphCommPlan
    dest_graph_plan: Optional[NCCLGraphCommPlan] = None

    def to(self, device: torch.device):
        self.source_graph_plan = self.source_graph_plan.to(device)
        if self.dest_graph_plan is not None:
            self.dest_graph_plan = self.dest_graph_plan.to(device)
        return self

    def reverse(self):
        if self.dest_graph_plan is None:
            raise ValueError("Destination graph plan is None, cannot reverse.")
        return NCCLEdgeConditionedGraphCommPlan(
            rank=self.rank,
            world_size=self.world_size,
            source_graph_plan=self.dest_graph_plan,
            dest_graph_plan=self.source_graph_plan,
        )


def compute_edge_slices(dest_ranks, rank, my_dst_global, offset):

    is_internal = dest_ranks == rank
    internal_dst_global = my_dst_global[is_internal]
    internal_node_idx = internal_dst_global - offset[rank]

    internal_edge_indices = torch.nonzero(is_internal, as_tuple=True)[0]

    remote_mask = ~is_internal

    boundary_edge_indices = torch.nonzero(remote_mask, as_tuple=True)[0]

    b_dst_global = my_dst_global[remote_mask]
    b_dest_ranks = dest_ranks[remote_mask]

    return (
        internal_node_idx,
        internal_edge_indices,
        b_dst_global,
        b_dest_ranks,
        boundary_edge_indices,
    )


def fast_2D_unique(indices_1, indices_2):
    packed_keys = indices_1.to(torch.int64) << 32 | indices_2.to(torch.int64)
    unique_packed, inverse_indices = torch.unique(
        packed_keys, return_inverse=True, sorted=False
    )
    unique_1 = unique_packed >> 32
    unique_2 = unique_packed & 0xFFFFFFFF
    return unique_1, unique_2, inverse_indices


def COO_to_NCCLCommPlan(
    rank: int,
    world_size: int,
    global_edges_dst: torch.Tensor,
    local_edge_list: torch.Tensor,
    offset: torch.Tensor,
    dst_offset: Optional[torch.Tensor] = None,
) -> NCCLGraphCommPlan:
    """

    Convert COO (Coordinate List) format graph to NCCLGraphCommPlan for distributed gather-scatter operations.

    Args:
        rank (int): Local rank
        world_size (int): World size
        global_edges_dst (torch.Tensor): Global destination indices of edges
        local_edge_list (torch.Tensor): List of indices of local edges
        offset (torch.Tensor): Offset for each rank.
            The vertices are partitioned among ranks in a contiguous manner.
            All vertices in the range [offset[rank], offset[rank + 1]) are assigned to the rank.
        dst_offset (Optional[torch.Tensor]): Offset for each rank for destination vertices, for heterogeneous graphs where
            source and destination vertices are different. The vertices are partitioned among ranks in a contiguous manner.

    """

    device = local_edge_list.device
    my_dst_global = global_edges_dst[local_edge_list].to(device)

    if int(offset[-1].item()) > (2**32):
        raise ValueError(
            f"{offset[-1]}, Number of vertices exceeding {2**32}, which is not supported"
        )

    my_start = offset[rank].item()
    my_end = offset[rank + 1].item()
    num_local_vertices = int(my_end - my_start)
    num_local_edges = local_edge_list.size(0)

    dst_offset = offset if dst_offset is None else dst_offset

    dest_ranks = torch.bucketize(my_dst_global, dst_offset[1:], right=False)

    # Seperate this out to reduce memory usage
    (
        internal_node_idx,
        internal_edge_indices,
        b_dst_global,
        b_dest_ranks,
        boundary_edge_indices,
    ) = compute_edge_slices(dest_ranks, rank, my_dst_global, dst_offset)

    unique_ranks, unique_global_ids, inverse_indices = fast_2D_unique(
        b_dest_ranks, b_dst_global
    )

    # print(f"Rank {rank} has {len(boundary_edge_indices)} edges to send ")
    # print(f"Rank {rank} has {len(unique_ranks)} unique messages to send ")

    # if len(unique_ranks) > 0:
    #     print(
    #         f"Rank {rank} message reduction ratio: {len(boundary_edge_indices)/len(unique_ranks)}"
    #     )

    boundary_edge_buffer_map = inverse_indices

    boundary_edge_splits = torch.bincount(unique_ranks, minlength=world_size).tolist()

    recv_counts_tensor = torch.zeros(world_size, dtype=torch.long, device=device)
    send_counts_tensor = torch.tensor(
        boundary_edge_splits, dtype=torch.long, device=device
    )
    if recv_counts_tensor.device == torch.device("cpu"):
        recv_counts_tensor = recv_counts_tensor.cuda()

    if send_counts_tensor.device == torch.device("cpu"):
        send_counts_tensor = send_counts_tensor.cuda()

    print(f"rank: {rank} send_counts_tensor: {send_counts_tensor}")

    # if rank == 0:
    #     breakpoint()
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    print(f"rank: {rank} recv_counts_tensor: {recv_counts_tensor}")

    boundary_node_splits = recv_counts_tensor.tolist()

    total_recv_nodes = sum(boundary_node_splits)
    if total_recv_nodes > 0:
        recv_global_ids = torch.empty(total_recv_nodes, dtype=torch.long, device=device)
    else:
        recv_global_ids = torch.empty(0, dtype=torch.long, device=device)

    if sum(send_counts_tensor) == 0:
        unique_global_ids = torch.empty(0, dtype=torch.long, device=device)

    if recv_global_ids.device == torch.device("cpu"):
        recv_global_ids = recv_global_ids.cuda()

    if unique_global_ids.device == torch.device("cpu"):
        unique_global_ids = unique_global_ids.cuda()

    dist.barrier()
    dist.all_to_all_single(
        recv_global_ids,
        unique_global_ids,
        output_split_sizes=boundary_node_splits,
        input_split_sizes=boundary_edge_splits,
    )

    boundary_node_idx = recv_global_ids - my_start

    return NCCLGraphCommPlan(
        rank=rank,
        world_size=world_size,
        num_local_vertices=num_local_vertices,
        num_local_edges=num_local_edges,
        local_edge_idx=internal_edge_indices,
        local_vertex_idx=internal_node_idx,
        boundary_edge_idx=boundary_edge_indices,
        boundary_edge_buffer_map=boundary_edge_buffer_map,
        boundary_edge_splits=boundary_edge_splits,
        boundary_vertex_idx=boundary_node_idx,
        boundary_vertex_splits=boundary_node_splits,
    )


def COO_to_NCCLEdgeConditionedCommPlan(
    rank: int,
    world_size: int,
    global_edges_src: torch.Tensor,
    global_edges_dst: torch.Tensor,
    local_edge_list: torch.Tensor,
    src_offset: torch.Tensor,
    dest_offset: Optional[torch.Tensor],
) -> NCCLEdgeConditionedGraphCommPlan:
    """

    Convert COO (Coordinate List) format graph to NCCLEdgeConditionedGraphCommPlan for distributed gather-scatter operations.

    Args:
        rank (int): Local rank
        world_size (int): World size
        global_edges_src (torch.Tensor): Global source indices of edges
        global_edges_dst (torch.Tensor): Global destination indices of edges
        local_edge_list (torch.Tensor): List of indices of local edges
        src_offset (torch.Tensor): Offset for each rank for source vertices.
            The vertices are partitioned among ranks in a contiguous manner.
            All vertices in the range [src_offset[rank], src_offset[rank + 1]) are assigned to the rank.
        dest_offset (Optional[torch.Tensor]): Offset for each rank for destination vertices.
            The vertices are partitioned among ranks in a contiguous manner.
            All vertices in the range [dest_offset[rank], dest_offset[rank + 1]) are assigned to the rank.
    """
    device = local_edge_list.device

    source_plan = COO_to_NCCLCommPlan(
        rank,
        world_size,
        global_edges_src,
        local_edge_list,
        src_offset,
        dst_offset=dest_offset,
    )

    if dest_offset is None:
        dest_offset = src_offset

    dest_plan = COO_to_NCCLCommPlan(
        rank,
        world_size,
        global_edges_dst,
        local_edge_list,
        dest_offset,
        dst_offset=src_offset,
    )

    return NCCLEdgeConditionedGraphCommPlan(
        rank=rank,
        world_size=world_size,
        source_graph_plan=source_plan,
        dest_graph_plan=dest_plan,
    )
