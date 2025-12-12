import torch
from dataclasses import dataclass
from typing import List
import torch.distributed as dist


@dataclass
class NCCLGraphCommPlan:
    """
    Class to store communication plan for distributed gather-scatter (vector addressing)

    Attributes:
        rank (int): Local rank
        world_size (int): World size
        local_num_vertices (int): Number of local vertices
        local_src_idx (torch.Tensor): Local source indices for scatter-sum
        local_dst_idx (torch.Tensor): Local destination indices for scatter-sum
        send_src_idx (torch.Tensor): Source indices to send to other ranks
        send_buffer_idx (torch.Tensor): Buffer indices to store data to send to other ranks
        send_comm_vector (torch.Tensor): Communication vector of shape [world_size] of messages to send to each rank
        recv_dst_idx (torch.Tensor): Destination indices to receive from other ranks
        recv_comm_vector (torch.Tensor): Communication vector of shape [world_size] of messages to
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


def COO_to_NCCLCommPlan(
    rank: int,
    world_size: int,
    global_edges_src: torch.Tensor,
    global_edges_dst: torch.Tensor,
    vertex_rank_placement: torch.Tensor,
    local_edge_list: torch.Tensor,
    offset: torch.Tensor,
) -> NCCLGraphCommPlan:
    device = local_edge_list.device
    my_src_global = global_edges_src[local_edge_list].to(device)
    my_dst_global = global_edges_dst[local_edge_list].to(device)

    my_start = offset[rank].item()
    my_end = offset[rank + 1].item()

    nodes_per_rank = torch.bincount(vertex_rank_placement, minlength=world_size)
    num_local_vertices = int(nodes_per_rank[rank].item())
    num_local_edges = local_edge_list.size(0)

    dest_ranks = torch.bucketize(global_edges_dst, offset, right=True) - 1

    is_internal = dest_ranks == rank
    internal_dst_global = my_dst_global[is_internal]
    internal_node_idx = internal_dst_global - offset

    internal_edge_indices = torch.nonzero(is_internal, as_tuple=True)[0]

    remote_mask = ~is_internal

    boundary_edge_indices = torch.nonzero(remote_mask, as_tuple=True)[0]

    b_dst_global = my_dst_global[remote_mask]
    b_dest_ranks = dest_ranks[remote_mask]

    sort_idx = torch.argsort(b_dest_ranks)
    boundary_edge_indices = boundary_edge_indices[sort_idx]
    b_dst_global = b_dst_global[sort_idx]
    b_dest_ranks = b_dest_ranks[sort_idx]

    unique_dests, inverse_indices = torch.unique(
        torch.stack([b_dest_ranks, b_dst_global]), dim=1, return_inverse=True
    )
    unique_ranks = unique_dests[0]
    unique_global_ids = unique_dests[1]

    boundary_edge_buffer_map = inverse_indices

    boundary_edge_splits = torch.bincount(unique_ranks, minlength=world_size).tolist()

    recv_counts_tensor = torch.empty(world_size, dtype=torch.long, device=device)
    send_counts_tensor = torch.tensor(
        boundary_edge_splits, dtype=torch.long, device=device
    )
    dist.all_to_all_single(recv_counts_tensor, send_counts_tensor)
    boundary_node_splits = recv_counts_tensor.tolist()

    total_recv_nodes = sum(boundary_node_splits)
    recv_global_ids = torch.empty(total_recv_nodes, dtype=torch.long, device=device)

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
