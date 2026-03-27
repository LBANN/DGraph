import torch
from typing import Optional, Tuple
from DGraph.data.graph import DistributedGraph


def node_renumbering(node_rank_placement) -> Tuple[torch.Tensor, torch.Tensor]:
    """The nodes are renumbered based on the rank mappings so the node features and
    numbers are contiguous."""

    contiguous_rank_mapping, renumbered_nodes = torch.sort(node_rank_placement)
    return renumbered_nodes, contiguous_rank_mapping


def edge_renumbering(
    edge_indices, renumbered_nodes, vertex_mapping, edge_features=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    src_indices = edge_indices[0, :]
    dst_indices = edge_indices[1, :]
    src_indices = renumbered_nodes[src_indices]
    dst_indices = renumbered_nodes[dst_indices]

    edge_src_rank_mapping = vertex_mapping[src_indices]
    edge_dest_rank_mapping = vertex_mapping[dst_indices]

    sorted_src_rank_mapping, sorted_indices = torch.sort(edge_src_rank_mapping)
    dst_indices = dst_indices[sorted_indices]
    src_indices = src_indices[sorted_indices]

    sorted_dest_rank_mapping = edge_dest_rank_mapping[sorted_indices]

    if edge_features is not None:
        # Sort the edge features based on the sorted indices
        edge_features = edge_features[sorted_indices]

    return (
        torch.stack([src_indices, dst_indices], dim=0),
        sorted_src_rank_mapping,
        sorted_dest_rank_mapping,
        edge_features,
    )


def process_homogenous_data(
    graph_data,
    labels,
    rank: int,
    world_Size: int,
    split_idx: dict,
    node_rank_placement: torch.Tensor,
    *args,
    **kwargs,
) -> DistributedGraph:
    """For processing homogenous graph with node features, edge index and labels"""
    assert "node_feat" in graph_data, "Node features not found"
    assert "edge_index" in graph_data, "Edge index not found"
    assert "num_nodes" in graph_data, "Number of nodes not found"
    assert graph_data["edge_feat"] is None, "Edge features not supported"

    node_features = torch.Tensor(graph_data["node_feat"]).float()
    edge_index = torch.Tensor(graph_data["edge_index"]).long()
    num_nodes = graph_data["num_nodes"]
    labels = torch.Tensor(labels).long()
    # For bidirectional graphs the number of edges are double counted
    num_edges = edge_index.shape[1]

    assert node_rank_placement.shape[0] == num_nodes, "Node mapping mismatch"
    assert "train" in split_idx, "Train mask not found"
    assert "valid" in split_idx, "Validation mask not found"
    assert "test" in split_idx, "Test mask not found"

    train_nodes = torch.from_numpy(split_idx["train"])
    valid_nodes = torch.from_numpy(split_idx["valid"])
    test_nodes = torch.from_numpy(split_idx["test"])

    # Renumber the nodes and edges to make them contiguous
    renumbered_nodes, contiguous_rank_mapping = node_renumbering(node_rank_placement)
    node_features = node_features[renumbered_nodes]

    # Sanity check to make sure we placed the nodes in the correct spots

    assert torch.all(node_rank_placement[renumbered_nodes] == contiguous_rank_mapping)

    # First renumber the edges
    # Then we calculate the location of the source and destination vertex of each edge
    # based on the rank mapping
    # Then we sort the edges based on the source vertex rank mapping
    # When determining the location of the edge, we use the rank of the source vertex
    # as the location of the edge

    edge_index, edge_rank_mapping, edge_dest_rank_mapping, _ = edge_renumbering(
        edge_index, renumbered_nodes, contiguous_rank_mapping, edge_features=None
    )

    train_nodes = renumbered_nodes[train_nodes]
    valid_nodes = renumbered_nodes[valid_nodes]
    test_nodes = renumbered_nodes[test_nodes]

    labels = labels[renumbered_nodes]

    graph_obj = DistributedGraph(
        node_features=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_loc=contiguous_rank_mapping.long(),
        edge_loc=edge_rank_mapping.long(),
        edge_dest_rank_mapping=edge_dest_rank_mapping.long(),
        world_size=world_Size,
        labels=labels,
        train_mask=train_nodes,
        val_mask=valid_nodes,
        test_mask=test_nodes,
    )
    return graph_obj
