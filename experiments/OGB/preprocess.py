import metis
import torch
import numpy as np


from DGraph.data.ogbn_datasets import DistributedOGBWrapper
from experiments.GraphCast.dist_utils import SingleProcessDummyCommunicator
import pickle

import networkx as nx
import numpy as np
from fire import Fire


def partition_graph(coo_list: np.ndarray, num_ranks: int):

    G = nx.Graph()
    G.add_edges_from(coo_list)
    metis_graph = metis.networkx_to_metis(G)
    (edgecuts, node_rank_placement) = metis.part_graph(metis_graph, nparts=num_ranks)
    # Node_rank_placement is of shape (num_nodes, ), where each element is the
    # rank of the node in the partitioning.

    node_rank_placement = np.array(node_rank_placement)
    # Renumber the nodes and edges so they are contiguous in memory.
    renumbered_nodes = node_renumbering(node_rank_placement, num_ranks)
    renumbered_edges = edge_renumbering(coo_list, renumbered_nodes)

    return renumbered_nodes, renumbered_edges


def partition_directed_graph(coo_list, num_nodes, num_parts):
    G = nx.DiGraph()
    nodes = np.arange(num_nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from(coo_list)
    metis_graph = metis.networkx_to_metis(G)

    print("Starting partitioning")
    (edgecuts, node_rank_placement) = metis.part_graph(metis_graph, nparts=num_parts)
    print("Finished partitioning")

    breakpoint()
    node_rank_placement = np.array(node_rank_placement)
    renumbered_nodes = node_renumbering(node_rank_placement, num_parts)
    renumbered_edges = edge_renumbering(coo_list, renumbered_nodes)
    return renumbered_nodes, renumbered_edges


def node_renumbering(node_rank_placement, num_parts):
    """The nodes are renumbered based on the rank mappings so the node features and
    numbers are contiguous."""
    rearranged_indices = []
    for rank in range(num_parts):
        mask = node_rank_placement == rank
        indices = np.where(mask)[0]
        rearranged_indices.append(indices)
    renumbered_nodes = np.concatenate(rearranged_indices)
    return renumbered_nodes


def edge_renumbering(edge_indices, renumbered_nodes):
    """The edges are renumbered based on the renumbered nodes."""
    src_indices = edge_indices[:, 0]
    dst_indices = edge_indices[:, 1]
    src_indices = renumbered_nodes[src_indices]
    dst_indices = renumbered_nodes[dst_indices]
    renumbered_edges = np.stack((src_indices, dst_indices))
    return renumbered_edges


def add_opposite_edges(edge_indices):
    """Add the opposite edges to the edge indices.
    Converts a directed graph to an undirected graph. Useful for METIS which only
    supports undirected graphs.
    """
    src_indices = edge_indices[0]
    dst_indices = edge_indices[1]
    opposite_edges = np.stack((dst_indices, src_indices))
    return np.concatenate((edge_indices, opposite_edges), axis=1)


def save_networkx_graph(coo_list, num_nodes, dname, directed=False):

    if not directed:
        G = nx.Graph()
        nodes = np.arange(num_nodes)
        G.add_nodes_from(nodes)
        G.add_edges_from(coo_list)

    else:
        G = nx.DiGraph()
        nodes = np.arange(num_nodes)
        G.add_nodes_from(nodes)
        G.add_edges_from(coo_list)
        G.add_edges_from([(dst, src) for src, dst in coo_list])

    with open(f"{dname}_directed={directed}.pkl", "wb") as f:
        pickle.dump(G, f)


def load_networkx_graph(dname):
    with open(f"{dname}.pkl", "rb") as f:
        G = pickle.load(f)
    return G


def main(dset_name: str):
    from ogb.nodeproppred import NodePropPredDataset

    assert dset_name in ["ogbn-arxiv", "ogbn-products", "ogbn-proteins"]

    is_directed = False
    if dset_name == "ogbn-arxiv":
        is_directed = True

    dataset = NodePropPredDataset(
        dset_name,
    )
    graph_data, labels = dataset[0]

    edge_index = torch.Tensor(graph_data["edge_index"]).long()
    num_nodes = graph_data["num_nodes"]

    coo_list = edge_index.numpy().T

    save_networkx_graph(coo_list, num_nodes, dset_name, directed=is_directed)


if __name__ == "__main__":
    Fire(main)
