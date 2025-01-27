import metis
import torch
import numpy as np


from DGraph.data.ogbn_datasets import DistributedOGBWrapper
from experiments.GraphCast.dist_utils import SingleProcessDummyCommunicator
import pickle

import networkx as nx
import numpy as np


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
    """"""
    src_indices = edge_indices[:, 0]
    dst_indices = edge_indices[:, 1]
    src_indices = renumbered_nodes[src_indices]
    dst_indices = renumbered_nodes[dst_indices]
    renumbered_edges = np.stack((src_indices, dst_indices))
    return renumbered_edges


def save_networkx_digraph(coo_list, num_nodes, dname):
    G = nx.DiGraph()
    nodes = np.arange(num_nodes)
    G.add_nodes_from(nodes)
    G.add_edges_from(coo_list)

    with open(f"{dname}.pkl", "wb") as f:
        pickle.dump(G, f)


def read_networkx_digraph(dname):
    with open(f"{dname}.pkl", "rb") as f:
        G = pickle.load(f)
    return G


if __name__ == "__main__":
    dset_name = "products"
    comm = SingleProcessDummyCommunicator()
    training_dataset = DistributedOGBWrapper(f"ogbn-{dset_name}", comm)

    node_features, edge_indices, rank_mappings, labels = training_dataset[0]
    print(node_features.shape)
    num_nodes = node_features.shape[0]
    coo_list = edge_indices.numpy().T
    node_rank_placement = partition_directed_graph(coo_list, num_nodes, 4)
