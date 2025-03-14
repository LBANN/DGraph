import numpy as np
import networkx as nx


def graphcast_graph_to_nxgraph(mesh_graph):
    G = nx.Graph()
    src_indices = mesh_graph[2].numpy()
    dst_indices = mesh_graph[3].numpy()
    edge_indices = np.stack((src_indices, dst_indices)).T
    G.add_edges_from(edge_indices)
    return G


def partition_graph(G: nx.Graph, num_ranks: int):
    try:
        import metis
    except ImportError:
        raise ImportError("Please install metis to use this function.")
    metis_graph = metis.networkx_to_metis(G)
    (edgecuts, node_rank_placement) = metis.part_graph(metis_graph, nparts=num_ranks)
    # Node_rank_placement is of shape (num_nodes, ), where each element is the
    # rank of the node in the partitioning.

    node_rank_placement = np.array(node_rank_placement)
    return node_rank_placement
