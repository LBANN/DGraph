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
    if num_ranks < 1:
        raise ValueError("Number of ranks must be greater than 0.")
    if num_ranks == 1:
        return np.zeros(G.number_of_nodes(), dtype=int)

    try:
        import pymetis
    except ImportError:
        raise ImportError(
            "Please install pymetis to use this function (`pip install pymetis`)."
        )

    # pymetis takes an adjacency list indexed by contiguous node id 0..N-1
    # (the mesh graph's vertices are numbered this way).
    adjacency = [sorted(G.neighbors(node)) for node in range(G.number_of_nodes())]
    (edgecuts, node_rank_placement) = pymetis.part_graph(num_ranks, adjacency=adjacency)
    # Node_rank_placement is of shape (num_nodes, ), where each element is the
    # rank of the node in the partitioning.

    node_rank_placement = np.array(node_rank_placement)
    return node_rank_placement
