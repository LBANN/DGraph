import metis
import torch
import numpy as np
import pickle
import graphcast_config
import networkx as nx
import numpy as np
from data_utils.utils import create_graph
from data_utils.icosahedral_mesh import (
    faces_to_edges,
    get_hierarchy_of_triangular_meshes_for_sphere,
    merge_meshes,
)
from fire import Fire


def partition_graph(G: nx.Graph, num_ranks: int):
    metis_graph = metis.networkx_to_metis(G)
    (edgecuts, node_rank_placement) = metis.part_graph(metis_graph, nparts=num_ranks)
    # Node_rank_placement is of shape (num_nodes, ), where each element is the
    # rank of the node in the partitioning.

    node_rank_placement = np.array(node_rank_placement)
    return node_rank_placement


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


def create_graphcast_meshgraph(mesh_level=6):
    _meshes = get_hierarchy_of_triangular_meshes_for_sphere(splits=mesh_level)
    finest_mesh = _meshes[-1]  # get the last one in the list of meshes
    mesh = merge_meshes(_meshes)
    mesh_src, mesh_dst = faces_to_edges(mesh.faces)
    mesh_vertices = np.array(mesh.vertices)
    mesh_src = torch.tensor(mesh_src, dtype=torch.int32)
    mesh_dst = torch.tensor(mesh_dst, dtype=torch.int32)
    mesh_pos = torch.tensor(mesh_vertices, dtype=torch.float32)
    mesh_graph = create_graph(mesh_src, mesh_dst, mesh_pos, to_bidirected=True)
    return mesh_graph


def graphcast_graph_to_nxgraph(mesh_graph):
    G = nx.Graph()
    src_indices = mesh_graph[2].numpy()
    dst_indices = mesh_graph[3].numpy()
    edge_indices = np.stack((src_indices, dst_indices)).T
    G.add_edges_from(edge_indices)
    return G


def save_nx_meshgraph(nx_graph, filename):
    with open(filename, "wb") as f:
        pickle.dump(nx_graph, f)


def metis_partition_graph(nx_graph, num_ranks):
    mesh_vertex_rank_placement = partition_graph(nx_graph, num_ranks)

    # Renumber the mesh graph vertices and edges
    renumbered_nodes = node_renumbering(mesh_vertex_rank_placement, num_ranks)
    renumbered_edges = edge_renumbering(nx_graph.edges, renumbered_nodes)

    # Generate rank placement for the grid nodes

    # Recalculate thee Grid2Mesh and Mesh2Grid edges using
    # the new node numbering


def main(load_graph="", mesh_level=6, num_ranks=1):
    if load_graph:
        with open(load_graph, "rb") as f:
            G = pickle.load(f)
    else:
        mesh_graph = create_graphcast_meshgraph(mesh_level=mesh_level)
        G = graphcast_graph_to_nxgraph(mesh_graph)

    # if num_ranks > 1:


if __name__ == "__main__":
    Fire(main)
