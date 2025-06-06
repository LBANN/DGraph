import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import concurrent.futures

args = argparse.ArgumentParser()
args.add_argument("--g", type=str, default="default.graph", help="graph file name")
args.add_argument(
    "--p", type=str, default="partition.graph.N", help="partition file name"
)
args.add_argument("--np", type=int, default=8, help="number of partitions")


def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


args = args.parse_args()
graph_file = args.g
partition_file = args.p
num_partitions = args.np


def reorder_vertices(partition_file):
    """
    Reorder vertices in the partition file to ensure that the vertex IDs are continuous.
    """
    vertex_rank_placement = []
    with open(partition_file, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            vertex_rank_placement.append(int(line))
    # This sorts the vertices so that the vertex IDs are continuous in each
    # partition.
    vertex_rank_placement = torch.from_numpy(np.array(vertex_rank_placement))
    sorted_rank_placement, sorted_indices = torch.sort(vertex_rank_placement)

    # We need the inverse mapping so we can take the COO list and map it to the new vertex IDs.
    # This is the inverse mapping from the sorted indices to the original indices.

    _, reverse_maps = torch.sort(sorted_indices)

    return sorted_rank_placement, sorted_indices, reverse_maps


def process_chunk(process_local_adj_list, start_index, end_index, reverse_map):
    local_coo_list = []
    for local_idx, line in enumerate(process_local_adj_list):
        glocal_idx = start_index + local_idx
        src_vertex = reverse_map[glocal_idx].item()
        for dst_vertex in line.split():
            dst_vertex = reverse_map[int(dst_vertex) - 1].item()
            local_coo_list.append((src_vertex, dst_vertex))
    return local_coo_list


def reorder_edge_list(graph_file, reverse_map):

    # This file is quite large usually, so try to do multiprocessing
    adj_list = []

    with open(graph_file, "r") as f:
        first_line = f.readline()
        num_vertices, num_edges = map(int, first_line.strip().split())
        assert num_vertices == len(reverse_map)

        for i in tqdm(range(num_vertices)):
            line = f.readline()
            line = line.strip()
            adj_list.append(line)

    num_cpus = os.cpu_count() or 1
    # num_workers = max(num_cpus - 2, 1)
    num_workers = 8
    chunk_size = max(1, num_vertices // num_workers)

    worker_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, num_vertices, chunk_size):
            start_index = i
            end_index = min(i + chunk_size, num_vertices)
            worker_results.append(
                executor.submit(
                    process_chunk,
                    adj_list[start_index:end_index],
                    start_index,
                    end_index,
                    reverse_map,
                )
            )

        for future in tqdm(concurrent.futures.as_completed(worker_results)):
            result = future.result()
            if result is not None:
                worker_results.extend(result)
            else:
                print("Worker failed to process chunk.")

    coo_list = torch.tensor(
        np.array(worker_results), dtype=torch.int64
    )  # shape: (num_edges, 2)
    assert coo_list.shape[1] == 2
    assert coo_list.shape[0] == 2 * num_edges

    #     src_vertex = reverse_map[i]
    #     for dst_vertex in line.split():

    #         dst_vertex = reverse_map[int(dst_vertex) - 1]
    #         coo_list[edge_counter, 0] = src_vertex
    #         coo_list[edge_counter, 1] = dst_vertex
    #         edge_counter += 1
    # assert edge_counter == 2 * num_edges

    return coo_list


def main():
    # Reorder vertices in the partition file
    # safe_mkdir("processed")
    # sorted_rank_placement, sorted_indices, reverse_maps = reorder_vertices(
    #     partition_file
    # )
    # torch.save(
    #     sorted_rank_placement,
    #     os.path.join("processed", f"sorted_rank_placement_{num_partitions}.pt"),
    # )

    # torch.save(
    #     sorted_indices, os.path.join("processed", f"forward_map_{num_partitions}.pt")
    # )
    # torch.save(
    #     reverse_maps, os.path.join("processed", f"reverse_map_{num_partitions}.pt")
    # )
    sorted_rank_placement = None
    sorted_indices = None

    reverse_maps = torch.load(
        os.path.join("processed", f"reverse_map_{num_partitions}.pt")
    )
    # Reorder the edge list in the graph file
    coo_list = reorder_edge_list(graph_file, reverse_maps)
    torch.save(
        coo_list,
        os.path.join("processed", f"edges_{num_partitions}.pt"),
    )


if __name__ == "__main__":
    main()
