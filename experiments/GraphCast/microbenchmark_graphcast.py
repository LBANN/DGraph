import torch

import torch.distributed as dist
from dataset import SyntheticWeatherDataset
from layers import MeshGraphMLP
from DGraph.Communicator import Communicator
from DGraph.distributed.nccl import TIMINGS
from DGraph.distributed.nccl._nccl_cache import (
    NCCLGatherCacheGenerator,
    NCCLScatterCache,
    NCCLScatterCacheGenerator,
    NCCLGatherCache,
)
from tqdm import tqdm
import numpy as np
import nvtx


def load_data(rank, world_size, num_days=2, batch_size=1):

    latlon_res = (721, 1440)
    num_samples_per_year_train = num_days
    num_workers = 8
    num_channels_climate = 73
    num_history = 0
    dt = 6.0
    start_year = 1980
    use_time_of_year_index = True
    channels_list = [i for i in range(num_channels_climate)]

    cos_zenith_args = {
        "dt": dt,
        "start_year": start_year,
    }
    mesh_vertex_placement = torch.load("mesh_vertex_rank_placement_4.pt")
    test_dataset = SyntheticWeatherDataset(
        channels=channels_list,
        num_samples_per_year=num_samples_per_year_train,
        num_steps=1,
        grid_size=latlon_res,
        cos_zenith_args=cos_zenith_args,
        batch_size=batch_size,
        num_workers=num_workers,
        num_history=num_history,
        use_time_of_year_index=use_time_of_year_index,
        mesh_vertex_placement=mesh_vertex_placement,
        world_size=world_size,
        rank=rank,
    )

    static_graph = test_dataset.get_static_graph()
    if rank == 0:
        print(len(test_dataset))
        print("=" * 80)
        print("Mesh src indices:\t", static_graph.mesh_graph_src_indices.shape)
        print("Mesh dst indices:\t", static_graph.mesh_graph_dst_indices.shape)
        print("Mesh src placement:\t", static_graph.mesh_graph_src_rank_placement.shape)
        print("Mesh dst placement:\t", static_graph.mesh_graph_dst_rank_placement.shape)

    return static_graph


def time_mesh_edge_block(
    local_vertex_features,
    local_edge_features,
    edge_src_indices,
    edge_dst_indices,
    edge_rank_placement,
    edge_src_rank,
    edge_dst_rank,
    comm,
    iters=100,
    inp_dim=512,
    out_dim=512,
    hidden_dim=512,
):

    mesh_mlp = MeshGraphMLP(
        input_dim=inp_dim + inp_dim,
        output_dim=out_dim,
        hidden_dim=hidden_dim,
        hidden_layers=1,
    )
    mesh_mlp = mesh_mlp.cuda()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    comm_times = torch.zeros(iters)
    process_times = torch.zeros(iters)

    rank = dist.get_rank()

    main_process = rank == 0

    num_input_rows = local_vertex_features.shape[1]
    src_gather_cache = NCCLGatherCacheGenerator(
        edge_src_indices,
        edge_rank_placement.view(-1),
        edge_src_rank.view(-1),
        num_input_rows,
        rank,
        comm.get_world_size(),
    )
    dst_gather_cache = NCCLGatherCacheGenerator(
        edge_dst_indices,
        edge_rank_placement.view(-1),
        edge_dst_rank.view(-1),
        num_input_rows,
        rank,
        comm.get_world_size(),
    )

    # if rank == 0:
    #     breakpoint()
    dist.barrier()
    if rank == 0:
        gather_forward_times = TIMINGS["Gather_Index_Forward"]
        print(f"Gather forward times: {gather_forward_times} ms")
    dist.barrier()

    local_edges = edge_rank_placement == rank
    local_edge_src_indices = edge_src_indices[local_edges]

    num_features = local_vertex_features.shape[-1]

    with nvtx.annotate("mesh_edge", color="green"):
        for i in tqdm(range(iters), disable=not main_process):
            start_comm_time = torch.cuda.Event(enable_timing=True)
            end_comm_time = torch.cuda.Event(enable_timing=True)
            start_process_time = torch.cuda.Event(enable_timing=True)
            end_process_time = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_comm_time.record(stream)

            # src_node_features = comm.gather(
            #     local_vertex_features,
            #     edge_src_indices,
            #     torch.cat([edge_rank_placement, edge_src_rank], dim=0),
            #     cache=src_gather_cache,
            # )

            src_node_features = torch.gather(
                local_vertex_features,
                1,
                local_edge_src_indices.view(1, -1, 1).expand(-1, -1, num_features)
                % num_input_rows,
            )

            dst_node_features = src_node_features
            # dst_node_features = comm.gather(
            #     local_vertex_features,
            #     edge_dst_indices,
            #     torch.cat([edge_rank_placement, edge_dst_rank], dim=0),
            #     cache=dst_gather_cache,
            # )

            end_comm_time.record(stream)
            dist.barrier()
            torch.cuda.synchronize()
            start_process_time.record(stream)
            concatenated_features = torch.cat(
                [src_node_features, dst_node_features], dim=-1
            )
            out = mesh_mlp(concatenated_features) + local_edge_features

            end_process_time.record(stream)
            torch.cuda.synchronize()
            dist.barrier()
            comm_times[i] = start_comm_time.elapsed_time(end_comm_time)
            process_times[i] = start_process_time.elapsed_time(end_process_time)
    dist.barrier()

    if rank == 0:
        gather_forward_times = TIMINGS["Gather_Index_Forward"]
        gather_forward_times = np.array(gather_forward_times)

        local_gather_times = TIMINGS["Gather_Forward_Local"]
        local_gather_times = np.array(local_gather_times)
        print(f"Local gather times: {local_gather_times[:10].mean()} ms")
        print(f"Local index calculation times: {gather_forward_times[:10].mean()} ms")
    return comm_times.numpy(), process_times.numpy()


def main():
    dist.init_process_group(backend="nccl")
    comm = Communicator.init_process_group("nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f"Rank {rank} out of {world_size} is running the benchmark.")

    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank} is using device {device}.")
    torch.cuda.set_device(device)

    # Load the data
    static_graph = load_data(rank, world_size)

    num_local_vertices = int(
        (static_graph.mesh_graph_node_rank_placement == rank).sum().item()
    )

    num_local_edges = int(
        (static_graph.mesh_graph_edge_rank_placement == rank).sum().item()
    )
    num_hidden_features = 512

    local_vertex_data = torch.randn(
        1, num_local_vertices, num_hidden_features, device=device
    )
    local_edge_data = torch.randn(
        1, num_local_edges, num_hidden_features, device=device
    )
    edge_rank_placement = static_graph.mesh_graph_edge_rank_placement.view(1, -1).to(
        device
    )
    edge_src_rank = static_graph.mesh_graph_src_rank_placement.view(1, -1).to(device)
    edge_dst_rank = static_graph.mesh_graph_dst_rank_placement.view(1, -1).to(device)
    src_edge_indices = static_graph.mesh_graph_src_indices.view(1, -1).to(device)
    dst_edge_indices = static_graph.mesh_graph_dst_indices.view(1, -1).to(device)

    # Time the mesh edge block
    comm_times, process_times = time_mesh_edge_block(
        local_vertex_data,
        local_edge_data,
        src_edge_indices,
        dst_edge_indices,
        edge_rank_placement,
        edge_src_rank,
        edge_dst_rank,
        comm,
        iters=100,
    )

    if rank == 0:
        # Save the results
        np.save("comm_times_mesh_edge_block.npy", comm_times)
        np.save("process_times_mesh_edge_block.npy", process_times)

        warmup_iters = 10
        comm_times = comm_times[warmup_iters:]
        process_times = process_times[warmup_iters:]
        print(f"Communication time: {np.mean(comm_times)} +/- {np.std(comm_times)} ms")
        print(
            f"Mean processing time: {np.mean(process_times)}+/- {np.std(process_times)} ms"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
