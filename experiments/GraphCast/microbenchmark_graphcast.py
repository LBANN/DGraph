import torch

import torch.distributed as dist
from dataset import SyntheticWeatherDataset
from layers import MeshGraphMLP
from DGraph.Communicator import Communicator


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
    print(len(test_dataset))
    print("=" * 80)
    static_graph = test_dataset.get_static_graph()

    print("Mesh src indices:\t", static_graph.mesh_graph_src_indices.shape)
    print("Mesh dst indices:\t", static_graph.mesh_graph_dst_indices.shape)
    print("Mesh src placement:\t", static_graph.mesh_graph_src_rank_placement.shape)
    print("Mesh dst placement:\t", static_graph.mesh_graph_dst_rank_placement.shape)


def time_mesh_edge_block(comm, iters=100):

    mesh_mlp = MeshGraphMLP(
        input_dim=2,
        output_dim=2,
        hidden_dim=2,
        hidden_layers=1,
    )
    mesh_mlp = mesh_mlp.cuda()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    comm_times = torch.zeros(iters)
    process_times = torch.zeros(iters)
    for i in range(iters):
        start_comm_time = torch.cuda.Event(enable_timing=True)
        end_comm_time = torch.cuda.Event(enable_timing=True)
        start_process_time = torch.cuda.Event(enable_timing=True)
        end_process_time = torch.cuda.Event(enable_timing=True)
        start_comm_time.record(stream)

        src_node_features = comm.gather()
        dst_node_features = comm.gather()

        end_comm_time.record(stream)
        start_process_time.record(stream)
        concatenated_features = torch.cat(
            [src_node_features, dst_node_features], dim=-1
        )
        edge_features = mesh_mlp(concatenated_features) + edge_features
        end_process_time.record(stream)
        torch.cuda.synchronize()
        dist.barrier()
        comm_times[i] = start_comm_time.elapsed_time(end_comm_time)
        process_times[i] = start_process_time.elapsed_time(end_process_time)
    return comm_times.numpy(), process_times.numpy()


def main():
    dist.init_process_group(backend="nccl")
    comm = Communicator.init_process_group("nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print(f"Rank {rank} out of {world_size} is running the benchmark.")

    # Load the data
    load_data(rank, world_size)


if __name__ == "__main__":
    main()
