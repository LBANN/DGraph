from typing import Optional
import torch
import torch.distributed as dist
import argparse
from DGraph.Communicator import Communicator
import numpy as np
import numpy.typing as npt
from DGraph.distributed.nccl._nccl_cache import (
    NCCLGatherCacheGenerator,
    NCCLScatterCache,
    NCCLScatterCacheGenerator,
    NCCLGatherCache,
)
from graph_utils import (
    GatherGraphData,
    ScatterGraphData,
    get_nccl_gather_benchmark_data,
    get_nccl_scatter_benchmark_data,
    safe_create_dir,
)


class NCCLBenchmark:
    def __init__(self, comm, world_size, rank):
        self.comm = comm
        self.world_size = world_size
        self.rank = rank

    def comm_stats_gather(self, edge_rank_placement, edge_src_rank):
        local_ranks = edge_rank_placement == self.rank

        # Number of messages recevied by the local rank
        local_comm_rev = (edge_src_rank[local_ranks] != self.rank).sum()

        # Number of messages sent by the local rank
        local_comm_send = (edge_src_rank[~local_ranks] == self.rank).sum()

        return {"local_comm_rev": local_comm_rev, "local_comm_send": local_comm_send}

    def benchmark_gather(
        self,
        data,
        edge_rank_placement,
        edge_src_rank,
        edge_indices,
        cache: Optional[NCCLGatherCache] = None,
        num_iters: int = 1000,
    ) -> npt.NDArray:
        dist.barrier()
        _times = np.zeros(num_iters)
        stream = torch.cuda.Stream()

        for i in range(num_iters):
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record(stream)
            edge_placement_data = torch.cat([edge_rank_placement, edge_src_rank], dim=0)
            gathered_data = self.comm.gather(
                data, edge_indices, edge_placement_data, cache=cache
            )
            end_time.record(stream)
            torch.cuda.synchronize()
            dist.barrier()
            _iter_time = start_time.elapsed_time(end_time)
            _times[i] = _iter_time

        return _times

    def benchmark_scatter(
        self,
        data,
        edge_rank_placement,
        edge_dest_rank,
        edge_indices,
        num_local_vertices,
        cache: Optional[NCCLScatterCache] = None,
        num_iters: int = 1000,
    ) -> npt.NDArray:
        dist.barrier()
        _times = np.zeros(num_iters)
        stream = torch.cuda.Stream()

        for i in range(num_iters):
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record(stream)
            edge_placement_data = torch.cat(
                [edge_rank_placement, edge_dest_rank], dim=0
            )
            scattered_data = self.comm.scatter(
                data, edge_indices, edge_placement_data, num_local_vertices, cache=cache
            )
            end_time.record(stream)
            torch.cuda.synchronize()
            dist.barrier()
            _iter_time = start_time.elapsed_time(end_time)
            _times[i] = _iter_time

        return _times

    def print(self, message):
        if self.rank == 0:
            print(message)

    def save_np(self, np_array: npt.NDArray, filename, rank_to_save=0):
        if self.rank == rank_to_save:
            np.save(filename, np_array)


def run_gather_benchmark(
    benchmark: NCCLBenchmark,
    num_iters: int,
    graph_data: GatherGraphData,
    cache: Optional[NCCLGatherCache] = None,
):
    rank = benchmark.rank
    vertex_data = graph_data.vertex_data
    vertex_mapping = graph_data.vertex_rank_mapping
    local_vertex_data = vertex_data[vertex_mapping == rank, :].unsqueeze(0)
    edge_placement = graph_data.edge_rank_placement
    edge_src_rank = graph_data.edge_src_rank
    edge_indices = graph_data.edge_indices

    times = benchmark.benchmark_gather(
        local_vertex_data,
        edge_placement,
        edge_src_rank,
        edge_indices,
        num_iters=num_iters,
        cache=cache,
    )

    return times


def run_scatter_benchmark(
    benchmark: NCCLBenchmark,
    num_iters: int,
    graph_data: ScatterGraphData,
    cache: Optional[NCCLScatterCache] = None,
):
    rank = benchmark.rank
    vertex_data = graph_data.vertex_data
    vertex_mapping = graph_data.data_rank_mapping
    local_vertex_data = vertex_data[vertex_mapping == rank].unsqueeze(0)
    edge_placement = graph_data.edge_rank_placement
    edge_src_rank = graph_data.edge_dest_rank
    edge_indices = graph_data.edge_indices
    num_local_vertices = graph_data.num_local_vertices
    times = benchmark.benchmark_scatter(
        local_vertex_data,
        edge_placement,
        edge_src_rank,
        edge_indices,
        num_local_vertices,
        num_iters=num_iters,
        cache=cache,
    )

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message_size", type=int, default=128)
    parser.add_argument("--benchmark_cache", action="store_true")
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    message_size = args.message_size
    benchmark_cache = args.benchmark_cache
    num_iters = args.num_iters
    log_dir = args.log_dir

    dist.init_process_group(backend="nccl")
    comm = Communicator.init_process_group("nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    safe_create_dir(log_dir, rank)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    torch.set_default_device(device)

    benchmark = NCCLBenchmark(comm, world_size, rank)
    benchmark.print(f"Running NCCL Benchmark on {world_size} ranks")

    # Built in small message benchmarks, in future we can add more
    gather_graph_data = get_nccl_gather_benchmark_data(message_size, world_size, device)

    benchmark.print("*" * 50)
    benchmark.print("Running Gather Benchmark")
    times = run_gather_benchmark(benchmark, num_iters, gather_graph_data, cache=None)

    benchmark.print("Saving Gather Benchmark Times")

    for i in range(world_size):
        benchmark.save_np(times, f"{log_dir}/gather_times_{i}.npy", rank_to_save=i)

    benchmark.print("Gather Benchmark Complete")
    benchmark.print("*" * 50)

    if benchmark_cache:
        vertex_data = gather_graph_data.vertex_data
        edge_placement = gather_graph_data.edge_rank_placement
        edge_src_rank = gather_graph_data.edge_src_rank

        gather_cache = NCCLGatherCacheGenerator.generate_cache(
            vertex_data, edge_placement, edge_src_rank, rank, world_size
        )
        benchmark.print("*" * 50)
        benchmark.print("Running Gather Benchmark with Cache")
        times = run_gather_benchmark(
            benchmark, num_iters, gather_graph_data, cache=gather_cache
        )

        benchmark.print("Saving Gather Benchmark with Cache Times")
        for i in range(world_size):
            benchmark.save_np(
                times, f"{log_dir}/gather_with_cache_times_{i}.npy", rank_to_save=i
            )

        benchmark.print("Gather Benchmark with Cache Complete")
        benchmark.print("*" * 50)

    scatter_graph_data = get_nccl_scatter_benchmark_data(
        message_size, world_size, device
    )
    benchmark.print("*" * 50)
    benchmark.print("Running Scatter Benchmark")
    times = run_scatter_benchmark(benchmark, num_iters, scatter_graph_data, cache=None)

    benchmark.print("Saving Scatter Benchmark Times")
    for i in range(world_size):
        benchmark.save_np(times, f"{log_dir}/scatter_times_{i}.npy", rank_to_save=i)

    benchmark.print("Scatter Benchmark Complete")
    benchmark.print("*" * 50)
    if benchmark_cache:
        vertex_data = scatter_graph_data.vertex_data
        edge_placement = scatter_graph_data.edge_rank_placement
        edge_dest_rank = scatter_graph_data.edge_dest_rank

        scatter_cache = NCCLScatterCacheGenerator.generate_cache(
            vertex_data, edge_placement, edge_dest_rank, rank, world_size
        )
        benchmark.print("*" * 50)
        benchmark.print("Running Scatter Benchmark with Cache")
        times = run_scatter_benchmark(
            benchmark, num_iters, scatter_graph_data, cache=scatter_cache
        )

        benchmark.print("Saving Scatter Benchmark with Cache Times")
        for i in range(world_size):
            benchmark.save_np(
                times, f"{log_dir}/scatter_with_cache_times_{i}.npy", rank_to_save=i
            )

        benchmark.print("Scatter Benchmark with Cache Complete")
        benchmark.print("*" * 50)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
