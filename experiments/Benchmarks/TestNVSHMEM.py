import torch
import torch.distributed as dist
import argparse
from DGraph.Communicator import CommunicatorBase
import DGraph.Communicator as Comm
import numpy as np
import numpy.typing as npt
import torch.distributed as dist
from graph_utils import (
    GatherGraphData,
    ScatterGraphData,
    get_nvshmem_gather_benchmark_data,
    get_nvshmem_scatter_benchmark_data,
    safe_create_dir,
)
import os


class NVSHMEMBenchmark:
    def __init__(self, comm_object: CommunicatorBase, *args, **kwargs) -> None:
        super().__init__()
        assert comm_object._is_initialized, "Communicator not initialized"

        self.comm_object = comm_object
        self.rank = self.comm_object.get_rank()
        self.world_size = self.comm_object.get_world_size()

    def benchmark_gather(
        self, data, edge_src_rank, edge_indices, num_iters: int = 1000
    ) -> npt.NDArray:
        dist.barrier()
        _times = np.zeros(num_iters)
        stream = torch.cuda.Stream()

        for i in range(num_iters):
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record(stream)
            gathered_data = self.comm_object.gather(
                data,
                edge_src_rank,
                edge_indices,
            )
            end_time.record(stream)
            torch.cuda.synchronize()
            _times[i] = start_time.elapsed_time(end_time)

        return _times

    def benchmark_scatter(
        self,
        data,
        edge_dest_rank,
        edge_indices,
        num_local_vertices,
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
            scattered_data = self.comm_object.scatter(
                data,
                edge_dest_rank,
                edge_indices,
                num_local_vertices,
            )
            end_time.record(stream)
            torch.cuda.synchronize()
            _times[i] = start_time.elapsed_time(end_time)

        return _times

    def print(self, message):
        if self.rank == 0:
            print(message)

    def save_np(self, np_array: npt.NDArray, filename, rank_to_save=0):
        if self.rank == rank_to_save:
            np.save(filename, np_array)


def run_gather_benchmark(
    benchmark: NVSHMEMBenchmark,
    num_iters: int,
    gather_graph_data: GatherGraphData,
) -> npt.NDArray:
    benchmark.print("Running Gather Benchmark")

    data = gather_graph_data.vertex_data
    edge_src_rank = gather_graph_data.edge_src_rank
    edge_indices = gather_graph_data.edge_indices

    times = benchmark.benchmark_gather(
        data, edge_src_rank, edge_indices, num_iters=num_iters
    )
    benchmark.print("Finished Gather Benchmark")
    return times


def run_scatter_benchmark(
    benchmark: NVSHMEMBenchmark,
    num_iters: int,
    scatter_graph_data: ScatterGraphData,
) -> npt.NDArray:
    benchmark.print("Running Scatter Benchmark")

    data = scatter_graph_data.vertex_data
    edge_dest_rank = scatter_graph_data.edge_dest_rank
    edge_indices = scatter_graph_data.edge_indices
    num_local_vertices = scatter_graph_data.num_local_vertices

    times = benchmark.benchmark_scatter(
        data, edge_dest_rank, edge_indices, num_local_vertices, num_iters=num_iters
    )
    benchmark.print("Finished Scatter Benchmark")
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message_size", type=int, default=128)
    parser.add_argument("--benchmark_cache", action="store_true")
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    message_size = args.message_size
    num_iters = args.num_iters
    log_dir = args.log_dir

    comm = Comm.Communicator.init_process_group("nvshmem")
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{os.getcwd()}/DGraph_tmpfile",
    )

    safe_create_dir(log_dir, rank)

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    torch.set_default_device(device)

    benchmark = NVSHMEMBenchmark(comm_object=comm)
    benchmark.print("*" * 50)
    benchmark.print("Running Gather Benchmark")

    gather_graph_data = get_nvshmem_gather_benchmark_data(
        message_size, rank, world_size, device
    )
    times = run_gather_benchmark(benchmark, num_iters, gather_graph_data)

    benchmark.print("Saving Gather Benchmark Times")

    for i in range(world_size):
        benchmark.save_np(
            times, f"{log_dir}/NVSHMEM_gather_times_{i}.npy", rank_to_save=i
        )

    benchmark.print("Gather Benchmark Complete")
    benchmark.print("*" * 50)

    scatter_graph_data = get_nvshmem_scatter_benchmark_data(
        message_size, rank, world_size, device
    )

    benchmark.print("Running Scatter Benchmark")
    times = run_scatter_benchmark(benchmark, num_iters, scatter_graph_data)

    benchmark.print("Saving Scatter Benchmark Times")

    for i in range(world_size):
        benchmark.save_np(
            times, f"{log_dir}/NVSHMEM_scatter_times_{i}.npy", rank_to_save=i
        )

    benchmark.print("Scatter Benchmark Complete")
    benchmark.print("*" * 50)


if __name__ == "__main__":
    main()
