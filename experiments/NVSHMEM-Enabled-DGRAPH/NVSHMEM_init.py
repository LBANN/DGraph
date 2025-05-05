import DGraph.Communicator as Comm
import torch.distributed as dist


def main():
    comm = Comm.Communicator.init_process_group("nvshmem")
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    assert dist.is_initialized(), "NCCL process group not initialized"
    assert dist.get_backend() == "nccl", "NCCL process group not initialized"
    assert dist.get_rank() == rank, "NCCL process group rank mismatch"
    assert dist.get_world_size() == world_size, "NCCL process group world size mismatch"

    for i in range(world_size):
        if rank == i:
            print(f"Rank {rank} checking in")
        dist.barrier()


if __name__ == "__main__":
    main()
