import DGraph.Communicator as Comm
import torch.distributed as dist
import torch
import DGraph.torch_nvshmem_p2p as nvshmem


def main():
    comm = Comm.Communicator.init_process_group("nvshmem")
    rank = comm.get_rank()
    world_size = comm.get_world_size()

    dist.barrier()
    assert dist.is_initialized(), "NCCL process group not initialized"
    assert dist.get_backend() == "nccl", "NCCL process group not initialized"
    assert dist.get_rank() == rank, "NCCL process group rank mismatch"
    assert dist.get_world_size() == world_size, "NCCL process group world size mismatch"

    dist.barrier()
    for i in range(world_size):
        if rank == i:
            print(
                f"Rank {rank} checking in. ",
                f"Number of available GPUs: {torch.cuda.device_count()}",
            )
        dist.barrier()

    # Set device for this process
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Allocate a tensor on the GPU
    num_elements = world_size
    nvshmem_tensor = nvshmem.NVSHMEMP2P.allocate_symmetric_memory(
        num_elements, local_rank
    )
    nvshmem_tensor.fill_(rank).float()

    dist.barrier()
    for i in range(world_size):
        if rank == i:
            print(
                f"Rank {rank}: ",
                f"Tensor: {nvshmem_tensor}",
            )
        dist.barrier()
    assert torch.allclose(
        nvshmem_tensor, torch.full((num_elements,), rank).cuda().float()
    ), "Tensor values do not match expected values"

    indices = torch.arange(num_elements, dtype=torch.int64).cuda()
    output_tensor = nvshmem.NVSHMEMP2P.allocate_symmetric_memory(
        num_elements, local_rank
    )
    output_tensor.fill_(0).float()
    ranks = torch.arange(world_size).cuda()
    nvshmem.NVSHMEMP2P.dist_get(
        nvshmem_tensor, output_tensor, indices, ranks, 1, num_elements, 1, num_elements
    )
    dist.barrier()
    for i in range(world_size):
        if rank == i:
            print(
                f"Rank {rank}: ",
                f"Output Tensor: {output_tensor}",
            )
        dist.barrier()

    assert torch.allclose(
        output_tensor, torch.arange(world_size).cuda().float()
    ), "Output tensor values do not match expected values"


if __name__ == "__main__":
    main()
