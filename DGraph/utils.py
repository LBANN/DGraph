import torch.distributed as dist


def largest_split(global_size, world_size):
    return (global_size + world_size) // world_size


def split_per_rank(global_size, current_rank, world_size):
    _split = largest_split(global_size, world_size)
    if current_rank != world_size - 1:
        return _split
    else:
        return global_size - (current_rank * _split)


def try_barrier():
    """Attempt a barrier but ignore any exceptions"""
    try:
        dist.barrier()
    except:
        pass
