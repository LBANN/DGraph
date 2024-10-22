import torch.distributed as dist


def check_dist_initialized():
    if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
    if not dist.is_initialized():
        raise RuntimeError("Requires distributed package to be initialized")


def check_nccl_availability():
    if not dist.is_nccl_available():
        raise RuntimeError("Requires NCCL backend")
