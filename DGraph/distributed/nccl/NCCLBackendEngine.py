import torch
import torch.distributed as dist
from DGraph.distributed.Engine import BackendEngine
from DGraph.distributed.nccl.gather_op_impl import _nccl_gather_op
from DGraph.distributed.nccl.scatter_op_impl import _nccl_scatter_op
from torch.autograd import Function


class GatherFunction(Function):
    @staticmethod
    def forward(ctx, send_tensor, recv_tensor, indices):
        ctx.save_for_backward(send_tensor, recv_tensor, indices)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_gather_op(send_tensor, recv_tensor, indices, rank, world_size)
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        send_tensor, recv_tensor, indices = ctx.saved_tensors
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output


class ScatterFunction(Function):
    @staticmethod
    def forward(
        ctx, send_tensor: torch.Tensor, recv_tensor: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(send_tensor, recv_tensor, indices)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_scatter_op(send_tensor, recv_tensor, indices, rank, world_size)
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output) -> torch.Tensor:
        send_tensor, recv_tensor, indices = ctx.saved_tensors
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output


def scatter(
    send_tensor: torch.Tensor, recv_tensor: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return ScatterFunction.apply(send_tensor, recv_tensor, indices)  # type: ignore


def gather(send_tensor, recv_tensor, indices) -> torch.Tensor:
    return GatherFunction.apply(send_tensor, recv_tensor, indices)  # type: ignore


class NCCLBackendEngine(BackendEngine):
    def __init__(self, *args, **kwargs):
        # check if already initialized
        self._initialized = dist.is_initialized()

    def init_process_group(self, *args, **kwargs):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", *args, **kwargs)

            self._initialized = True

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_world_size(self) -> int:
        return dist.get_world_size()

    def scatter(self, *args, **kwargs) -> torch.Tensor:
        input_tensor: torch.Tensor = args[0]
        indices: torch.Tensor = args[1]
        local_size: int = args[2]
        batch_size: int = input_tensor.shape[0]
        feature_size: int = input_tensor.shape[2]

        output_tensor: torch.Tensor = torch.zeros(
            (
                batch_size,
                local_size,
                feature_size,
            )
        )
        return scatter(input_tensor, output_tensor, indices)

    def gather(self, *args, **kwargs) -> torch.Tensor:
        input_tensor = args[0]
        indices = args[1]
        indices_shape = indices.shape
        b_size = indices_shape[0]
        n = indices_shape[1]
        feature_size = input_tensor.shape[2]
        output_tensor = torch.zeros(b_size, n, feature_size, device=input_tensor.device)
        return gather(input_tensor, output_tensor, indices)
