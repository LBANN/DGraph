import torch

from DGraph import Communicator
from torch.autograd import Function
from DGraph.distributed.commInfo import CommunicationPattern


class HaloExchangeImpl(Function):
    @staticmethod
    def forward(
        ctx, send_buffer, comm, comm_pattern: CommunicationPattern
    ) -> torch.Tensor:
        # Allocate the receive buffer
        feature_dim = send_buffer.shape[1] if send_buffer.ndim == 2 else 1
        # Allocate the receive buffer
        total_recv = int(comm_pattern.recv_offset[-1].item())

        ctx.comm_pattern = comm_pattern
        ctx.feature_dim = feature_dim
        ctx.comm = comm
        recv_buffer = comm.allocate_buffer(
            (total_recv, feature_dim) if send_buffer.ndim == 2 else (total_recv,),
            dtype=send_buffer.dtype,
            device=send_buffer.device,
        )
        # TODO: complete
        send_offsets = comm_pattern.send_offset
        recv_offsets = comm_pattern.recv_offset
        put_forward_remote_offset = comm_pattern.put_forward_remote_offset
        comm.put(
            send_buffer,
            recv_buffer,
            send_offsets,
            recv_offsets,
            remote_offsets=put_forward_remote_offset,
        )

        return recv_buffer

    @staticmethod
    def backward(ctx, grad_recv_buffer):
        total_sent = ctx.comm_pattern.send_offset[-1].item()
        feature_dim = ctx.feature_dim
        comm = ctx.comm

        grad_input_tensor = comm.allocate_buffer(
            (total_sent, feature_dim) if grad_recv_buffer.ndim == 2 else (total_sent,),
            dtype=grad_recv_buffer.dtype,
            device=grad_recv_buffer.device,
        )
        send_offsets = ctx.comm_pattern.send_offset
        recv_offsets = ctx.comm_pattern.recv_offset
        put_backward_remote_offset = ctx.comm_pattern.put_backward_remote_offset
        comm.put(
            grad_recv_buffer,
            grad_input_tensor,
            recv_offsets,
            send_offsets,
            remote_offsets=put_backward_remote_offset,
        )

        return grad_input_tensor, None, None


class HaloExchange:
    """Halo exchange class for communication backends."""

    def __init__(self, comm: Communicator):
        self.comm = comm

    def __call__(
        self, x_local: torch.Tensor, comm_pattern: CommunicationPattern
    ) -> torch.Tensor:
        send_buffer = x_local[comm_pattern.send_local_idx]
        recv_buffer = HaloExchangeImpl.apply(send_buffer, self.comm, comm_pattern)
        return recv_buffer  # type: ignore
