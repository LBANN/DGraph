import torch.distributed as dist
from DGraph.utils import largest_split
from DGraph.distributed.nccl.scatter_op_impl import _nccl_scatter_op
from torch.autograd import Function


def _nccl_scatter_op(send_tensor_buffer, recv_tensor_buffer, indices, rank, world_size):
    p2p_op_list = []
    mb_size = send_tensor_buffer.shape[0]
    num_input_rows = largest_split(send_tensor_buffer.shape[1], world_size)
    num_output_rows = largest_split(recv_tensor_buffer.shape[1], world_size)

    for mb in range(mb_size):
        for i, ind_i in enumerate(indices[mb]):
            src_rank = i // num_input_rows
            recv_rank = ind_i // num_output_rows
            send_tensor = send_tensor_buffer[mb, i % num_input_rows]
            recv_tensor = recv_tensor_buffer[mb, ind_i % num_output_rows]

            if src_rank == recv_rank:
                continue

            if rank == src_rank:
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, recv_rank))

            if rank == recv_rank:
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, src_rank))
    reqs = dist.batch_isend_irecv(p2p_op_list)

    for req in reqs:
        req.wait()


class ScatterFunction(Function):
    @staticmethod
    def forward(ctx, send_tensor, recv_tensor, indices):
        ctx.save_for_backward(send_tensor, recv_tensor, indices)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_scatter_op(send_tensor, recv_tensor, indices, rank, world_size)
        return recv_tensor

    @staticmethod
    def backward(ctx, grad_output):
        send_tensor, recv_tensor, indices = ctx.saved_tensors
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _nccl_scatter_op(grad_output, send_tensor, indices, rank, world_size)
        return grad_output, None, None


def scatter(send_tensor, recv_tensor, indices):
    return ScatterFunction.apply(send_tensor, recv_tensor, indices)
