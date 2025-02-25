import torch
import torch.distributed as dist
from DGraph.distributed.nccl._nccl_cache import NCCLGatherCache, NCCLScatterCache
from typing import Optional, Union


def _nccl_alltoall_v(
    local_send_tensor: torch.Tensor,
    local_recv_tensor: torch.Tensor,
    indices: torch.Tensor,
    local_rank_mapping: torch.Tensor,
    edge_rank_loc: torch.Tensor,
    src_rank_loc: torch.Tensor,
    rank: int,
    world_size: int,
    cache: Optional[Union[NCCLGatherCache, NCCLScatterCache]] = None,
):
    num_features = local_send_tensor.shape[2]
    num_src_rows = local_send_tensor.shape[1]

    recv_buffer_dict = {}
    if cache is None:

    # These are all the rows participating in communication
    all_comm_mask = edge_rank_loc != src_rank_loc

    if all_comm_mask.sum() == 0:
        return local_recv_tensor
    # These ranks will send a message
    comm_senders = src_rank_loc[all_comm_mask]
    # These ranks will recieve a message
    comm_receivers = edge_rank_loc[all_comm_mask]

    # Current rank will send to these ranks
    send_to_ranks = comm_receivers[comm_senders == rank]

    # Current rank will receive from these ranks
    receive_from_ranks = comm_senders[comm_receivers == rank]
    send_comm_vector = torch.bincount(send_to_ranks, minlength=world_size).long()
    recv_comm_vector = torch.bincount(receive_from_ranks, minlength=world_size).long()
    recv_local_placement = {}
    send_local_placement = {}

    for i, num_messages in enumerate(recv_comm_vector):
        if num_messages == 0:
            continue

        if i == rank:
            continue

        recv_buffer = torch.zeros(1, int(num_messages.item()), num_features).to(
            local_send_tensor.device
        )
        recv_buffer_dict[i] = recv_buffer
        _local_placement_indices = torch.argwhere(local_rank_mapping == i)
        assert (
            _local_placement_indices.shape[0] == num_messages
        ), f"{i} {(local_rank_mapping == i).sum()} {_local_placement_indices.shape} {num_messages} {local_rank_mapping.shape}"
        recv_local_placement[i] = _local_placement_indices

    for i, num_messages in enumerate(send_comm_vector):
        if num_messages == 0:
            # Not sending any messages current_rank to rank i
            continue

        if i == rank:
            continue

        _mask = (src_rank_loc == rank) & (edge_rank_loc == i)
        _send_row = indices[0][_mask] % num_src_rows
        send_local_placement[i] = _send_row

    p2p_op_list = []
    for send_rank_index in range(world_size):
        for recv_rank_index in range(world_size):
            if send_rank_index == recv_rank_index:
                # No self-sends allowed. Should be done in the local gather.
                continue
            if (send_rank_index != rank) and (recv_rank_index != rank):
                # Current rank not involved in this p2p communication pair.
                continue

            if send_rank_index == rank:
                # Current rank is sending data to recv_rank_index
                if send_comm_vector[recv_rank_index].item() == 0:
                    continue
                send_tensor = local_send_tensor[
                    :, send_local_placement[recv_rank_index], :
                ]
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, recv_rank_index))

            if recv_rank_index == rank:
                if recv_comm_vector[send_rank_index].item() == 0:
                    # print(f"Rank {rank}")
                    continue
                recv_tensor = recv_buffer_dict[send_rank_index]
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, send_rank_index))

    if len(p2p_op_list) > 0:
        reqs = dist.batch_isend_irecv(p2p_op_list)

        for req in reqs:
            req.wait()

    for key, recv_buffer in recv_buffer_dict.items():

        local_recv_tensor[:, recv_local_placement[key].view(-1), :] = (
            recv_buffer.unsqueeze(0)
        )

    return local_recv_tensor


def _nccl_alltoallv_with_dict(send_buffer_dict, recv_buffer_dict, rank, world_size):
    p2p_op_list = []
    for _sender in range(world_size):
        for _receiver in range(world_size):
            if _sender == _receiver:
                continue
            if (_sender != rank) and (_receiver != rank):
                continue

            if _sender == rank:
                if _receiver in send_buffer_dict:
                    send_buffer = send_buffer_dict[_receiver]
                    p2p_op_list.append(dist.P2POp(dist.isend, send_buffer, _receiver))

            if _receiver == rank:
                if _sender in recv_buffer_dict:
                    recv_buffer = recv_buffer_dict[_sender]
                    p2p_op_list.append(dist.P2POp(dist.irecv, recv_buffer, _sender))

    # print(p2p_op_list)
    if len(p2p_op_list) > 0:
        reqs = dist.batch_isend_irecv(p2p_op_list)

        for req in reqs:
            req.wait()
    for key, recv_buffer in recv_buffer_dict.items():
        recv_buffer_dict[key] = recv_buffer.float()
    return recv_buffer_dict
