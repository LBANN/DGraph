from typing import Dict, Tuple
import torch


def _get_send_comm_vector(
    comm_senders: torch.Tensor,
    comm_receivers: torch.Tensor,
    rank: int,
    world_size: int,
):
    """Returns the number of messages to be sent to each rank based on the
    src and dest ranks.

    Args:
        src_ranks (torch.Tensor): The source ranks of the messages
        dest_ranks (torch.Tensor): The destination ranks of the messages
        rank (int): The current rank
        world_size (int): The total number of ranks

    Returns:
        torch.Tensor: The number of messages to be sent to each rank.
                      Shape: (world_size,)
    """
    send_to_ranks = comm_receivers[comm_senders == rank]
    send_comm_vector = torch.bincount(send_to_ranks, minlength=world_size).long()
    return send_comm_vector


def _get_recv_comm_vector(
    comm_senders: torch.Tensor,
    comm_receivers: torch.Tensor,
    rank: int,
    world_size: int,
):
    """Returns the number of messages to be received from each rank based on the
    src and dest ranks.

    Args:
        comm_senders (torch.Tensor): The source ranks of the messages
        comm_receivers (torch.Tensor): The destination ranks of the messages
        rank (int): The current rank
        world_size (int): The total number of ranks

    Returns:
        torch.Tensor: The number of messages to be received from each rank.
                      Shape: (world_size,)
    """
    receive_from_ranks = comm_senders[comm_receivers == rank]
    recv_comm_vector = torch.bincount(receive_from_ranks, minlength=world_size).long()
    return recv_comm_vector


def _get_send_recv_comm_vectors(
    src_ranks: torch.Tensor, dest_ranks: torch.Tensor, rank: int, world_size: int
):
    """Returns the number of messages to be sent and received from each rank based on the
    src and dest ranks.

    Args:
        src_ranks (torch.Tensor): The source ranks of the messages
        dest_ranks (torch.Tensor): The destination ranks of the messages
        rank (int): The current rank
        world_size (int): The total number of ranks

    Returns:
        torch.Tensor: The number of messages to be sent to each rank.
                      Shape: (world_size,)
        torch.Tensor: The number of messages to be received from each rank.
                      Shape: (world_size,)
    """
    all_comm_mask = src_ranks != dest_ranks
    comm_senders = src_ranks[all_comm_mask]
    comm_receivers = dest_ranks[all_comm_mask]

    recv_comm_vector = _get_recv_comm_vector(
        comm_senders, comm_receivers, rank, world_size
    )

    send_comm_vector = _get_send_comm_vector(
        comm_senders, comm_receivers, rank, world_size
    )
    return send_comm_vector, recv_comm_vector


def _get_local_send_placement(
    send_comm_vector: torch.Tensor,
    indices: torch.Tensor,
    src_ranks: torch.Tensor,
    dest_ranks: torch.Tensor,
    rank: int,
    num_src_rows: int,
) -> Dict[int, torch.Tensor]:
    send_local_placement: Dict[int, torch.Tensor] = {}
    for i, num_messages in enumerate(send_comm_vector):
        if num_messages == 0:
            # Not sending any messages current_rank to rank i
            continue

        if i == rank:
            # Not sending any messages to self
            continue

        _mask = (src_ranks == rank) & (dest_ranks == i)
        _send_row = indices[0][_mask] % num_src_rows
        send_local_placement[i] = _send_row

    return send_local_placement


def _get_local_unique_recv_placement(
    indices: torch.Tensor,
    src_ranks: torch.Tensor,
    receive_from_mask: torch.Tensor,
    num_local_output_rows: int,
    rank: int,
    world_size: int,
) -> Dict[int, torch.Tensor]:
    """Returns the unique indices that will be received from each rank. As aggregation
    is done locally before sending the data, the destination rank will only receive
    unique indices from the source rank. This function returns the placement of
    these unique indices in the destination rank, to be called
    by the destination rank. This is used to allocate the receive buffer.

    Args:
        indices (torch.Tensor): The indices tensor
        src_ranks (torch.Tensor): The source ranks of the messages
        receive_from_mask (torch.Tensor): The mask of the messages to be received.
                                          Element i is True if
                                          src_ranks[i] != dest_ranks[i] and
                                          dest_ranks[i] == rank
        num_local_output_rows (int): The number of rows in the output tensor of the
                                     scattered data
        rank (int): The current rank (destination rank)
        world_size (int): The total number of ranks
    Returns:
        Dict[int, torch.Tensor]: A dictionary with the rank as the key and the
                                local scatter location of the received data
                                from that rank as the value.
    """
    send_local_placement: Dict[int, torch.Tensor] = {}
    if torch.any(receive_from_mask):
        receive_from_ranks = src_ranks[receive_from_mask]
        for _sender in range(world_size):
            if _sender == rank:
                continue
            _mask = receive_from_ranks == _sender
            if torch.any(_mask):
                _send_mask = (src_ranks == _sender) & receive_from_mask
                _send_indices = indices[_send_mask] % num_local_output_rows
                _unique_indices = torch.unique(_send_indices, sorted=False)
                send_local_placement[_sender] = _unique_indices
    else:
        return send_local_placement

    return send_local_placement


def _get_local_recv_buffer_w_placement(
    recv_comm_vector: torch.Tensor,
    local_rank_mapping: torch.Tensor,
    num_features: int,
    rank: int,
    device: torch.device,
) -> Tuple[Dict, Dict]:

    recv_local_placement: Dict[int, torch.Tensor] = {}
    recv_buffer_dict: Dict[int, torch.Tensor] = {}

    for i, num_messages in enumerate(recv_comm_vector):
        if num_messages == 0:
            continue

        if i == rank:
            continue

        _local_placement_indices = torch.argwhere(local_rank_mapping == i)
        recv_local_placement[i] = _local_placement_indices
        num_rows = int(num_messages.item())
        recv_buffer = torch.zeros(1, num_rows, num_features).to(device)
        recv_buffer_dict[i] = recv_buffer

    return recv_buffer_dict, recv_local_placement


def _get_local_recv_placement(
    recv_comm_vector: torch.Tensor,
    local_rank_mapping: torch.Tensor,
    rank: int,
) -> Dict[int, torch.Tensor]:
    recv_local_placement: Dict[int, torch.Tensor] = {}
    for i, num_messages in enumerate(recv_comm_vector):
        if num_messages == 0:
            continue

        if i == rank:
            continue

        _local_placement_indices = torch.argwhere(local_rank_mapping == i)
        recv_local_placement[i] = _local_placement_indices

    return recv_local_placement


def _allocate_local_recv_buffers(
    recv_comm_vector: torch.Tensor,
    num_features: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    recv_buffer_dict: Dict[int, torch.Tensor] = {}
    for i, num_messages in enumerate(recv_comm_vector):
        if num_messages == 0:
            continue

        if i == rank:
            continue

        num_rows = int(num_messages.item())
        recv_buffer = torch.zeros(1, num_rows, num_features).to(device)
        recv_buffer_dict[i] = recv_buffer
    return recv_buffer_dict


def _generate_local_rank_mapping(
    _global_rank_mapping: torch.Tensor, world_size: int
) -> torch.Tensor:

    local_rank_mapping = torch.zeros_like(_global_rank_mapping).view(-1)

    _needs_padding = _global_rank_mapping.shape[0] % world_size != 0

    if _needs_padding:
        padding = world_size - (local_rank_mapping.shape[0] % world_size)
        local_rank_mapping = torch.cat(
            [local_rank_mapping, torch.zeros(padding).long()]
        )

    local_rank_mapping = local_rank_mapping.view(world_size, -1)
    _fill_val = (
        torch.arange(world_size)
        .view(-1, 1)
        .expand(world_size, local_rank_mapping.shape[1])
    )
    local_rank_mapping += _fill_val
    local_rank_mapping = local_rank_mapping.view(-1)
    _max_size = max(local_rank_mapping.shape[0], local_rank_mapping.shape[0])
    return local_rank_mapping[:_max_size]
