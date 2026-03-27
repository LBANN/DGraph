import torch
from typing import Optional
import torch.nn as nn
from DGraph import Communicator
from torch.autograd import Function
from DGraph.distributed.commInfo import CommunicationPattern


class HaloExchangeImpl(Function):
    """Backend-agnostic autograd function for halo vertex feature exchange.

    Performs the inter-rank communication portion of a halo exchange.  The
    gather step (indexing ``x_local`` by ``send_local_idx``) is intentionally
    kept *outside* this class so that PyTorch's built-in autograd handles
    gradient accumulation for vertices sent to multiple ranks via
    ``scatter_add_`` automatically.

    Forward:
        Allocates a receive buffer, then calls ``comm.put`` to deliver each
        per-rank slice of ``send_buffer`` into the corresponding slot of the
        remote rank's receive buffer.  Uses ``comm_pattern.put_forward_remote_offset``
        as the one-sided write offset (ignored by two-sided backends).

    Backward:
        Transposes the forward communication: ``send_offset`` and ``recv_offset``
        swap roles, and ``put_backward_remote_offset`` is used as the write
        offset.  Returns ``grad_send_buffer``; gradients for ``comm`` and
        ``comm_pattern`` are ``None`` (non-differentiable).

    See Also:
        ``HaloExchange`` for the user-facing wrapper.
        ``DGraphMessagePassing`` for an end-to-end usage example.
        ``DGraph/distributed/HaloExchangeDocument.md`` for full design details.
    """

    @staticmethod
    def forward(
        ctx, send_buffer, comm, comm_pattern: CommunicationPattern
    ) -> torch.Tensor:
        # Allocate the receive buffer
        feature_dim = send_buffer.shape[1] if send_buffer.ndim == 2 else 1
        total_recv = int(comm_pattern.recv_offset[-1].item())

        ctx.comm_pattern = comm_pattern
        ctx.feature_dim = feature_dim
        ctx.comm = comm
        recv_buffer = comm.alloc_buffer(
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

        grad_input_tensor = comm.alloc_buffer(
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
    """Exchanges halo vertex features between ranks for distributed GNN computation.

    Wraps the full gather → communicate → return pipeline behind a single
    callable.  The result is autograd-compatible: gradients flow back through
    the communication and through the gather indexing step automatically.

    Usage::

        exchanger = HaloExchange(communicator)
        halo_features = exchanger(local_node_features, comm_pattern)
        # halo_features: [num_halo, F] — features of remote neighbour vertices

    The returned ``halo_features`` can be concatenated with ``local_node_features``
    to form the augmented subgraph consumed by a GNN layer.  See
    ``DGraphMessagePassing`` for a complete example.

    Args:
        comm (Communicator): Initialised communicator backed by any supported
            engine (NCCL, MPI, NVSHMEM).

    See Also:
        ``DGraphMessagePassing`` — end-to-end message-passing wrapper.
        ``commInfo.build_communication_pattern`` — how to build ``comm_pattern``.
        ``DGraph/distributed/HaloExchangeDocument.md`` — full design details.
    """

    def __init__(self, comm: Communicator):
        self.comm = comm

    def __call__(
        self, x_local: torch.Tensor, comm_pattern: CommunicationPattern
    ) -> torch.Tensor:
        """Exchange boundary vertex features with neighbouring ranks.

        Args:
            x_local (torch.Tensor): Local node feature matrix of shape
                ``[num_local, F]``.
            comm_pattern (CommunicationPattern): Precomputed communication
                pattern for this rank (see ``commInfo.build_communication_pattern``).

        Returns:
            torch.Tensor: Halo node features of shape ``[num_halo, F]``,
                one row per remote neighbour vertex ordered by
                ``comm_pattern.recv_offset``.
        """
        send_buffer = x_local[comm_pattern.send_local_idx]
        recv_buffer = HaloExchangeImpl.apply(send_buffer, self.comm, comm_pattern)
        return recv_buffer  # type: ignore


class DGraphMessagePassing(nn.Module):
    """Distributed GNN message-passing layer wrapper.

    Combines halo exchange with a user-supplied message-passing layer into a
    single ``nn.Module``.  This is the recommended way to build a distributed
    GNN layer with DGraph.

    The forward pass performs three steps:

    1. **Halo exchange** — fetch features of remote neighbour vertices from
       other ranks via ``HaloExchange``.
    2. **Augment** — concatenate local and halo features into a single tensor
       indexed by ``comm_pattern.local_edge_list``.
    3. **Message passing** — run the wrapped ``message_passing_layer`` on the
       augmented local subgraph.

    The ``message_passing_layer`` must accept the following positional arguments::

        output = message_passing_layer(
            node_features,   # [num_local + num_halo, F]
            edge_index,      # [num_local_edges, 2]  (local integer indices)
            edge_features,   # [num_local_edges, E] or None
        )

    and must return updated features only for the *local* vertices (i.e.
    ``output`` has shape ``[num_local, F_out]``).

    Usage::

        conv = MyGNNConv(in_channels=64, out_channels=64)
        layer = DGraphMessagePassing(exchanger=HaloExchange(comm), message_passing_layer=conv)

        # Inside the training loop (all tensors already on the correct device):
        updated = layer(local_node_features, comm_pattern, local_edge_features)

    Args:
        exchanger (HaloExchange): Initialised ``HaloExchange`` instance.
        message_passing_layer (nn.Module): Any GNN conv / message-passing
            module that matches the calling convention above.

    See Also:
        ``HaloExchange`` — the underlying exchange primitive.
        ``commInfo.build_communication_pattern`` — how to build ``comm_pattern``.
    """

    def __init__(self, exchanger: HaloExchange, message_passing_layer: nn.Module):
        super(DGraphMessagePassing, self).__init__()
        self.message_passing_layer = message_passing_layer
        self.exchanger = exchanger

    def forward(
        self,
        local_node_features: torch.Tensor,
        comm_pattern: CommunicationPattern,
        local_edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run halo exchange then message passing on the augmented local subgraph.

        Args:
            local_node_features (torch.Tensor): Feature matrix for locally-owned
                vertices, shape ``[num_local, F]``.
            comm_pattern (CommunicationPattern): Precomputed communication pattern
                for this rank.
            local_edge_features (Optional[torch.Tensor]): Edge feature matrix for
                local edges, shape ``[num_local_edges, E]``, or ``None`` if the
                underlying layer does not use edge features.

        Returns:
            torch.Tensor: Updated feature matrix for locally-owned vertices,
                shape ``[num_local, F_out]``, where ``F_out`` is determined by
                ``message_passing_layer``.
        """

        halo_node_features = self.exchanger(local_node_features, comm_pattern)

        local_subgraph = torch.cat([local_node_features, halo_node_features], dim=0)

        local_updated_node_features = self.message_passing_layer(
            local_subgraph, comm_pattern.local_edge_list, local_edge_features
        )

        return local_updated_node_features
