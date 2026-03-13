from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from ogb.nodeproppred import NodePropPredDataset
from DGraph.Communicator import CommunicatorBase
from DGraph.distributed import CommunicationPattern, build_communication_pattern


def generate_communication_pattern(
    edge_index: torch.Tensor,
    node_rank_placement: torch.Tensor,
    rank: int,
    world_size: int,
) -> CommunicationPattern:
    comm_pattern = build_communication_pattern(
        edge_index, node_rank_placement, rank, world_size
    )
    return comm_pattern


class DGraphOGBDataset(Dataset):
    def __init__(
        self,
        dname: str,
        comm: CommunicatorBase,
        node_rank_placement: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            dname (str): Name of the dataset
            comm (CommunicatorBase): Communicator object
            node_rank_placement (torch.Tensor): Node rank placement, where node_rank_placement[i] is the rank of the node i
            *args:
            **kwargs:

        """
        super().__init__()
        self.comm_object = comm
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        comm.barrier()

        if self.rank == 0:
            # Load dataset on rank 0 first
            self.dataset = NodePropPredDataset(name=dname)

        comm.barrier()

        # Load dataset on all other ranks
        if self.rank != 0:
            self.dataset = NodePropPredDataset(name=dname)

        comm.barrier()

        graph_data, labels = self.dataset[0]
        split_idx = self.dataset.get_idx_split()

        num_nodes = graph_data["num_nodes"]
        node_features = torch.from_numpy(graph_data["node_feat"]).float()
        edge_index = torch.from_numpy(graph_data["edge_index"]).long()
        labels = torch.from_numpy(labels).long()

        self.comm_pattern = generate_communication_pattern(
            edge_index, node_rank_placement, self.rank, self.world_size
        )

        local_nodes = node_rank_placement == self.rank
        local_node_features = node_features[local_nodes, :]
        local_labels = labels[local_nodes]

        # local_features =
