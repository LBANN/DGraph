from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
from ogb.nodeproppred import NodePropPredDataset
from DGraph.Communicator import CommunicatorBase
from DGraph.distributed import CommunicationPattern, build_communication_pattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_local_split_masks(
    node_rank_placement: torch.Tensor,
    split_idx: dict,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Convert global OGB split indices into boolean masks over *local* nodes.

    Args:
        node_rank_placement: [V] tensor mapping each global vertex to its rank.
        split_idx: dict with keys 'train', 'valid', 'test', each a 1-D tensor
            of global node indices (as returned by ogb's ``get_idx_split``).
        rank: this process's rank.

    Returns:
        Dict with keys 'train', 'valid', 'test', each a boolean tensor of
        shape [num_local] that is True for local nodes belonging to that split.
    """
    V = node_rank_placement.shape[0]
    local_node_global_ids = torch.where(node_rank_placement == rank)[0]

    masks = {}
    for split_name, global_ids in split_idx.items():
        global_mask = torch.zeros(V, dtype=torch.bool)
        global_mask[global_ids] = True
        masks[split_name] = global_mask[local_node_global_ids]
    return masks


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
        self.local_node_features = local_node_features
        self.local_labels = local_labels

        rank = comm.get_rank()
        assert split_idx is not None

        local_masks = _build_local_split_masks(node_rank_placement, split_idx, rank)
        self.train_mask = local_masks["train"]
        self.val_mask = local_masks["valid"]
        self.test_mask = local_masks["test"]

    def get_masks(self):
        local_masks = {
            "train_mask": self.train_mask,
            "val_mask": self.val_mask,
            "test_mask": self.test_mask,
        }
        return local_masks

    def __len__(self) -> int:
        return 1

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, CommunicationPattern]:
        return (
            self.local_node_features,
            self.local_labels,
            self.comm_pattern,
        )
