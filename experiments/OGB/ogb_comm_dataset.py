import os
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset

from local_papers100m_dataset import LocalPapers100MDataset

# ogb's cached .pt files predate torch 2.6's weights_only=True default and
# contain plain dicts/numpy arrays that the safe unpickler rejects, so force
# the legacy behavior for every torch.load call ogb makes internally.
_torch_load = torch.load


def _weights_only_false_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _weights_only_false_load

from ogb.nodeproppred import NodePropPredDataset
from DGraph.Communicator import CommunicatorBase
from DGraph.distributed import CommunicationPattern, build_communication_pattern
from DGraph.data.graph import get_round_robin_node_rank_map

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
        node_rank_placement: Optional[torch.Tensor] = None,
        root_dir: Optional[str] = None,
        feature_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            dname (str): Name of the dataset
            comm (CommunicatorBase): Communicator object
            node_rank_placement (torch.Tensor): Node rank placement, where node_rank_placement[i] is the rank of the node i
            feature_dim (Optional[int]): If set, truncate node features to the
                first `feature_dim` columns before sharding. A memory-pressure
                lever for timing-only benchmarks (no real accuracy tracked) on
                graphs whose full feature width doesn't fit -- not meaningful
                for anything that reports learned accuracy.
            *args:
            **kwargs:

        """
        super().__init__()
        self.comm_object = comm
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

        comm.barrier()

        local_papers100m_dir = os.path.join(
            root_dir or "dataset", "papers100M_dgl"
        )
        extract_dir = os.path.join(
            local_papers100m_dir, "ogbn-papers100M-seeds"
        )
        converted_dir = os.path.join(local_papers100m_dir, "converted")
        if dname == "ogbn-papers100M" and os.path.exists(
            os.path.join(converted_dir, "edge_index.pt")
        ):
            self.dataset = LocalPapers100MDataset(extract_dir, converted_dir)
        else:
            if self.rank == 0:
                # Load dataset on rank 0 first
                self.dataset = NodePropPredDataset(
                    name=dname, root=root_dir if root_dir else "dataset"
                )

            comm.barrier()

            # Load dataset on all other ranks
            if self.rank != 0:
                self.dataset = NodePropPredDataset(
                    name=dname, root=root_dir if root_dir else "dataset"
                )

        comm.barrier()

        import time as _time

        _t0 = _time.time()

        def _log(msg):
            if self.rank == 0:
                print(f"[DGraphOGBDataset][rank0] +{_time.time() - _t0:6.1f}s {msg}", flush=True)

        _log("calling self.dataset[0] ...")
        graph_data, labels = self.dataset[0]
        _log("self.dataset[0] done; calling get_idx_split() ...")
        split_idx = self.dataset.get_idx_split()
        _log("get_idx_split() done")

        num_nodes = graph_data["num_nodes"]
        node_features = torch.from_numpy(graph_data["node_feat"]).float()
        if feature_dim is not None:
            # Column slice on the (possibly mmap-backed) array stays a view;
            # only the row-gather below actually materializes memory.
            node_features = node_features[:, :feature_dim]
        edge_index = torch.from_numpy(graph_data["edge_index"]).long().T
        labels = torch.from_numpy(labels).long()
        _log(f"tensors wrapped (num_nodes={num_nodes}, edge_index={tuple(edge_index.shape)})")

        if node_rank_placement is None:
            node_rank_placement = get_round_robin_node_rank_map(
                num_nodes, self.world_size
            )
        _log("node_rank_placement ready; building communication pattern ...")

        self.comm_pattern = generate_communication_pattern(
            edge_index, node_rank_placement, self.rank, self.world_size
        )
        _log("communication pattern built")

        local_nodes = node_rank_placement == self.rank
        local_node_features = node_features[local_nodes, :]
        local_labels = labels[local_nodes]
        _log("local features/labels sliced")
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


if __name__ == "__main__":
    from DGraph.Communicator import Communicator

    comm = Communicator.init_process_group("nccl")

    rank = comm.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = comm.get_world_size()
    torch.cuda.set_device(local_rank)

    node_rank_placement = torch.load(
        f"/p/vast1/zaman2/matrix/DGraph/experiments/OGB/ogbn-arxiv-mappings/ogbn-arxiv_vertex_rank_mapping_{world_size}.pt"
    )
    dataset = DGraphOGBDataset(
        dname="ogbn-arxiv", comm=comm, node_rank_placement=node_rank_placement
    )

    data, labels, comm_pattern = dataset[0]
    if rank == 0:

        breakpoint()
        import os

        print(comm_pattern.comm_map)
        file_path = os.path.abspath(__file__)
        # Get the directory containing the current script
        file_dir = os.path.dirname(file_path)
        print(f"Saving to {file_dir}/comm_map_{world_size}.pt")
        torch.save(comm_pattern.comm_map, f"{file_dir}/comm_map_{world_size}.pt")

    comm.barrier()
