import torch
import numpy as np
import os.path as osp
from DGraph.CommunicatorBase import CommunicatorBase
from DGraph.data.ogbn_datasets import process_homogenous_data


def assign_node_rank(num_nodes, world_size):
    _div = num_nodes // world_size
    _mod = num_nodes % world_size
    arr = np.arange(world_size).repeat(_div)
    if _mod > 0:
        arr = np.concatenate((arr, np.arange(_mod)))
    np.random.shuffle(arr)
    return torch.from_numpy(arr).long()


class DistributedIGBWrapper:
    def __init__(
        self,
        root,
        comm,
        node_rank_placement=None,
        sim_node_features=True,
        num_features=1,
    ):
        self.root = root
        self.comm_object = comm
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.num_features = num_features
        self.num_nodes = 227130858
        self.num_edges = 3727095830
        self.num_classes = 19
        self.sim_node_features = sim_node_features
        if node_rank_placement is None:
            node_rank_placement = assign_node_rank(self.num_nodes, self.world_size)
        self.load_graph_data(node_rank_placement)

    def load_graph_data(self, node_rank_placement):
        processed_dir = osp.join(self.root, "processed")
        edge_dir = osp.join(processed_dir, "paper__cites__paper")
        node_features_dir = osp.join(processed_dir, "paper")
        edges = np.load(osp.join(edge_dir, "edge_index.npy"), mmap_mode="r")

        graph_data = {"edge_index": edges}

        if self.sim_node_features:
            node_features = torch.randn(
                (self.num_nodes, self.num_features), dtype=torch.float32
            )
        else:
            node_features = np.memmap(
                osp.join(node_features_dir, "node_feat.npy"),
                mode="r",
                dtype="float32",
                shape=(self.num_nodes, 1024),
            )
            self.num_features = 1024

        graph_data["node_feat"] = node_features
        labels = np.memmap(
            osp.join(node_features_dir, "node_label_19.npy"),
            mode="r",
            dtype="float32",
        )

        n_train = int(self.num_nodes * 0.6)
        n_val = int(self.num_nodes * 0.2)

        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val : self.num_nodes] = True

        split_idx = {
            "train": train_mask,
            "valid": val_mask,
            "test": test_mask,
        }
        graph_obj = process_homogenous_data(
            graph_data,
            labels,
            0,
            self.world_size,
            split_idx,
            node_rank_placement,
        )
        self.graph_obj = graph_obj

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        rank = self.comm_object.get_rank()
        local_node_features = self.graph_obj.get_local_node_features(rank=rank)
        labels = self.graph_obj.get_local_labels(rank=rank)

        # TODO: Move this to a backend-specific collator in the future
        if self.comm_object.backend == "nccl":
            # Return Graph object with Rank placement data

            # NOTE: Two-sided comm needs all the edge indices not the local ones
            edge_indices = self.graph_obj.get_global_edge_indices()
            rank_mappings = self.graph_obj.get_global_rank_mappings()
        else:
            # One-sided communication, no need for rank placement data

            edge_indices = self.graph_obj.get_local_edge_indices(rank=rank)
            rank_mappings = self.graph_obj.get_local_rank_mappings(rank=rank)

        return local_node_features, edge_indices, rank_mappings, labels
