import torch
import numpy as np
import os.path as osp
from DGraph.CommunicatorBase import CommunicatorBase
from DGraph.data.ogbn_datasets import process_homogenous_data
import os
from utils import DummyComm


def assign_node_rank(num_nodes, world_size):
    _div = num_nodes // world_size
    _mod = num_nodes % world_size
    arr = np.arange(world_size).repeat(_div)
    if _mod > 0:
        arr = np.concatenate((arr, np.arange(_mod)))
    np.random.shuffle(arr)
    return torch.from_numpy(arr).long()


def partitioned_saver(graph_obj, graph_file_path, rank):
    # Only save what we need
    torch.save(graph_obj, graph_file_path + f".part{rank}")


class DistributedIGBWrapper:
    def __init__(
        self,
        root,
        comm,
        graph_file_path=None,
        node_rank_placement=None,
        sim_node_features=True,
        num_features=1,
    ):
        self.root = root
        self.comm_object = comm
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.num_features = num_features
        self.num_nodes = 269346174
        self.num_edges = 3727095830
        self.num_classes = 19
        self.sim_node_features = sim_node_features
        if node_rank_placement is None:
            node_rank_placement = assign_node_rank(self.num_nodes, self.world_size)
            print(f"Node Rank Placement shape: {node_rank_placement.shape}")
        graph_file_path = (
            graph_file_path
            if graph_file_path is not None
            else osp.join("graph_cache", f"IGB260M_graph_data_{self.world_size}.pt")
        )

        self.graph_file_path = graph_file_path

        if os.path.exists(graph_file_path + f".part{self.rank}"):
            print(f"Loading cached graph from {graph_file_path}.part{self.rank}")
            self.graph_obj = torch.load(
                graph_file_path + f".part{self.rank}", weights_only=False
            )
        else:
            if os.path.exists(graph_file_path):
                print(f"Loading cached graph from {graph_file_path}")
                graph_obj = torch.load(graph_file_path, weights_only=False)
                self._load_slices_graph(graph_obj)
            else:
                print("Processing graph data")
                self.load_graph_data(node_rank_placement)

    def _load_slices_graph(self, graph_obj):
        print("Slicing and saving the graph data")
        tensor_dict = {
            "node_feat": graph_obj.get_local_node_features(rank=self.rank),
            "edge_index": graph_obj.get_global_edge_indices(),
            "rank_mapping": graph_obj.get_global_rank_mappings(),
            "labels": graph_obj.get_local_labels(rank=self.rank),
        }

        self.graph_obj = tensor_dict
        os.makedirs("graph_cache", exist_ok=True)
        partitioned_saver(self.graph_obj, self.graph_file_path, self.rank)

    def load_graph_data(self, node_rank_placement):
        processed_dir = osp.join(self.root, "processed")
        edge_dir = osp.join(processed_dir, "paper__cites__paper")
        node_features_dir = osp.join(processed_dir, "paper")
        edges = np.load(osp.join(edge_dir, "edge_index.npy"), mmap_mode="r")
        edges = edges.T
        print(edges.shape)

        graph_data = {"edge_index": edges, "num_nodes": self.num_nodes}

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
        graph_data["edge_feat"] = None
        labels = np.memmap(
            osp.join(node_features_dir, "node_label_19.npy"),
            mode="r",
            dtype="float32",
        )
        print(labels.shape)

        n_train = int(self.num_nodes * 0.6)
        n_val = int(self.num_nodes * 0.2)

        train_mask = np.zeros(self.num_nodes, dtype=bool)
        val_mask = np.zeros(self.num_nodes, dtype=bool)
        test_mask = np.zeros(self.num_nodes, dtype=bool)

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

        self._load_slices_graph(graph_obj)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        local_node_features = self.graph_obj["node_feat"]
        labels = self.graph_obj["labels"]
        edge_indices = self.graph_obj["edge_index"]
        rank_mappings = self.graph_obj["rank_mapping"]

        return local_node_features, edge_indices, rank_mappings, labels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/IGB")
    parser.add_argument("--world_size", type=int, default=2)

    args = parser.parse_args()
    root = args.root
    world_size = args.world_size
    node_rank_placement_file = None

    for i in range(world_size):
        comm = DummyComm(world_size, rank=i)
        dataset = DistributedIGBWrapper(root, comm)
