# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
import sys
from time import perf_counter
from typing import Optional
from DGraph.data.datasets import DistributedOGBWrapper
from DGraph.Communicator import CommunicatorBase, Communicator

from DGraph.distributed.nccl._nccl_cache import (
    NCCLGatherCacheGenerator,
    NCCLScatterCacheGenerator,
)
import fire
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from GCN import CommAwareGCN as GCN
from utils import (
    dist_print_ephemeral,
    make_experiment_log,
    write_experiment_log,
    cleanup,
    visualize_trajectories,
    safe_create_dir,
    calculate_accuracy,
)
import numpy as np
import os


class SingleProcessDummyCommunicator(CommunicatorBase):
    def __init__(self):
        super().__init__()
        self._rank = 0
        self._world_size = 1
        self._is_initialized = True
        self.backend = "single"

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def scatter(
        self,
        tensor: torch.Tensor,
        src: torch.Tensor,
        rank_mappings,
        num_local_nodes,
        **kwargs,
    ):
        # TODO: Wrap this in the datawrapper class
        src = src.unsqueeze(-1).expand(1, -1, tensor.shape[-1])
        out = torch.zeros(1, num_local_nodes, tensor.shape[-1]).to(tensor.device)
        out.scatter_add(1, src, tensor)
        return out

    def gather(self, tensor, dst, rank_mappings, **kwargs):
        # TODO: Wrap this in the datawrapper class
        dst = dst.unsqueeze(-1).expand(1, -1, tensor.shape[-1])
        out = torch.gather(tensor, 1, dst)
        return out

    def __str__(self) -> str:
        return self.backend

    def rank_cuda_device(self):
        device = torch.cuda.current_device()
        return device

    def barrier(self):
        # No-op for single process
        pass


def _run_experiment(
    dataset,
    comm,
    lr: float,
    epochs: int,
    log_prefix: str,
    in_dim: int = 128,
    hidden_dims: int = 128,
    num_classes: int = 40,
    use_cache: bool = False,
    dset_name: str = "arxiv",
):
    local_rank = comm.get_rank() % torch.cuda.device_count()
    print(f"Rank: {local_rank} Local Rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    model = GCN(
        in_channels=in_dim, hidden_dims=hidden_dims, num_classes=num_classes, comm=comm
    )
    rank = comm.get_rank()
    model = model.to(device)

    model = (
        DDP(model, device_ids=[local_rank], output_device=local_rank)
        if comm.get_world_size() > 1
        else model
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stream = torch.cuda.Stream()

    node_features, edge_indices, rank_mappings, labels = dataset[0]

    node_features = node_features.to(device).unsqueeze(0)
    edge_indices = edge_indices.to(device).unsqueeze(0)
    labels = labels.to(device).unsqueeze(0)
    rank_mappings = rank_mappings

    if rank == 0:
        print("*" * 80)
    for i in range(comm.get_world_size()):
        if i == rank:
            print(f"Rank: {rank} Mapping: {rank_mappings.shape}")
            print(f"Rank: {rank} Node Features: {node_features.shape}")
            print(f"Rank: {rank} Edge Indices: {edge_indices.shape}")

        comm.barrier()
    criterion = torch.nn.CrossEntropyLoss()

    train_mask = dataset.graph_obj.get_local_mask("train", rank)
    validation_mask = dataset.graph_obj.get_local_mask("val", rank)
    training_loss_scores = []
    validation_loss_scores = []
    validation_accuracy_scores = []

    world_size = comm.get_world_size()

    print(f"Rank: {rank} training_mask: {train_mask.shape}")
    print(f"Rank: {rank} validation_mask: {validation_mask.shape}")

    gather_cache = None
    scatter_cache = None

    if use_cache:
        print(f"Rank: {rank} Using Cache. Generating Cache")
        start_time = perf_counter()
        src_indices = edge_indices[:, 0, :]
        dst_indices = edge_indices[:, 1, :]

        # This says where the edges are located
        edge_placement = rank_mappings[0]
        
        cache_prefix = f"cache/{dset_name}"
        scatter_cache_file = f"{cache_prefix}_scatter_cache_{world_size}_{rank}.pt"
        gather_cache_file = f"{cache_prefix}_gather_cache_{world_size}_{rank}.pt"

        if os.path.exists(gather_cache_file):
            gather_cache = torch.load(gather_cache_file, weights_only=False)

        if os.path.exists(scatter_cache_file):
            scatter_cache = torch.load(scatter_cache_file, weight_only=False)

        # These say where the source and destination nodes are located
        edge_src_placement = rank_mappings[
            0
        ]  # Redundant but making explicit for clarity
        edge_dest_placement = rank_mappings[1]

        num_input_rows = node_features.size(1)
        local_num_edges = (edge_placement == rank).sum().item()

        if gather_cache is None:
            gather_cache = NCCLGatherCacheGenerator(
                dst_indices,
                edge_placement,
                edge_dest_placement,
                num_input_rows,
                rank,
                world_size,
            )
            with open(f"{log_prefix}_gather_cache_{world_size}_{rank}.pt", "wb") as f:
                torch.save(gather_cache, f) 
        
        if scatter_cache is None:
            nodes_per_rank = dataset.graph_obj.get_nodes_per_rank()

            scatter_cache = NCCLScatterCacheGenerator(
                src_indices,
                edge_placement,
                edge_src_placement,
                nodes_per_rank[rank],
                rank,
                world_size,
            )
            with open(f"{log_prefix}_scatter_cache_{world_size}_{rank}.pt", "wb") as f:
                torch.save(scatter_cache, f)

        # Sanity checks for the cache
        for key, value in gather_cache.gather_send_local_placement.items():
            assert value.max().item() < num_input_rows
            assert key < world_size
            assert key != rank
            assert value.shape[0] == gather_cache.gather_send_comm_vector[key]

        for key, value in gather_cache.gather_recv_local_placement.items():
            assert value.max().item() < local_num_edges
            assert key < world_size
            assert key != rank
            assert value.shape[0] == gather_cache.gather_recv_comm_vector[key]

        for rank, value in scatter_cache.gather_send_local_placement.items():
            assert value.max().item() < local_num_edges
            assert rank < world_size
            assert rank != rank
            assert value.shape[0] == scatter_cache.gather_send_comm_vector

        for rank, value in scatter_cache.gather_recv_local_placement.items():
            assert value.max().item() < num_input_rows
            assert rank < world_size
            assert rank != rank
            assert value.shape[0] == scatter_cache.gather_recv_comm_vector
        end_time = perf_counter()
        print(f"Rank: {rank} Cache Generation Time: {end_time - start_time:.4f} s")

        
        #with open(f"{log_prefix}_gather_cache_{world_size}_{rank}.pt", "wb") as f:
        #    torch.save(gather_cache, f)
        #with open(f"{log_prefix}_scatter_cache_{world_size}_{rank}.pt", "wb") as f:
        #    torch.save(scatter_cache, f)
        #print(f"Rank: {rank} Cache Generated")

    training_times = []
    for i in range(epochs):
        comm.barrier()
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record(stream)
        optimizer.zero_grad()
        _output = model(
            node_features, edge_indices, rank_mappings, gather_cache, scatter_cache
        )
        # Must flatten along the batch dimension for the loss function
        output = _output[:, train_mask].view(-1, num_classes)
        gt = labels[:, train_mask].view(-1)
        loss = criterion(output, gt)
        loss.backward()
        dist_print_ephemeral(f"Epoch {i} \t Loss: {loss.item()}", rank)
        optimizer.step()

        comm.barrier()
        end_time.record(stream)
        torch.cuda.synchronize()
        training_times.append(start_time.elapsed_time(end_time))
        training_loss_scores.append(loss.item())
        write_experiment_log(str(loss.item()), f"{log_prefix}_training_loss.log", rank)

        model.eval()
        with torch.no_grad():
            validation_preds = _output[:, validation_mask].view(-1, num_classes)
            label_validation = labels[:, validation_mask].view(-1)
            validation_score = criterion(
                validation_preds,
                label_validation,
            )
            write_experiment_log(
                str(validation_score.item()), f"{log_prefix}_validation_loss.log", rank
            )

            validation_loss_scores.append(validation_score.item())

            val_pred = torch.log_softmax(validation_preds, dim=1)
            accuracy = calculate_accuracy(val_pred, label_validation)
            validation_accuracy_scores.append(accuracy)
            write_experiment_log(
                f"Validation Accuracy: {accuracy:.2f}",
                f"{log_prefix}_validation_accuracy.log",
                rank,
            )
        model.train()

    torch.cuda.synchronize()

    model.eval()

    with torch.no_grad():
        test_idx = dataset.graph_obj.get_local_mask("test", rank)
        test_labels = labels[:, test_idx].view(-1)
        test_preds = model(node_features, edge_indices, rank_mappings)[:, test_idx]
        test_preds = test_preds.view(-1, num_classes)
        test_loss = criterion(test_preds, test_labels)
        test_preds = torch.log_softmax(test_preds, dim=1)
        test_accuracy = calculate_accuracy(test_preds, test_labels)
        test_log_file = f"{log_prefix}_test_results.log"
        write_experiment_log(
            "loss,accuracy",
            test_log_file,
            rank,
        )
        write_experiment_log(f"{test_loss.item()},{test_accuracy}", test_log_file, rank)

    make_experiment_log(f"{log_prefix}_training_times.log", rank)
    make_experiment_log(f"{log_prefix}_runtime_experiment.log", rank)

    for times in training_times:
        write_experiment_log(str(times), f"{log_prefix}_training_times.log", rank)

    average_time = np.mean(training_times[1:])
    log_str = f"Average time per epoch: {average_time:.4f} ms"
    write_experiment_log(log_str, f"{log_prefix}_runtime_experiment.log", rank)

    return (
        np.array(training_loss_scores),
        np.array(validation_loss_scores),
        np.array(validation_accuracy_scores),
    )


def main(
    backend: str = "single",
    dataset: str = "arxiv",
    epochs: int = 3,
    lr: float = 0.001,
    runs: int = 1,
    log_dir: str = "logs",
    node_rank_placement_file: Optional[str] = None,
    use_cache: bool = False,
):
    _communicator = backend.lower()
    dset_name = dataset
    assert _communicator.lower() in [
        "single",
        "nccl",
        "nvshmem",
        "mpi",
    ], "Invalid backend"

    in_dims = {"arxiv": 128, "products": 100}

    assert dataset in ["arxiv", "products"], "Invalid dataset"

    node_rank_placement = None
    if _communicator.lower() == "single":
        # Dummy communicator for single process testing
        comm = SingleProcessDummyCommunicator()

    else:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        comm = Communicator.init_process_group(_communicator)

        # Must pass the node rank placement file the first time
        if node_rank_placement_file is not None:
            assert os.path.exists(
                node_rank_placement_file
            ), "Node rank placement file not found"
            node_rank_placement = torch.load(
                node_rank_placement_file, weights_only=False
            )

    safe_create_dir(log_dir, comm.get_rank())
    training_dataset = DistributedOGBWrapper(
        f"ogbn-{dataset}",
        comm,
        node_rank_placement=node_rank_placement,
        force_reprocess=True,
    )

    num_classes = training_dataset.num_classes

    training_trajectores = np.zeros((runs, epochs))
    validation_trajectores = np.zeros((runs, epochs))
    validation_accuracies = np.zeros((runs, epochs))
    world_size = comm.get_world_size()
    for i in range(runs):
        log_prefix = f"{log_dir}/{dataset}_{world_size}_cache={use_cache}_run_{i}"
        training_traj, val_traj, val_accuracy = _run_experiment(
            training_dataset,
            comm,
            lr,
            epochs,
            log_prefix,
            use_cache=use_cache,
            num_classes=num_classes,
            dset_name=dset_name,
            in_dim=in_dims[dset_name]
        )
        training_trajectores[i] = training_traj
        validation_trajectores[i] = val_traj
        validation_accuracies[i] = val_accuracy

    visualize_trajectories(
        training_trajectores,
        "Training Loss",
        f"{log_dir}/training_loss.png",
        comm.get_rank(),
    )
    visualize_trajectories(
        validation_trajectores,
        "Validation Loss",
        f"{log_dir}/validation_loss.png",
        comm.get_rank(),
    )
    visualize_trajectories(
        validation_accuracies,
        "Validation Accuracy",
        f"{log_dir}/validation_accuracy.png",
        comm.get_rank(),
    )
    cleanup()


if __name__ == "__main__":
    fire.Fire(main)
