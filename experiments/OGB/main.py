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
from DGraph.data.datasets import DistributedOGBWrapper
from DGraph.Communicator import CommunicatorBase, Communicator

import fire
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from GCN import CommAwareGCN as GCN
from utils import (
    dist_print_ephemeral,
    write_experiment_log,
    cleanup,
    visualize_trajectories,
    safe_create_dir,
    calculate_accuracy,
)
import numpy as np


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
        self, tensor: torch.Tensor, src: torch.Tensor, rank_mappings, num_local_nodes
    ):
        # TODO: Wrap this in the datawrapper class
        src = src.unsqueeze(-1).expand(1, -1, tensor.shape[-1])
        out = torch.zeros(1, num_local_nodes, tensor.shape[-1]).to(tensor.device)
        out.scatter_add(1, src, tensor)
        return out

    def gather(self, tensor, dst, rank_mappings):
        # TODO: Wrap this in the datawrapper class
        dst = dst.unsqueeze(-1).expand(1, -1, tensor.shape[-1])
        out = torch.gather(tensor, 1, dst)
        return out

    def __str__(self) -> str:
        return self.backend

    def rank_cuda_device(self):
        device = torch.cuda.current_device()
        return device


def _run_experiment(
    dataset,
    comm,
    lr: float,
    epochs: int,
    log_prefix: str,
    hidden_dims: int = 128,
    num_classes: int = 40,
):
    torch.cuda.set_device(comm.get_rank())
    device = torch.cuda.current_device()
    model = GCN(
        in_channels=128, hidden_dims=hidden_dims, num_classes=num_classes, comm=comm
    )
    rank = comm.get_rank()
    model = model.to(device)

    model = (
        DDP(model, device_ids=[rank], output_device=rank)
        if comm.get_world_size() > 1
        else model
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    stream = torch.cuda.Stream()

    start_time.record(stream)
    node_features, edge_indices, rank_mappings, labels = dataset[0]

    node_features = node_features.to(device).unsqueeze(0)
    edge_indices = edge_indices.to(device)[:, :-1].unsqueeze(0)
    labels = labels.to(device).unsqueeze(0)
    rank_mappings = rank_mappings[:, :-1]

    if rank == 0:
        print("*" * 80)
    for i in range(comm.get_world_size()):
        if i == rank:
            print(f"Rank: {rank} Mapping: {rank_mappings.shape}")
            print(f"Rank: {rank} Node Features: {node_features.shape}")
            print(f"Rank: {rank} Edge Indices: {edge_indices.shape}")
        dist.barrier()
    criterion = torch.nn.CrossEntropyLoss()

    train_mask = dataset.graph_obj.get_local_mask("train")
    validation_mask = dataset.graph_obj.get_local_mask("val")
    training_loss_scores = []
    validation_loss_scores = []
    validation_accuracy_scores = []

    print(f"Rank: {rank} training_mask: {train_mask.shape}")
    print(f"Rank: {rank} validation_mask: {validation_mask.shape}")

    for i in range(epochs):
        optimizer.zero_grad()
        _output = model(node_features, edge_indices, rank_mappings)
        # Must flatten along the batch dimension for the loss function
        output = _output[:, train_mask].view(-1, num_classes)
        gt = labels[:, train_mask].view(-1)
        loss = criterion(output, gt)
        loss.backward()
        dist_print_ephemeral(f"Epoch {i} \t Loss: {loss.item()}", rank)
        optimizer.step()

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
    end_time.record(stream)
    torch.cuda.synchronize()

    model.eval()

    with torch.no_grad():
        test_idx = dataset.graph_obj.get_local_mask("test")
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

    average_time = start_time.elapsed_time(end_time) / epochs
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
    epochs: int = 100,
    lr: float = 0.001,
    runs: int = 1,
    log_dir: str = "logs",
):
    _communicator = backend.lower()

    assert _communicator.lower() in [
        "single",
        "nccl",
        "nvshmem",
        "mpi",
    ], "Invalid backend"

    assert dataset in ["arxiv"], "Invalid dataset"

    if _communicator.lower() == "single":
        # Dummy communicator for single process testing
        comm = SingleProcessDummyCommunicator()
    else:
        dist.init_process_group(backend="nccl")
        comm = Communicator.init_process_group(_communicator)

    safe_create_dir(log_dir, comm.get_rank())
    training_dataset = DistributedOGBWrapper(f"ogbn-{dataset}", comm)

    training_trajectores = np.zeros((runs, epochs))
    validation_trajectores = np.zeros((runs, epochs))
    validation_accuracies = np.zeros((runs, epochs))
    for i in range(runs):
        log_prefix = f"{log_dir}/run_{i}"
        training_traj, val_traj, val_accuracy = _run_experiment(
            training_dataset, comm, lr, epochs, log_prefix
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
