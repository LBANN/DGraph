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
from typing import Optional
from DGraph.Communicator import CommunicatorBase, Communicator

import fire
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from DGraph.utils.TimingReport import TimingReport
from GCN import CommAwareGCN as GCN
from nccl_ogb_dataset import NCCLOGBDataset
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
import json


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

    def scatter(self, tensor, src, rank_mappings, num_local_nodes, **kwargs):
        src = src.unsqueeze(-1).expand(1, -1, tensor.shape[-1])
        out = torch.zeros(1, num_local_nodes, tensor.shape[-1]).to(tensor.device)
        out.scatter_add(1, src, tensor)
        return out

    def gather(self, tensor, dst, rank_mappings, **kwargs):
        dst = dst.unsqueeze(-1).expand(1, -1, tensor.shape[-1])
        out = torch.gather(tensor, 1, dst)
        return out

    def __str__(self) -> str:
        return self.backend

    def rank_cuda_device(self):
        device = torch.cuda.current_device()
        return device

    def barrier(self):
        pass


def _run_experiment(
    dataset: NCCLOGBDataset,
    comm,
    lr: float,
    epochs: int,
    log_prefix: str,
    in_dim: int = 128,
    hidden_dims: int = 128,
    num_classes: int = 40,
    dset_name: str = "arxiv",
):
    local_rank = comm.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    rank = comm.get_rank()

    model = GCN(
        in_channels=in_dim, hidden_dims=hidden_dims, num_classes=num_classes, comm=comm
    )
    model = model.to(device)
    model = (
        DDP(model, device_ids=[local_rank], output_device=local_rank)
        if comm.get_world_size() > 1
        else model
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stream = torch.cuda.Stream()

    graph_data = dataset[0]
    node_features = graph_data.node_features.to(device).unsqueeze(0)
    labels = graph_data.labels.to(device).unsqueeze(0)
    comm_plan = graph_data.comm_plan.to(device)
    train_mask = graph_data.train_mask
    validation_mask = graph_data.val_mask

    if rank == 0:
        print("*" * 80)
    for i in range(comm.get_world_size()):
        if i == rank:
            print(f"Rank: {rank} Node Features: {node_features.shape}")
            print(f"Rank: {rank} Num local nodes: {graph_data.num_local_nodes}")
            print(f"Rank: {rank} Num local edges: {graph_data.num_local_edges}")
        comm.barrier()

    criterion = torch.nn.CrossEntropyLoss()

    print(f"Rank: {rank} training_mask: {train_mask.shape}")
    print(f"Rank: {rank} validation_mask: {validation_mask.shape}")

    training_loss_scores = []
    validation_loss_scores = []
    validation_accuracy_scores = []
    training_times = []

    for i in range(epochs):
        comm.barrier()
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record(stream)

        optimizer.zero_grad()
        _output = model(node_features, comm_plan)

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
            validation_score = criterion(validation_preds, label_validation)
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
        test_mask = graph_data.test_mask
        test_labels = labels[:, test_mask].view(-1)
        test_preds = model(node_features, comm_plan)[:, test_mask].view(-1, num_classes)
        test_loss = criterion(test_preds, test_labels)
        test_preds = torch.log_softmax(test_preds, dim=1)
        test_accuracy = calculate_accuracy(test_preds, test_labels)
        test_log_file = f"{log_prefix}_test_results.log"
        write_experiment_log("loss,accuracy", test_log_file, rank)
        write_experiment_log(f"{test_loss.item()},{test_accuracy}", test_log_file, rank)

    make_experiment_log(f"{log_prefix}_training_times.log", rank)
    make_experiment_log(f"{log_prefix}_runtime_experiment.log", rank)
    for times in training_times:
        write_experiment_log(str(times), f"{log_prefix}_training_times.log", rank)

    average_time = np.mean(training_times[1:])
    write_experiment_log(
        f"Average time per epoch: {average_time:.4f} ms",
        f"{log_prefix}_runtime_experiment.log",
        rank,
    )

    return (
        np.array(training_loss_scores),
        np.array(validation_loss_scores),
        np.array(validation_accuracy_scores),
    )


def main(
    backend: str = "nccl",
    dataset: str = "arxiv",
    epochs: int = 3,
    lr: float = 0.001,
    runs: int = 1,
    log_dir: str = "logs",
    node_rank_placement_file: Optional[str] = None,
    force_reprocess: bool = False,
):
    _communicator = backend.lower()
    assert _communicator in ["nccl", "nvshmem", "mpi"], (
        f"Invalid backend '{_communicator}'. NCCLOGBDataset requires NCCL."
    )
    assert dataset in ["arxiv", "products"], "Invalid dataset"

    in_dims = {"arxiv": 128, "products": 100}

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    comm = Communicator.init_process_group(_communicator)

    node_rank_placement = None
    if node_rank_placement_file is not None:
        assert os.path.exists(node_rank_placement_file), (
            "Node rank placement file not found"
        )
        node_rank_placement = torch.load(node_rank_placement_file, weights_only=False)

    TimingReport.init(comm)
    safe_create_dir(log_dir, comm.get_rank())

    training_dataset = NCCLOGBDataset(
        f"ogbn-{dataset}",
        comm,
        node_rank_placement=node_rank_placement,
        force_reprocess=force_reprocess,
    )

    num_classes = training_dataset.num_classes
    world_size = comm.get_world_size()
    rank = comm.get_rank()

    training_trajectores = np.zeros((runs, epochs))
    validation_trajectores = np.zeros((runs, epochs))
    validation_accuracies = np.zeros((runs, epochs))

    for i in range(runs):
        log_prefix = f"{log_dir}/{dataset}_{world_size}_run_{i}"
        training_traj, val_traj, val_accuracy = _run_experiment(
            training_dataset,
            comm,
            lr,
            epochs,
            log_prefix,
            num_classes=num_classes,
            dset_name=dataset,
            in_dim=in_dims[dataset],
        )
        training_trajectores[i] = training_traj
        validation_trajectores[i] = val_traj
        validation_accuracies[i] = val_accuracy

    write_experiment_log(
        json.dumps(TimingReport._timers),
        f"{log_dir}/{dataset}_timing_report_world_size_{world_size}.json",
        rank,
    )

    visualize_trajectories(
        training_trajectores,
        "Training Loss",
        f"{log_dir}/training_loss.png",
        rank,
    )
    visualize_trajectories(
        validation_trajectores,
        "Validation Loss",
        f"{log_dir}/validation_loss.png",
        rank,
    )
    visualize_trajectories(
        validation_accuracies,
        "Validation Accuracy",
        f"{log_dir}/validation_accuracy.png",
        rank,
    )
    cleanup()


if __name__ == "__main__":
    fire.Fire(main)
