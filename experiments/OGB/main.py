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
"""
Distributed GCN benchmark on OGB node-property-prediction datasets.
"""
import os
import json
from typing import Optional

import fire
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from DGraph.Communicator import Communicator
from DGraph.distributed import HaloExchange
from DGraph.utils.TimingReport import TimingReport

from GCN import CommAwareGCN as GCN
from ogb_comm_dataset import DGraphOGBDataset
from utils import (
    calculate_accuracy,
    cleanup,
    dist_print_ephemeral,
    make_experiment_log,
    safe_create_dir,
    visualize_trajectories,
    write_experiment_log,
)


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------


def _run_experiment(
    dataset: DGraphOGBDataset,
    comm: Communicator,
    lr: float,
    epochs: int,
    log_prefix: str,
    hidden_dims: int = 256,
    num_classes: int = 40,
    device: str | torch.device = "cuda",
    rank: int = 0,
    local_rank: int = 0,
):
    """Run one full training + validation + test experiment.

    Args:
        dataset: Loaded ``DGraphOGBDataset`` for this rank.
        node_rank_placement: [V] global vertex→rank assignment tensor.
        comm: Initialised communicator.
        lr: Adam learning rate.
        epochs: Number of training epochs.
        log_prefix: Path prefix for all output log files.
        hidden_dims: Hidden layer width for the GCN.
        num_classes: Number of output classes.

    Returns:
        Tuple of (training_loss, validation_loss, validation_accuracy) numpy arrays,
        each of length ``epochs``.
    """

    # ---- Extract local data from the dataset --------------------------------
    comm_pattern = dataset.comm_pattern

    local_node_features, local_labels, comm_pattern = dataset[0]
    local_node_features = local_node_features.to(device)
    local_labels = local_labels.to(device)

    in_dim = local_node_features.shape[1]

    local_masks = dataset.get_masks()
    train_mask = local_masks["train_mask"].to(device)
    val_mask = local_masks["val_mask"].to(device)
    test_mask = local_masks["test_mask"].to(device)

    if rank == 0:
        print(
            f"Dataset loaded — "
            f"local nodes: {local_node_features.shape[0]}, "
            f"in_dim: {in_dim}, "
            f"num_classes: {num_classes}"
        )

    # ---- Model setup --------------------------------------------------------
    halo_exchanger = HaloExchange(comm)
    model = GCN(
        in_channels=in_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        halo_exchanger=halo_exchanger,
        comm=comm,
    ).to(device)

    if comm.get_world_size() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    stream = torch.cuda.Stream()

    # ---- Training loop ------------------------------------------------------
    training_loss_scores = []
    validation_loss_scores = []
    validation_accuracy_scores = []
    training_times = []

    make_experiment_log(f"{log_prefix}_training_loss.log", rank)
    make_experiment_log(f"{log_prefix}_validation_loss.log", rank)
    make_experiment_log(f"{log_prefix}_validation_accuracy.log", rank)

    for epoch in range(epochs):
        model.train()
        comm.barrier()
        torch.cuda.synchronize()

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record(stream)

        optimizer.zero_grad()

        with TimingReport("forward"):
            output = model(local_node_features, comm_pattern)

        train_output = output[train_mask]
        train_labels = local_labels[train_mask]
        loss = criterion(train_output, train_labels.reshape(-1))

        with TimingReport("backward"):
            loss.backward()

        optimizer.step()

        comm.barrier()
        end_time.record(stream)
        torch.cuda.synchronize()

        elapsed_ms = start_time.elapsed_time(end_time)
        training_times.append(elapsed_ms)
        training_loss_scores.append(loss.item())

        dist_print_ephemeral(
            f"Epoch {epoch:4d} | loss: {loss.item():.4f} | {elapsed_ms:.1f} ms",
            rank,
        )
        write_experiment_log(str(loss.item()), f"{log_prefix}_training_loss.log", rank)

        # ---- Validation -----------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_output = output[val_mask]
            val_labels = local_labels[val_mask]
            val_loss = criterion(val_output, val_labels.reshape(-1))
            val_preds = torch.log_softmax(val_output, dim=1)
            val_accuracy = calculate_accuracy(val_preds, val_labels)

        validation_loss_scores.append(val_loss.item())
        validation_accuracy_scores.append(val_accuracy)
        write_experiment_log(
            str(val_loss.item()), f"{log_prefix}_validation_loss.log", rank
        )
        write_experiment_log(
            f"Validation Accuracy: {val_accuracy:.2f}",
            f"{log_prefix}_validation_accuracy.log",
            rank,
        )

    torch.cuda.synchronize()

    # ---- Test evaluation ----------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_output = model(local_node_features, comm_pattern)
        test_preds = test_output[test_mask]
        test_labels = local_labels[test_mask]
        test_loss = criterion(test_preds, test_labels.reshape(-1))
        test_preds_log = torch.log_softmax(test_preds, dim=1)
        test_accuracy = calculate_accuracy(test_preds_log, test_labels)

    test_log_file = f"{log_prefix}_test_results.log"
    make_experiment_log(test_log_file, rank)
    write_experiment_log("loss,accuracy", test_log_file, rank)
    write_experiment_log(f"{test_loss.item()},{test_accuracy}", test_log_file, rank)

    if rank == 0:
        print(
            f"\nTest  | loss: {test_loss.item():.4f} | accuracy: {test_accuracy:.2f}%"
        )

    # ---- Timing summary -----------------------------------------------------
    make_experiment_log(f"{log_prefix}_training_times.log", rank)
    for t in training_times:
        write_experiment_log(str(t), f"{log_prefix}_training_times.log", rank)

    average_time = (
        np.mean(training_times[1:]) if len(training_times) > 1 else training_times[0]
    )
    make_experiment_log(f"{log_prefix}_runtime_experiment.log", rank)
    write_experiment_log(
        f"Average time per epoch (excl. first): {average_time:.4f} ms",
        f"{log_prefix}_runtime_experiment.log",
        rank,
    )

    return (
        np.array(training_loss_scores),
        np.array(validation_loss_scores),
        np.array(validation_accuracy_scores),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    backend: str = "nccl",
    dataset: str = "arxiv",
    epochs: int = 10,
    lr: float = 0.001,
    runs: int = 1,
    hidden_dims: int = 256,
    log_dir: str = "logs",
    node_rank_placement_file: Optional[str] = None,
    root_dir: Optional[str] = None,
):
    """Distributed GCN benchmark on OGB node-property-prediction datasets.

    Args:
        backend: Communication backend — one of ``nccl``, ``mpi``, ``nvshmem``.
        dataset: OGB dataset name — one of ``arxiv``, ``products``.
        epochs: Number of training epochs per run.
        lr: Adam learning rate.
        runs: Number of independent runs (for mean/std reporting).
        hidden_dims: Hidden layer width for the GCN.
        log_dir: Directory to write log files and plots.
        node_rank_placement_file: Path to a ``.pt`` file containing a [V]
            int64 tensor mapping each global vertex to its assigned rank.
            Required for all distributed backends.
    """
    assert backend.lower() in (
        "nccl",
        "mpi",
        "nvshmem",
    ), f"Unsupported backend '{backend}'. Choose from: nccl, mpi, nvshmem."
    assert dataset in (
        "arxiv",
        "products",
    ), f"Unsupported dataset '{dataset}'. Choose from: arxiv, products."

    num_classes = {"arxiv": 40, "products": 47}[dataset]

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    comm = Communicator.init_process_group(backend.lower())

    rank = comm.get_rank()
    world_size = comm.get_world_size()

    rank = comm.get_rank()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    TimingReport.init(comm)
    safe_create_dir(log_dir, rank)

    # ---- Node rank placement ------------------------------------------------
    assert node_rank_placement_file is not None, (
        "--node_rank_placement_file is required. "
        "Generate one with preprocess.py before running this script."
    )
    assert os.path.exists(
        node_rank_placement_file
    ), f"Node rank placement file not found: {node_rank_placement_file}"
    node_rank_placement = torch.load(node_rank_placement_file, weights_only=False)

    # ---- Dataset ------------------------------------------------------------
    training_dataset = DGraphOGBDataset(
        dname=f"ogbn-{dataset}",
        comm=comm,
        node_rank_placement=node_rank_placement,
        root_dir=root_dir,
    )

    # ---- Runs ---------------------------------------------------------------
    training_trajectories = np.zeros((runs, epochs))
    validation_trajectories = np.zeros((runs, epochs))
    validation_accuracies = np.zeros((runs, epochs))

    for run in range(runs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Run {run + 1}/{runs}")
            print(f"{'='*60}")

        log_prefix = f"{log_dir}/{dataset}_world{world_size}_run{run}"
        train_loss, val_loss, val_acc = _run_experiment(
            dataset=training_dataset,
            comm=comm,
            lr=lr,
            epochs=epochs,
            log_prefix=log_prefix,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            device=device,
            rank=rank,
            local_rank=local_rank,
        )
        training_trajectories[run] = train_loss
        validation_trajectories[run] = val_loss
        validation_accuracies[run] = val_acc

    # ---- Timing report ------------------------------------------------------
    write_experiment_log(
        json.dumps(TimingReport._timers),
        f"{log_dir}/{dataset}_timing_report_world{world_size}.json",
        rank,
    )

    # ---- Plots --------------------------------------------------------------
    visualize_trajectories(
        training_trajectories,
        "Training Loss",
        f"{log_dir}/training_loss.png",
        rank,
    )
    visualize_trajectories(
        validation_trajectories,
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
