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

from GCN import GCNModel, create_sparse_adj
from ogb_comm_dataset import DGraphOGBDataset


def benchmark_ogb_end_to_end(
    dataset: DGraphOGBDataset,
    comm: Communicator,
    lr: float,
    epochs: int,
    log_prefix: str,
    hidden_dims: int = 256,
    num_classes: int = 40,
    num_layers: int = 2,
    device: str | torch.device = "cuda",
    rank: int = 0,
    local_rank: int = 0,
    warm_up: int = 10,
    cur_dir: Optional[str] = None,
    convert_to_sparse_adj: bool = True,
):

    graph_dataset = dataset[0]
    local_node_features, local_labels, comm_pattern = graph_dataset

    in_dim = local_node_features.shape[1]
    local_masks = dataset.get_masks()
    train_mask = local_masks["train_mask"].to(device)
    val_mask = local_masks["val_mask"].to(device)
    test_mask = local_masks["test_mask"].to(device)

    halo_exchanger = HaloExchange(comm)
    if convert_to_sparse_adj:
        comm_pattern.local_edge_list = create_sparse_adj(
            comm_pattern.local_edge_list,
            num_local_nodes=comm_pattern.num_local_vertices,
            num_halo_nodes=comm_pattern.num_halo_vertices,
            device=device,
        )
    model = GCNModel(
        in_channels=in_dim,
        hidden_channels=hidden_dims,
        out_channels=num_classes,
        num_layers=num_layers,
        halo_exchanger=halo_exchanger,
    ).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    stream = torch.cuda.Stream()
    TimingReport.init(communicator=comm)

    if rank == 0:
        print(
            f"Starting benchmark with {comm.get_world_size()} processes, local rank {local_rank}, global rank {rank}",
            f" Local node features shape: {local_node_features.shape}, local labels shape: {local_labels.shape}",
        )

    for epoch in range(epochs):
        with TimingReport(
            "total-training-time",
        ):
            y = model(local_node_features.to(device), comm_pattern)
            loss = y.sum()
            # loss = criterion(y[train_mask], local_labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if rank == 0:
        timers = TimingReport._timers
        report = {name: np.mean(times[warm_up:]) for name, times in timers.items()}
        print(report)
        os.makedirs(
            f"{cur_dir}/benchmark_results/benchmark_results",
            exist_ok=True,
        )
        with open(
            f"{cur_dir}/benchmark_results/{log_prefix}_timing_report.json",
            "w",
        ) as f:
            json.dump(report, f, indent=4)


def main(dataset: str = "arxiv"):

    assert dataset in (
        "arxiv",
        "products",
    ), f"Unsupported dataset '{dataset}'. Choose from: arxiv, products."

    num_classes = {"arxiv": 40, "products": 47}[dataset]

    comm = Communicator.init_process_group("nccl")
    rank = comm.get_rank()
    local_rank = rank % torch.cuda.device_count()
    world_size = comm.get_world_size()

    torch.cuda.set_device(local_rank)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dataset = DGraphOGBDataset(
        dname=f"ogbn-{dataset}",
        comm=comm,
        root_dir=f"{cur_dir}/data",
    )

    benchmark_ogb_end_to_end(
        dataset=dist_dataset,
        comm=comm,
        lr=0.01,
        epochs=100,
        log_prefix=f"ogbn_{dataset}-{world_size}",
        hidden_dims=128,
        num_classes=num_classes,
        num_layers=3,
        device="cuda",
        rank=rank,
        local_rank=local_rank,
        cur_dir=cur_dir,
    )


if __name__ == "__main__":
    fire.Fire(main)
