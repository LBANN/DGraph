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

from fire import Fire
import torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR
import graphcast_config
from model import DGraphCast
import torch
from torch.utils.data import DataLoader
from dataset import SyntheticWeatherDataset
from dist_utils import SingleProcessDummyCommunicator, CommAwareDistributedSampler
from torch.profiler import profile, ProfilerActivity
from DGraph.Communicator import Communicator
from torch.nn.parallel import DistributedDataParallel as DDP


def compute_loss(ground_truth, prediction, comm):
    loss = (ground_truth - prediction).pow(2)
    loss = loss.mean(dim=(1, 2))
    loss = loss.sum()
    loss = loss * comm._ranks_per_sample / comm.get_world_size()
    return loss


def main(
    batch_size: int = 1,
    is_distributed: bool = False,
    test_run: bool = False,
    use_synthetic_data: bool = False,
    benchmark: bool = False,
    backend: str = "single",
    procs_per_graph: int = 2,
):
    _communicator = backend.lower()

    assert _communicator.lower() in [
        "single",
        "nccl",
        "nvshmem",
        "mpi",
    ], "Invalid backend"

    cfg = graphcast_config.Config()

    # Create the model

    if is_distributed:
        comm = Communicator.init_process_group(
            _communicator, ranks_per_graph=procs_per_graph
        )
    else:
        comm = SingleProcessDummyCommunicator()
    if not use_synthetic_data:
        raise NotImplementedError("Real data is not yet supported yet.")

    model = DGraphCast(cfg, comm)

    torch.cuda.set_device(comm.get_rank())
    device = torch.cuda.current_device()

    model = model.to(device)

    model = (
        DDP(model, device_ids=[device], output_device=device)
        if is_distributed
        else model
    )

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler1 = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg.training.num_iters_step1,
    )
    scheduler2 = CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_iters_step2, eta_min=0.0
    )
    scheduler3 = LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (cfg.training.num_iters_step3 / cfg.training.lr),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2, scheduler3],
        milestones=[
            cfg.training.num_iters_step1,
            cfg.training.num_iters_step1 + cfg.training.num_iters_step2,
        ],
    )

    # Create the dataset
    dataset = SyntheticWeatherDataset(
        channels=[x for x in range(cfg.data.num_channels_climate)],
        num_samples_per_year=cfg.data.num_samples_per_year_train,
        num_steps=cfg.data.num_history,
        device=torch.device("cpu"),
    )

    static_graph = dataset.get_static_graph()
    assert batch_size == 1, "Per Rank batch size must be 1 for distributed training."

    sampler = CommAwareDistributedSampler(dataset, comm)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    # Train the model

    _iter = 0
    while (
        _iter
        < cfg.training.num_iters_step1
        + cfg.training.num_iters_step2
        + cfg.training.num_iters_step3
    ):
        break_training = False

        for data in dataloader:
            in_data = data["invar"]
            ground_truth = data["outvar"]

            model.train()
            optimizer.zero_grad()
            predicted_grid = model(in_data, static_graph)
            loss = compute_loss(ground_truth, predicted_grid, comm)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _iter += 1

            scheduler.step()

            if test_run:
                break_training = True
                break

            # Save the model
            if _iter % cfg.training.save_freq == 0:
                torch.save(model.state_dict(), f"model_{_iter}.pth")

            if _iter % cfg.training.val_freq == 0:
                model.eval()
                val_loss = model(data)
                print(f"Validation loss: {val_loss}")

        if break_training:
            break

    if benchmark:
        inputs = next(iter(dataloader))["invar"]

        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
        ) as prof:
            model(inputs, static_graph)

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


if __name__ == "__main__":
    Fire(main)
