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


def main(
    batch_size: int = 2, is_distributed: bool = False, use_synthetic_data: bool = False
):
    cfg = graphcast_config.Config()

    # Create the model
    model = DGraphCast(cfg)

    if is_distributed:
        raise NotImplementedError("Distributed training is not yet supported.")

    if not use_synthetic_data:
        raise NotImplementedError("Real data is not yet supported yet.")

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
        num_history=cfg.data.num_history,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Train the model

    while (
        _iter
        < cfg.training.num_iters_step1
        + cfg.training.num_iters_step2
        + cfg.training.num_iters_step3
    ):

        for data in dataloader:
            model.train()
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _iter += 1

            scheduler.step()

            # Save the model
            if _iter % cfg.training.save_freq == 0:
                torch.save(model.state_dict(), f"model_{_iter}.pth")

            if _iter % cfg.training.val_freq == 0:
                model.eval()
                val_loss = model(data)
                print(f"Validation loss: {val_loss}")


if __name__ == "__main__":
    Fire(main)
