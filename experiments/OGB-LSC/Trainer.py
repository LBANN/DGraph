# Copyright (c) 2014-2025, Lawrence Livermore National Security, LLC.
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
import torch
from RGAT import CommAwareRGAT
from config import ModelConfig, TrainingConfig


class Trainer:
    def __init__(self, dataset, comm):
        self.dataset = dataset
        self.comm = comm
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        # TODO: We need some better way to set the device but
        # difficult to do that since systems have different bindings.
        # self.device = torch.device(f"cuda:{comm.get_local_rank()}")
        rank = comm.get_rank()
        print(f"Rank {rank} using GPU {rank % torch.cuda.device_count()}")
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        self.device = torch.device("cuda")
        self.model = CommAwareRGAT(
            in_channels=self.model_config.num_features,
            out_channels=self.model_config.num_classes,
            hidden_channels=self.model_config.hidden_channels,
            num_relations=self.model_config.num_relations,
            num_layers=self.model_config.num_layers,
            heads=self.model_config.heads,
            comm=comm,
            dropout=self.model_config.dropout,
        ).to(self.device)

    def prepare_data(self):
        self.dataset = self.dataset.add_batch_dimension()
        self.dataset = self.dataset.to(self.device)

    def train(self):
        self.model.train()

        xs, edge_index, edge_type, rank_mapping = self.dataset[0]

        for epoch in range(1, self.training_config.epochs + 1):
            out = self.model(xs, edge_index, edge_type, rank_mapping)
            loss = torch.nn.functional.cross_entropy(
                out[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask]
            )
            loss.backward()
        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        out = self.model(
            self.dataset.x, self.dataset.edge_index, self.dataset.rank_mapping
        )
        y_true = self.dataset.y.cpu().numpy()
        y_pred = out.argmax(dim=-1, keepdim=True).cpu().numpy()

        train_acc = (
            y_pred[self.dataset.train_mask] == y_true[self.dataset.train_mask]
        ).sum() / int(self.dataset.train_mask.sum())
        val_acc = (
            y_pred[self.dataset.val_mask] == y_true[self.dataset.val_mask]
        ).sum() / int(self.dataset.val_mask.sum())
        test_acc = (
            y_pred[self.dataset.test_mask] == y_true[self.dataset.test_mask]
        ).sum() / int(self.dataset.test_mask.sum())

        return train_acc, val_acc, test_acc
