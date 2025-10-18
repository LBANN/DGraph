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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from distributed_layers import GetGlobalVal
import os


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
        # Enable unused-parameter detection only if requested (reduces sync errors with moderate overhead)
        ddp_find_unused = bool(int(os.getenv("RGAT_DDP_FIND_UNUSED", "0")))
        self.model = DDP(
            self.model,
            device_ids=[rank % num_gpus],
            find_unused_parameters=ddp_find_unused,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_config.lr, weight_decay=5e-4
        )

    def prepare_data(self):
        self.dataset = self.dataset.add_batch_dimension()
        self.dataset = self.dataset.to(self.device)

    def train(self):
        self.model.train()

        xs, edge_index, edge_type, rank_mapping = self.dataset[0]

        for epoch in range(1, self.training_config.epochs + 1):
            out = self.model(xs, edge_index, edge_type, rank_mapping)
            train_mask = self.dataset.get_mask("train")
            local_train_vertices = out[:, train_mask, :].squeeze(0)
            target = self.dataset.get_target("train")

            loss = torch.nn.functional.cross_entropy(
                local_train_vertices, target, reduction="sum"
            )
            local_num_targets = target.size(0)
            global_num_targets = GetGlobalVal(local_num_targets)
            loss = loss / global_num_targets  # Average the loss
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        xs, edge_index, edge_type, rank_mapping = self.dataset[0]
        out = self.model(xs, edge_index, edge_type, rank_mapping)

        y_pred = out.argmax(dim=-1, keepdim=True).cpu().numpy()
        train_mask = self.dataset.get_mask("train").cpu().numpy()
        val_mask = self.dataset.get_mask("val").cpu().numpy()
        test_mask = self.dataset.get_mask("test").cpu().numpy()
        y_true_train = self.dataset.get_target("train").cpu().numpy()
        y_pred_val = self.dataset.get_target("val").cpu().numpy()
        y_pred_test = self.dataset.get_target("test").cpu().numpy()

        train_acc = (y_pred[train_mask] == y_true_train).sum() / int(train_mask.sum())
        # Not guaranteed to have validation or test samples on every rank
        num_local_val_samples = int(val_mask.sum())
        num_local_test_samples = int(test_mask.sum())
        if num_local_val_samples == 0:
            val_acc = 0.0
        else:
            val_acc = (y_pred[val_mask] == y_pred_val).sum().item()
        val_acc = GetGlobalVal(val_acc)

        num_global_val_samples = GetGlobalVal(num_local_val_samples)
        val_acc = val_acc / int(num_global_val_samples)

        if num_local_test_samples == 0:
            test_acc = 0.0
        else:
            test_acc = (y_pred[test_mask] == y_pred_test).sum().item()

        test_acc = GetGlobalVal(test_acc)
        num_global_test_samples = GetGlobalVal(num_local_test_samples)
        test_acc = test_acc / int(num_global_test_samples)

        # All ranks should have the same accuracy values

        return train_acc, val_acc, test_acc
