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
from lsc_datasets.distributed_graph_dataset import DistributedHeteroGraphDataset

import os


def stop():
    import sys

    dist.destroy_process_group()
    sys.exit(0)


def print_on_rank_zero(*args, **kwargs):
    dist.barrier()
    if dist.get_rank() == 0:
        print(*args, **kwargs)
    dist.barrier()


def print_on_all_ranks(*args, **kwargs):
    dist.barrier()
    for rank in range(dist.get_world_size()):
        if dist.get_rank() == rank:
            print(*args, **kwargs)
        dist.barrier()


class Trainer:
    def __init__(self, dataset, comm):
        self.dataset: DistributedHeteroGraphDataset = dataset
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
            in_channels=self.dataset.num_features,
            out_channels=self.dataset.num_classes,
            num_relations=self.dataset.num_relations,
            hidden_channels=self.model_config.hidden_channels,
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
            find_unused_parameters=True,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_config.lr, weight_decay=5e-4
        )

    def prepare_data(self):
        self.dataset = self.dataset.add_batch_dimension()
        # self.dataset = self.dataset.to(self.device)

    def train(self):
        self.model.train()

        xs, _, edge_type, _ = self.dataset[0]
        comm_plans = self.dataset.get_NCCL_comm_plans()

        # Fetch once; masks/targets are static across epochs
        train_mask = self.dataset.get_mask("train")
        target = self.dataset.get_target("train").flatten()

        xs = [x.to(self.device) for x in xs]
        target = target.to(self.device)
        train_mask = train_mask.to(self.device)

        print_on_all_ranks(f"Rank {self.comm.get_rank()} starting training...")

        print_on_rank_zero("*" * 20 + " Starting Training " + "*" * 20)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        dist.barrier()

        print_on_rank_zero(
            f"Current memory usage before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

        for plan in comm_plans:
            source_plan = plan.source_graph_plan
            dest_plan = plan.dest_graph_plan
            assert (
                dest_plan is not None
            ), "Destination plan should not be None for NCCL communication"
            source_memory = source_plan.memory_usage(unit="GB")["gpu"]
            dest_memory = dest_plan.memory_usage(unit="GB")["gpu"]
            print_on_rank_zero(
                f"Comm Plan - Source Memory: {source_memory:.2f} GB, Destination Memory: {dest_memory:.2f} GB"
            )

        loss = torch.tensor([0.0]).to(self.device)

        for epoch in range(1, self.training_config.epochs + 1):
            # zero grads before forward to avoid dangling reduction state
            self.optimizer.zero_grad(set_to_none=True)

            out = self.model(xs, edge_type, comm_plans)
            #     # print(
            #     #     f"Rank {self.comm.get_rank()} completed forward pass for epoch {epoch}"
            #     # )

            local_train_vertices = out[:, train_mask, :].squeeze(0)

            loss = torch.nn.functional.cross_entropy(
                local_train_vertices, target, reduction="sum"
            )
            local_num_targets = target.size(0)
            global_num_targets = GetGlobalVal(local_num_targets)
            loss = loss / global_num_targets  # Average the loss

            loss.backward()
            self.optimizer.step()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            print_on_rank_zero(f"Epoch {epoch:03d} | loss {loss.item():.4f}")
        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        xs, _, edge_type, _ = self.dataset[0]
        comm_plans = self.dataset.get_NCCL_comm_plans()
        out = self.model(xs, edge_type, comm_plans)

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
