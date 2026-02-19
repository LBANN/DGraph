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
from torch import nn
import torch.distributed as dist
from torch.autograd import Function
from typing import Callable


def _compute_bn_forward(input, learned_gamma=None, learned_beta=None):
    local_sum = torch.mean(input, dim=0)
    global_sum = local_sum.clone()
    num_rows = torch.tensor([input.size(0)], dtype=torch.float32, device=input.device)

    global_num_rows = num_rows.clone()

    dist.all_reduce(global_num_rows, op=dist.ReduceOp.SUM)
    global_mean = global_sum / global_num_rows
    local_var = ((input - global_mean) ** 2).sum(dim=0)
    global_var = local_var.clone()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_var, op=dist.ReduceOp.SUM)
    global_var = global_var / global_num_rows

    x_hat = (input - global_mean) / torch.sqrt(global_var + 1e-5)
    if learned_gamma is not None and learned_beta is not None:
        output = x_hat * learned_gamma + learned_beta

    return output, x_hat, global_mean, global_var, global_num_rows


def _compute_bn_backward(
    grad_output, x, x_hat, mean, var, num_rows, learned_gamma=None, learned_beta=None
):
    if learned_gamma is not None and learned_beta is not None:
        local_dbeta = torch.sum(grad_output, dim=0)
        global_dbeta = local_dbeta.clone().unsqueeze(0)
        dist.all_reduce(global_dbeta, op=dist.ReduceOp.SUM)
        local_dgamma = torch.sum(grad_output * x_hat, dim=0)
        global_dgamma = local_dgamma.clone().unsqueeze(0)
        dist.all_reduce(global_dgamma, op=dist.ReduceOp.SUM)
        dx_hat = grad_output * learned_gamma
    else:
        dx_hat = grad_output
        global_dgamma = None
        global_dbeta = None

    local_dvar = torch.sum(dx_hat * (x - mean) * -0.5 * (var + 1e-5) ** 2, dim=0)
    global_dvar = local_dvar.clone()
    dist.all_reduce(global_dvar, op=dist.ReduceOp.SUM)

    local_dmean = torch.sum(
        dx_hat * -1 / torch.sqrt(var + 1e-5), dim=0
    ) + global_dvar * torch.mean(-2 * (x - mean), dim=0)
    global_dmean = local_dmean.clone()
    dist.all_reduce(global_dmean, op=dist.ReduceOp.SUM)
    dx = (
        (dx_hat / torch.sqrt(var + 1e-5))
        + (global_dvar * 2 * (x - mean) / num_rows)
        + (global_dmean / num_rows)
    )
    return dx, global_dgamma, global_dbeta


class DistributedBN_with_Recompute(Function):
    @staticmethod
    def forward(ctx, input, learned_gamma=None, learned_beta=None):
        ctx.save_for_backward(input)
        ctx.learned_gamma = learned_gamma
        ctx.learned_beta = learned_beta
        output, _, global_mean, global_var, global_num_rows = _compute_bn_forward(
            input, learned_gamma, learned_beta
        )
        ctx.mean = global_mean
        ctx.var = global_var
        ctx.input = input
        ctx.num_rows = global_num_rows
        return output, global_mean, global_var

    @staticmethod
    def backward(ctx, grad_output, grad_mean, grad_var):
        x = ctx.input
        mean = ctx.mean
        var = ctx.var
        # recompute x_hat to save memory
        x_hat = (x - mean) / torch.sqrt(var + 1e-5)
        learned_gamma = ctx.learned_gamma
        learned_beta = ctx.learned_beta
        num_rows = ctx.num_rows

        dx, global_dgamma, global_dbeta = _compute_bn_backward(
            grad_output, x, x_hat, mean, var, num_rows, learned_gamma, learned_beta
        )

        return dx, global_dgamma, global_dbeta


class DistributedBN_Impl(Function):
    @staticmethod
    def forward(ctx, input, learned_gamma=None, learned_beta=None):
        output, x_hat, global_mean, global_var, global_num_rows = _compute_bn_forward(
            input, learned_gamma, learned_beta
        )

        ctx.save_for_backward(x_hat)
        ctx.learned_gamma = learned_gamma
        ctx.learned_beta = learned_beta
        ctx.mean = global_mean
        ctx.var = global_var
        ctx.num_rows = global_num_rows
        ctx.input = input
        ctx.x_hat = x_hat
        return output, global_mean, global_var

    @staticmethod
    def backward(ctx, grad_output, grad_mean, grad_var):

        learned_gamma = ctx.learned_gamma
        learned_beta = ctx.learned_beta
        mean = ctx.mean
        var = ctx.var
        x_hat = ctx.x_hat
        num_rows = ctx.num_rows
        x = ctx.input
        dx, global_dgamma, global_dbeta = _compute_bn_backward(
            grad_output, x, x_hat, mean, var, num_rows, learned_gamma, learned_beta
        )

        return dx, global_dgamma, global_dbeta


class DistributedBatchNorm1D(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        recompute=False,
    ):
        super(DistributedBatchNorm1D, self).__init__()
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(1, num_features))
            self.register_buffer("running_var", torch.ones(1, num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)
        self.recompute = recompute
        if recompute:
            self.bn: Callable = DistributedBN_with_Recompute.apply
        else:
            self.bn: Callable = DistributedBN_Impl.apply

    def forward(self, x):
        if x.dim() == 3:
            assert x.size(0) == 1, "only mini-batch size 1 is supported"
            x = x.squeeze(0)
        elif x.dim() != 2:
            raise ValueError("Expected 2D or 3D input (got {}D input)".format(x.dim()))

        if self.training:
            if self.track_running_stats:
                self.num_batches_tracked += 1
            y, mean, var = self.bn(x, self.gamma, self.beta)

            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (
                        1 - self.momentum
                    ) * self.running_mean + self.momentum * mean
                    self.running_var = (
                        1 - self.momentum
                    ) * self.running_var + self.momentum * var
        else:
            y = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            if self.gamma is not None and self.beta is not None:
                y = y * self.gamma + self.beta

        if y.dim() == 2:
            y = y.unsqueeze(0)
        return y


def GetGlobalVal(local_val):
    """Get the global sum of a local value across all ranks."""
    global_val = torch.tensor([local_val]).cuda()
    dist.all_reduce(global_val, op=dist.ReduceOp.SUM)
    return global_val.item()
