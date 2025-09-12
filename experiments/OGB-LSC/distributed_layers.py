import torch
from torch import nn
import torch.distributed as dist
from torch.autograd import Function


def _compute_bn_forward(input, learned_gamma=None, learned_beta=None):
    local_sum = torch.mean(input, dim=0)
    global_sum = local_sum.clone()
    num_rows = torch.tensor([input.size(0)], dtype=torch.float32, device=input.device)

    global_num_rows = num_rows.clone()

    dist.all_reduce(global_num_rows, op=dist.ReduceOp.SUM)
    global_mean = global_sum / global_num_rows
    local_var = (input - global_mean) ** 2
    global_var = local_var.clone()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_var, op=dist.ReduceOp.SUM)
    global_var = global_var / global_num_rows

    x_hat = (input - global_mean) / torch.sqrt(global_var + 1e-5)
    if learned_gamma is not None and learned_beta is not None:
        output = x_hat * learned_gamma + learned_beta

    return output, x_hat, global_mean, global_var, global_num_rows


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
    def backward(ctx, grad_output):
        x = ctx.input
        mean = ctx.mean
        var = ctx.var
        # recompute x_hat to save memory
        x_hat = (x - mean) / torch.sqrt(var + 1e-5)
        learned_gamma = ctx.learned_gamma
        learned_beta = ctx.learned_beta
        num_rows = ctx.num_rows

        if learned_gamma is not None and learned_beta is not None:
            local_dbeta = torch.sum(grad_output, dim=0)
            global_dbeta = local_dbeta.clone()
            dist.all_reduce(global_dbeta, op=dist.ReduceOp.SUM)
            local_dgamma = torch.sum(grad_output * x_hat, dim=0)
            global_dgamma = local_dgamma.clone()
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
    def backward(ctx, grad_output):

        learned_gamma = ctx.learned_gamma
        learned_beta = ctx.learned_beta
        mean = ctx.mean
        var = ctx.var
        x_hat = ctx.x_hat
        num_rows = ctx.num_rows
        x = ctx.input

        if learned_gamma is not None and learned_beta is not None:
            local_dbeta = torch.sum(grad_output, dim=0)
            global_dbeta = local_dbeta.clone()
            dist.all_reduce(global_dbeta, op=dist.ReduceOp.SUM)
            local_dgamma = torch.sum(grad_output * x_hat, dim=0)
            global_dgamma = local_dgamma.clone()
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


class DistributedBatchNorm1D(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(DistributedBatchNorm1D, self).__init__()
        self.bn = nn.BatchNorm1d(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, x):
        return self.bn(x)
