import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import os


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def make_experiment_log(fname, rank):
    if rank == 0:
        with open(fname, "w") as f:
            f.write("")


def write_experiment_log(log: str, fname: str, rank: int):
    if rank == 0:
        with open(fname, "a") as f:
            f.write(log + "\n")


def dist_print_ephemeral(
    msg,
    rank,
):
    if rank == 0:
        print(msg, end="\r")


def visualize_trajectories(trajectories, title, figsave, rank):
    if rank != 0:
        return
    mean = np.mean(trajectories, axis=0)
    std = np.std(trajectories, axis=0)
    x = np.arange(len(mean))

    fig, ax = plt.subplots()

    ax.plot(x, mean, "-")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    fig.savefig(figsave)


def safe_create_dir(directory, rank):
    if rank == 0:
        if not os.path.exists(directory):
            os.makedirs(directory)


def calculate_accuracy(pred, labels):
    pred = pred.argmax(dim=1)
    correct = pred.eq(labels).sum().item()
    if len(labels) > 0:
        return correct / len(labels) * 100
    else:
        return 0.0
