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
import fire
import torch
import torch.distributed as dist
from typing import Literal
from DGraph import Communicator
from config import SyntheticDatasetConfig
from functools import partial
import os
import os.path as osp


def main(
    comm_type: Literal["nccl", "nvshmem"] = "nccl",
    dataset: Literal["synthetic", "mag240m"] = "mag240m",
    num_papers: int = 2048,
    num_authors: int = 512,
    num_institutions: int = 16,
    data_dir: str = "lsc_datasets/data/MAG240M",
):
    assert comm_type in ["nccl", "nvshmem"]
    comm = Communicator.init_process_group(comm_type)

    device_id = comm.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    def comm_print(*args, **kwargs):
        comm.barrier()
        if comm.get_rank() == 0:
            print(*args, **kwargs)
        comm.barrier()

    world_size = comm.get_world_size()
    comm_print(f"Communicator initialized with World Size: {world_size}")

    assert dataset in ["synthetic", "mag240m"]
    if dataset == "synthetic":
        from lsc_datasets.synthetic_dataset import (
            SyntheticHeterogeneousDataset as Dataset,
        )

        synthetic_config = SyntheticDatasetConfig(
            num_papers=num_papers,
            num_authors=num_authors,
            num_institutions=num_institutions,
        )

        comm_print(
            f"Setting up synthetic dataset with configuration {synthetic_config}"
        )
        graph_dataset = partial(
            Dataset,
            synthetic_config=synthetic_config,
        )
        comm_print(f"Finished setting up synthetic dataset")

    elif dataset == "mag240m":
        from lsc_datasets.MAG240M_dataset import DGraph_MAG240M_Dataset as Dataset

        comm_print(f"Setting up MAG240M dataset")
        cur_dir = osp.dirname(os.path.abspath(__file__))
        data_dir = osp.join(cur_dir, data_dir)
        # print(f"Data directory: {data_dir}")
        graph_dataset = partial(
            Dataset,
            data_dir=data_dir,
            comm_plan_only=True,
        )
        comm_print(f"Finished setting up MAG240M dataset")

    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    graph_dataset = graph_dataset(comm=comm)

    comm_plans = graph_dataset.get_NCCL_comm_plans()

    for i, comm_plan in enumerate(comm_plans):
        comm_plan = comm_plan.source_graph_plan
        comm_print(f"Comm Plan # {i}")
        comm_print(f"Num Local Vertices: {comm_plan.num_local_vertices}")
        comm_print(f"Num Boundary Vertices: {comm_plan.boundary_vertex_splits}")
        comm_print(f"Num Local Edges: {comm_plan.num_local_edges}")
        comm_print(f"Num Boundary Edges: {comm_plan.boundary_edge_splits}")


if __name__ == "__main__":
    fire.Fire(main)
