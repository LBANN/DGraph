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
from functools import partial
import os.path as osp
from DGraph import Communicator
from Trainer import Trainer
from config import SyntheticDatasetConfig
import torch.distributed as dist


def _load_optional_file(file_path: str):
    if osp.exists(file_path):
        return torch.load(file_path, weights_only=False)
    return None


def main(
    comm_type: str = "nccl",
    dataset: str = "synthetic",
    num_papers: int = 2048,
    num_authors: int = 512,
    num_institutions: int = 16,
    optimized_graph_structure: bool = True,
    paper_rank_mapping_file: str = "",
    author_rank_mapping_file: str = "",
    institution_rank_mapping_file: str = "",
    data_dir: str = f"{osp.dirname(osp.abspath(__file__))}/lsc_datasets/data/MAG240M",
):
    """Main function to run DGraph experiments on OGB-LSC datasets.

    Args:
        comm_type (str): Type of communicator to use. Options are 'nccl' and
                         'nvshmem'. Default is 'nccl'.
        dataset (str): Dataset to use. Options are 'synthetic' and 'mag240m'.
                       Default is 'synthetic'.
        num_papers (int): Number of paper nodes to use in the synthetic dataset.
                          Default is 2048.
        num_authors (int): Number of author nodes to use in the synthetic dataset.
                           Default is 512.
        num_institutions (int): Number of institution nodes to use in the synthetic
                                dataset. Default is 16.
        paper_rank_mapping_file (str): Path to the paper rank mapping file for
                                       mag240m dataset. Default is ''.
        author_rank_mapping_file (str): Path to the author rank mapping file for
                                        mag240m dataset. Default is not set.
        institution_rank_mapping_file (str): Path to the institution rank mapping
                                           file for mag240m dataset. Default is not set.
        data_dir (str): Path to the mag240m dataset directory. Default is
                        'mag240m/data/MAG240M'.
    """
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
        graph_dataset = partial(
            Dataset,
            synthetic_config=synthetic_config,
        )

    elif dataset == "mag240m":
        from lsc_datasets.MAG240M_dataset import DGraph_MAG240M_Dataset as Dataset

        graph_dataset = partial(
            Dataset,
            paper_rank_mappings=_load_optional_file(paper_rank_mapping_file),
            author_rank_mappings=_load_optional_file(author_rank_mapping_file),
            institution_rank_mappings=_load_optional_file(
                institution_rank_mapping_file
            ),
            data_dir=data_dir,
            comm_plan_only=optimized_graph_structure,
        )
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    assert comm_type in ["nccl", "nvshmem"]
    comm = Communicator.init_process_group(comm_type)

    device_id = comm.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    comm.barrier()
    print(f"Running with {comm.get_world_size()} ranks. Rank: {comm.get_rank()}")

    graph_dataset = graph_dataset(comm=comm)

    trainer = Trainer(graph_dataset, comm)
    trainer.prepare_data()
    trainer.train()
    comm.destroy()

    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    fire.Fire(main)
