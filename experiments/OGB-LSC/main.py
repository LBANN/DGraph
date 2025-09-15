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
import DGraph.Communicator as Comm
from Trainer import Trainer
from config import ModelConfig
import torch.distributed as dist


def main(
    comm_type: str = "nccl",
    dataset: str = "synthetic",
    num_papers: int = 2048,
    num_authors: int = 512,
    num_institutions: int = 16,
    paper_rank_mapping_file: str = "",
    author_rank_mapping_file: str = "",
    institution_rank_mapping_file: str = "",
    data_dir: str = "mag240m/data/MAG240M",
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
        from synthetic.synthetic_dataset import HeterogeneousDataset as Dataset

        graph_dataset = partial(
            Dataset,
            num_papers=num_papers,
            num_authors=num_authors,
            num_institutions=num_institutions,
            num_features=ModelConfig().num_features,
            num_classes=ModelConfig().num_classes,
        )

    elif dataset == "mag240m":
        from mag240m.DGraph_MAG240M import DGraph_MAG240M as Dataset

        assert osp.exists(paper_rank_mapping_file)
        assert osp.exists(author_rank_mapping_file)
        assert osp.exists(institution_rank_mapping_file)
        paper_rank_mapping = torch.load(paper_rank_mapping_file, weights_only=False)
        author_rank_mapping = torch.load(author_rank_mapping_file, weights_only=False)
        institution_rank_mapping = torch.load(
            institution_rank_mapping_file, weights_only=False
        )

        graph_dataset = partial(
            Dataset,
            paper_rank_mappings=paper_rank_mapping,
            author_rank_mappings=author_rank_mapping,
            institution_rank_mappings=institution_rank_mapping,
            data_dir=data_dir,
        )
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    assert comm_type in ["nccl", "nvshmem"]
    comm = Comm.Communicator.init_process_group(comm_type)

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
