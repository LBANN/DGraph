# Directed Heterogeneous Graphs on DGraph

`DGraph` supports arbitrary graph types, GNNs, and structures for distributed training. This example shows how to use `DGraph` to train a Relational Graph Attention Network ([RGAT](https://arxiv.org/abs/1703.06103)) on the [OGB-LSC MAG240M](https://ogb.stanford.edu/docs/lsc/mag240m/) dataset, which is a large-scale heterogeneous graph with three types of nodes (paper, author, institution) and three types of edges (paper->paper, paper->author, author->institution). 

## Requirements

- fire

## Data preparation
The dataset is fairly large (over 100GB). Please follow the instructions in the `mag240m` folder to download and preprocess the dataset.

## Training
To train RGAT on a synthetic dataset, run the following command:

```bash
torchrun-hpc -N <number of nodes> -n <number of processes> main.py \
--dataset synthetic --num_papers <number of paper vertices> \
--num_authors <number of author vertices> --num_institutions <number of institution
```

To train RGAT on the MAG240M dataset, run the following command:

```bash
torchrun-hpc -N <number of nodes> -n <number of processes> main.py --dataset mag240m \
--data-path <path to the mag240m folder> 
```
