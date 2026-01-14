# Directed Heterogeneous Graphs on DGraph

`DGraph` supports arbitrary graph types, GNNs, and structures for distributed training. This example shows how to use `DGraph` to train a Relational Graph Attention Network ([RGAT](https://arxiv.org/abs/1703.06103)) on the [OGB-LSC MAG240M](https://ogb.stanford.edu/docs/lsc/mag240m/) dataset, which is a large-scale heterogeneous graph with three types of nodes (paper, author, institution) and three types of edges (paper->paper, paper->author, author->institution). 

## Requirements

Make sure you have the following packages installed:
- `torch`
- `torch_geometric`
- `ogb`
- `torch_sparse`
- `numpy`
- `tqdm`
- `fire`

## Preprocessing the dataset
The MAG240M dataset is a fairly large graph dataset and requires some preprocessing before it can be used with DGraph, and takes a while to process. The following script processes the dataset and saves the processed data in a directory.

```bash
torchrun-hpc -N <number of nodes> -n <number of processes> setup_dataset_comms.py --comm_type nccl --dataset mag240m --data_dir <path_to_data_directory>
```

Make sure to replace `<path_to_data_directory>` with the path where you want to store the processed data. The script will download the dataset if it is not already present in the specified directory. The processed data will be saved in the same directory.

The processing machine requires at least `128GB` of RAM to process the dataset.


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
