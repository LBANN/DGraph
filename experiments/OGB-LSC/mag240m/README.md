# Processing OGB-LSC MAG240M Dataset

This directory contains the code to preprocess and load the OGB-LSC MAG240M dataset to use with DGraph.

## Prerequisites

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
python DGraph_MAG240M.py --data_dir <path_to_data_directory>
```

Make sure to replace `<path_to_data_directory>` with the path where you want to store the processed data. The script will download the dataset if it is not already present in the specified directory. The processed data will be saved in the same directory.

The processing machine requires at least `128GB` of RAM to process the dataset.




