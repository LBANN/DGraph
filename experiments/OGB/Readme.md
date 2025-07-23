# OGBN Experiments

This folder contains the experiments for the OGBN datasets. The experiments are run on the OGBN datasets using the `main.py` script. The script is run with the following command:

```bash
python main.py --dataset ogbn-dataset-name  --backend backend --lr lr --epochs epochs --runs runs --log_dir log-dir
```

The arguments are as follows:
- `--dataset`: The name of the OGBN dataset to run the experiment on. Possible values are 'arxiv', 'products', 'proteins', 'papers100M'
- `--backend`: The backend to use for the experiment. Possible values are 'single', 'nccl', 'nvshmem', and 'mpi'. Default is 'single'
- `--lr`: The learning rate for the experiment. Default is 0.01
- `--epochs`: The number of epochs to run the experiment for. Default is 100
- `--runs`: The number of runs to run the experiment for. Default is 10
- `--log_dir`: The directory to save the logs for the experiment. Default is 'logs'

All runs are on the full graph. The script saves the logs in the `log_dir` directory. The logs contain the training and validation loss and accuracy for each run. The script also saves the model with the best validation accuracy.

### Distributed Training

DGraph supports distributed training using the `nccl`, `nvshmem`, and `mpi` backends. 

In order to run the experiments with the `nccl` backend, run the following command:

```bash
torchrun-hpc -N <nodes> -n <gpus> main.py --backend nccl --lr lr --epochs epochs --runs runs --node_rank_placement_file <file_dir> --log_dir log-dir
```
You may have to turn ``--xargs=--mpibind=off`` and ``--xargs=--gpu-bind=none`` in your Slurm script to avoid binding issues.

**Note that we use `torchrun-hpc` instead of `torchrun` **, the run command may vary based on your environment.



### Additional Notes
The experiments use some additional libraries. Use the [ogb] option
when installing with pip.

- fire: For command line argument parsing
