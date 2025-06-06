# NVSHMEM-Enabled DGraph

Enabling the NVSHMEM backend in DGraph requires MPI and NVSHMEM to be installed on the system. 

## Pre-requisites

DGraph must be built with NVSHMEM, MPI, and CUDA in order to use the NVSHMEM backend. The setup script will install the appropriate submodules but the dependencies must be installed and available on the system.

DGraph searches for NVSHMEM, MPI, and CUDA based on the following environment variables:
- `NVSHMEM_HOME`
- `MPI_HOME`
- `CUDA_HOME`

If these environment variables are not set, DGraph will not be able to find the necessary. 
Furthermore, the `LD_LIBRARY_PATH` must be set to include the library paths for NVSHMEM, MPI, and CUDA.

### Pre-install check

To check if the necessary dependencies are installed, run the following command:

```shell
python PreInstallCheck.py
```

Note: This script currently only checks for the presence of the environment variables and libraries. It does not check the library or compiler versions and their compatibility with DGraph. 

Since DGraph extends PyTorch with custom CUDA modules, the CUDA version used to compile PyTorch must match the CUDA version used to compile DGraph.

You can check the CUDA version used to compile PyTorch by running the following command:

```shell
python -c "import torch; print(torch.version.cuda)"
```

NVSHMEM compilation information can be usually found by running the `nvshmem-info` command, usually located in the `bin` directory of the NVSHMEM installation.

```shell
$NVSHMEM_HOME/bin/nvshmem-info -b
```

## Building DGraph with NVSHMEM
To build DGraph with NVSHMEM, make sure the environment variables are set and run the following command:

```shell
pip install -e .
```

## Running DGraph with NVSHMEM

DGraph builds on top of the PyTorch distributed package, so it is important to initialize the PyTorch distributed package before using DGraph, but also initialize MPI. Using a distributed launcher simplifies this process. We recommend using [`torchrun-hpc](https://github.com/lbann/HPC-launcher) 

```shell
torchrun-hpc -N<NUM_NODES> -n<NUM_PROCESSES_PER_NODE> NVSHMEM_init.py
```

The script assumes that the launcher starts the processes and `torch.dist` is initialized. If not using a launcher, you must initialize the PyTorch distributed package. 