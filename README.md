# LBANN: Livermore Big Artificial Neural Network Toolkit

The Livermore Big Artificial Neural Network toolkit (LBANN) is an
open-source, HPC-centric, deep learning training framework that is
optimized to compose multiple levels of parallelism.

LBANN provides model-parallel acceleration through domain
decomposition to optimize for strong scaling of network training.  It
also allows for composition of model-parallelism with both data
parallelism and ensemble training methods for training large neural
networks with massive amounts of data.  LBANN is able to advantage of
tightly-coupled accelerators, low-latency high-bandwidth networking,
and high-bandwidth parallel file systems.

##  DGraph
DGraph is deep learning library for training graph neural networks at scale that is built on top of PyTorch.


To install DGraph, clone the repository and install with pip:
```shell
pip install -e .[ogb]
```

### Running tests
To run the tests, use the following command:
```shell
python -m pytest tests/
```

### Requirements
DGraph requires the following packages:
- PyTorch >= 2.1.0
- NumPy
- pytest
- mpi4py
- cupy

For the full list of requirements, see `requirements.txt`.

DGraph also requires the following libraries:
- NCCL
- NVSHMEM

## Publications

A list of publications, presentations and posters are shown
[here](https://lbann.readthedocs.io/en/latest/publications.html).

## Reporting issues
Issues, questions, and bugs can be raised on the [Github issue
tracker](https://github.com/LBANN/lbann/issues).
