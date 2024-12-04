## Running Distributed Tests

An easy way to run the distributed tests I have is to use torchrun in combination with pytest. 

For example, the following code will run the `test_nccl_backend` test on 2 GPUs on a single node:

```
torchrun --nnodes 1 --nproc-per-node 2 -m pytest test_nccl_backend.py
```

### Testing MPI-Enabled Backends

To run the MPI tests, you will need to use the `mpirun` (or equivalant wrapper) command. For example, the following code will run the `test_mpi_backend` test on 2 GPUs on a single node using SLURM:

```
srun -n 2 python -m pytest test_mpi_backend.py
```
