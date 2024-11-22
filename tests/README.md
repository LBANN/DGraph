## Running Distributed Tests

An easy way to run the distributed tests I have is to use torchrun in combination with pytest. 

For example, the following code will run the `test_nccl_backend` test on 2 GPUs on a single node:

```
torchrun --nnodes 1 --nproc-per-node 2 -m pytest test_nccl_backend.py
```