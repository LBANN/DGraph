## Microbenchmarks for DGraph Communication primitives

### Overview

This directory contains microbenchmarks for DGraph communication primitives with the various backends. 


### Communication Patterns

DGraph does fine-grained communication that is dependent on the graph structure or data. 
So for benchmarks, a communication pattern must be provided to the benchmarking code.

The `Gather` and `Scatter` patterns are provided using two dataclasses that encapsulate all the information needed. They can be found in `graph_utils.py`.

```python
@dataclass
class GatherGraphData:
    """Dataclass to store graph data."""

    vertex_data: torch.Tensor           # Data associated with each vertex
    vertex_rank_mapping: torch.Tensor   # Where each vertex is located
    edge_rank_placement: torch.Tensor   # Where each edge is located
    edge_src_rank: torch.Tensor         # Rank of the source vertex of each edge
    edge_indices: torch.Tensor          # Vertex index of the source vertex of each edge 
```


```python
@dataclass
class ScatterGraphData:
    """Dataclass to store graph data."""

    vertex_data: torch.Tensor           # Data associated with each vertex
    data_rank_mapping: torch.Tensor     # Where each data is located
    edge_rank_placement: torch.Tensor   # Where each edge is located
    edge_dst_rank: torch.Tensor         # Rank of the destination vertex of each edge
    edge_indices: torch.Tensor          # Vertex index of the destination vertex of each num_local_vertices: int             # Number of vertices on each rank   
```

*** New communication patterns can be added to the benchmarking code by creating new instances of these dataclasses. ***

### Running the benchmarks

Run the benchmarks using the following command:

```shell
torchrun --nnodes <N> --nproc-per-node <P> testNCCL.py
```

To run the NVSHMEM benchmarks, use the following command:

```shell
srun -N <N> --ntasks-per-node=<P> testNVSHMEM.py
```