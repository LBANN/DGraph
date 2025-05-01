## Billion Vertex Graphs

As DGraph allows us to learn extremely large graphs, we push the size of the graphs beyond to train with full graph GNN training. We generate a synthetic graphs with 1 billion vertices.

## Data Generation

### Building the Graph Generator
We provide a fast graph generator to generate large graphs. The generator generates a graph with a given number of vertices and a maximum degree. The generator just requires a `GCC>10.3`. Build the generator in the `Generator` directory
```bash
cd Generator
make
```

### Generating the Graph
The generator takes the number of vertices and the maximum degree as input, and outputs a text file in the METIS graph format. Run the following command to generate a graph with 1 billion vertices with a maximum degree of 5:

```bash
./Generator/graph_generator 1000000000 5 1B5D.graph
```

This will generate an undirected graph with 1 billion vertices and a maximum degree of 5. The graph will be saved in the file `1B5D.graph`. The generator will take a few minutes to run and require `~150GB` of memory. 

The graph will be generated in the METIS format, which is a simple text format that describes the graph. The first line of the file contains the number of vertices and edges. The i-th line of the file contains the neighbors of the i-th vertex. 

### Partition the graph

We assume a there is a working `METIS` installation with flags `i64=1` and `r64=1`. `Parametis` may be useful as well.

To partition the graph in to `<num_partitions>` partitions, run the following command:
```bash
gpmetis 1B5D.graph <num_partitions>
```
This will generate a file `1B5D.graph.part.<num_partitions>` which contains the partitioning of the graph. The i-th line of the file contains the partition id of the i-th vertex. The partition ids are 0-indexed. This also requires `~150GB` of memory (with the flag `-ondisk`).

### Preprocess for DGraph

To finish the graph generation and make the data ready for DGraph, we take the graph file and partition file and run the following command:
```bash 
python preprocess.py --g <graph_file> --p <partition_file> --np <num_partitions> 
```

The script will generate the necessary files for DGraph to run a distributed training partitioned in `<num_partitions>` partitions. 

