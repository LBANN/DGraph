## GraphCast

DGraph implementatio of GraphCast.

The GraphCast implementation is on the implementation available from [DeepMind](https://github.com/google-deepmind/graphcast)
 and [NVIDIA's Modulus implementation](https://github.com/NVIDIA/modulus/tree/main/examples/weather/graphcast).

### Simulated Dataset

GraphCast is originally trained on the ERA5 dataset from 1979-2017. The dataset is quite large and require substantial computational resources. There for we provide a simulated dataset that matches the original graph structure and connectivity but uses random data for atmospheric variables.

### Running Test

Run the single process test with the following command from the
`tests` directory:

```bash
DGRAPH_GRAPHCAST_DISTRIBUTED_TESTS=0 python3 -m pytest
```

### How to run

To run the GraphCast model no additional dependencies are required. Install DGraph with

```bash
pip install dgraph[graphcast]
```

The model, data, and training configuration are in `graphcast_config.py`. Change the configuration as needed.

Run the single process GraphCast model as a test run with the following command:
```bash
python main.py --test_run
```

Run with benchmarking with the following command:
```bash
python main.py --benchmark
```
***Note: *** The graph requires a large amount of memory so better to do run on the CPU and a machine with a large amount of memory.
