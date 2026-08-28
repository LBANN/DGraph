python -m benchmarks.bench_compute --model edge --sweep edges --min 1000 --max 1000000 --steps 10 \
  --fixed-value 200000 --feature-dim 512 --warmup 5 --trials 20 --output data/compute_edge_eswp_test.json --seed 4
  
python -m benchmarks.bench_compute --model gcn --sweep edges --min 1000 --max 1000000 --steps 10 \
  --fixed-value 200000 --feature-dim 512 --warmup 5 --trials 20 --output data/compute_gcn_eswp_test.json --seed 4

python -m analysis.fit_primitives     --compute-gcn data/compute_gcn_*swp_test.json  \
  --compute-edge data/compute_edge_*swp_test.json   --output data/fitted_primitives.json

python -m visualization.plot_compute     --gcn-vertex  data/compute_gcn_vswp_test.json \
  --gcn-edge data/compute_gcn_eswp_test.json  --edge-vertex data/compute_edge_vswp_test.json \
  --edge-edge data/compute_edge_eswp_test.json --primitives  data/fitted_primitives.json --output figures/compute
