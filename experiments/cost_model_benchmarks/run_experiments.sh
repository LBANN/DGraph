#!/usr/bin/env bash
# Step-by-step driver for the cost-model benchmark pipeline (see README.md's
# "Pipeline" section) on a single node, sweeping the distributed
# (torchrun-launched) benchmarks over GPU_COUNTS.
#
# Single-node only: no inter-node ping-pong/concurrency/crossover runs are
# included, since this script never launches more than one node.
#
# Usage:
#   ./run_experiments.sh
#
# Override any config variable via the environment, e.g.:
#   GPU_COUNTS="2 4 8" MODEL=edge ./run_experiments.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# An unmatched glob (e.g. a partial re-run missing an expected output file)
# should surface as a clear argparse error, not a literal glob-pattern
# string silently passed through to a file open() call.
shopt -s nullglob

# --- Config ------------------------------------------------------------
GPU_COUNTS=(${GPU_COUNTS:-2 4})        # world sizes to sweep via torchrun --nproc_per_node
SEED=${SEED:-42}
FEATURE_DIM=${FEATURE_DIM:-128}
MODEL=${MODEL:-gcn}                    # gcn | edge
GRAPH=${GRAPH:-erdos_renyi}            # erdos_renyi | sbm
PARTITIONER=${PARTITIONER:-balanced}   # random | balanced | metis
WARMUP=${WARMUP:-10}
TRIALS=${TRIALS:-50}
CROSSOVER_SIZES=${CROSSOVER_SIZES:-1000,10000,100000,1000000}
E2E_VERTEX_SIZES=(${E2E_VERTEX_SIZES:-10000 100000 1000000})
DATA_DIR=${DATA_DIR:-data}
FIG_DIR=${FIG_DIR:-figures}

# fit_overhead.py / compute_predictions.py default to "world_size <= 8",
# which never holds anything out for a single-node run limited to <=8 GPUs
# (held-out MAPE would always report NaN over an empty set). Hold out the
# largest swept GPU count instead, so held-out MAPE reflects genuine
# extrapolation across the GPU_COUNTS actually being run.
MAX_GPU_COUNT=$(printf '%s\n' "${GPU_COUNTS[@]}" | sort -n | tail -1)
FIT_FILTER=${FIT_FILTER:-"world_size < ${MAX_GPU_COUNT}"}

mkdir -p "$DATA_DIR" "$FIG_DIR"

step() { echo; echo "=== $* ==="; }

# -------------------------------------------------------------------------
# 1. Single-GPU primitive microbenchmarks (1.3 compute, 1.4 gather) — no
#    torchrun, GPU-count independent.
#
#    fit_primitives.py fits GCN and edge-conditioned compute costs
#    independently (--compute-gcn / --compute-edge), regardless of which
#    single MODEL the distributed crossover/end-to-end sweep below uses —
#    so both model types are always benchmarked here.
# -------------------------------------------------------------------------
for m in gcn edge; do
  step "1.3 compute -- ${m}, vertex sweep"
  python -m benchmarks.bench_compute --model "${m}" --sweep vertices \
    --min 1000 --max 1000000 --steps 10 --fixed-value 200000 \
    --feature-dim "${FEATURE_DIM}" --warmup "${WARMUP}" --trials "${TRIALS}" \
    --output "${DATA_DIR}/compute_${m}_vswp.json" --seed "${SEED}"

  step "1.3 compute -- ${m}, edge sweep"
  python -m benchmarks.bench_compute --model "${m}" --sweep edges \
    --min 1000 --max 1000000 --steps 10 --fixed-value 200000 \
    --feature-dim "${FEATURE_DIM}" --warmup "${WARMUP}" --trials "${TRIALS}" \
    --output "${DATA_DIR}/compute_${m}_eswp.json" --seed "${SEED}"
done

step "1.4 gather -- contiguous / clustered / random"
for dist_name in contiguous clustered random; do
  python -m benchmarks.bench_gather --distribution "${dist_name}" \
    --min-k 1000 --max-k 10000000 --steps 20 --N 20000000 \
    --feature-dim "${FEATURE_DIM}" --cluster-size 64 \
    --warmup "${WARMUP}" --trials "${TRIALS}" \
    --output "${DATA_DIR}/gather_${dist_name}.json" --seed "${SEED}"
done

# -------------------------------------------------------------------------
# 2. Intra-node ping-pong (1.1) -- fixed 2-rank NVLink/PCIe pair test,
#    independent of GPU_COUNTS.
# -------------------------------------------------------------------------
step "1.1 pingpong -- intra-node (2 ranks)"
torchrun --nnodes 1 --nproc_per_node 2 -m benchmarks.bench_pingpong \
  --mode intra --min-bytes 64 --max-bytes 67108864 --steps 21 \
  --warmup 20 --trials 100 \
  --output "${DATA_DIR}/pingpong_intra.json" --seed "${SEED}"

# -------------------------------------------------------------------------
# 3. Distributed benchmarks (2.1 end-to-end, 2.2 crossover), swept over
#    GPU_COUNTS via torchrun.
# -------------------------------------------------------------------------
for K in "${GPU_COUNTS[@]}"; do
  step "2.2 crossover -- K=${K} GPUs"
  torchrun --nnodes 1 --nproc_per_node "${K}" -m benchmarks.bench_crossover \
    --graph "${GRAPH}" --graph-sizes "${CROSSOVER_SIZES}" \
    --avg-degree 20 --feature-dim "${FEATURE_DIM}" --model "${MODEL}" \
    --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
    --output "${DATA_DIR}/crossover_K${K}.json" --seed "${SEED}"

  for N in "${E2E_VERTEX_SIZES[@]}"; do
    step "2.1 end-to-end -- K=${K} GPUs, N=${N}"
    torchrun --nnodes 1 --nproc_per_node "${K}" -m benchmarks.bench_end_to_end \
      --graph "${GRAPH}" --num-vertices "${N}" --avg-degree 20 \
      --feature-dim "${FEATURE_DIM}" --model "${MODEL}" \
      --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
      --output "${DATA_DIR}/e2e_K${K}_N${N}.json" --seed "${SEED}"
  done
done

# Explicit list of exactly the e2e files this script just produced — not a
# e2e_*.json wildcard glob, which would also match any stray file sharing
# that prefix (e.g. a leftover data/e2e_K8_F128_er_bal.json from manually
# running a docstring example) and silently pool it into the fit/held-out
# split as if it were part of this run's matrix.
E2E_FILES=()
for K in "${GPU_COUNTS[@]}"; do
  for N in "${E2E_VERTEX_SIZES[@]}"; do
    E2E_FILES+=("${DATA_DIR}/e2e_K${K}_N${N}.json")
  done
done

# -------------------------------------------------------------------------
# 4. Fit primitives, fit overhead, apply the assembled model.
# -------------------------------------------------------------------------
step "fit_primitives"
# NOTE: explicit filenames, not a wildcard glob. A glob like
# compute_gcn_*.json would also match unrelated stale files that happen to
# share the prefix (e.g. data/compute_gcn_eswp_test.json left over from
# run_local_compute_tests.sh, which uses a different --feature-dim) and
# silently pool incompatible data into one regression.
python -m analysis.fit_primitives \
  --pingpong-intra "${DATA_DIR}/pingpong_intra.json" \
  --compute-gcn "${DATA_DIR}/compute_gcn_vswp.json" "${DATA_DIR}/compute_gcn_eswp.json" \
  --compute-edge "${DATA_DIR}/compute_edge_vswp.json" "${DATA_DIR}/compute_edge_eswp.json" \
  --gather-contiguous "${DATA_DIR}/gather_contiguous.json" \
  --gather-clustered "${DATA_DIR}/gather_clustered.json" \
  --gather-random "${DATA_DIR}/gather_random.json" \
  --output "${DATA_DIR}/fitted_primitives.json"

step "fit_overhead (fit-filter: ${FIT_FILTER})"
python -m analysis.fit_overhead \
  --primitives "${DATA_DIR}/fitted_primitives.json" \
  --e2e-runs "${E2E_FILES[@]}" \
  --fit-filter "${FIT_FILTER}" \
  --output "${DATA_DIR}/fitted_overhead.json"

step "compute_predictions (fit-filter: ${FIT_FILTER})"
python -m analysis.compute_predictions \
  --primitives "${DATA_DIR}/fitted_primitives.json" \
  --overhead "${DATA_DIR}/fitted_overhead.json" \
  --e2e-runs "${E2E_FILES[@]}" \
  --fit-filter "${FIT_FILTER}" \
  --output "${DATA_DIR}/predictions.json"

# -------------------------------------------------------------------------
# 5. Visualization.
# -------------------------------------------------------------------------
step "visualization"
python -m visualization.plot_compute \
  --gcn-vertex "${DATA_DIR}/compute_gcn_vswp.json" --gcn-edge "${DATA_DIR}/compute_gcn_eswp.json" \
  --edge-vertex "${DATA_DIR}/compute_edge_vswp.json" --edge-edge "${DATA_DIR}/compute_edge_eswp.json" \
  --primitives "${DATA_DIR}/fitted_primitives.json" --output "${FIG_DIR}/compute"

python -m visualization.plot_gather \
  --contiguous "${DATA_DIR}/gather_contiguous.json" --clustered "${DATA_DIR}/gather_clustered.json" \
  --random "${DATA_DIR}/gather_random.json" --fitted "${DATA_DIR}/fitted_primitives.json" \
  --output "${FIG_DIR}/gather"

python -m visualization.plot_pingpong \
  --intra "${DATA_DIR}/pingpong_intra.json" --primitives "${DATA_DIR}/fitted_primitives.json" \
  --output "${FIG_DIR}/pingpong"

for K in "${GPU_COUNTS[@]}"; do
  python -m visualization.plot_crossover --input "${DATA_DIR}/crossover_K${K}.json" \
    --output "${FIG_DIR}/crossover_K${K}.png"
done

python -m visualization.plot_validation --predictions "${DATA_DIR}/predictions.json" \
  --color-by world_size --output "${FIG_DIR}/validation"

echo
echo "Done. Data in ${DATA_DIR}/, figures in ${FIG_DIR}/."
