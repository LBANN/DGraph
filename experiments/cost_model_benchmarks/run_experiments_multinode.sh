set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
shopt -s nullglob

# =======================================================================
# Multi-node benchmark driver -- uses hpc-launcher's `torchrun-hpc`
# (https://github.com/llnl/HPC-launcher) instead of srun+torchrun.
#
# torchrun-hpc builds and submits its own SLURM batch script per
# invocation, which changes the operating model versus the old
# srun-inside-an-allocation approach:
#   - This script no longer needs to run inside a pre-existing
#     salloc/sbatch allocation. Run it directly, e.g. from a login node.
#   - Every launch() call below requests its own (nnodes, procs/node)
#     topology and is scheduled independently. Fixed-topology steps
#     (pingpong-inter, concurrency) are no longer constrained to fit
#     inside whatever NNODES was chosen for the full-scale crossover/e2e
#     runs -- see the notes at each call site below.
#   - MASTER_ADDR/MASTER_PORT/node_rank are handled internally by
#     torchrun-hpc; no rendezvous port bookkeeping is needed here.
#   - torchrun-hpc changes into its own launch directory (timestamped, by
#     default) before running the command, and *relative* path arguments
#     resolve against that directory, not this script's cwd. DATA_DIR and
#     FIG_DIR are therefore canonicalized to absolute paths below.
#
# Assumption: torchrun-hpc blocks until the submitted job finishes
# (mirroring srun's blocking behavior), so that the files each step
# produces are on disk before the next step (or the fit/plot stage) reads
# them. Each launch() call is followed by a check that its expected
# output file actually landed, so the script fails fast with a clear
# message if that assumption doesn't hold for your installed version.
# If in doubt, sanity-check with `torchrun-hpc --dry-run` first.
# =======================================================================

if ! command -v torchrun-hpc >/dev/null 2>&1; then
  echo "[error] torchrun-hpc not found on PATH." >&2
  echo "        Install hpc-launcher with: pip install hpc-launcher" >&2
  exit 1
fi

# --- Config ------------------------------------------------------------
NNODES=${NNODES:-2}
if [[ "${NNODES}" -lt 2 ]]; then
  echo "[error] NNODES=${NNODES}; this script needs >= 2." >&2
  echo "        For single-node runs use ./run_experiments.sh instead." >&2
  exit 1
fi

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
SEED=${SEED:-42}
FEATURE_DIM=${FEATURE_DIM:-128}
MODEL=${MODEL:-gcn}
GRAPH=${GRAPH:-erdos_renyi}
PARTITIONER=${PARTITIONER:-balanced}
WARMUP=${WARMUP:-10}
TRIALS=${TRIALS:-50}
CONCURRENCY_BYTES=${CONCURRENCY_BYTES:-16777216}   # 16 MiB
CROSSOVER_SIZES=${CROSSOVER_SIZES:-1000,10000,100000,1000000}
E2E_VERTEX_SIZES=(${E2E_VERTEX_SIZES:-10000 100000 1000000})
SINGLE_NODE_GPU_COUNTS=(${SINGLE_NODE_GPU_COUNTS:-2 4})  # must match run_experiments.sh
DATA_DIR=${DATA_DIR:-data}
FIG_DIR=${FIG_DIR:-figures}

# torchrun-hpc scheduling options. All optional -- leave unset to let
# hpc-launcher auto-detect system defaults (queue, time limit, account).
JOB_NAME_PREFIX=${JOB_NAME_PREFIX:-gnn_bench}
TORCHRUN_HPC_QUEUE=${TORCHRUN_HPC_QUEUE:-}
TORCHRUN_HPC_TIME_LIMIT=${TORCHRUN_HPC_TIME_LIMIT:-}   # minutes
TORCHRUN_HPC_ACCOUNT=${TORCHRUN_HPC_ACCOUNT:-}
TORCHRUN_HPC_EXCLUSIVE=${TORCHRUN_HPC_EXCLUSIVE:-1}    # non-empty = pass --exclusive
TORCHRUN_HPC_EXTRA_ARGS=${TORCHRUN_HPC_EXTRA_ARGS:-}   # e.g. "-r mpi --comm-backend NCCL"

K=$((NNODES * GPUS_PER_NODE))   # total world size for the full-scale runs

mkdir -p "$DATA_DIR" "$FIG_DIR"
# Canonicalize to absolute paths -- see launch-directory note above.
DATA_DIR=$(cd "${DATA_DIR}" && pwd)
FIG_DIR=$(cd "${FIG_DIR}" && pwd)

echo "Target topology: ${NNODES} nodes x ${GPUS_PER_NODE} GPU/node -> K=${K}"
echo "Each step below submits (and waits on) its own torchrun-hpc job."

step() { echo; echo "=== $* ==="; }

# Fails loudly if a step's expected output file didn't land -- see the
# blocking-behavior assumption noted at the top of this file.
expect_output() {
  if [[ ! -s "$1" ]]; then
    echo "[error] expected output file '$1' was not produced by the last step." >&2
    echo "        If torchrun-hpc returned before the job finished, this" >&2
    echo "        script's blocking assumption doesn't hold -- see header." >&2
    exit 1
  fi
}

# Runs one Python module under torchrun-hpc. torchrun-hpc builds and
# submits the SLURM batch script itself; no srun/torchrun/rendezvous
# plumbing is needed here.
#   $1 = nnodes, $2 = procs/node, $3 = python module (dotted path),
#   rest = module args.
launch() {
  local nnodes=$1 nproc=$2 module=$3; shift 3
  local -a extra=()
  [[ -n "${TORCHRUN_HPC_EXCLUSIVE}" ]] && extra+=(--exclusive)
  [[ -n "${TORCHRUN_HPC_QUEUE}" ]] && extra+=(--queue "${TORCHRUN_HPC_QUEUE}")
  [[ -n "${TORCHRUN_HPC_TIME_LIMIT}" ]] && extra+=(--time-limit "${TORCHRUN_HPC_TIME_LIMIT}")
  [[ -n "${TORCHRUN_HPC_ACCOUNT}" ]] && extra+=(--account "${TORCHRUN_HPC_ACCOUNT}")
  # Intentional word-splitting of a user-supplied flag string.
  local -a passthrough=(${TORCHRUN_HPC_EXTRA_ARGS})

  torchrun-hpc \
    --nodes "${nnodes}" \
    --procs-per-node "${nproc}" \
    "${module}" "$@"
}

# =======================================================================
# Multi-node benchmarks. Every step below needs more than one node; the
# single-node primitives come from run_experiments.sh.
# =======================================================================

# 1.1 inter-node ping-pong: fixed at exactly 2 nodes, 1 rank/node, so
# rank 0 and rank 1 are guaranteed to be on different nodes (bench_pingpong
# asserts it). This no longer depends on the full-scale NNODES setting.
step "1.1 pingpong -- inter-node (2 nodes, 1 rank/node)"
launch 2 1 benchmarks/bench_pingpong.py \
  --mode inter --min-bytes 64 --max-bytes 67108864 --steps 21 \
  --warmup 20 --trials 100 \
  --output "${DATA_DIR}/pingpong_inter.json" --seed "${SEED}"
expect_output "${DATA_DIR}/pingpong_inter.json"

# 1.2 concurrency: fixed at exactly 4 ranks, 2/node, giving each rank both
# an intra-node peer and an inter-node peer. This is the benchmark that
# justifies T_comm = max(T_intra, T_inter). Since this now schedules its
# own independent 2-node job, it no longer needs to be gated behind
# NNODES == 2 -- it always runs, regardless of the full-scale topology.
step "1.2 concurrency -- intra/inter overlap (2 nodes, 2 ranks/node)"
launch 2 2 benchmarks/bench_concurrency.py \
  --message-bytes "${CONCURRENCY_BYTES}" --warmup 20 --trials 100 \
  --output "${DATA_DIR}/concurrency.json" --seed "${SEED}"
expect_output "${DATA_DIR}/concurrency.json"

# 2.2 crossover and 2.1 end-to-end at full scale across all nodes.
step "2.2 crossover -- K=${K} GPUs (${NNODES} nodes)"
launch "${NNODES}" "${GPUS_PER_NODE}" benchmarks/bench_crossover.py \
  --graph "${GRAPH}" --graph-sizes "${CROSSOVER_SIZES}" \
  --avg-degree 20 --feature-dim "${FEATURE_DIM}" --model "${MODEL}" \
  --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
  --output "${DATA_DIR}/crossover_K${K}.json" --seed "${SEED}"
expect_output "${DATA_DIR}/crossover_K${K}.json"

for N in "${E2E_VERTEX_SIZES[@]}"; do
  step "2.1 end-to-end -- K=${K} GPUs, N=${N}"
  launch "${NNODES}" "${GPUS_PER_NODE}" benchmarks/bench_end_to_end.py \
    --graph "${GRAPH}" --num-vertices "${N}" --avg-degree 20 \
    --feature-dim "${FEATURE_DIM}" --model "${MODEL}" \
    --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
    --output "${DATA_DIR}/e2e_K${K}_N${N}.json" --seed "${SEED}"
  expect_output "${DATA_DIR}/e2e_K${K}_N${N}.json"
done

# =======================================================================
# Refit and plot (CPU-only post-processing, runs locally -- no launcher
# involved).
# =======================================================================

# Explicit file list (single-node K values plus this run's K), never a
# e2e_*.json glob — a glob would also pick up stray files that were not part
# of this run matrix.
E2E_FILES=()
for k in "${SINGLE_NODE_GPU_COUNTS[@]}" "${K}"; do
  for N in "${E2E_VERTEX_SIZES[@]}"; do
    f="${DATA_DIR}/e2e_K${k}_N${N}.json"
    if [[ -f "$f" ]]; then
      E2E_FILES+=("$f")
    elif [[ "$k" -eq "$K" ]]; then
      echo "[warn] missing ${f} — this run should have produced it; check the K=${K} step above for errors"
    else
      echo "[warn] missing ${f} — run ./run_experiments.sh first to produce the K=${k} points"
    fi
  done
done

if [[ ${#E2E_FILES[@]} -eq 0 ]]; then
  echo "[error] no end-to-end run files found in ${DATA_DIR}/ — nothing to fit against." >&2
  exit 1
fi

# Hold out the largest world size, so held-out MAPE measures extrapolation
# into the multi-node regime specifically.
FIT_FILTER=${FIT_FILTER:-"world_size < ${K}"}

step "fit_primitives (intra + inter network)"
python -m analysis.fit_primitives \
  --pingpong-intra "${DATA_DIR}/pingpong_intra.json" \
  --pingpong-inter "${DATA_DIR}/pingpong_inter.json" \
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

step "visualization"
# Now that an inter-node fit exists, plot both network tiers.
python -m visualization.plot_pingpong \
  --intra "${DATA_DIR}/pingpong_intra.json" \
  --inter "${DATA_DIR}/pingpong_inter.json" \
  --primitives "${DATA_DIR}/fitted_primitives.json" \
  --output "${FIG_DIR}/pingpong"

python -m visualization.plot_crossover --input "${DATA_DIR}/crossover_K${K}.json" \
  --output "${FIG_DIR}/crossover_K${K}.png"

python -m visualization.plot_validation --predictions "${DATA_DIR}/predictions.json" \
  --color-by world_size --output "${FIG_DIR}/validation"

echo
echo "Done. Data in ${DATA_DIR}/, figures in ${FIG_DIR}/."