#!/usr/bin/env bash
#SBATCH --job-name=cost_model_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=cost_model_multinode_%j.out
# NOTE: add your cluster's --partition/--account/--gres here, e.g.
#   #SBATCH --gres=gpu:4          (or --gpus-per-node=4, cluster-dependent)
#
# Multi-node extension of run_experiments.sh: adds the inter-node primitives
# and the K = NNODES*GPUS_PER_NODE end-to-end/crossover points that a
# single-node run cannot produce.
#
# PREREQUISITE: run ./run_experiments.sh on one node first. This script reuses
# its compute_*/gather_*/pingpong_intra.json outputs and its K=2,4 e2e runs;
# it only produces what genuinely requires >1 node.
#
# Run it ONCE — it fans out to the other nodes itself via srun. Either:
#
#   sbatch ./run_experiments_multinode.sh
#
# or from an interactive allocation (salloc -N 2 gives one shell on the head
# node; srun distributes from there):
#
#   salloc -N 2 --ntasks-per-node=1 --gres=gpu:4 -t 2:00:00
#   ./run_experiments_multinode.sh
#
# Each step runs `srun --ntasks-per-node=1`, placing exactly one torchrun
# launcher on each node, which then spawns nproc_per_node worker processes
# locally. torchrun uses *static* rendezvous (explicit --node_rank taken from
# SLURM_NODEID) rather than c10d dynamic rendezvous, so global ranks are
# deterministically block-assigned (global_rank = node_rank*nproc + local_rank).
# bench_concurrency hardcodes a "ranks 0,1 on node A; ranks 2,3 on node B"
# layout, and c10d assigns node ranks in join order — which would scramble
# that mapping nondeterministically between runs.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
shopt -s nullglob

# --- Allocation discovery ----------------------------------------------
if ! command -v srun >/dev/null 2>&1; then
  echo "[error] srun not found — this script must run inside a SLURM allocation" >&2
  echo "        (sbatch ./run_experiments_multinode.sh, or salloc then run it)" >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[error] no SLURM allocation detected (SLURM_JOB_ID unset)." >&2
  echo "        Run under sbatch, or inside 'salloc -N 2 ... '." >&2
  exit 1
fi

NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-2}}}
if [[ "${NNODES}" -lt 2 ]]; then
  echo "[error] allocation has ${NNODES} node(s); this script needs >= 2." >&2
  echo "        For single-node runs use ./run_experiments.sh instead." >&2
  exit 1
fi

# Head node hostname, used as the torchrun rendezvous master by every node.
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}

# --- Config ------------------------------------------------------------
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29500}
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

K=$((NNODES * GPUS_PER_NODE))   # total world size for the full-scale runs

mkdir -p "$DATA_DIR" "$FIG_DIR"

echo "Allocation: ${NNODES} nodes, ${GPUS_PER_NODE} GPU/node -> K=${K}"
echo "Rendezvous master: ${MASTER_ADDR}"

step() { echo; echo "=== $* ==="; }

# Each torchrun invocation gets its own port. Back-to-back runs reusing one
# port can fail with "address already in use" while the previous rendezvous
# socket is still in TIME_WAIT. Incremented by a plain assignment in this
# shell — NOT inside $(...), whose subshell assignment would be discarded,
# silently pinning every step to the same port.
_port=$MASTER_PORT

# Launch one torchrun per node via srun.
#   $1 = nproc_per_node, rest = the module + its args.
# --node_rank must expand per-node, so the torchrun command runs under a
# remote shell where $SLURM_NODEID is set; args are %q-quoted so anything
# containing spaces survives that extra round of shell parsing intact.
launch() {
  local nproc=$1; shift
  _port=$((_port + 1))
  local quoted_cmd
  quoted_cmd=$(printf '%q ' "$@")
  srun --nodes="${NNODES}" --ntasks-per-node=1 --cpu-bind=none \
    bash -c "torchrun \
      --nnodes ${NNODES} \
      --nproc_per_node ${nproc} \
      --node_rank \${SLURM_NODEID} \
      --master_addr ${MASTER_ADDR} \
      --master_port ${_port} \
      ${quoted_cmd}"
}

# =======================================================================
# Multi-node benchmarks. Every step below needs more than one node; the
# single-node primitives come from run_experiments.sh.
# =======================================================================

# 1.1 inter-node ping-pong: exactly 2 ranks, one per node, so rank 0 and
# rank 1 are guaranteed to be on different nodes (bench_pingpong asserts it).
step "1.1 pingpong -- inter-node (1 rank/node)"
launch 1 -m benchmarks.bench_pingpong \
  --mode inter --min-bytes 64 --max-bytes 67108864 --steps 21 \
  --warmup 20 --trials 100 \
  --output "${DATA_DIR}/pingpong_inter.json" --seed "${SEED}"

# 1.2 concurrency: exactly 4 ranks laid out 2-per-node, giving each rank both
# an intra-node peer and an inter-node peer. This is the benchmark that
# justifies T_comm = max(T_intra, T_inter); it cannot run on one node.
if [[ "${NNODES}" -eq 2 ]]; then
  step "1.2 concurrency -- intra/inter overlap (2 ranks/node)"
  launch 2 -m benchmarks.bench_concurrency \
    --message-bytes "${CONCURRENCY_BYTES}" --warmup 20 --trials 100 \
    --output "${DATA_DIR}/concurrency.json" --seed "${SEED}"
else
  echo "[skip] bench_concurrency requires exactly 4 ranks (NNODES=2, 2/node); NNODES=${NNODES}"
fi

# 2.2 crossover and 2.1 end-to-end at full scale across all nodes.
step "2.2 crossover -- K=${K} GPUs (${NNODES} nodes)"
launch "${GPUS_PER_NODE}" -m benchmarks.bench_crossover \
  --graph "${GRAPH}" --graph-sizes "${CROSSOVER_SIZES}" \
  --avg-degree 20 --feature-dim "${FEATURE_DIM}" --model "${MODEL}" \
  --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
  --output "${DATA_DIR}/crossover_K${K}.json" --seed "${SEED}"

for N in "${E2E_VERTEX_SIZES[@]}"; do
  step "2.1 end-to-end -- K=${K} GPUs, N=${N}"
  launch "${GPUS_PER_NODE}" -m benchmarks.bench_end_to_end \
    --graph "${GRAPH}" --num-vertices "${N}" --avg-degree 20 \
    --feature-dim "${FEATURE_DIM}" --model "${MODEL}" \
    --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
    --output "${DATA_DIR}/e2e_K${K}_N${N}.json" --seed "${SEED}"
done

# =======================================================================
# Refit and plot (CPU-only post-processing, runs on the head node).
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
