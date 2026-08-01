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
#   GPU_COUNTS="2 4 8" E2E_MODELS="gcn edge" ./run_experiments.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# An unmatched glob (e.g. a partial re-run missing an expected output file)
# should surface as a clear argparse error, not a literal glob-pattern
# string silently passed through to a file open() call.
shopt -s nullglob

# --- Config ------------------------------------------------------------
GPU_COUNTS=(${GPU_COUNTS:-2 4})        # world sizes to sweep via torchrun --nproc_per_node
SEED=${SEED:-42}
FEATURE_DIM=${FEATURE_DIM:-128}        # F for the gather microbenchmark (1.4)
# Layer variants and feature dims for the distributed (crossover + e2e) runs.
# Defaults are the paper configuration, so a plain ./run_experiments.sh
# produces exactly what the paper needs with nothing to override.
#
# F is a single value here while COMPUTE_FEATURE_DIMS below sweeps five: the
# F-dependence result is a compute-microbenchmark finding (it yields F*), and
# every extra F multiplies the whole distributed grid. Add to
# E2E_FEATURE_DIMS if you want the assembled model validated at other F too.
E2E_MODELS=(${E2E_MODELS:-gcn edge gcn_spmm})
E2E_FEATURE_DIMS=(${E2E_FEATURE_DIMS:-128})
# Layer variants and feature dims for the single-GPU compute sweeps (1.3).
# Sweeping F is what makes T_comp's F dependence identifiable: at a single F
# the F^2 and F terms are collinear, so a compute-bound layer and a
# bandwidth-bound one fit the data equally well. See analysis/compute_forms.py.
COMPUTE_MODELS=(${COMPUTE_MODELS:-gcn edge gcn_spmm})
COMPUTE_FEATURE_DIMS=(${COMPUTE_FEATURE_DIMS:-32 64 128 256 512})
# Defaults to FEATURE_DIM so overriding one does not silently leave the
# figures pointing at a different feature dim than the distributed runs use.
REF_FEATURE_DIM=${REF_FEATURE_DIM:-$FEATURE_DIM}
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

# Fail before spending GPU hours rather than at the plotting step at the end:
# plot_compute reads the REF_FEATURE_DIM cell, which only exists if that F was
# actually swept, and each model in E2E_MODELS only gets a T_comp fit if it was
# also benchmarked as a compute model.
if [[ ! " ${COMPUTE_FEATURE_DIMS[*]} " =~ " ${REF_FEATURE_DIM} " ]]; then
  echo "[error] REF_FEATURE_DIM=${REF_FEATURE_DIM} is not in COMPUTE_FEATURE_DIMS" \
       "(${COMPUTE_FEATURE_DIMS[*]}); the figures would have no data to plot." >&2
  exit 1
fi
for m in "${E2E_MODELS[@]}"; do
  if [[ ! " ${COMPUTE_MODELS[*]} " =~ " ${m} " ]]; then
    echo "[error] E2E_MODELS contains '${m}', which is not in COMPUTE_MODELS" \
         "(${COMPUTE_MODELS[*]}); its distributed runs would have no T_comp" \
         "fit to predict with." >&2
    exit 1
  fi
done
for f in "${E2E_FEATURE_DIMS[@]}"; do
  if [[ ! " ${COMPUTE_FEATURE_DIMS[*]} " =~ " ${f} " ]]; then
    echo "[warn] E2E_FEATURE_DIMS contains F=${f}, which is not in" \
         "COMPUTE_FEATURE_DIMS (${COMPUTE_FEATURE_DIMS[*]}). T_comp will be" \
         "extrapolated in F rather than interpolated there." >&2
  fi
done

mkdir -p "$DATA_DIR" "$FIG_DIR"

step() { echo; echo "=== $* ==="; }

# -------------------------------------------------------------------------
# 1. Single-GPU primitive microbenchmarks (1.3 compute, 1.4 gather) — no
#    torchrun, GPU-count independent.
#
#    fit_primitives.py fits each layer variant's compute cost independently
#    (--compute-gcn / --compute-edge / --compute-gcn-spmm), so every model in
#    COMPUTE_MODELS is benchmarked here regardless of which subset the
#    distributed crossover/end-to-end sweep below exercises.
# -------------------------------------------------------------------------
# Both sweep directions are required per (model, F) cell: fit_compute needs a
# vertex sweep (|E| fixed) and an edge sweep (|V| fixed) to identify the |V|
# and |E| coefficients independently. Filenames carry F so the cells stay
# separable and a stale file from a different F cannot be pooled in.
#
# FIT_COMPUTE_ARGS is accumulated here rather than reconstructed later: the
# outer loop is per-model, so each --compute-* flag is followed contiguously
# by exactly the files just generated for it, which is what nargs="+" wants.
# (Built as a flat array, not an associative one — bash 3.2, still the default
# on macOS, has no `declare -A`.)
FIT_COMPUTE_ARGS=()
for m in "${COMPUTE_MODELS[@]}"; do
  case "$m" in
    gcn)      FIT_COMPUTE_ARGS+=(--compute-gcn) ;;
    edge)     FIT_COMPUTE_ARGS+=(--compute-edge) ;;
    gcn_spmm) FIT_COMPUTE_ARGS+=(--compute-gcn-spmm) ;;
    *) echo "[error] unknown compute model '${m}'" >&2; exit 1 ;;
  esac
  for f in "${COMPUTE_FEATURE_DIMS[@]}"; do
    for sweep in vertices edges; do
      case "$sweep" in
        vertices) tag=vswp ;;
        edges)    tag=eswp ;;
      esac
      out="${DATA_DIR}/compute_${m}_F${f}_${tag}.json"
      step "1.3 compute -- ${m}, F=${f}, ${sweep} sweep"
      python -m benchmarks.bench_compute --model "${m}" --sweep "${sweep}" \
        --min 1000 --max 1000000 --steps 10 --fixed-value 200000 \
        --feature-dim "${f}" --warmup "${WARMUP}" --trials "${TRIALS}" \
        --output "${out}" --seed "${SEED}"
      FIT_COMPUTE_ARGS+=("${out}")
    done
  done
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
# Output filenames carry every axis that varies (K, model, F, N). They used to
# be e2e_K${K}_N${N}.json, which silently overwrote one model's results with
# the next as soon as more than one model was swept — the two runs differ only
# in fields that were not in the name.
#
# E2E_FILES is accumulated here rather than rebuilt afterwards from the same
# nested loops, so the list of files fed to the fit cannot drift out of sync
# with the list actually produced. It is an explicit list, not an e2e_*.json
# glob, which would also match stray files from ad-hoc runs (e.g. a leftover
# data/e2e_K8_F128_er_bal.json) and pool them into the fit/held-out split.
E2E_FILES=()
CROSSOVER_TAGS=()
for K in "${GPU_COUNTS[@]}"; do
  for m in "${E2E_MODELS[@]}"; do
    for f in "${E2E_FEATURE_DIMS[@]}"; do
      tag="K${K}_${m}_F${f}"

      step "2.2 crossover -- K=${K} GPUs, model=${m}, F=${f}"
      torchrun --nnodes 1 --nproc_per_node "${K}" -m benchmarks.bench_crossover \
        --graph "${GRAPH}" --graph-sizes "${CROSSOVER_SIZES}" \
        --avg-degree 20 --feature-dim "${f}" --model "${m}" \
        --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
        --output "${DATA_DIR}/crossover_${tag}.json" --seed "${SEED}"
      CROSSOVER_TAGS+=("${tag}")

      for N in "${E2E_VERTEX_SIZES[@]}"; do
        step "2.1 end-to-end -- K=${K} GPUs, model=${m}, F=${f}, N=${N}"
        torchrun --nnodes 1 --nproc_per_node "${K}" -m benchmarks.bench_end_to_end \
          --graph "${GRAPH}" --num-vertices "${N}" --avg-degree 20 \
          --feature-dim "${f}" --model "${m}" \
          --partitioner "${PARTITIONER}" --warmup "${WARMUP}" --trials "${TRIALS}" \
          --output "${DATA_DIR}/e2e_${tag}_N${N}.json" --seed "${SEED}"
        E2E_FILES+=("${DATA_DIR}/e2e_${tag}_N${N}.json")
      done
    done
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
  "${FIT_COMPUTE_ARGS[@]}" \
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
# plot_compute shows one (model, F) cell per panel, so it gets the reference
# feature dim. NOTE: there is no figure for the F sweep itself yet — the
# fitted F* values are printed by fit_primitives and stored in
# fitted_primitives.json, but nothing plots T vs F.
PLOT_COMPUTE_ARGS=()
for m in "${COMPUTE_MODELS[@]}"; do
  case "$m" in
    gcn)      vflag=--gcn-vertex;  eflag=--gcn-edge ;;
    edge)     vflag=--edge-vertex; eflag=--edge-edge ;;
    gcn_spmm) vflag=--spmm-vertex; eflag=--spmm-edge ;;
  esac
  PLOT_COMPUTE_ARGS+=("${vflag}" "${DATA_DIR}/compute_${m}_F${REF_FEATURE_DIM}_vswp.json")
  PLOT_COMPUTE_ARGS+=("${eflag}" "${DATA_DIR}/compute_${m}_F${REF_FEATURE_DIM}_eswp.json")
done
python -m visualization.plot_compute "${PLOT_COMPUTE_ARGS[@]}" \
  --primitives "${DATA_DIR}/fitted_primitives.json" --output "${FIG_DIR}/compute"

# bench_gather records BOTH gather_trials_seconds and
# scatter_add_trials_seconds, and fit_primitives fits both, but plot_gather
# renders one --operation per invocation (default: gather). Without the
# second call the scatter-add fit is never looked at by a human -- which is
# how an O(N) zero-fill inside the timed region went unnoticed there while
# the (clean) gather curve looked fine.
for op in gather scatter_add; do
  python -m visualization.plot_gather \
    --contiguous "${DATA_DIR}/gather_contiguous.json" --clustered "${DATA_DIR}/gather_clustered.json" \
    --random "${DATA_DIR}/gather_random.json" --fitted "${DATA_DIR}/fitted_primitives.json" \
    --operation "${op}" --output "${FIG_DIR}/${op}"
done

python -m visualization.plot_pingpong \
  --intra "${DATA_DIR}/pingpong_intra.json" --primitives "${DATA_DIR}/fitted_primitives.json" \
  --output "${FIG_DIR}/pingpong"

for tag in "${CROSSOVER_TAGS[@]}"; do
  python -m visualization.plot_crossover --input "${DATA_DIR}/crossover_${tag}.json" \
    --output "${FIG_DIR}/crossover_${tag}.png"
done

python -m visualization.plot_validation --predictions "${DATA_DIR}/predictions.json" \
  --color-by world_size --output "${FIG_DIR}/validation"

echo
echo "Done. Data in ${DATA_DIR}/, figures in ${FIG_DIR}/."
