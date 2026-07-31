"""Analysis — Fit Library Overhead Bias (T_overhead).

Reads ``fitted_primitives.json`` and the small-K subset of end-to-end runs.
For each run it computes the model-predicted T_layer (without overhead), then
fits a single scalar T_overhead that minimises MAPE:

    MAPE = mean( |T_measured - (T_model + T_overhead)| / T_measured )

The subset used for fitting is controlled by ``--fit-filter``, which is
evaluated as a Python expression where each run's config fields are available
as local variables (e.g. ``"world_size <= 8"``).

Outputs ``data/fitted_overhead.json``.

Usage::

    python -m analysis.fit_overhead \\
        --primitives data/fitted_primitives.json \\
        --e2e-runs   data/e2e_*.json \\
        --fit-filter "world_size <= 8" \\
        --output     data/fitted_overhead.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Cost model (without overhead)
# ---------------------------------------------------------------------------

def _gather_piecewise_time(nbytes: float, params: dict) -> float:
    """Evaluate the fitted piecewise L2-cache/HBM-bandwidth gather model for
    a single byte count. Mirrors ``time_model`` in
    ``analysis/fit_primitives.py::fit_gather`` exactly — this is the single
    place both ``predict_layer_time`` and ``compute_predictions.py`` should
    call, rather than re-deriving a simplified linear approximation.
    """
    overhead = params.get("launch_overhead_seconds", 0.0)
    bw_HBM = params.get("bandwidth_bytes_per_sec")
    bw_L2 = params.get("L2_bandwidth_bytes_per_sec")
    if not bw_HBM or not bw_L2:
        return overhead

    inv_bw_L2 = 1.0 / bw_L2
    inv_bw_HBM = 1.0 / bw_HBM
    L2_thresh = params.get("L2_inflection_bytes", 0.0)
    HBM_thresh = params.get("HBM_inflection_bytes", 0.0)

    bytes_L2 = min(max(nbytes, 0.0), L2_thresh)
    bytes_HBM = max(0.0, nbytes - HBM_thresh)
    t_mem = bytes_L2 * inv_bw_L2 + bytes_HBM * inv_bw_HBM
    return max(overhead, t_mem)


def predict_layer_time(run_config: dict, per_rank_stats: dict,
                       primitives: dict) -> dict:
    """Predict T_layer for one rank using the assembled primitive model.

    T_layer = T_comp + max(T_intra, T_inter) + T_buffer_copy

    Parameters
    ----------
    run_config : dict
        Config block from the end-to-end JSON (feature_dim, model, etc.)
    per_rank_stats : dict
        Stats for rank 0 from per_rank_stats list.
    primitives : dict
        Loaded fitted_primitives.json.

    Returns
    -------
    dict with keys T_comp, T_intra, T_inter, T_comm, T_buffer_copy, T_total
    (T_total = T_comp + T_comm + T_buffer_copy, i.e. without T_overhead).
    Every component is floored at 0.0, since noisy fitted intercepts can
    otherwise extrapolate to physically-meaningless negative times.
    """
    F = run_config["feature_dim"]
    model_type = run_config.get("model", "gcn")

    n_local = per_rank_stats.get("n_local", 0)
    n_halo  = per_rank_stats.get("n_halo", 0)
    n_total = n_local + n_halo

    # Prefer the real local edge count recorded by bench_end_to_end.py
    # (edge_index.shape[1], from DGraph.distributed.build_communication_pattern's
    # comm_pattern.local_edge_list); fall back to the avg_degree*n_local proxy
    # only for older run data that predates recording it. The proxy ignores
    # partitioner-dependent edge-cut effects (random/balanced/metis produce
    # very different local edge densities for the same avg_degree).
    n_edges_local = per_rank_stats.get("n_edges_local")
    if n_edges_local is None:
        avg_degree = run_config.get("avg_degree", 20.0)
        n_edges_local = int(n_local * avg_degree)

    # T_comp — the end-to-end benchmark times forward + backward + grad-zero
    # per iteration (bench_end_to_end.py::one_layer), so T_comp must be the
    # forward+backward cost, not forward alone.
    #
    # NOTE: the fit stored under the "backward" key is ALREADY that combined
    # cost. bench_compute.py's bwd() closure re-runs the forward pass inside
    # its own timed region (it has to, to rebuild the autograd graph each
    # trial), so "backward_trials_seconds" measures grad-zero + forward +
    # backward — exactly the op set one_layer() performs. The key is a
    # misnomer; "forward" is the only fit that is forward-alone.
    # Using "forward" here undercounted T_comp and cost ~30% MAPE; adding
    # forward+backward together instead would double-count the forward pass.
    comp_params = primitives.get("compute", {}).get(model_type, {}).get("backward", None)
    if comp_params:
        T_comp = (comp_params["coeff_V"] * n_total
                  + comp_params["coeff_E"] * n_edges_local
                  + comp_params["intercept"])
    else:
        T_comp = 0.0
    T_comp = max(T_comp, 0.0)

    # T_intra and T_inter
    intra_bytes = per_rank_stats.get("c_intra_bytes", 0)
    inter_bytes = per_rank_stats.get("c_inter_bytes", 0)

    net = primitives.get("network", {})

    def net_time(nbytes: int, mode: str) -> float:
        if nbytes == 0:
            return 0.0
        params = net.get(mode, None)
        if params is None:
            # A missing network fit is fine when this mode never carries
            # traffic (e.g. single-node runs never have inter-node bytes,
            # so "inter" is legitimately absent). But if a run actually
            # has nonzero bytes for this mode, silently treating them as
            # zero-cost would bias T_comm and every downstream prediction/
            # overhead fit without any signal that it happened.
            raise ValueError(
                f"per_rank_stats has {nbytes} bytes of {mode}-node "
                f"communication but fitted_primitives.json has no '{mode}' "
                f"network fit. Re-run fit_primitives.py with "
                f"--pingpong-{mode} data, or confirm this run should never "
                f"have {mode}-node traffic in the first place."
            )
        B = params.get("bandwidth_bytes_per_sec", 1e10)
        t_L = params.get("latency_seconds", 0.0)
        return t_L + nbytes / B

    T_intra = max(net_time(intra_bytes, "intra"), 0.0)
    T_inter = max(net_time(inter_bytes, "inter"), 0.0)
    T_comm = max(T_intra, T_inter)

    # T_buffer_copy (gather of send buffer) — uses the fitted piecewise
    # L2/HBM model, not a simplified single-slope linear stand-in.
    send_bytes = per_rank_stats.get("send_total", 0) * F * 4
    gath_params = primitives.get("gather", {}).get("clustered", {}).get("gather", None)
    if gath_params and send_bytes > 0:
        T_buffer_copy = _gather_piecewise_time(send_bytes, gath_params)
    else:
        T_buffer_copy = 0.0
    T_buffer_copy = max(T_buffer_copy, 0.0)

    return {
        "T_comp": T_comp,
        "T_intra": T_intra,
        "T_inter": T_inter,
        "T_comm": T_comm,
        "T_buffer_copy": T_buffer_copy,
        "T_total": T_comp + T_comm + T_buffer_copy,
    }


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_e2e_runs(paths: list) -> list:
    runs = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        config = data.get("config", {})
        for meas in data.get("measurements", []):
            per_rank_stats = meas.get("per_rank_stats", [{}])
            rank0_stats = per_rank_stats[0] if per_rank_stats else {}
            trials = meas.get("rank0_trials_seconds", [])
            if not trials:
                continue
            runs.append({
                "config": config,
                "per_rank_stats": rank0_stats,
                "measured_median": float(np.median(trials)),
                "source_file": str(p),
            })
    return runs


def apply_filter(runs: list, filter_expr: str) -> tuple:
    """Split runs into fit and held-out sets using filter_expr."""
    if not filter_expr:
        return runs, []
    fit_runs, held_runs = [], []
    for r in runs:
        env = dict(r["config"])
        env.update(r["per_rank_stats"])
        try:
            if eval(filter_expr, {"__builtins__": {}}, env):
                fit_runs.append(r)
            else:
                held_runs.append(r)
        except Exception as e:
            print(f"[fit_overhead] Warning: filter eval failed for run ({e}), including in fit set")
            fit_runs.append(r)
    return fit_runs, held_runs


# ---------------------------------------------------------------------------
# Scalar overhead fitting
# ---------------------------------------------------------------------------

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return m minimising sum(weights * |values - m|)."""
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    cutoff = cum[-1] / 2.0
    idx = int(np.searchsorted(cum, cutoff))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def fit_overhead_scalar(fit_runs: list, primitives: dict) -> tuple:
    """Fit T_overhead to minimise MAPE on fit_runs. Returns (overhead, mape_in_sample)."""
    if not fit_runs:
        return 0.0, float("nan")

    residuals = np.array([
        r["measured_median"] - predict_layer_time(r["config"], r["per_rank_stats"], primitives)["T_total"]
        for r in fit_runs
    ])
    T_meas = np.array([r["measured_median"] for r in fit_runs])

    # minimize sum(|residual - overhead| / T_meas) over the scalar overhead
    # == minimize sum(weight * |residual - overhead|) with weight = 1/T_meas,
    # whose minimizer is the weighted median (NOT the plain median, which
    # only minimizes the unweighted sum |residual - overhead|).
    overhead = weighted_median(residuals, 1.0 / T_meas)

    mape = float(np.mean([
        abs(r["measured_median"] - (predict_layer_time(r["config"], r["per_rank_stats"], primitives)["T_total"] + overhead))
        / r["measured_median"]
        for r in fit_runs
    ]))
    return overhead, mape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fit T_overhead from end-to-end runs")
    p.add_argument("--primitives", type=str, required=True,
                   help="Path to fitted_primitives.json")
    p.add_argument("--e2e-runs", nargs="+", required=True, metavar="FILE")
    p.add_argument("--fit-filter", type=str, default="world_size <= 8",
                   help="Python expression evaluated per run; True → fit set")
    p.add_argument("--output", type=str, default="data/fitted_overhead.json")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.primitives) as f:
        primitives = json.load(f)

    all_runs = load_e2e_runs(args.e2e_runs)
    print(f"[fit_overhead] Loaded {len(all_runs)} run(s)")

    fit_runs, held_runs = apply_filter(all_runs, args.fit_filter)
    print(f"[fit_overhead] Fit set: {len(fit_runs)}  Held-out: {len(held_runs)}")

    overhead, mape_in = fit_overhead_scalar(fit_runs, primitives)
    print(f"[fit_overhead] T_overhead = {overhead*1e3:.3f} ms  in-sample MAPE = {mape_in*100:.2f}%")

    result = {
        "overhead_seconds": overhead,
        "fit_filter": args.fit_filter,
        "num_fit_points": len(fit_runs),
        "num_held_out": len(held_runs),
        "in_sample_mape": mape_in,
        "fit_subset_runs": [
            {
                "source_file": r["source_file"],
                "world_size": r["config"].get("world_size"),
                "feature_dim": r["config"].get("feature_dim"),
                "measured_median_seconds": r["measured_median"],
            }
            for r in fit_runs
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fit_overhead] Written to {out_path}")


if __name__ == "__main__":
    main()
