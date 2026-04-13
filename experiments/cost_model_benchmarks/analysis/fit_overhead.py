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

def predict_layer_time(run_config: dict, per_rank_stats: dict,
                       primitives: dict) -> float:
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
    """
    F = run_config["feature_dim"]
    model_type = run_config.get("model", "gcn")

    n_local = per_rank_stats.get("n_local", 0)
    n_halo  = per_rank_stats.get("n_halo", 0)
    n_total = n_local + n_halo

    # A rough edge count estimate: use avg_degree * n_local as a proxy
    avg_degree = run_config.get("avg_degree", 20.0)
    n_edges_local = int(n_local * avg_degree)

    # T_comp
    comp_params = primitives.get("compute", {}).get(model_type, {}).get("forward", None)
    if comp_params:
        T_comp = (comp_params["coeff_V"] * n_total
                  + comp_params["coeff_E"] * n_edges_local
                  + comp_params["intercept"])
        T_comp = max(T_comp, 0.0)
    else:
        T_comp = 0.0

    # T_intra and T_inter
    intra_bytes = per_rank_stats.get("c_intra_bytes", 0)
    inter_bytes = per_rank_stats.get("c_inter_bytes", 0)

    net = primitives.get("network", {})

    def net_time(nbytes: int, mode: str) -> float:
        params = net.get(mode, None)
        if params is None or nbytes == 0:
            return 0.0
        B = params.get("bandwidth_bytes_per_sec", 1e10)
        t_L = params.get("latency_seconds", 0.0)
        return t_L + nbytes / B

    T_intra = net_time(intra_bytes, "intra")
    T_inter = net_time(inter_bytes, "inter")
    T_comm = max(T_intra, T_inter)

    # T_buffer_copy (gather of send buffer)
    send_bytes = per_rank_stats.get("send_total", 0) * F * 4
    gath_params = primitives.get("gather", {}).get("clustered", {}).get("gather", None)
    if gath_params and send_bytes > 0:
        B_g = gath_params.get("bandwidth_bytes_per_sec", 1e12)
        T_buffer_copy = gath_params.get("intercept_seconds", 0.0) + send_bytes / B_g
    else:
        T_buffer_copy = 0.0

    return T_comp + T_comm + T_buffer_copy


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

def fit_overhead_scalar(fit_runs: list, primitives: dict) -> tuple:
    """Fit T_overhead to minimise MAPE on fit_runs. Returns (overhead, mape_in_sample)."""
    if not fit_runs:
        return 0.0, float("nan")

    residuals = []
    for r in fit_runs:
        T_model = predict_layer_time(r["config"], r["per_rank_stats"], primitives)
        residuals.append(r["measured_median"] - T_model)

    # Optimal scalar overhead that minimises sum of |err - overhead| / T_meas
    # is the weighted median; for uniform weights it's just the median of residuals.
    overhead = float(np.median(residuals))

    mape = float(np.mean([
        abs(r["measured_median"] - (predict_layer_time(r["config"], r["per_rank_stats"], primitives) + overhead))
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
