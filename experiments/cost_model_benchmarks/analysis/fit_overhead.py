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

from analysis import compute_forms


# ---------------------------------------------------------------------------
# Cost model (without overhead)
# ---------------------------------------------------------------------------

def _gather_time(nbytes: float, params: dict) -> float:
    """Evaluate the fitted Hockney gather model for a single byte count:

        T(bytes) = launch_overhead + bytes / B_gather

    Mirrors ``time_model`` in ``analysis/fit_primitives.py::fit_gather``
    exactly — this is the single place both ``predict_layer_time`` and
    ``compute_predictions.py`` should call, rather than re-deriving an
    approximation of it.
    """
    overhead = params.get("launch_overhead_seconds", 0.0)
    bw = params.get("bandwidth_bytes_per_sec")
    if not bw or not np.isfinite(bw):
        return overhead
    return overhead + max(nbytes, 0.0) / bw


def predict_layer_time(run_config: dict, per_rank_stats: dict,
                       primitives: dict, nics_per_node: int = 1) -> dict:
    """Predict T_layer for one rank using the assembled primitive model.

    T_layer = T_comp + (T_intra + T_inter) + T_buffer_copy

    Parameters
    ----------
    run_config : dict
        Config block from the end-to-end JSON (feature_dim, model, etc.)
    per_rank_stats : dict
        Stats for rank 0 from per_rank_stats list.
    primitives : dict
        Loaded fitted_primitives.json.
    nics_per_node : int
        Number of network interfaces per node. Ranks co-located on a node
        share them, so the per-pair inter-node bandwidth measured by
        bench_pingpong (1 rank/node, hence 1 rank/NIC) is not what a rank
        sees once several ranks per node contend for the same rail. See
        the contention note on ``net_time`` below.

    Returns
    -------
    dict with keys T_comp, T_intra, T_inter, T_comm, T_buffer_copy, T_total
    (T_total = T_comp + T_comm + T_buffer_copy, i.e. without T_overhead).
    Every component is floored at 0.0, since noisy fitted intercepts can
    otherwise extrapolate to physically-meaningless negative times.
    """
    F = run_config["feature_dim"]
    model_type = run_config.get("model", "gcn")
    ranks_per_node = run_config.get("ranks_per_node", 1)

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
    if comp_params is None:
        # Never silently fall back to zero: T_comp is the dominant term in every
        # run measured so far (44 ms of a 49 ms layer at K=2), so a missing fit
        # would not look like an error, it would look like a fast prediction.
        available = sorted(primitives.get("compute", {}))
        raise ValueError(
            f"No compute fit for model {model_type!r} in fitted_primitives.json "
            f"(have: {available or 'none'}), but an end-to-end run uses it. "
            f"Re-run fit_primitives.py with the matching --compute-* sweeps "
            f"for {model_type!r}."
        )
    # Which columns this evaluates depends on the form fit_compute chose
    # (fixed-F legacy, edge-centric, or vertex-centric). compute_forms owns
    # both sides so the fit and this evaluation cannot drift apart.
    T_comp = compute_forms.evaluate(comp_params, n_total, n_edges_local, F)
    T_comp = max(T_comp, 0.0)

    # T_intra and T_inter
    intra_bytes = per_rank_stats.get("c_intra_bytes", 0)
    inter_bytes = per_rank_stats.get("c_inter_bytes", 0)

    net = primitives.get("network", {})

    # Ranks co-located on a node share that node's NICs. bench_pingpong
    # measures the inter-node rate with 1 rank per node -- i.e. one rank per
    # NIC -- so its B_inter is a *per-NIC* capacity, not a per-rank one. With
    # r ranks per node and n NICs, ceil(r/n) ranks contend for each rail and
    # each sees B_inter / ceil(r/n).
    #
    # Not a fitted parameter: nics_per_node is a topology constant (read from
    # `nvidia-smi topo -m`), exactly like ranks_per_node above. Ignoring it
    # (equivalently, pretending every rank owns a NIC) under-predicts K=8 on
    # 2 nodes x 4 GPU by 15-31%, because 4 ranks share 2 NICs and each rank
    # gets half the ping-pong rate.
    ranks_per_nic = max(1, -(-ranks_per_node // max(1, nics_per_node)))

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
        # Intra-node traffic goes over NVLink, which is all-to-all: co-located
        # ranks do not share a single rail the way they share NICs, so only
        # the inter-node rate is divided by the contention factor.
        if mode == "inter":
            B = B / ranks_per_nic
        return t_L + nbytes / B

    T_intra = max(net_time(intra_bytes, "intra"), 0.0)
    T_inter = max(net_time(inter_bytes, "inter"), 0.0)
    # Additive, not max(). bench_concurrency measures exactly this overlap on
    # this machine and finds none: with 16 MiB exchanges issued on separate
    # CUDA streams, T_concurrent / max(T_intra, T_inter) = 1.60 while
    # T_concurrent / (T_intra + T_inter) = 0.97. NVLink and IB transfers
    # serialize, so the optimistic max() under-predicts every multi-node run.
    T_comm = T_intra + T_inter

    # T_buffer_copy (gather of send buffer) — uses the fitted Hockney gather
    # model, evaluated by the same function fit_gather fits.
    send_bytes = per_rank_stats.get("send_total", 0) * F * 4
    gath_params = primitives.get("gather", {}).get("clustered", {}).get("gather", None)
    if gath_params and send_bytes > 0:
        T_buffer_copy = _gather_time(send_bytes, gath_params)
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


def fit_overhead_scalar(fit_runs: list, primitives: dict,
                        nics_per_node: int = 1) -> tuple:
    """Fit T_overhead to minimise MAPE on fit_runs. Returns (overhead, mape_in_sample)."""
    if not fit_runs:
        return 0.0, float("nan")

    residuals = np.array([
        r["measured_median"] - predict_layer_time(r["config"], r["per_rank_stats"], primitives, nics_per_node)["T_total"]
        for r in fit_runs
    ])
    T_meas = np.array([r["measured_median"] for r in fit_runs])

    # minimize sum(|residual - overhead| / T_meas) over the scalar overhead
    # == minimize sum(weight * |residual - overhead|) with weight = 1/T_meas,
    # whose minimizer is the weighted median (NOT the plain median, which
    # only minimizes the unweighted sum |residual - overhead|).
    overhead = weighted_median(residuals, 1.0 / T_meas)

    mape = float(np.mean([
        abs(r["measured_median"] - (predict_layer_time(r["config"], r["per_rank_stats"], primitives, nics_per_node)["T_total"] + overhead))
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
    p.add_argument("--nics-per-node", type=int, default=1,
                   help="Network interfaces per node (topology constant, from "
                        "`nvidia-smi topo -m`; not fitted). Co-located ranks "
                        "share them, so each rank sees B_inter divided by "
                        "ceil(ranks_per_node / nics_per_node).")
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

    overhead, mape_in = fit_overhead_scalar(fit_runs, primitives, args.nics_per_node)
    print(f"[fit_overhead] T_overhead = {overhead*1e3:.3f} ms  in-sample MAPE = {mape_in*100:.2f}%")

    result = {
        "overhead_seconds": overhead,
        "fit_filter": args.fit_filter,
        "nics_per_node": args.nics_per_node,
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
