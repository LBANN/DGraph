"""Analysis — Fit Primitive Cost-Model Parameters.

Reads JSON outputs from benchmarks 1.1, 1.3, and 1.4, fits the following
parameters by linear regression on **medians** of per-trial times:

* Network:   T = t_L + bytes / B  → fit (t_L, B) for intra and inter
* Compute:   T = coeff_V * |V| + coeff_E * |E| + intercept
             (separate fits for GCN and edge-conditioned models)
* Gather:    T = intercept + bytes / B_gather
             (separate fits for contiguous, clustered, random distributions)

Writes ``data/fitted_primitives.json``.

Usage::

    python -m analysis.fit_primitives \\
        --pingpong-intra  data/pingpong_intra_*.json \\
        --pingpong-inter  data/pingpong_inter_*.json \\
        --compute-gcn     data/compute_gcn_*.json \\
        --compute-edge    data/compute_edge_*.json \\
        --gather-contiguous data/gather_contiguous_*.json \\
        --gather-clustered  data/gather_clustered_*.json \\
        --gather-random     data/gather_random_*.json \\
        --output data/fitted_primitives.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit
from scipy.special import expit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_json_files(paths: list) -> list:
    records = []
    for p in paths:
        with open(p) as f:
            records.append(json.load(f))
    return records


def median_of_trials(trials: list) -> float:
    return float(np.median(trials))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def linear_fit(x: np.ndarray, y: np.ndarray):
    """Fit y = slope * x + intercept via scipy linregress. Returns dict."""
    result = sp_stats.linregress(x, y)
    y_pred = result.slope * x + result.intercept
    r2 = r_squared(y, y_pred)
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": r2,
    }


# ---------------------------------------------------------------------------
# Network fit: T = t_L + bytes / B
# ---------------------------------------------------------------------------


def fit_network(records: list) -> dict:
    """Fit (t_L, B) from ping-pong records (one mode per call)."""
    bytes_arr = []
    time_arr = []
    for rec in records:
        for meas in rec["measurements"]:
            nbytes = meas["params"]["message_bytes"]
            t_med = median_of_trials(meas["trials_seconds"])
            bytes_arr.append(nbytes)
            time_arr.append(t_med)

    bytes_arr = np.array(bytes_arr, dtype=float)
    time_arr = np.array(time_arr, dtype=float)

    # T = t_L + bytes / B  →  T = intercept + slope * bytes
    # so slope = 1/B, intercept = t_L
    fit = linear_fit(bytes_arr, time_arr)
    bandwidth = 1.0 / fit["slope"] if fit["slope"] > 0 else float("nan")
    latency = fit["intercept"]
    return {
        "bandwidth_bytes_per_sec": bandwidth,
        "latency_seconds": latency,
        "r_squared": fit["r_squared"],
        "_raw_slope": fit["slope"],
        "_raw_intercept": fit["intercept"],
        "_num_points": len(bytes_arr),
    }


# ---------------------------------------------------------------------------
# Compute fit: T = coeff_V * |V| + coeff_E * |E| + intercept
# ---------------------------------------------------------------------------


def fit_compute(records: list, timing_key: str = "forward_trials_seconds") -> dict:
    """Fit compute cost as a function of |V| and |E|.

    Uses multiple linear regression: T = a * |V| + b * |E| + c
    """
    V_arr, E_arr, T_arr = [], [], []
    for rec in records:
        for meas in rec["measurements"]:
            V_arr.append(meas["params"]["num_vertices"])
            E_arr.append(meas["params"]["num_edges"])
            T_arr.append(median_of_trials(meas[timing_key]))

    V_arr = np.array(V_arr, dtype=float)
    E_arr = np.array(E_arr, dtype=float)
    T_arr = np.array(T_arr, dtype=float)

    # Design matrix: [V, E, 1]
    A = np.column_stack([V_arr, E_arr, np.ones_like(V_arr)])
    result, _, _, _ = np.linalg.lstsq(A, T_arr, rcond=None)
    coeff_V, coeff_E, intercept = result
    T_pred = A @ result
    r2 = r_squared(T_arr, T_pred)

    return {
        "coeff_V": float(coeff_V),
        "coeff_E": float(coeff_E),
        "intercept": float(intercept),
        "r_squared": r2,
        "_num_points": len(T_arr),
    }


# ---------------------------------------------------------------------------
# Gather fit: T = intercept + max(overhead, bytes / B_gather)
# ---------------------------------------------------------------------------


def fit_gather(records: list, timing_key: str = "gather_trials_seconds") -> dict:
    k_arr, T_arr, F_arr = [], [], []
    for rec in records:
        F = rec["config"]["feature_dim"]
        for meas in rec["measurements"]:
            k = meas["params"]["k"]
            t_med = median_of_trials(meas[timing_key])
            k_arr.append(k)
            T_arr.append(t_med)
            F_arr.append(F)

    k_arr = np.array(k_arr, dtype=float)
    F_arr = np.array(F_arr, dtype=float)
    T_arr = np.array(T_arr, dtype=float)

    bytes_arr = k_arr * F_arr * 4.0  # float32

    # 1. The Piecewise Linear Model (No Logarithms)
    def time_model(b, overhead, inv_bw_L2, inv_bw_HBM, L2_thresh, HBM_thresh):
        # 1. Bucket the bytes into their respective physical regimes
        # Bytes processed exclusively at L2 speeds
        bytes_L2 = np.clip(b, 0, L2_thresh)
        # Bytes processed exclusively at HBM speeds
        bytes_HBM = np.maximum(0, b - HBM_thresh)

        # 2. Apply the specific bandwidth (slope) to each bucket
        t_mem = (bytes_L2 * inv_bw_L2) + (bytes_HBM * inv_bw_HBM)

        # 3. Floor the total time by the kernel launch overhead
        return np.maximum(overhead, t_mem)

    # 2. Strategic initial guesses
    min_T, max_T = float(np.min(T_arr)), float(np.max(T_arr))
    max_b = float(np.max(bytes_arr))

    # The asymptotic slope is the difference in max/min time over max bytes
    inv_bw_HBM_guess = (max_T - min_T) / (max_b + 1e-9)

    p0 = [
        min_T,  # overhead
        inv_bw_HBM_guess * 0.3,  # inv_bw_L2
        inv_bw_HBM_guess,  # inv_bw_HBM
        max_b * 0.00001,  # L2_thresh
        max_b * 0.001,  # HBM_thresh
    ]

    try:
        bounds = ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, max_b, max_b])

        # 3. The Magic Fix: sigma=T_arr weights the fit by relative error
        popt, _ = curve_fit(
            time_model,
            bytes_arr,
            T_arr,
            p0=p0,
            bounds=bounds,
            method="trf",
            sigma=T_arr,
            absolute_sigma=False,
        )
        overhead, inv_bw_L2, inv_bw_HBM, L2_thresh, HBM_thresh = popt

    except Exception as e:
        print(f"Fit failed: {e}")
        overhead = inv_bw_L2 = inv_bw_HBM = L2_thresh = np.nan

    bw_HBM = 1.0 / inv_bw_HBM if inv_bw_HBM > 0 else float("nan")
    bw_L2 = 1.0 / inv_bw_L2 if inv_bw_L2 > 0 else float("nan")

    # 3. Calculate linear R-squared in linear space
    if not np.isnan(overhead):
        T_pred = time_model(
            bytes_arr, overhead, inv_bw_L2, inv_bw_HBM, L2_thresh, HBM_thresh
        )
        ss_res = np.sum((T_arr - T_pred) ** 2)
        ss_tot = np.sum((T_arr - np.mean(T_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r_squared = float("nan")

    print(
        f" Fitted gather: overhead={overhead*1e3:.3f} ms ",
        f"BW_L2={bw_L2/1e9:.2f} GB/s  BW_HBM={bw_HBM/1e9:.2f} GB/s  "
        f"L2_thresh={L2_thresh/1e6:.2f} MB  HBM_thresh={HBM_thresh/1e6:.2f} MB  "
        f"R²={r_squared:.4f}",
    )

    return {
        "bandwidth_bytes_per_sec": bw_HBM,
        "L2_bandwidth_bytes_per_sec": bw_L2,
        "L2_inflection_bytes": L2_thresh,
        "HBM_inflection_bytes": HBM_thresh,
        "launch_overhead_seconds": overhead,
        "r_squared": r_squared,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Fit cost-model primitive parameters")
    p.add_argument("--pingpong-intra", nargs="+", default=[], metavar="FILE")
    p.add_argument("--pingpong-inter", nargs="+", default=[], metavar="FILE")
    p.add_argument("--compute-gcn", nargs="+", default=[], metavar="FILE")
    p.add_argument("--compute-edge", nargs="+", default=[], metavar="FILE")
    p.add_argument("--gather-contiguous", nargs="+", default=[], metavar="FILE")
    p.add_argument("--gather-clustered", nargs="+", default=[], metavar="FILE")
    p.add_argument("--gather-random", nargs="+", default=[], metavar="FILE")
    p.add_argument("--output", type=str, default="data/fitted_primitives.json")
    return p.parse_args()


def main():
    args = parse_args()
    result = {}

    # Network
    net = {}
    if args.pingpong_intra:
        recs = load_json_files(args.pingpong_intra)
        net["intra"] = fit_network(recs)
        print(
            f"[network/intra] B={net['intra']['bandwidth_bytes_per_sec']/1e9:.2f} GB/s  "
            f"t_L={net['intra']['latency_seconds']*1e6:.2f} µs  "
            f"R²={net['intra']['r_squared']:.4f}"
        )
    if args.pingpong_inter:
        recs = load_json_files(args.pingpong_inter)
        net["inter"] = fit_network(recs)
        print(
            f"[network/inter] B={net['inter']['bandwidth_bytes_per_sec']/1e9:.2f} GB/s  "
            f"t_L={net['inter']['latency_seconds']*1e6:.2f} µs  "
            f"R²={net['inter']['r_squared']:.4f}"
        )
    result["network"] = net

    # Compute
    comp = {}
    if args.compute_gcn:
        recs = load_json_files(args.compute_gcn)
        comp["gcn"] = {
            "forward": fit_compute(recs, "forward_trials_seconds"),
            "backward": fit_compute(recs, "backward_trials_seconds"),
        }
        print(
            f"[compute/gcn] coeff_V={comp['gcn']['forward']['coeff_V']:.3e}  "
            f"coeff_E={comp['gcn']['forward']['coeff_E']:.3e}  "
            f"R²={comp['gcn']['forward']['r_squared']:.4f}"
        )
    if args.compute_edge:
        recs = load_json_files(args.compute_edge)
        comp["edge"] = {
            "forward": fit_compute(recs, "forward_trials_seconds"),
            "backward": fit_compute(recs, "backward_trials_seconds"),
        }
        print(
            f"[compute/edge] coeff_V={comp['edge']['forward']['coeff_V']:.3e}  "
            f"coeff_E={comp['edge']['forward']['coeff_E']:.3e}  "
            f"R²={comp['edge']['forward']['r_squared']:.4f}"
        )
    result["compute"] = comp

    # Gather
    gath = {}
    for dist_name, files_attr in [
        ("contiguous", "gather_contiguous"),
        ("clustered", "gather_clustered"),
        ("random", "gather_random"),
    ]:
        files = getattr(args, files_attr.replace("-", "_"))
        if files:
            recs = load_json_files(files)
            gath[dist_name] = {
                "gather": fit_gather(recs, "gather_trials_seconds"),
                "scatter_add": fit_gather(recs, "scatter_add_trials_seconds"),
            }
            print(
                f"[gather/{dist_name}] "
                f"B_gather={gath[dist_name]['gather']['bandwidth_bytes_per_sec']/1e9:.2f} GB/s  "
                f"R²={gath[dist_name]['gather']['r_squared']:.4f}"
            )
    result["gather"] = gath

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[fit_primitives] Written to {out_path}")


if __name__ == "__main__":
    main()
