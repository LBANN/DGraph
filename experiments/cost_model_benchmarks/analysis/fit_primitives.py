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

from analysis import compute_forms


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


def weighted_linear_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray = None):
    """Fit y = slope * x + intercept via weighted least squares.

    ``sigma`` follows the same convention as ``scipy.optimize.curve_fit``:
    residuals are weighted by ``1/sigma``, i.e. this minimises
    ``sum(((y - pred) / sigma) ** 2)``. Passing ``sigma=y`` (as ``fit_gather``
    already does via ``curve_fit``) weights by *relative* error so that
    points spanning many orders of magnitude don't dominate the fit.
    """
    if sigma is None:
        sigma = np.ones_like(y)
    w = 1.0 / sigma
    A = np.column_stack([x, np.ones_like(x)]) * w[:, None]
    b = y * w
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    slope, intercept = result
    y_pred = slope * x + intercept
    r2 = r_squared(y, y_pred)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
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
    # Weighted by relative error (sigma=time_arr) for consistency with
    # fit_gather's curve_fit(sigma=T_arr) — message sizes span 64B-64MiB, so
    # an unweighted fit would be dominated by the largest sizes and leave the
    # latency intercept (small messages) poorly constrained.
    fit = weighted_linear_fit(bytes_arr, time_arr, sigma=time_arr)
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


def fit_compute(records: list, timing_key: str = "forward_trials_seconds",
                model_type: str = "gcn") -> dict:
    """Fit compute cost as a function of |V|, |E| and the feature dim F.

    The design matrix depends on the layer's structure and on whether F was
    actually swept — see ``analysis/compute_forms.py``, which owns the form
    definitions so that fitting and prediction cannot disagree. With single-F
    data this degrades to the original ``T = a|V| + b|E| + c``.
    """
    V_arr, E_arr, F_arr, T_arr = [], [], [], []
    for rec in records:
        default_F = rec.get("config", {}).get("feature_dim")
        for meas in rec["measurements"]:
            params = meas["params"]
            V_arr.append(params["num_vertices"])
            E_arr.append(params["num_edges"])
            F_arr.append(params.get("feature_dim", default_F))
            T_arr.append(median_of_trials(meas[timing_key]))

    if any(f is None for f in F_arr):
        raise ValueError(
            "Some compute measurements have no feature_dim, in params or in "
            "the file's config block. F is a design-matrix input now, so it "
            "cannot be inferred. Re-run bench_compute.py to regenerate."
        )

    V_arr = np.array(V_arr, dtype=float)
    E_arr = np.array(E_arr, dtype=float)
    F_arr = np.array(F_arr, dtype=float)
    T_arr = np.array(T_arr, dtype=float)

    form = compute_forms.select_form(model_type, len(np.unique(F_arr)))

    # Weighted by relative error (1/T) for consistency with fit_gather's
    # curve_fit(sigma=T_arr) — |V|/|E| sweeps are log-spaced over several
    # orders of magnitude, so an unweighted fit would be dominated by the
    # largest graphs and leave the small-graph intercept (which matters most
    # for the crossover/tipping-point analysis) poorly constrained.
    A = np.column_stack(compute_forms.design_columns(form, V_arr, E_arr, F_arr))
    w = 1.0 / T_arr
    result, _, _, _ = np.linalg.lstsq(A * w[:, None], T_arr * w, rcond=None)
    T_pred = A @ result
    r2 = r_squared(T_arr, T_pred)

    out = {
        "form": form,
        "r_squared": r2,
        "_num_points": len(T_arr),
        "_distinct_feature_dims": sorted(float(f) for f in np.unique(F_arr)),
    }
    out.update(
        {name: float(v) for name, v in zip(compute_forms.FORM_COEFFS[form], result)}
    )
    return out


# ---------------------------------------------------------------------------
# Gather fit: T = launch_overhead + bytes / B_gather  (Hockney)
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

    # Hockney model: a fixed per-kernel launch/setup cost plus a linear
    # bandwidth term.
    #
    #     T(bytes) = launch_overhead + bytes / B_gather
    #
    # This replaces an earlier 5-parameter piecewise "L2-cache vs HBM
    # bandwidth" model. That model was unidentifiable on this benchmark's
    # data and fit garbage: because it capped the L2 term at L2_thresh but
    # only started the HBM term at a *separate*, larger HBM_thresh, every
    # byte count between the two thresholds had zero marginal cost — a flat
    # plateau spanning most of the sweep. It routinely converged with
    # B_L2 far SLOWER than B_HBM (physically backwards) and R² as low as
    # 0.60.
    #
    # The measured curve shows no two-regime structure to fit in the first
    # place: effective bandwidth rises smoothly to a single asymptote, and
    # the smallest transfers in the sweep are already large enough that any
    # cache effect is masked by the launch-overhead floor. Two parameters
    # are what the data supports, and this is the same Hockney form
    # fit_network uses for the interconnect (R² >= 0.999 here).
    def time_model(b, overhead, inv_bw):
        return overhead + b * inv_bw

    min_T, max_T = float(np.min(T_arr)), float(np.max(T_arr))
    max_b = float(np.max(bytes_arr))
    p0 = [min_T, (max_T - min_T) / (max_b + 1e-9)]

    try:
        # sigma=T_arr weights residuals by relative error, consistent with
        # fit_network/fit_compute — byte counts span ~4 orders of magnitude,
        # so an unweighted fit would be dominated by the largest transfers.
        popt, _ = curve_fit(
            time_model,
            bytes_arr,
            T_arr,
            p0=p0,
            bounds=([0, 0], [np.inf, np.inf]),
            method="trf",
            sigma=T_arr,
            absolute_sigma=False,
        )
        overhead, inv_bw = popt

    except Exception as e:
        print(f"Fit failed: {e}")
        overhead = inv_bw = np.nan

    bw = 1.0 / inv_bw if inv_bw > 0 else float("nan")

    if not np.isnan(overhead):
        T_pred = time_model(bytes_arr, overhead, inv_bw)
        ss_res = np.sum((T_arr - T_pred) ** 2)
        ss_tot = np.sum((T_arr - np.mean(T_arr)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    else:
        r_squared = float("nan")

    print(
        f" Fitted gather: launch_overhead={overhead*1e6:.2f} us  "
        f"BW={bw/1e9:.2f} GB/s  R²={r_squared:.4f}"
    )

    return {
        "bandwidth_bytes_per_sec": bw,
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
    p.add_argument("--compute-gcn-spmm", nargs="+", default=[], metavar="FILE",
                   help="Sweeps for the transform-then-aggregate SpMM GCN "
                        "layer (--model gcn_spmm). Fit with the vertex-centric "
                        "form; see analysis/compute_forms.py.")
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
    if args.pingpong_intra and args.pingpong_inter:
        # Flat (single-tier) network fit: pools intra + inter data into one
        # (B, t_L) pair, ignoring the intra/inter distinction. Used as a
        # genuine baseline for the hierarchical-vs-flat ablation, rather than
        # an ad hoc multiplier on the hierarchical prediction.
        recs = load_json_files(args.pingpong_intra) + load_json_files(args.pingpong_inter)
        net["flat"] = fit_network(recs)
        print(
            f"[network/flat] B={net['flat']['bandwidth_bytes_per_sec']/1e9:.2f} GB/s  "
            f"t_L={net['flat']['latency_seconds']*1e6:.2f} µs  "
            f"R²={net['flat']['r_squared']:.4f}"
        )
    result["network"] = net

    # Compute
    comp = {}
    for model_type, files in [
        ("gcn", args.compute_gcn),
        ("edge", args.compute_edge),
        ("gcn_spmm", args.compute_gcn_spmm),
    ]:
        if not files:
            continue
        recs = load_json_files(files)
        comp[model_type] = {
            "forward": fit_compute(recs, "forward_trials_seconds", model_type),
            "backward": fit_compute(recs, "backward_trials_seconds", model_type),
        }
        fwd = comp[model_type]["forward"]
        coeff_str = "  ".join(
            f"{name}={fwd[name]:.3e}"
            for name in compute_forms.FORM_COEFFS[fwd["form"]]
        )
        print(f"[compute/{model_type}] form={fwd['form']}  {coeff_str}  "
              f"R²={fwd['r_squared']:.4f}")
        if fwd["form"] == "legacy_VE":
            print(f"[compute/{model_type}] NOTE: only one feature dim "
                  f"({fwd['_distinct_feature_dims']}) in the input, so the "
                  "F-dependent form is unidentifiable and T_comp was fit at "
                  "fixed F. Sweep --feature-dim to get F*.")
        else:
            avg_deg = None
            if fwd["form"] == "vertex_centric":
                # F* depends on |E|/|V| for this form; report at the degree the
                # benchmark actually used rather than inventing one.
                degs = {
                    m["params"]["num_edges"] / m["params"]["num_vertices"]
                    for r in recs for m in r["measurements"]
                }
                avg_deg = sum(degs) / len(degs)
            f_star = compute_forms.crossover_feature_dim(fwd, avg_deg)
            print(f"[compute/{model_type}] F* = {f_star:.1f}" + (
                f"  (at mean degree {avg_deg:.1f})" if avg_deg else ""))
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
