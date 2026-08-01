"""Analysis — Apply Assembled Cost Model and Compare to Measurements.

Reads:
    - data/fitted_primitives.json
    - data/fitted_overhead.json
    - All end-to-end JSON run files

For every run, computes the predicted T_layer:

    T_layer = T_comp + max(T_intra, T_inter) + T_buffer_copy + T_overhead

and records the measured median, predicted value, absolute error, and relative
error (as a fraction).

Also computes aggregate MAPE for the fit subset and the held-out subset
(using the same --fit-filter expression as fit_overhead.py).

Outputs ``data/predictions.json``.

Usage::

    python -m analysis.compute_predictions \\
        --primitives    data/fitted_primitives.json \\
        --overhead      data/fitted_overhead.json \\
        --e2e-runs      data/e2e_*.json \\
        --fit-filter    "world_size <= 8" \\
        --output        data/predictions.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.fit_overhead import (
    apply_filter,
    load_e2e_runs,
    predict_layer_time,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Apply cost model and compute predictions")
    p.add_argument("--primitives", type=str, required=True)
    p.add_argument("--overhead", type=str, required=True)
    p.add_argument("--e2e-runs", nargs="+", required=True, metavar="FILE")
    p.add_argument("--fit-filter", type=str, default="world_size <= 8",
                   help="Same expression used when fitting overhead (determines train/test split)")
    p.add_argument("--output", type=str, default="data/predictions.json")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.primitives) as f:
        primitives = json.load(f)
    with open(args.overhead) as f:
        overhead_data = json.load(f)

    T_overhead = overhead_data.get("overhead_seconds", 0.0)
    # Read from fitted_overhead.json rather than taking a second CLI flag:
    # T_overhead was fitted under a specific contention assumption, so passing
    # a different nics_per_node here would silently pair an overhead constant
    # with a network model it was never fitted against.
    nics_per_node = overhead_data.get("nics_per_node", 1)

    all_runs = load_e2e_runs(args.e2e_runs)
    fit_runs, held_runs = apply_filter(all_runs, args.fit_filter)

    fit_set = set(id(r) for r in fit_runs)

    prediction_entries = []
    for r in all_runs:
        breakdown = predict_layer_time(r["config"], r["per_rank_stats"], primitives, nics_per_node)
        T_pred = breakdown["T_total"] + T_overhead
        T_meas = r["measured_median"]
        abs_err = abs(T_meas - T_pred)
        rel_err = abs_err / T_meas if T_meas > 0 else float("nan")

        entry = {
            "source_file": r["source_file"],
            "config": r["config"],
            "partition_stats": r["per_rank_stats"],
            "measured_median_seconds": T_meas,
            "predicted_seconds": T_pred,
            "absolute_error_seconds": abs_err,
            "relative_error": rel_err,
            "in_fit_set": id(r) in fit_set,
            "breakdown": {
                "T_comp_seconds": breakdown["T_comp"],
                "T_comm_seconds": breakdown["T_comm"],
                "T_buffer_copy_seconds": breakdown["T_buffer_copy"],
                "T_overhead_seconds": T_overhead,
            },
        }
        prediction_entries.append(entry)

    # Aggregate MAPE
    def mape(entries):
        errs = [e["relative_error"] for e in entries
                if not np.isnan(e["relative_error"])]
        return float(np.mean(errs)) if errs else float("nan")

    fit_entries  = [e for e in prediction_entries if e["in_fit_set"]]
    held_entries = [e for e in prediction_entries if not e["in_fit_set"]]

    mape_fit   = mape(fit_entries)
    mape_held  = mape(held_entries)
    mape_total = mape(prediction_entries)

    print(f"[predictions] Fit MAPE={mape_fit*100:.2f}%  "
          f"Held-out MAPE={mape_held*100:.2f}%  "
          f"Total MAPE={mape_total*100:.2f}%")

    result = {
        "fit_filter": args.fit_filter,
        "T_overhead_seconds": T_overhead,
        "aggregate": {
            "mape_fit_set":    mape_fit,
            "mape_held_out":   mape_held,
            "mape_all":        mape_total,
            "num_fit":         len(fit_entries),
            "num_held_out":    len(held_entries),
        },
        "predictions": prediction_entries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[predictions] Written to {out_path}")


if __name__ == "__main__":
    main()
