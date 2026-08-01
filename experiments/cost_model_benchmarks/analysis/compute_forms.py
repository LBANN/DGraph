"""Design matrices for the T_comp fit — the single definition shared by
``fit_primitives.fit_compute`` (which fits the coefficients) and
``fit_overhead.predict_layer_time`` (which evaluates them).

Keeping one definition matters: a mismatch between the columns used to fit and
the expression used to predict is silent — it produces plausible numbers that
are simply wrong, with no exception and no bad R^2 to notice.

Forms
-----
``legacy_VE``      T = a|V| + b|E| + c
    The original fixed-F form. Exact as a linearization at one feature
    dimension, but says nothing about how cost scales with F.

``edge_centric``   T = a|E|F^2 + b|E|F + c|V|F + d
    For layers that apply the dense transform *per edge*: ``GCNLayer``
    (``linear(x[src])``) and ``EdgeConditionedLayer`` (an MLP over
    ``[h_src, h_dst, e]``). The F^2 term is the GEMM, the |E|F terms are the
    gather and scatter traffic, the |V|F term is the output/zero-init.

``vertex_centric`` T = a|V|F^2 + b|E|F + c|V|F + d
    For transform-then-aggregate layers: ``GCNSpMMLayer``. The GEMM is now
    over vertices, and the SpMM contributes |E|F.

Why F matters
-------------
At a single F, |E|F^2 and |E|F are perfectly collinear, so fixed-F data cannot
distinguish a compute-bound layer from a bandwidth-bound one — both hypotheses
fit equally well. Sweeping F separates them, and the fitted ratio gives the
feature dimension at which a layer crosses over:

    F* = b/a   (edge_centric)      the F where bandwidth and compute terms match

``fit_compute`` therefore refuses to fit an F-dependent form to single-F data
and falls back to ``legacy_VE``; see ``select_form``.
"""

import numpy as np


# Which form each --model value takes.
MODEL_FORM = {
    "gcn": "edge_centric",
    "edge": "edge_centric",
    "gcn_spmm": "vertex_centric",
}

# Coefficient names per form, in design-matrix column order.
FORM_COEFFS = {
    "legacy_VE": ["coeff_V", "coeff_E", "intercept"],
    "edge_centric": ["coeff_EF2", "coeff_EF", "coeff_VF", "intercept"],
    "vertex_centric": ["coeff_VF2", "coeff_EF", "coeff_VF", "intercept"],
}


def design_columns(form: str, V, E, F):
    """Return the design-matrix columns for *form* as a list of arrays.

    V, E, F may be scalars or arrays; they are broadcast together.
    """
    V = np.asarray(V, dtype=float)
    E = np.asarray(E, dtype=float)
    F = np.asarray(F, dtype=float)
    ones = np.ones_like(V * E * F)

    if form == "legacy_VE":
        return [V * ones, E * ones, ones]
    if form == "edge_centric":
        return [E * F * F, E * F, V * F, ones]
    if form == "vertex_centric":
        return [V * F * F, E * F, V * F, ones]
    raise ValueError(
        f"Unknown compute form {form!r}; expected one of {sorted(FORM_COEFFS)}"
    )


def select_form(model_type: str, distinct_feature_dims: int) -> str:
    """Pick the fit form for *model_type*, downgrading if F was not swept.

    An F-dependent form needs at least two distinct feature dimensions to be
    identifiable: at a single F, |V|F^2 and |V|F differ only by a constant
    factor, so the columns are collinear and the split between them is
    arbitrary. lstsq would still return *an* answer — one that fits the
    training data and extrapolates nonsensically in F. Fall back instead.
    """
    if distinct_feature_dims < 2:
        return "legacy_VE"
    return MODEL_FORM.get(model_type, "edge_centric")


def evaluate(params: dict, V, E, F) -> float:
    """Evaluate a fitted compute model. *params* is a fit_compute() result."""
    form = params.get("form", "legacy_VE")
    coeffs = [params[name] for name in FORM_COEFFS[form]]
    cols = design_columns(form, V, E, F)
    return float(sum(c * col for c, col in zip(coeffs, cols)))


def crossover_feature_dim(params: dict, avg_degree: float = None) -> float:
    """F* — the feature dim where the GEMM term overtakes the aggregation term.

    Below F*, the layer's cost is dominated by moving features (linear in F);
    above it, by the dense transform (quadratic in F).

    The two forms differ in whether F* depends on the graph:

    - ``edge_centric``: compute is ``a|E|F^2``, aggregation ``b|E|F``. The |E|
      cancels, so ``F* = b/a`` — a property of the layer alone.
    - ``vertex_centric``: compute is ``a|V|F^2``, aggregation ``b|E|F``. These
      have different geometry, so ``F* = (b/a) * (|E|/|V|)`` = ``(b/a) *
      avg_degree`` — the crossover moves with graph density, which is the
      whole point of transform-then-aggregate. *avg_degree* is required here.

    Returns NaN for the legacy form, which has no F dependence.
    """
    form = params.get("form", "legacy_VE")
    if form == "legacy_VE":
        return float("nan")
    a = params[FORM_COEFFS[form][0]]  # the F^2 coefficient
    b = params["coeff_EF"]
    if not a > 0:
        return float("nan")
    if form == "edge_centric":
        return float(b / a)
    if avg_degree is None:
        raise ValueError(
            "avg_degree is required for the vertex_centric form: its crossover "
            "F* = (b/a) * |E|/|V| depends on graph density, unlike the "
            "edge_centric form where |E| cancels."
        )
    return float((b / a) * avg_degree)
