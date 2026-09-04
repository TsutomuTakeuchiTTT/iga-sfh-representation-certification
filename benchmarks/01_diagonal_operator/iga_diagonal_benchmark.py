#!/usr/bin/env python3
"""Analytic diagonal-operator benchmark for reportable representation selection.

The benchmark independently compares:
1. a closed-form trust-region solution,
2. a generic constrained numerical optimizer, and
3. Monte Carlo verification of the worst-case MSE identity.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from decimal import Decimal, getcontext
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "iga_diagonal_benchmark_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize


OUTPUT_DIR = Path(__file__).resolve().parent
SEED = 20260826
MONTE_CARLO_SIZE = 1_000_000
EPSILON = 0.55
R_X = 1.5
KAPPA_VALUES = (1.0, 0.2, 0.08)

SIGMA_STRINGS = (
    "12",
    "8",
    "5",
    "3",
    "1.8",
    "1.1",
    "0.65",
    "0.4",
    "0.25",
    "0.15",
    "0.09",
    "0.05",
)
G_STRINGS = (
    "0.50",
    "0.42",
    "0.36",
    "0.31",
    "0.27",
    "0.23",
    "0.19",
    "0.16",
    "0.13",
    "0.10",
    "0.075",
    "0.05",
)

SIGMA = np.array([float(value) for value in SIGMA_STRINGS], dtype=float)
G = np.array([float(value) for value in G_STRINGS], dtype=float)
P = len(SIGMA)


def _analytic_weights_float(m: int, kappa: float) -> tuple[np.ndarray, float]:
    """Return the exact-form optimizer and Lagrange multiplier in float arithmetic."""
    sigma = SIGMA[:m]
    g = G[:m]
    h = 1.0 + R_X**2 * sigma**2
    c = R_X**2 * sigma * g
    weights = c / h
    if np.linalg.norm(weights) <= kappa:
        return weights, 0.0

    lower = 0.0
    upper = 1.0
    while np.linalg.norm(c / (h + upper)) > kappa:
        upper *= 2.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if np.linalg.norm(c / (h + middle)) > kappa:
            lower = middle
        else:
            upper = middle
    multiplier = upper
    return c / (h + multiplier), multiplier


def _risk_components(m: int, weights: np.ndarray) -> tuple[float, float, float]:
    residual = np.concatenate((SIGMA[:m] * weights - G[:m], -G[m:]))
    bias = R_X * np.linalg.norm(residual)
    stability = np.linalg.norm(weights)
    risk = np.hypot(bias, stability)
    return float(bias), float(stability), float(risk)


def _decimal_reference(m: int, kappa: float) -> tuple[float, float, float]:
    """Compute a high-precision reference risk and a two-float enclosing bracket."""
    getcontext().prec = 90
    decimal_sigma = [Decimal(value) for value in SIGMA_STRINGS[:m]]
    decimal_g = [Decimal(value) for value in G_STRINGS[:m]]
    decimal_tail_g = [Decimal(value) for value in G_STRINGS[m:]]
    decimal_r = Decimal(str(R_X))
    decimal_kappa = Decimal(str(kappa))
    h = [Decimal(1) + decimal_r**2 * value**2 for value in decimal_sigma]
    c = [decimal_r**2 * value * target for value, target in zip(decimal_sigma, decimal_g)]

    def weights_at(multiplier: Decimal) -> list[Decimal]:
        return [value / (curvature + multiplier) for value, curvature in zip(c, h)]

    def norm(values: list[Decimal]) -> Decimal:
        return sum(value * value for value in values).sqrt()

    weights = weights_at(Decimal(0))
    if norm(weights) > decimal_kappa:
        lower = Decimal(0)
        upper = Decimal(1)
        while norm(weights_at(upper)) > decimal_kappa:
            upper *= Decimal(2)
        for _ in range(400):
            middle = (lower + upper) / Decimal(2)
            if norm(weights_at(middle)) > decimal_kappa:
                lower = middle
            else:
                upper = middle
        weights = weights_at(upper)

    residual = [
        value * weight - target
        for value, weight, target in zip(decimal_sigma, weights, decimal_g)
    ]
    residual.extend([-target for target in decimal_tail_g])
    bias = decimal_r * norm(residual)
    stability = norm(weights)
    risk = (bias * bias + stability * stability).sqrt()
    risk_float = float(risk)
    lower_float = float(np.nextafter(risk_float, -np.inf))
    upper_float = float(np.nextafter(risk_float, np.inf))
    return risk_float, lower_float, upper_float


def _numerical_solution(m: int, kappa: float) -> tuple[np.ndarray, bool, int]:
    """Solve the same convex problem with a generic SLSQP optimizer."""
    sigma = SIGMA[:m]
    g = G[:m]
    tail_constant = R_X**2 * np.dot(G[m:], G[m:])

    def objective(weights: np.ndarray) -> float:
        residual = sigma * weights - g
        return float(R_X**2 * np.dot(residual, residual) + np.dot(weights, weights) + tail_constant)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return 2.0 * (R_X**2 * sigma * (sigma * weights - g) + weights)

    result = minimize(
        objective,
        np.zeros(m, dtype=float),
        jac=gradient,
        method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": lambda weights: kappa**2 - float(np.dot(weights, weights)),
            "jac": lambda weights: -2.0 * weights,
        },
        options={"ftol": 1.0e-14, "maxiter": 5000, "disp": False},
    )
    weights = np.asarray(result.x, dtype=float)
    weight_norm = np.linalg.norm(weights)
    if weight_norm > kappa:
        weights *= kappa / weight_norm
    return weights, bool(result.success), int(result.nit)


def _candidate_status(lower: float, upper: float) -> str:
    if upper <= EPSILON:
        return "feasible"
    if lower > EPSILON:
        return "infeasible"
    return "unresolved"


def _run_benchmark() -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    rng = np.random.default_rng(SEED)
    projected_standard_noise = rng.standard_normal(MONTE_CARLO_SIZE)

    for kappa in KAPPA_VALUES:
        for m in range(1, P + 1):
            analytic_weights, multiplier = _analytic_weights_float(m, kappa)
            analytic_bias, analytic_stability, analytic_risk = _risk_components(m, analytic_weights)
            reference_risk, lower_risk, reference_upper = _decimal_reference(m, kappa)
            numerical_weights, numerical_success, numerical_iterations = _numerical_solution(m, kappa)
            numerical_bias, numerical_stability, numerical_risk = _risk_components(m, numerical_weights)

            primal_upper = max(reference_upper, np.nextafter(numerical_risk, np.inf))
            status = _candidate_status(lower_risk, primal_upper)
            theoretical_mse = analytic_risk**2
            empirical_mse = np.nan
            relative_mse_error = np.nan
            if np.isclose(kappa, 0.2):
                errors = analytic_bias + analytic_stability * projected_standard_noise
                empirical_mse = float(np.mean(errors**2))
                relative_mse_error = abs(empirical_mse / theoretical_mse - 1.0)

            rows.append(
                {
                    "kappa": kappa,
                    "candidate_dimension": m,
                    "singular_value_at_boundary": SIGMA[m - 1],
                    "analytic_risk": analytic_risk,
                    "reference_risk": reference_risk,
                    "numerical_risk": numerical_risk,
                    "risk_lower_bound": lower_risk,
                    "risk_upper_bound": primal_upper,
                    "worst_case_bias": analytic_bias,
                    "stability_factor": analytic_stability,
                    "lagrange_multiplier": multiplier,
                    "stability_constraint_active": bool(multiplier > 0.0),
                    "status": status,
                    "numerical_success": numerical_success,
                    "numerical_iterations": numerical_iterations,
                    "analytic_numerical_absolute_error": abs(analytic_risk - numerical_risk),
                    "analytic_numerical_relative_error": abs(numerical_risk / analytic_risk - 1.0),
                    "theoretical_worst_case_mse": theoretical_mse,
                    "empirical_worst_case_mse": empirical_mse,
                    "empirical_mse_relative_error": relative_mse_error,
                }
            )

    frame = pd.DataFrame(rows)
    selections: dict[str, int | None] = {}
    for kappa in KAPPA_VALUES:
        subset = frame.loc[frame["kappa"] == kappa]
        feasible = subset.loc[subset["status"] == "feasible", "candidate_dimension"]
        selections[f"{kappa:g}"] = None if feasible.empty else int(feasible.min())

    monotonic_violations: dict[str, int] = {}
    for kappa in KAPPA_VALUES:
        risks = frame.loc[frame["kappa"] == kappa, "analytic_risk"].to_numpy()
        monotonic_violations[f"{kappa:g}"] = int(np.sum(np.diff(risks) > 5.0e-13))

    moderate = frame.loc[frame["kappa"] == 0.2]
    boundary_epsilon = float(moderate.loc[moderate["candidate_dimension"] == 6, "reference_risk"].iloc[0])
    boundary_width = 1.0e-6
    boundary_status = (
        "unresolved"
        if boundary_epsilon - boundary_width <= boundary_epsilon < boundary_epsilon + boundary_width
        else "unexpected"
    )

    summary = {
        "benchmark": "analytic diagonal operator",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "numerical_optimizer": {
            "method": "SLSQP",
            "ftol": 1.0e-14,
            "maxiter": 5000,
        },
        "seed": SEED,
        "monte_carlo_size": MONTE_CARLO_SIZE,
        "epsilon": EPSILON,
        "state_radius": R_X,
        "candidate_dimensions": list(range(1, P + 1)),
        "singular_values": SIGMA.tolist(),
        "target_coefficients": G.tolist(),
        "kappa_values": list(KAPPA_VALUES),
        "selected_dimensions": selections,
        "monotonicity_violations": monotonic_violations,
        "max_analytic_numerical_absolute_error": float(frame["analytic_numerical_absolute_error"].max()),
        "max_analytic_numerical_relative_error": float(frame["analytic_numerical_relative_error"].max()),
        "max_monte_carlo_mse_relative_error": float(moderate["empirical_mse_relative_error"].max()),
        "boundary_diagnostic": {
            "candidate_dimension": 6,
            "epsilon_equal_to_reference_risk": boundary_epsilon,
            "illustrative_bracket_half_width": boundary_width,
            "status": boundary_status,
        },
    }
    return frame, summary


def _validate(frame: pd.DataFrame, summary: dict) -> None:
    if not frame["numerical_success"].all():
        failed = frame.loc[~frame["numerical_success"]]
        raise RuntimeError(f"Numerical optimization failed for rows:\n{failed}")
    if summary["max_analytic_numerical_relative_error"] > 2.0e-8:
        raise RuntimeError("Analytic and numerical risks do not agree to the required tolerance.")
    if any(summary["monotonicity_violations"].values()):
        raise RuntimeError("Nested candidate risk is not monotone.")
    if summary["selected_dimensions"] != {"1": 6, "0.2": 7, "0.08": None}:
        raise RuntimeError(f"Unexpected candidate selections: {summary['selected_dimensions']}")
    if summary["max_monte_carlo_mse_relative_error"] > 1.0e-2:
        raise RuntimeError("Monte Carlo MSE check exceeds one percent relative error.")
    if summary["boundary_diagnostic"]["status"] != "unresolved":
        raise RuntimeError("Boundary diagnostic did not produce the unresolved status.")


def _plot(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))
    colors = {1.0: "#0072B2", 0.2: "#D55E00", 0.08: "#009E73"}
    labels = {1.0: r"$\kappa=1.0$", 0.2: r"$\kappa=0.20$", 0.08: r"$\kappa=0.08$"}

    ax = axes[0]
    for kappa in KAPPA_VALUES:
        subset = frame.loc[frame["kappa"] == kappa]
        ax.plot(
            subset["candidate_dimension"],
            subset["analytic_risk"],
            marker="o",
            markersize=3.8,
            linewidth=1.5,
            color=colors[kappa],
            label=labels[kappa],
        )
        feasible = subset.loc[subset["status"] == "feasible"]
        if not feasible.empty:
            selected = feasible.iloc[0]
            ax.scatter(
                selected["candidate_dimension"],
                selected["analytic_risk"],
                s=65,
                facecolors="none",
                edgecolors=colors[kappa],
                linewidths=1.6,
                zorder=5,
            )
    ax.axhline(EPSILON, color="0.25", linestyle="--", linewidth=1.1, label=r"$\epsilon=0.55$")
    ax.set_xlabel("Candidate dimension $m$")
    ax.set_ylabel(r"Candidate risk $\rho_m$")
    ax.set_xticks(range(1, P + 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1]
    moderate = frame.loc[frame["kappa"] == 0.2]
    ax.plot(
        moderate["candidate_dimension"],
        moderate["worst_case_bias"],
        marker="o",
        markersize=3.8,
        linewidth=1.5,
        color="#CC79A7",
        label="Worst-case bias",
    )
    ax.plot(
        moderate["candidate_dimension"],
        moderate["stability_factor"],
        marker="s",
        markersize=3.5,
        linewidth=1.5,
        color="#56B4E9",
        label="Stability factor",
    )
    ax.axhline(0.2, color="0.35", linestyle="--", linewidth=1.1, label=r"$\kappa=0.20$")
    ax.set_xlabel("Candidate dimension $m$")
    ax.set_ylabel("Risk component")
    ax.set_xticks(range(1, P + 1))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[2]
    numerical_relative = np.maximum(
        moderate["analytic_numerical_relative_error"].to_numpy(),
        np.finfo(float).eps,
    )
    mse_relative = np.maximum(
        moderate["empirical_mse_relative_error"].to_numpy(),
        np.finfo(float).eps,
    )
    ax.semilogy(
        moderate["candidate_dimension"],
        numerical_relative,
        marker="o",
        markersize=3.8,
        linewidth=1.5,
        color="#E69F00",
        label="Numerical vs analytic risk",
    )
    ax.semilogy(
        moderate["candidate_dimension"],
        mse_relative,
        marker="s",
        markersize=3.5,
        linewidth=1.5,
        color="#0072B2",
        label="Monte Carlo vs exact MSE",
    )
    ax.set_xlabel("Candidate dimension $m$")
    ax.set_ylabel("Absolute relative error")
    ax.set_xticks(range(1, P + 1))
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, va="top", fontweight="bold")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUTPUT_DIR / "fig_diagonal_operator_benchmark.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig_diagonal_operator_benchmark.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame, summary = _run_benchmark()
    _validate(frame, summary)
    frame.to_csv(OUTPUT_DIR / "diagonal_operator_benchmark_results.csv", index=False)
    with (OUTPUT_DIR / "diagonal_operator_benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _plot(frame)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
