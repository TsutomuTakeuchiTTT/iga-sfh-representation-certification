#!/usr/bin/env python3
"""Robustness and post-selection benchmark for adaptive representation.

The benchmark has two deliberately separate modules.

1. A finite-dimensional inverse problem compares nominal certification with a
   robust outer-envelope certification under a structured operator mismatch and
   an explicit quadratic nonlinear remainder.  The coupled worst-case bias is
   available in closed form and is also checked by direct adversarial
   optimization.
2. A Gaussian two-candidate experiment isolates post-selection coverage.  Each
   fixed candidate has 95 percent marginal coverage, while selecting with the
   same data reduces coverage to 0.95**2.  Independent selection and a
   simultaneous calibration both restore 95 percent coverage.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "iga_robust_postselection_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq, minimize
from scipy.stats import norm


try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    # Jupyter Notebook does not define __file__.
    OUTPUT_DIR = Path.cwd()

P = 6
R_STATE = 0.90
EPSILON = 0.480
KAPPA = 0.65
ETA_OPERATOR = 0.30
Q_NONLINEAR = 0.20
LOWER_BOUND_PADDING = 1.0e-10

SIGMA = np.array([5.0, 2.6, 1.4, 0.75, 0.38, 0.18], dtype=float)
G = np.array([0.55, 0.45, 0.34, 0.25, 0.16, 0.10], dtype=float)
H_DIAGONAL = np.array([1.0, 0.8, 0.6, 0.45, 0.30, 0.20], dtype=float)
H = np.diag(H_DIAGONAL)
NONLINEAR_DIRECTION = np.array([0.65, 0.50, 0.40, 0.30, 0.20, 0.10], dtype=float)
NONLINEAR_DIRECTION /= np.linalg.norm(NONLINEAR_DIRECTION)

COVERAGE_PROBABILITY = 0.95
COVERAGE_REPETITIONS = 1_000_000
COVERAGE_SEED = 20260826


@dataclass(frozen=True)
class Candidate:
    dimension: int

    @property
    def name(self) -> str:
        return f"V{self.dimension}"

    @property
    def basis(self) -> np.ndarray:
        return np.eye(P)[:, : self.dimension]


CANDIDATES = tuple(Candidate(dimension) for dimension in range(1, P + 1))


def _bias_and_gradient(
    reduced_weight: np.ndarray,
    candidate: Candidate,
    robust: bool,
) -> tuple[float, np.ndarray, dict[str, float]]:
    basis = candidate.basis
    weight = basis @ reduced_weight
    response_residual = SIGMA * weight - G

    residual_norm = np.linalg.norm(response_residual)
    state_bias = R_STATE * residual_norm
    if residual_norm > 0.0:
        state_gradient = (
            R_STATE
            * basis.T
            @ (SIGMA * response_residual)
            / residual_norm
        )
    else:
        state_gradient = np.zeros(candidate.dimension, dtype=float)

    operator_vector = H.T @ weight
    operator_norm = np.linalg.norm(operator_vector)
    operator_bias = ETA_OPERATOR * R_STATE * operator_norm if robust else 0.0
    if robust and operator_norm > 0.0:
        operator_gradient = (
            ETA_OPERATOR
            * R_STATE
            * basis.T
            @ (H @ operator_vector)
            / operator_norm
        )
    else:
        operator_gradient = np.zeros(candidate.dimension, dtype=float)

    nonlinear_projection = float(NONLINEAR_DIRECTION @ weight)
    nonlinear_bias = Q_NONLINEAR * abs(nonlinear_projection) if robust else 0.0
    if robust and nonlinear_projection != 0.0:
        nonlinear_gradient = (
            Q_NONLINEAR
            * np.sign(nonlinear_projection)
            * basis.T
            @ NONLINEAR_DIRECTION
        )
    else:
        nonlinear_gradient = np.zeros(candidate.dimension, dtype=float)

    total_bias = state_bias + operator_bias + nonlinear_bias
    bias_gradient = state_gradient + operator_gradient + nonlinear_gradient
    return (
        float(total_bias),
        bias_gradient,
        {
            "state_bias": float(state_bias),
            "operator_bias": float(operator_bias),
            "nonlinear_bias": float(nonlinear_bias),
            "total_bias": float(total_bias),
        },
    )


def _objective_and_gradient(
    reduced_weight: np.ndarray,
    candidate: Candidate,
    robust: bool,
) -> tuple[float, np.ndarray, dict[str, float]]:
    bias, bias_gradient, components = _bias_and_gradient(
        reduced_weight,
        candidate,
        robust,
    )
    stability = float(np.linalg.norm(reduced_weight))
    objective = bias**2 + stability**2
    gradient = 2.0 * bias * bias_gradient + 2.0 * reduced_weight
    components = {
        **components,
        "stability_factor": stability,
        "objective_squared_risk": float(objective),
        "risk": float(np.sqrt(objective)),
    }
    return float(objective), gradient, components


def _make_feasible(weight: np.ndarray) -> np.ndarray:
    result = np.asarray(weight, dtype=float).copy()
    norm_value = np.linalg.norm(result)
    if norm_value > KAPPA:
        result *= KAPPA / norm_value
    return result


def _solve_slsqp(candidate: Candidate, robust: bool) -> dict:
    result = minimize(
        lambda weight: _objective_and_gradient(weight, candidate, robust)[0],
        np.zeros(candidate.dimension, dtype=float),
        jac=lambda weight: _objective_and_gradient(weight, candidate, robust)[1],
        method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": lambda weight: KAPPA**2 - float(weight @ weight),
            "jac": lambda weight: -2.0 * weight,
        },
        options={"ftol": 1.0e-14, "maxiter": 10000, "disp": False},
    )
    reduced_weight = _make_feasible(result.x)
    objective, gradient, components = _objective_and_gradient(
        reduced_weight,
        candidate,
        robust,
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "reduced_weight": reduced_weight,
        "weight": candidate.basis @ reduced_weight,
        "objective": objective,
        "gradient": gradient,
        "components": components,
    }


def _solve_cobyla(candidate: Candidate, robust: bool) -> dict:
    result = minimize(
        lambda weight: _objective_and_gradient(weight, candidate, robust)[0],
        np.zeros(candidate.dimension, dtype=float),
        method="COBYLA",
        constraints={
            "type": "ineq",
            "fun": lambda weight: KAPPA**2 - float(weight @ weight),
        },
        options={
            "catol": 1.0e-12,
            "tol": 1.0e-11,
            "maxiter": 100000,
            "rhobeg": 0.10,
        },
    )
    reduced_weight = _make_feasible(result.x)
    objective, gradient, components = _objective_and_gradient(
        reduced_weight,
        candidate,
        robust,
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nfev),
        "reduced_weight": reduced_weight,
        "weight": candidate.basis @ reduced_weight,
        "objective": objective,
        "gradient": gradient,
        "components": components,
    }


def _analytic_nominal_solution(candidate: Candidate) -> dict:
    indices = np.arange(candidate.dimension)
    singular_values = SIGMA[indices]
    target = G[indices]

    def weights(lagrange_multiplier: float) -> np.ndarray:
        return (
            R_STATE**2
            * singular_values
            * target
            / (1.0 + R_STATE**2 * singular_values**2 + lagrange_multiplier)
        )

    unconstrained = weights(0.0)
    if np.linalg.norm(unconstrained) <= KAPPA:
        lagrange_multiplier = 0.0
        reduced_weight = unconstrained
    else:
        upper = 1.0
        while np.linalg.norm(weights(upper)) > KAPPA:
            upper *= 2.0
        lagrange_multiplier = brentq(
            lambda value: np.linalg.norm(weights(value)) - KAPPA,
            0.0,
            upper,
            xtol=1.0e-14,
        )
        reduced_weight = weights(lagrange_multiplier)

    objective, _, components = _objective_and_gradient(
        reduced_weight,
        candidate,
        robust=False,
    )
    return {
        "lagrange_multiplier": float(lagrange_multiplier),
        "reduced_weight": reduced_weight,
        "weight": candidate.basis @ reduced_weight,
        "objective": objective,
        "components": components,
    }


def _convex_risk_bracket(solutions: tuple[dict, ...]) -> tuple[float, float]:
    upper_objective = min(solution["objective"] for solution in solutions)
    lower_candidates = []
    for solution in solutions:
        point = solution["reduced_weight"]
        objective = solution["objective"]
        gradient = solution["gradient"]
        lower_candidates.append(
            objective - float(gradient @ point) - KAPPA * np.linalg.norm(gradient)
        )
    scale = 1.0 + upper_objective
    lower_objective = max(
        0.0,
        max(lower_candidates) - LOWER_BOUND_PADDING * scale,
    )
    upper_objective += LOWER_BOUND_PADDING * scale
    return float(np.sqrt(lower_objective)), float(np.sqrt(upper_objective))


def _candidate_status(lower_risk: float, upper_risk: float) -> str:
    if upper_risk <= EPSILON:
        return "feasible"
    if lower_risk > EPSILON:
        return "infeasible"
    return "unresolved"


def _quadratic_ball_maximum(linear: np.ndarray, coefficient: float) -> float:
    linear_norm = float(np.linalg.norm(linear))
    if coefficient < 0.0 and linear_norm <= -2.0 * coefficient * R_STATE:
        return float(-linear_norm**2 / (4.0 * coefficient))
    return float(R_STATE * linear_norm + coefficient * R_STATE**2)


def _coupled_bias_closed_form(weight: np.ndarray) -> float:
    response_residual = SIGMA * weight - G
    nonlinear_projection = float(NONLINEAR_DIRECTION @ weight)
    values = []
    for outer_sign in (-1.0, 1.0):
        for operator_sign in (-1.0, 1.0):
            linear = (
                outer_sign * response_residual
                + ETA_OPERATOR * operator_sign * H.T @ weight
            )
            coefficient = (
                outer_sign
                * Q_NONLINEAR
                * nonlinear_projection
                / R_STATE**2
            )
            values.append(_quadratic_ball_maximum(linear, coefficient))
    return float(max(values))


def _coupled_bias_direct(weight: np.ndarray) -> float:
    response_residual = SIGMA * weight - G
    nonlinear_projection = float(NONLINEAR_DIRECTION @ weight)
    best = -np.inf
    for outer_sign in (-1.0, 1.0):
        for operator_sign in (-1.0, 1.0):
            linear = (
                outer_sign * response_residual
                + ETA_OPERATOR * operator_sign * H.T @ weight
            )
            coefficient = (
                outer_sign
                * Q_NONLINEAR
                * nonlinear_projection
                / R_STATE**2
            )

            def negative_objective(state: np.ndarray) -> float:
                return -float(linear @ state + coefficient * (state @ state))

            linear_norm = np.linalg.norm(linear)
            starts = [np.zeros(P, dtype=float)]
            if linear_norm > 0.0:
                starts.extend(
                    [
                        R_STATE * linear / linear_norm,
                        -R_STATE * linear / linear_norm,
                    ]
                )
            for axis in range(P):
                unit = np.zeros(P, dtype=float)
                unit[axis] = R_STATE
                starts.extend([unit, -unit])

            for start in starts:
                result = minimize(
                    negative_objective,
                    start,
                    method="SLSQP",
                    constraints={
                        "type": "ineq",
                        "fun": lambda state: R_STATE**2 - float(state @ state),
                        "jac": lambda state: -2.0 * state,
                    },
                    options={"ftol": 1.0e-14, "maxiter": 10000, "disp": False},
                )
                state = np.asarray(result.x, dtype=float)
                state_norm = np.linalg.norm(state)
                if state_norm > R_STATE:
                    state *= R_STATE / state_norm
                best = max(best, -negative_objective(state))
    return float(best)


def _evaluate_weight(weight: np.ndarray) -> dict[str, float]:
    response_residual = SIGMA * weight - G
    nominal_bias = R_STATE * np.linalg.norm(response_residual)
    operator_support = ETA_OPERATOR * R_STATE * np.linalg.norm(H.T @ weight)
    nonlinear_support = Q_NONLINEAR * abs(
        float(NONLINEAR_DIRECTION @ weight)
    )
    robust_bias = nominal_bias + operator_support + nonlinear_support
    coupled_bias = _coupled_bias_closed_form(weight)
    coupled_direct = _coupled_bias_direct(weight)
    stability = np.linalg.norm(weight)
    return {
        "nominal_bias": float(nominal_bias),
        "operator_support": float(operator_support),
        "nonlinear_support": float(nonlinear_support),
        "robust_bias": float(robust_bias),
        "coupled_bias_closed_form": float(coupled_bias),
        "coupled_bias_direct": float(coupled_direct),
        "coupled_bias_absolute_difference": float(
            abs(coupled_bias - coupled_direct)
        ),
        "stability_factor": float(stability),
        "nominal_risk": float(np.hypot(nominal_bias, stability)),
        "coupled_risk": float(np.hypot(coupled_bias, stability)),
        "robust_outer_risk": float(np.hypot(robust_bias, stability)),
    }


def _run_robustness_module() -> pd.DataFrame:
    rows: list[dict] = []
    for candidate in CANDIDATES:
        analytic = _analytic_nominal_solution(candidate)
        nominal_slsqp = _solve_slsqp(candidate, robust=False)
        nominal_cobyla = _solve_cobyla(candidate, robust=False)
        robust_slsqp = _solve_slsqp(candidate, robust=True)
        robust_cobyla = _solve_cobyla(candidate, robust=True)

        nominal_lower, nominal_upper = _convex_risk_bracket(
            (nominal_slsqp, nominal_cobyla)
        )
        robust_lower, robust_upper = _convex_risk_bracket(
            (robust_slsqp, robust_cobyla)
        )
        nominal_evaluation = _evaluate_weight(nominal_slsqp["weight"])
        robust_evaluation = _evaluate_weight(robust_slsqp["weight"])

        rows.append(
            {
                "candidate": candidate.name,
                "candidate_dimension": candidate.dimension,
                "epsilon": EPSILON,
                "kappa": KAPPA,
                "nominal_risk": nominal_slsqp["components"]["risk"],
                "nominal_risk_lower_bound": nominal_lower,
                "nominal_risk_upper_bound": nominal_upper,
                "nominal_risk_bracket_width": nominal_upper - nominal_lower,
                "nominal_status": _candidate_status(nominal_lower, nominal_upper),
                "robust_risk": robust_slsqp["components"]["risk"],
                "robust_risk_lower_bound": robust_lower,
                "robust_risk_upper_bound": robust_upper,
                "robust_risk_bracket_width": robust_upper - robust_lower,
                "robust_status": _candidate_status(robust_lower, robust_upper),
                "nominal_analytic_risk": analytic["components"]["risk"],
                "nominal_analytic_relative_difference": abs(
                    nominal_slsqp["components"]["risk"]
                    / analytic["components"]["risk"]
                    - 1.0
                ),
                "nominal_solver_relative_difference": abs(
                    nominal_cobyla["components"]["risk"]
                    / nominal_slsqp["components"]["risk"]
                    - 1.0
                ),
                "robust_solver_relative_difference": abs(
                    robust_cobyla["components"]["risk"]
                    / robust_slsqp["components"]["risk"]
                    - 1.0
                ),
                "nominal_solution_weight": " ".join(
                    f"{value:.16g}" for value in nominal_slsqp["weight"]
                ),
                "robust_solution_weight": " ".join(
                    f"{value:.16g}" for value in robust_slsqp["weight"]
                ),
                **{
                    f"nominal_solution_{key}": value
                    for key, value in nominal_evaluation.items()
                },
                **{
                    f"robust_solution_{key}": value
                    for key, value in robust_evaluation.items()
                },
                "nominal_slsqp_success": nominal_slsqp["success"],
                "nominal_cobyla_success": nominal_cobyla["success"],
                "robust_slsqp_success": robust_slsqp["success"],
                "robust_cobyla_success": robust_cobyla["success"],
            }
        )
    return pd.DataFrame(rows)


def _select_candidate(frame: pd.DataFrame, status_column: str) -> str | None:
    feasible = frame.loc[frame[status_column] == "feasible"].copy()
    if feasible.empty:
        return None
    feasible = feasible.sort_values("candidate_dimension", kind="stable")
    return str(feasible.iloc[0]["candidate"])


def _coverage_row(
    procedure: str,
    selected_estimate: np.ndarray,
    covered: np.ndarray,
    selection_index: np.ndarray | None,
    half_width: float,
    analytic_coverage: float,
    analytic_rmse: float,
) -> dict:
    repetitions = selected_estimate.size
    coverage = float(np.mean(covered))
    if selection_index is None:
        frequency_v1 = 1.0 if procedure == "fixed_V1" else 0.0
        frequency_v2 = 1.0 - frequency_v1
    else:
        frequency_v1 = float(np.mean(selection_index == 0))
        frequency_v2 = float(np.mean(selection_index == 1))
    conditional_bias = float(np.mean(selected_estimate))
    conditional_rmse = float(np.sqrt(np.mean(selected_estimate**2)))
    analytic_standard_error = float(
        np.sqrt(analytic_coverage * (1.0 - analytic_coverage) / repetitions)
    )
    return {
        "procedure": procedure,
        "repetitions": repetitions,
        "target_coverage": COVERAGE_PROBABILITY,
        "interval_half_width": float(half_width),
        "reporting_probability": 1.0,
        "coverage": coverage,
        "analytic_coverage": float(analytic_coverage),
        "coverage_minus_analytic": coverage - analytic_coverage,
        "coverage_standardized_difference": (
            coverage - analytic_coverage
        ) / analytic_standard_error,
        "monte_carlo_standard_error": float(
            np.sqrt(coverage * (1.0 - coverage) / repetitions)
        ),
        "conditional_bias": conditional_bias,
        "analytic_conditional_bias": 0.0,
        "conditional_rmse": conditional_rmse,
        "analytic_conditional_rmse": float(analytic_rmse),
        "conditional_rmse_minus_analytic": conditional_rmse - analytic_rmse,
        "selection_frequency_V1": frequency_v1,
        "selection_frequency_V2": frequency_v2,
    }


def _run_coverage_module() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(COVERAGE_SEED)
    inference_data = rng.standard_normal((COVERAGE_REPETITIONS, 2))
    independent_selection_data = rng.standard_normal((COVERAGE_REPETITIONS, 2))
    rows_index = np.arange(COVERAGE_REPETITIONS)

    marginal_half_width = float(norm.ppf((1.0 + COVERAGE_PROBABILITY) / 2.0))
    simultaneous_half_width = float(
        norm.ppf((1.0 + np.sqrt(COVERAGE_PROBABILITY)) / 2.0)
    )
    same_data_rmse = float(np.sqrt(1.0 + 2.0 / np.pi))

    same_data_index = np.argmax(np.abs(inference_data), axis=1)
    independent_index = np.argmax(np.abs(independent_selection_data), axis=1)
    same_data_estimate = inference_data[rows_index, same_data_index]
    independent_estimate = inference_data[rows_index, independent_index]

    rows = [
        _coverage_row(
            "fixed_V1",
            inference_data[:, 0],
            np.abs(inference_data[:, 0]) <= marginal_half_width,
            None,
            marginal_half_width,
            COVERAGE_PROBABILITY,
            1.0,
        ),
        _coverage_row(
            "fixed_V2",
            inference_data[:, 1],
            np.abs(inference_data[:, 1]) <= marginal_half_width,
            None,
            marginal_half_width,
            COVERAGE_PROBABILITY,
            1.0,
        ),
        _coverage_row(
            "same_data_naive",
            same_data_estimate,
            np.abs(same_data_estimate) <= marginal_half_width,
            same_data_index,
            marginal_half_width,
            COVERAGE_PROBABILITY**2,
            same_data_rmse,
        ),
        _coverage_row(
            "independent_selection",
            independent_estimate,
            np.abs(independent_estimate) <= marginal_half_width,
            independent_index,
            marginal_half_width,
            COVERAGE_PROBABILITY,
            1.0,
        ),
        _coverage_row(
            "same_data_calibrated",
            same_data_estimate,
            np.abs(same_data_estimate) <= simultaneous_half_width,
            same_data_index,
            simultaneous_half_width,
            COVERAGE_PROBABILITY,
            same_data_rmse,
        ),
    ]
    frame = pd.DataFrame(rows)
    metadata = {
        "coverage_probability": COVERAGE_PROBABILITY,
        "repetitions": COVERAGE_REPETITIONS,
        "seed": COVERAGE_SEED,
        "marginal_half_width": marginal_half_width,
        "simultaneous_half_width": simultaneous_half_width,
        "analytic_same_data_naive_coverage": COVERAGE_PROBABILITY**2,
        "analytic_same_data_selected_rmse": same_data_rmse,
    }
    return frame, metadata


def _build_summary(
    robustness: pd.DataFrame,
    coverage: pd.DataFrame,
    coverage_metadata: dict,
) -> dict:
    nominal_selected = _select_candidate(robustness, "nominal_status")
    robust_selected = _select_candidate(robustness, "robust_status")
    nominal_row = robustness.set_index("candidate").loc[nominal_selected]
    robust_row = robustness.set_index("candidate").loc[robust_selected]
    coverage_indexed = coverage.set_index("procedure")
    return {
        "benchmark": "operator mismatch, nonlinear remainder, and post-selection coverage",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "robustness_parameters": {
            "state_dimension": P,
            "state_radius": R_STATE,
            "epsilon": EPSILON,
            "kappa": KAPPA,
            "operator_scale": ETA_OPERATOR,
            "nonlinear_scale": Q_NONLINEAR,
            "singular_values": SIGMA.tolist(),
            "target_coefficients": G.tolist(),
            "operator_shape_diagonal": H_DIAGONAL.tolist(),
            "nonlinear_direction": NONLINEAR_DIRECTION.tolist(),
        },
        "selected_candidates": {
            "nominal": nominal_selected,
            "robust": robust_selected,
        },
        "selected_candidate_diagnostics": {
            "nominal_selected_nominal_risk": float(nominal_row["nominal_risk"]),
            "nominal_selected_coupled_risk": float(
                nominal_row["nominal_solution_coupled_risk"]
            ),
            "nominal_selected_robust_outer_risk": float(
                nominal_row["nominal_solution_robust_outer_risk"]
            ),
            "robust_selected_nominal_risk": float(
                robust_row["robust_solution_nominal_risk"]
            ),
            "robust_selected_coupled_risk": float(
                robust_row["robust_solution_coupled_risk"]
            ),
            "robust_selected_robust_outer_risk": float(
                robust_row["robust_solution_robust_outer_risk"]
            ),
            "nominal_selection_is_false_certification": bool(
                nominal_row["nominal_risk"] <= EPSILON
                and nominal_row["nominal_solution_coupled_risk"] > EPSILON
            ),
            "robust_selection_is_valid_for_coupled_problem": bool(
                robust_row["robust_solution_robust_outer_risk"] <= EPSILON
                and robust_row["robust_solution_coupled_risk"] <= EPSILON
            ),
        },
        "computational_checks": {
            "maximum_nominal_analytic_relative_difference": float(
                robustness["nominal_analytic_relative_difference"].max()
            ),
            "maximum_nominal_solver_relative_difference": float(
                robustness["nominal_solver_relative_difference"].max()
            ),
            "maximum_robust_solver_relative_difference": float(
                robustness["robust_solver_relative_difference"].max()
            ),
            "maximum_risk_bracket_width": float(
                max(
                    robustness["nominal_risk_bracket_width"].max(),
                    robustness["robust_risk_bracket_width"].max(),
                )
            ),
            "maximum_coupled_closed_form_direct_absolute_difference": float(
                max(
                    robustness[
                        "nominal_solution_coupled_bias_absolute_difference"
                    ].max(),
                    robustness[
                        "robust_solution_coupled_bias_absolute_difference"
                    ].max(),
                )
            ),
            "robust_outer_envelope_violations": int(
                (
                    robustness["nominal_solution_coupled_risk"]
                    > robustness["nominal_solution_robust_outer_risk"] + 1.0e-10
                ).sum()
                + (
                    robustness["robust_solution_coupled_risk"]
                    > robustness["robust_solution_robust_outer_risk"] + 1.0e-10
                ).sum()
            ),
            "unresolved_candidate_count": int(
                (robustness["nominal_status"] == "unresolved").sum()
                + (robustness["robust_status"] == "unresolved").sum()
            ),
        },
        "coverage_metadata": coverage_metadata,
        "coverage_results": {
            procedure: {
                key: float(coverage_indexed.loc[procedure, key])
                for key in (
                    "coverage",
                    "analytic_coverage",
                    "conditional_bias",
                    "analytic_conditional_bias",
                    "conditional_rmse",
                    "analytic_conditional_rmse",
                    "selection_frequency_V1",
                    "selection_frequency_V2",
                )
            }
            for procedure in coverage_indexed.index
        },
        "maximum_coverage_analytic_absolute_difference": float(
            np.max(np.abs(coverage["coverage_minus_analytic"]))
        ),
        "maximum_absolute_coverage_z_score": float(
            np.max(np.abs(coverage["coverage_standardized_difference"]))
        ),
        "maximum_rmse_analytic_absolute_difference": float(
            np.max(np.abs(coverage["conditional_rmse_minus_analytic"]))
        ),
        "maximum_selection_frequency_absolute_difference": float(
            np.max(
                np.abs(
                    coverage.loc[
                        coverage["procedure"].isin(
                            (
                                "same_data_naive",
                                "independent_selection",
                                "same_data_calibrated",
                            )
                        ),
                        "selection_frequency_V1",
                    ]
                    - 0.5
                )
            )
        ),
    }


def _validate(
    robustness: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: dict,
) -> None:
    success_columns = [
        "nominal_slsqp_success",
        "nominal_cobyla_success",
        "robust_slsqp_success",
        "robust_cobyla_success",
    ]
    if not robustness[success_columns].to_numpy().all():
        raise RuntimeError("At least one candidate optimization failed.")
    checks = summary["computational_checks"]
    if checks["maximum_nominal_analytic_relative_difference"] > 1.0e-10:
        raise RuntimeError("Nominal analytic and numerical solutions disagree.")
    if checks["maximum_nominal_solver_relative_difference"] > 1.0e-8:
        raise RuntimeError("Nominal numerical solvers disagree.")
    if checks["maximum_robust_solver_relative_difference"] > 1.0e-8:
        raise RuntimeError("Robust numerical solvers disagree.")
    if checks["maximum_risk_bracket_width"] > 2.0e-5:
        raise RuntimeError("A numerical risk bracket is too wide.")
    if checks["maximum_coupled_closed_form_direct_absolute_difference"] > 1.0e-9:
        raise RuntimeError("Closed-form and direct adversarial maxima disagree.")
    if checks["robust_outer_envelope_violations"] != 0:
        raise RuntimeError("The robust outer envelope failed to dominate coupled risk.")
    if checks["unresolved_candidate_count"] != 0:
        raise RuntimeError("The robustness module contains an unresolved candidate.")
    if summary["selected_candidates"] != {"nominal": "V2", "robust": "V3"}:
        raise RuntimeError(
            f"Unexpected selections: {summary['selected_candidates']}"
        )
    diagnostics = summary["selected_candidate_diagnostics"]
    if not diagnostics["nominal_selection_is_false_certification"]:
        raise RuntimeError("The nominal selection did not produce false certification.")
    if not diagnostics["robust_selection_is_valid_for_coupled_problem"]:
        raise RuntimeError("The robust selection did not certify the coupled problem.")
    if summary["maximum_coverage_analytic_absolute_difference"] > 1.5e-3:
        raise RuntimeError("Monte Carlo coverage differs excessively from theory.")
    if summary["maximum_absolute_coverage_z_score"] > 5.0:
        raise RuntimeError("A Monte Carlo coverage deviation exceeds five sigma.")
    if summary["maximum_rmse_analytic_absolute_difference"] > 2.0e-3:
        raise RuntimeError("Monte Carlo RMSE differs excessively from theory.")
    if summary["maximum_selection_frequency_absolute_difference"] > 2.0e-3:
        raise RuntimeError("A selection frequency differs excessively from one half.")
    if not np.allclose(coverage["reporting_probability"], 1.0):
        raise RuntimeError("Coverage module unexpectedly abstained from reporting.")


def _plot(robustness: pd.DataFrame, coverage: pd.DataFrame, summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.labelsize": 10,
            "legend.fontsize": 8.2,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.25))

    ax = axes[0]
    dimensions = robustness["candidate_dimension"].to_numpy()
    ax.plot(
        dimensions,
        robustness["nominal_risk"],
        marker="o",
        color="#0072B2",
        label="Nominal risk",
    )
    ax.plot(
        dimensions,
        robustness["robust_risk"],
        marker="s",
        color="#D55E00",
        label="Robust outer risk",
    )
    ax.axhline(EPSILON, color="0.25", linestyle="--", linewidth=1.1, label=r"$\epsilon=0.480$")
    nominal_selected_dimension = int(
        robustness.loc[
            robustness["candidate"]
            == summary["selected_candidates"]["nominal"],
            "candidate_dimension",
        ].iloc[0]
    )
    robust_selected_dimension = int(
        robustness.loc[
            robustness["candidate"]
            == summary["selected_candidates"]["robust"],
            "candidate_dimension",
        ].iloc[0]
    )
    ax.scatter(
        [nominal_selected_dimension],
        [robustness.loc[robustness["candidate_dimension"] == nominal_selected_dimension, "nominal_risk"].iloc[0]],
        s=105,
        facecolors="none",
        edgecolors="#0072B2",
        linewidths=1.6,
        zorder=5,
    )
    ax.scatter(
        [robust_selected_dimension],
        [robustness.loc[robustness["candidate_dimension"] == robust_selected_dimension, "robust_risk"].iloc[0]],
        s=105,
        facecolors="none",
        edgecolors="#D55E00",
        linewidths=1.6,
        zorder=5,
    )
    ax.set_xticks(dimensions)
    ax.set_xlabel("Candidate dimension")
    ax.set_ylabel("Certified candidate risk")
    ax.grid(alpha=0.22)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1]
    selected_labels = [r"Nominal-selected $V_2$", r"Robust-selected $V_3$"]
    nominal_row = robustness.set_index("candidate").loc["V2"]
    robust_row = robustness.set_index("candidate").loc["V3"]
    values = np.array(
        [
            [
                nominal_row["nominal_solution_nominal_risk"],
                nominal_row["nominal_solution_coupled_risk"],
                nominal_row["nominal_solution_robust_outer_risk"],
            ],
            [
                robust_row["robust_solution_nominal_risk"],
                robust_row["robust_solution_coupled_risk"],
                robust_row["robust_solution_robust_outer_risk"],
            ],
        ]
    )
    base = np.arange(2, dtype=float)
    width = 0.23
    labels = ("Nominal", "Coupled truth", "Robust envelope")
    colors = ("#56B4E9", "#009E73", "#E69F00")
    for offset, label, color, column in zip((-width, 0.0, width), labels, colors, range(3)):
        ax.bar(base + offset, values[:, column], width=width, color=color, label=label)
    ax.axhline(EPSILON, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_xticks(base)
    ax.set_xticklabels(selected_labels)
    ax.set_ylabel("Risk at selected weight")
    ax.grid(alpha=0.22, axis="y")
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.40,
    )
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[2]
    procedures = (
        "fixed_V1",
        "fixed_V2",
        "same_data_naive",
        "independent_selection",
        "same_data_calibrated",
    )
    display_labels = (
        r"Fixed $V_1$",
        r"Fixed $V_2$",
        "Same data\nnaive",
        "Independent\nselection",
        "Same data\ncalibrated",
    )
    subset = coverage.set_index("procedure").loc[list(procedures)]
    colors = ["#56B4E9", "#56B4E9", "#D55E00", "#009E73", "#CC79A7"]
    bars = ax.bar(
        np.arange(len(procedures)),
        subset["coverage"],
        yerr=1.96 * subset["monte_carlo_standard_error"],
        capsize=2.5,
        color=colors,
    )
    ax.axhline(COVERAGE_PROBABILITY, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_xticks(np.arange(len(procedures)))
    ax.set_xticklabels(display_labels)
    ax.set_ylim(0.885, 0.965)
    ax.set_ylabel("Empirical coverage")
    ax.grid(alpha=0.22, axis="y")
    for bar, value in zip(bars, subset["coverage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.0020,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
    ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, va="top", fontweight="bold")

    fig.tight_layout(w_pad=1.7)
    fig.savefig(
        OUTPUT_DIR / "fig_robust_postselection_benchmark.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "fig_robust_postselection_benchmark.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    robustness = _run_robustness_module()
    coverage, coverage_metadata = _run_coverage_module()
    summary = _build_summary(robustness, coverage, coverage_metadata)
    _validate(robustness, coverage, summary)
    robustness.to_csv(
        OUTPUT_DIR / "robustness_benchmark_results.csv",
        index=False,
    )
    coverage.to_csv(
        OUTPUT_DIR / "postselection_coverage_results.csv",
        index=False,
    )
    with (OUTPUT_DIR / "robust_postselection_benchmark_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _plot(robustness, coverage, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
