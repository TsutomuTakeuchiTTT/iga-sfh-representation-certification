#!/usr/bin/env python3
"""Non-nested anisotropic benchmark for reportable representation selection.

The benchmark checks four features of the theory:

1. candidate risk is not ordered by dimension for non-nested representations;
2. anisotropic state geometry changes the ranking of equal-dimensional candidates;
3. direction-dependent model discrepancy can reverse that ranking again; and
4. the signed support-function formula remains exact for shifted, non-centred sets.

Two independent numerical methods, gradient-based SLSQP and derivative-free
COBYLA, solve every candidate problem. A convex first-order lower bound and a
primal upper bound provide a conservative risk bracket for classification.
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
    str(Path(tempfile.gettempdir()) / "iga_nonnested_benchmark_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize


OUTPUT_DIR = Path(__file__).resolve().parent
P = 8
EPSILON = 0.600
KAPPA = 0.45
STRICT_KAPPA = 0.10
LOWER_BOUND_PADDING = 1.0e-10

SIGMA = np.array([6.0, 3.0, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05], dtype=float)
G = np.array([0.50, 0.42, 0.35, 0.28, 0.18, 0.12, 0.08, 0.05], dtype=float)

STATE_AXES = np.array([0.70, 1.80, 1.50, 0.55, 0.40, 0.30, 0.25, 0.20], dtype=float)
DISCREPANCY_AXES = np.array([0.03, 0.08, 1.10, 0.12, 0.06, 0.04, 0.03, 0.02], dtype=float)


def _givens(i: int, j: int, angle_degrees: float) -> np.ndarray:
    matrix = np.eye(P)
    angle = np.deg2rad(angle_degrees)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    matrix[i, i] = cosine
    matrix[j, j] = cosine
    matrix[i, j] = -sine
    matrix[j, i] = sine
    return matrix


STATE_ROTATION = _givens(1, 2, 18.0) @ _givens(0, 3, -12.0)
DISCREPANCY_ROTATION = _givens(2, 3, 15.0) @ _givens(1, 4, -10.0)
L_STATE = STATE_ROTATION @ np.diag(STATE_AXES) @ STATE_ROTATION.T
L_DISCREPANCY = (
    DISCREPANCY_ROTATION @ np.diag(DISCREPANCY_AXES) @ DISCREPANCY_ROTATION.T
)
ISOTROPIC_STATE_RADIUS = np.linalg.norm(L_STATE, ord="fro") / np.sqrt(P)


@dataclass(frozen=True)
class Candidate:
    name: str
    indices: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return len(self.indices)

    @property
    def basis(self) -> np.ndarray:
        return np.eye(P)[:, self.indices]


CANDIDATES = (
    Candidate("A1", (0,)),
    Candidate("A2", (1,)),
    Candidate("B12", (0, 1)),
    Candidate("B13", (0, 2)),
    Candidate("B23", (1, 2)),
    Candidate("C128", (0, 1, 7)),
    Candidate("C134", (0, 2, 3)),
    Candidate("C123", (0, 1, 2)),
    Candidate("D1234", (0, 1, 2, 3)),
    Candidate("D1256", (0, 1, 4, 5)),
    Candidate("E12345", (0, 1, 2, 3, 4)),
)


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    state_shape: np.ndarray
    discrepancy_shape: np.ndarray
    kappa: float


SCENARIOS = (
    Scenario(
        "full_anisotropic_directional",
        "Anisotropic state + directional discrepancy",
        L_STATE,
        L_DISCREPANCY,
        KAPPA,
    ),
    Scenario(
        "anisotropic_no_discrepancy",
        "Anisotropic state, no discrepancy",
        L_STATE,
        np.zeros((P, P), dtype=float),
        KAPPA,
    ),
    Scenario(
        "isotropic_no_discrepancy",
        "RMS-matched isotropic state, no discrepancy",
        ISOTROPIC_STATE_RADIUS * np.eye(P),
        np.zeros((P, P), dtype=float),
        KAPPA,
    ),
    Scenario(
        "full_strict_stability",
        "Full model with strict stability",
        L_STATE,
        L_DISCREPANCY,
        STRICT_KAPPA,
    ),
)


def _components_and_gradient(
    reduced_weight: np.ndarray,
    candidate: Candidate,
    scenario: Scenario,
) -> tuple[dict[str, float], np.ndarray]:
    basis = candidate.basis
    weight = basis @ reduced_weight
    response_residual = SIGMA * weight - G

    state_vector = scenario.state_shape.T @ response_residual
    discrepancy_vector = scenario.discrepancy_shape.T @ weight
    state_bias = np.linalg.norm(state_vector)
    discrepancy_bias = np.linalg.norm(discrepancy_vector)
    stability = np.linalg.norm(reduced_weight)
    total_bias = state_bias + discrepancy_bias
    objective = total_bias**2 + stability**2
    risk = np.sqrt(objective)

    state_jacobian = scenario.state_shape.T @ (SIGMA[:, None] * basis)
    if state_bias > 0.0:
        state_gradient = state_jacobian.T @ (state_vector / state_bias)
    else:
        state_gradient = np.zeros(candidate.dimension, dtype=float)

    discrepancy_jacobian = scenario.discrepancy_shape.T @ basis
    if discrepancy_bias > 0.0:
        discrepancy_gradient = discrepancy_jacobian.T @ (
            discrepancy_vector / discrepancy_bias
        )
    else:
        discrepancy_gradient = np.zeros(candidate.dimension, dtype=float)

    gradient = (
        2.0 * total_bias * (state_gradient + discrepancy_gradient)
        + 2.0 * reduced_weight
    )
    components = {
        "state_bias": float(state_bias),
        "discrepancy_bias": float(discrepancy_bias),
        "total_bias": float(total_bias),
        "stability_factor": float(stability),
        "objective_squared_risk": float(objective),
        "risk": float(risk),
    }
    return components, gradient


def _make_feasible(weight: np.ndarray, kappa: float) -> np.ndarray:
    result = np.asarray(weight, dtype=float).copy()
    norm = np.linalg.norm(result)
    if norm > kappa:
        result *= kappa / norm
    return result


def _solve_slsqp(candidate: Candidate, scenario: Scenario) -> dict:
    def objective(weight: np.ndarray) -> float:
        return _components_and_gradient(weight, candidate, scenario)[0][
            "objective_squared_risk"
        ]

    def gradient(weight: np.ndarray) -> np.ndarray:
        return _components_and_gradient(weight, candidate, scenario)[1]

    result = minimize(
        objective,
        np.zeros(candidate.dimension, dtype=float),
        jac=gradient,
        method="SLSQP",
        constraints={
            "type": "ineq",
            "fun": lambda weight: scenario.kappa**2 - float(weight @ weight),
            "jac": lambda weight: -2.0 * weight,
        },
        options={"ftol": 1.0e-13, "maxiter": 10000, "disp": False},
    )
    weight = _make_feasible(result.x, scenario.kappa)
    components, objective_gradient = _components_and_gradient(weight, candidate, scenario)
    return {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "weight": weight,
        "components": components,
        "gradient": objective_gradient,
        "message": str(result.message),
    }


def _solve_cobyla(candidate: Candidate, scenario: Scenario) -> dict:
    def objective(weight: np.ndarray) -> float:
        return _components_and_gradient(weight, candidate, scenario)[0][
            "objective_squared_risk"
        ]

    result = minimize(
        objective,
        np.zeros(candidate.dimension, dtype=float),
        method="COBYLA",
        constraints={
            "type": "ineq",
            "fun": lambda weight: scenario.kappa**2 - float(weight @ weight),
        },
        options={
            "catol": 1.0e-12,
            "tol": 1.0e-11,
            "maxiter": 100000,
            "rhobeg": min(0.1, 0.5 * scenario.kappa),
        },
    )
    weight = _make_feasible(result.x, scenario.kappa)
    components, objective_gradient = _components_and_gradient(weight, candidate, scenario)
    return {
        "success": bool(result.success),
        "iterations": int(result.nfev),
        "weight": weight,
        "components": components,
        "gradient": objective_gradient,
        "message": str(result.message),
    }


def _convex_risk_bracket(
    candidate: Candidate,
    scenario: Scenario,
    solutions: tuple[dict, ...],
) -> tuple[float, float]:
    """Bound the optimal risk using convex first-order minorants and primal points."""
    upper_objective = min(
        solution["components"]["objective_squared_risk"] for solution in solutions
    )
    lower_objectives = []
    for solution in solutions:
        weight = solution["weight"]
        objective = solution["components"]["objective_squared_risk"]
        gradient = solution["gradient"]
        tangent_minimum = (
            objective
            - float(gradient @ weight)
            - scenario.kappa * np.linalg.norm(gradient)
        )
        lower_objectives.append(tangent_minimum)

    scale = 1.0 + upper_objective
    lower_objective = max(0.0, max(lower_objectives) - LOWER_BOUND_PADDING * scale)
    upper_objective = upper_objective + LOWER_BOUND_PADDING * scale
    return float(np.sqrt(lower_objective)), float(np.sqrt(upper_objective))


def _status(lower_risk: float, upper_risk: float) -> str:
    if upper_risk <= EPSILON:
        return "feasible"
    if lower_risk > EPSILON:
        return "infeasible"
    return "unresolved"


def _run_candidates() -> pd.DataFrame:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        for candidate in CANDIDATES:
            slsqp = _solve_slsqp(candidate, scenario)
            cobyla = _solve_cobyla(candidate, scenario)
            lower_risk, upper_risk = _convex_risk_bracket(
                candidate,
                scenario,
                (slsqp, cobyla),
            )
            slsqp_risk = slsqp["components"]["risk"]
            cobyla_risk = cobyla["components"]["risk"]
            relative_difference = abs(cobyla_risk / slsqp_risk - 1.0)
            components = slsqp["components"]
            rows.append(
                {
                    "scenario": scenario.name,
                    "scenario_label": scenario.label,
                    "candidate": candidate.name,
                    "candidate_dimension": candidate.dimension,
                    "one_based_indices": " ".join(str(index + 1) for index in candidate.indices),
                    "kappa": scenario.kappa,
                    "epsilon": EPSILON,
                    "state_bias": components["state_bias"],
                    "discrepancy_bias": components["discrepancy_bias"],
                    "total_bias": components["total_bias"],
                    "stability_factor": components["stability_factor"],
                    "slsqp_risk": slsqp_risk,
                    "cobyla_risk": cobyla_risk,
                    "solver_relative_difference": relative_difference,
                    "risk_lower_bound": lower_risk,
                    "risk_upper_bound": upper_risk,
                    "risk_bracket_width": upper_risk - lower_risk,
                    "status": _status(lower_risk, upper_risk),
                    "slsqp_success": slsqp["success"],
                    "cobyla_success": cobyla["success"],
                    "slsqp_iterations": slsqp["iterations"],
                    "cobyla_evaluations": cobyla["iterations"],
                    "optimal_weight": " ".join(f"{value:.16g}" for value in slsqp["weight"]),
                }
            )
    return pd.DataFrame(rows)


def _select_candidate(subset: pd.DataFrame) -> str | None:
    feasible = subset.loc[subset["status"] == "feasible"].copy()
    if feasible.empty:
        return None
    minimum_dimension = int(feasible["candidate_dimension"].min())
    layer = feasible.loc[feasible["candidate_dimension"] == minimum_dimension]
    layer = layer.sort_values(["risk_upper_bound", "candidate"], kind="stable")
    return str(layer.iloc[0]["candidate"])


def _selected_weight(frame: pd.DataFrame) -> np.ndarray:
    row = frame.loc[
        (frame["scenario"] == "full_anisotropic_directional")
        & (frame["candidate"] == "B12")
    ].iloc[0]
    reduced = np.fromstring(row["optimal_weight"], sep=" ")
    return CANDIDATES[2].basis @ reduced


def _asymmetric_support_diagnostic(frame: pd.DataFrame) -> dict:
    weight = _selected_weight(frame)
    residual = SIGMA * weight - G
    state_centre_coordinate = np.array(
        [0.18, -0.12, 0.08, 0.05, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    discrepancy_centre_coordinate = np.array(
        [0.12, 0.08, -0.06, 0.04, 0.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    state_centre = L_STATE @ state_centre_coordinate
    discrepancy_centre = L_DISCREPANCY @ discrepancy_centre_coordinate

    symmetric_state_support = np.linalg.norm(L_STATE.T @ residual)
    symmetric_discrepancy_support = np.linalg.norm(L_DISCREPANCY.T @ weight)
    beta_plus = (
        float(residual @ state_centre)
        + symmetric_state_support
        + float(weight @ discrepancy_centre)
        + symmetric_discrepancy_support
    )
    beta_minus = (
        -float(residual @ state_centre)
        + symmetric_state_support
        - float(weight @ discrepancy_centre)
        + symmetric_discrepancy_support
    )

    numerical_signed_maxima: dict[str, float] = {}
    numerical_success: dict[str, bool] = {}
    for sign, label in ((1.0, "plus"), (-1.0, "minus")):
        def negative_signed_bias(adversary: np.ndarray) -> float:
            state_coordinate = adversary[:P]
            discrepancy_coordinate = adversary[P:]
            bias = float(
                residual @ (state_centre + L_STATE @ state_coordinate)
                + weight
                @ (
                    discrepancy_centre
                    + L_DISCREPANCY @ discrepancy_coordinate
                )
            )
            return -sign * bias

        result = minimize(
            negative_signed_bias,
            np.zeros(2 * P, dtype=float),
            method="SLSQP",
            constraints=(
                {
                    "type": "ineq",
                    "fun": lambda adversary: 1.0
                    - float(adversary[:P] @ adversary[:P]),
                },
                {
                    "type": "ineq",
                    "fun": lambda adversary: 1.0
                    - float(adversary[P:] @ adversary[P:]),
                },
            ),
            options={"ftol": 1.0e-14, "maxiter": 10000, "disp": False},
        )
        numerical_signed_maxima[label] = float(-result.fun)
        numerical_success[label] = bool(result.success)

    closed_form = {"plus": float(beta_plus), "minus": float(beta_minus)}
    errors = {
        label: abs(numerical_signed_maxima[label] - closed_form[label])
        for label in ("plus", "minus")
    }
    return {
        "state_centre_coordinate_norm": float(np.linalg.norm(state_centre_coordinate)),
        "discrepancy_centre_coordinate_norm": float(
            np.linalg.norm(discrepancy_centre_coordinate)
        ),
        "closed_form_beta_plus": float(beta_plus),
        "closed_form_beta_minus": float(beta_minus),
        "closed_form_worst_case_bias": float(max(beta_plus, beta_minus)),
        "active_sign": "plus" if beta_plus >= beta_minus else "minus",
        "numerical_beta_plus": numerical_signed_maxima["plus"],
        "numerical_beta_minus": numerical_signed_maxima["minus"],
        "absolute_error_beta_plus": float(errors["plus"]),
        "absolute_error_beta_minus": float(errors["minus"]),
        "maximum_absolute_error": float(max(errors.values())),
        "numerical_success": numerical_success,
    }


def _build_summary(frame: pd.DataFrame, asymmetric: dict) -> dict:
    selections = {}
    feasible_sets = {}
    for scenario in SCENARIOS:
        subset = frame.loc[frame["scenario"] == scenario.name]
        selections[scenario.name] = _select_candidate(subset)
        feasible_sets[scenario.name] = subset.loc[
            subset["status"] == "feasible", "candidate"
        ].tolist()

    full = frame.loc[frame["scenario"] == "full_anisotropic_directional"]
    full_by_candidate = full.set_index("candidate")
    ranking_examples = {
        "higher_dimension_can_be_worse": {
            "lower_dimension_candidate": "B12",
            "lower_dimension_risk": float(full_by_candidate.loc["B12", "slsqp_risk"]),
            "higher_dimension_candidate": "C134",
            "higher_dimension_risk": float(full_by_candidate.loc["C134", "slsqp_risk"]),
        },
        "equal_dimension_geometry_comparison": {
            "B12_full_risk": float(full_by_candidate.loc["B12", "slsqp_risk"]),
            "B23_full_risk": float(full_by_candidate.loc["B23", "slsqp_risk"]),
        },
    }
    return {
        "benchmark": "non-nested anisotropic candidate benchmark",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "epsilon": EPSILON,
        "nominal_kappa": KAPPA,
        "strict_kappa": STRICT_KAPPA,
        "singular_values": SIGMA.tolist(),
        "target_coefficients": G.tolist(),
        "state_axes": STATE_AXES.tolist(),
        "state_rotation_degrees": {"R23": 18.0, "R14": -12.0},
        "discrepancy_axes": DISCREPANCY_AXES.tolist(),
        "discrepancy_rotation_degrees": {"R34": 15.0, "R25": -10.0},
        "rms_matched_isotropic_state_radius": float(ISOTROPIC_STATE_RADIUS),
        "selection_rule": (
            "minimum feasible dimension; within that layer, minimum risk upper bound; "
            "remaining ties by candidate label"
        ),
        "selected_candidates": selections,
        "feasible_candidate_sets": feasible_sets,
        "all_strict_stability_candidates_infeasible": bool(
            (frame.loc[frame["scenario"] == "full_strict_stability", "status"]
            == "infeasible").all()
        ),
        "maximum_solver_relative_difference": float(
            frame["solver_relative_difference"].max()
        ),
        "maximum_risk_bracket_width": float(frame["risk_bracket_width"].max()),
        "unresolved_candidate_count": int((frame["status"] == "unresolved").sum()),
        "ranking_examples": ranking_examples,
        "asymmetric_support_diagnostic": asymmetric,
    }


def _validate(frame: pd.DataFrame, summary: dict) -> None:
    if not frame["slsqp_success"].all() or not frame["cobyla_success"].all():
        failed = frame.loc[~frame["slsqp_success"] | ~frame["cobyla_success"]]
        raise RuntimeError(f"At least one numerical solve failed:\n{failed}")
    if summary["maximum_solver_relative_difference"] > 5.0e-8:
        raise RuntimeError("Independent numerical solvers disagree beyond tolerance.")
    if summary["maximum_risk_bracket_width"] > 2.0e-5:
        raise RuntimeError("Convex first-order risk bracket is too wide.")
    if summary["unresolved_candidate_count"] != 0:
        raise RuntimeError("The benchmark contains an unresolved candidate.")
    expected_selections = {
        "full_anisotropic_directional": "B12",
        "anisotropic_no_discrepancy": "B23",
        "isotropic_no_discrepancy": "B12",
        "full_strict_stability": None,
    }
    if summary["selected_candidates"] != expected_selections:
        raise RuntimeError(
            f"Unexpected candidate selections: {summary['selected_candidates']}"
        )
    if not summary["all_strict_stability_candidates_infeasible"]:
        raise RuntimeError("Strict-stability setting did not produce certified failure.")
    if summary["asymmetric_support_diagnostic"]["maximum_absolute_error"] > 1.0e-10:
        raise RuntimeError("Signed support-function diagnostic exceeds tolerance.")


def _plot(frame: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.3,
            "axes.labelsize": 10,
            "legend.fontsize": 8.1,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.25))

    full = frame.loc[frame["scenario"] == "full_anisotropic_directional"].copy()
    x_positions = np.arange(len(full))
    dimension_colors = {
        1: "#56B4E9",
        2: "#D55E00",
        3: "#009E73",
        4: "#CC79A7",
        5: "#E69F00",
    }
    ax = axes[0]
    for dimension, subset in full.groupby("candidate_dimension", sort=True):
        indices = subset.index.to_numpy() - full.index.min()
        ax.scatter(
            indices,
            subset["slsqp_risk"],
            s=42,
            color=dimension_colors[int(dimension)],
            label=f"dimension {int(dimension)}",
            zorder=3,
        )
    selected_position = int(np.where(full["candidate"].to_numpy() == "B12")[0][0])
    selected_risk = float(full.loc[full["candidate"] == "B12", "slsqp_risk"].iloc[0])
    ax.scatter(
        selected_position,
        selected_risk,
        s=105,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    ax.axhline(EPSILON, color="0.25", linestyle="--", linewidth=1.1, label=r"$\epsilon=0.600$")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(full["candidate"], rotation=35, ha="right")
    ax.set_ylabel(r"Candidate risk $\rho_\alpha$")
    ax.set_xlabel("Non-nested candidate")
    ax.grid(alpha=0.22, axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.02, 0.97, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1]
    dimension_two = ("B12", "B13", "B23")
    ablation_scenarios = (
        "full_anisotropic_directional",
        "anisotropic_no_discrepancy",
        "isotropic_no_discrepancy",
    )
    scenario_labels = {
        "full_anisotropic_directional": "Full",
        "anisotropic_no_discrepancy": r"Anisotropic $\mathcal{K}$, $\Delta=0$",
        "isotropic_no_discrepancy": r"Isotropic $\mathcal{K}$, $\Delta=0$",
    }
    scenario_colors = {
        "full_anisotropic_directional": "#0072B2",
        "anisotropic_no_discrepancy": "#D55E00",
        "isotropic_no_discrepancy": "#009E73",
    }
    base = np.arange(len(dimension_two), dtype=float)
    offsets = (-0.20, 0.0, 0.20)
    for scenario_name, offset in zip(ablation_scenarios, offsets):
        subset = frame.loc[
            (frame["scenario"] == scenario_name)
            & frame["candidate"].isin(dimension_two)
        ].set_index("candidate").loc[list(dimension_two)]
        ax.bar(
            base + offset,
            subset["slsqp_risk"],
            width=0.19,
            color=scenario_colors[scenario_name],
            label=scenario_labels[scenario_name],
        )
    ax.axhline(EPSILON, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_xticks(base)
    ax.set_xticklabels(dimension_two)
    ax.set_xlabel("Dimension-two candidate")
    ax.set_ylabel(r"Candidate risk $\rho_\alpha$")
    ax.grid(alpha=0.22, axis="y")
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[2]
    component_candidates = ("B12", "B13", "B23")
    component_subset = full.set_index("candidate").loc[list(component_candidates)]
    base = np.arange(len(component_candidates), dtype=float)
    width = 0.21
    ax.bar(
        base - width,
        component_subset["state_bias"],
        width=width,
        color="#CC79A7",
        label="State bias",
    )
    ax.bar(
        base,
        component_subset["discrepancy_bias"],
        width=width,
        color="#E69F00",
        label="Discrepancy bias",
    )
    ax.bar(
        base + width,
        component_subset["stability_factor"],
        width=width,
        color="#56B4E9",
        label="Stability factor",
    )
    ax.scatter(
        base,
        component_subset["slsqp_risk"],
        marker="D",
        s=31,
        color="black",
        label="Total risk",
        zorder=4,
    )
    ax.set_xticks(base)
    ax.set_xticklabels(component_candidates)
    ax.set_xlabel("Dimension-two candidate")
    ax.set_ylabel("Risk component")
    ax.grid(alpha=0.22, axis="y")
    ax.legend(frameon=False)
    ax.text(0.02, 0.97, "(c)", transform=ax.transAxes, va="top", fontweight="bold")

    fig.tight_layout(w_pad=1.8)
    fig.savefig(OUTPUT_DIR / "fig_nonnested_anisotropic_benchmark.pdf", bbox_inches="tight")
    fig.savefig(
        OUTPUT_DIR / "fig_nonnested_anisotropic_benchmark.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    frame = _run_candidates()
    asymmetric = _asymmetric_support_diagnostic(frame)
    summary = _build_summary(frame, asymmetric)
    _validate(frame, summary)
    frame.to_csv(OUTPUT_DIR / "nonnested_anisotropic_benchmark_results.csv", index=False)
    with (OUTPUT_DIR / "nonnested_anisotropic_benchmark_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    _plot(frame)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
