#!/usr/bin/env python3
"""Physically structured SED proxy benchmark for adaptive representation.

This deterministic benchmark is the fourth validation stage of the
adaptive-representation manuscript.  It uses an internally generated compact
stellar-population proxy rather than an external SPS package, so that the
complete test remains self-contained and reproducible.  The proxy includes
age-dependent continua, a smooth 4000-Angstrom break, metallicity-dependent
blanketing, dust attenuation, and age-dependent nebular lines.

The scientific test is deliberately local: the state vector is a perturbation
of the formed stellar mass in eight age bins.  Candidate representations are
nested coarsenings of those bins.  Six one-factor template perturbations define
the finite mismatch library used by the robust outer envelope.  Its guarantee
therefore applies to the convex hull of that declared library, not to arbitrary
joint stellar-population mismatches.
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
    str(Path(tempfile.gettempdir()) / "iga_sed_proxy_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize


try:
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    # Jupyter Notebook does not define __file__.
    OUTPUT_DIR = Path.cwd()


WAVELENGTH_ANGSTROM = np.geomspace(1200.0, 24000.0, 1000)
AGE_GYR = np.array([0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 8.0, 12.0])
BASELINE_SFH = np.array([0.010, 0.015, 0.025, 0.050, 0.130, 0.220, 0.300, 0.250])
TARGET = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

BANDS = (
    ("FUV", 1530.0, 230.0),
    ("NUV", 2310.0, 400.0),
    ("u", 3550.0, 350.0),
    ("g", 4750.0, 600.0),
    ("r", 6220.0, 600.0),
    ("i", 7630.0, 700.0),
    ("z", 9050.0, 750.0),
    ("Y", 10200.0, 850.0),
    ("J", 12500.0, 1300.0),
    ("H", 16500.0, 1700.0),
    ("K", 22000.0, 1800.0),
)

STATE_RADIUS = 0.080
EPSILON = 0.047
KAPPA = 0.060
SNR_VALUES = (15, 30, 60)
MONTE_CARLO_REPETITIONS = 300_000
MONTE_CARLO_SEED = 20260827
LOWER_BOUND_PADDING = 1.0e-10


@dataclass(frozen=True)
class PopulationParameters:
    name: str
    metallicity_ratio: float
    dust_av: float
    nebular_scale: float


NOMINAL_POPULATION = PopulationParameters("nominal", 1.0, 0.40, 1.0)
MISMATCH_POPULATIONS = (
    PopulationParameters("metallicity_0.6", 0.6, 0.40, 1.0),
    PopulationParameters("metallicity_1.4", 1.4, 0.40, 1.0),
    PopulationParameters("dust_Av_0.25", 1.0, 0.25, 1.0),
    PopulationParameters("dust_Av_0.60", 1.0, 0.60, 1.0),
    PopulationParameters("nebular_0.6", 1.0, 0.40, 0.6),
    PopulationParameters("nebular_1.4", 1.0, 0.40, 1.4),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    groups: tuple[tuple[int, ...], ...]

    @property
    def dimension(self) -> int:
        return len(self.groups)

    @property
    def basis(self) -> np.ndarray:
        result = np.zeros((AGE_GYR.size, self.dimension), dtype=float)
        for column, indices in enumerate(self.groups):
            result[list(indices), column] = 1.0 / np.sqrt(len(indices))
        return result


CANDIDATES = (
    Candidate("C2", ((0, 1, 2, 3), (4, 5, 6, 7))),
    Candidate("C3", ((0, 1, 2), (3,), (4, 5, 6, 7))),
    Candidate("C4", ((0, 1), (2,), (3,), (4, 5, 6, 7))),
    Candidate("C6", ((0,), (1,), (2,), (3,), (4, 5), (6, 7))),
    Candidate("C8", tuple((index,) for index in range(8))),
)


@dataclass(frozen=True)
class Scenario:
    coverage: str
    snr: int
    band_mask: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.coverage}_SNR{self.snr}"


FULL_MASK = np.ones(len(BANDS), dtype=bool)
OPTICAL_MASK = np.array(
    [False, False, True, True, True, True, True, False, False, False, False]
)
SCENARIOS = tuple(
    Scenario(coverage, snr, mask)
    for coverage, mask in (("full", FULL_MASK), ("optical", OPTICAL_MASK))
    for snr in SNR_VALUES
)


def _planck_shape(temperature_kelvin: float) -> np.ndarray:
    exponent = 1.4387769e8 / (WAVELENGTH_ANGSTROM * temperature_kelvin)
    return WAVELENGTH_ANGSTROM**-5 / np.expm1(np.clip(exponent, 1.0e-8, 700.0))


def _population_spectrum(
    age_gyr: float,
    metallicity_ratio: float,
    nebular_scale: float,
) -> np.ndarray:
    hot_temperature = 5000.0 + 24000.0 / (1.0 + (age_gyr / 0.05) ** 0.70)
    cool_temperature = 4200.0 + 1800.0 / (1.0 + (age_gyr / 2.0) ** 0.60)
    hot_fraction = np.exp(-age_gyr / 0.25)
    spectrum = (
        hot_fraction * _planck_shape(hot_temperature)
        + (1.0 - hot_fraction) * 0.55 * _planck_shape(cool_temperature)
    )
    spectrum /= np.trapezoid(spectrum, WAVELENGTH_ANGSTROM)
    spectrum *= (age_gyr + 0.02) ** -0.72

    break_strength = (
        1.0
        + 0.70 * np.log10(1.0 + age_gyr / 0.08)
        + 0.25 * np.log10(metallicity_ratio)
    )
    blue_weight = 1.0 / (1.0 + np.exp((WAVELENGTH_ANGSTROM - 4000.0) / 120.0))
    spectrum *= 1.0 - blue_weight * (1.0 - 1.0 / break_strength)
    spectrum *= np.exp(
        -0.18
        * (metallicity_ratio - 1.0)
        * (5500.0 / WAVELENGTH_ANGSTROM) ** 1.2
    )

    young_weight = np.exp(-age_gyr / 0.04) * nebular_scale
    peak = float(np.max(spectrum))
    for center, amplitude, width in (
        (3727.0, 0.70, 30.0),
        (4861.0, 0.45, 35.0),
        (5007.0, 0.80, 35.0),
        (6563.0, 1.00, 40.0),
    ):
        spectrum += (
            young_weight
            * amplitude
            * peak
            * np.exp(-0.5 * ((WAVELENGTH_ANGSTROM - center) / width) ** 2)
        )
    return spectrum


def _dust_attenuation(dust_av: float) -> np.ndarray:
    return 10.0 ** (
        -0.4 * dust_av * (WAVELENGTH_ANGSTROM / 5500.0) ** -0.70
    )


def _raw_band_matrix(parameters: PopulationParameters) -> np.ndarray:
    spectra = np.column_stack(
        [
            _population_spectrum(
                age,
                parameters.metallicity_ratio,
                parameters.nebular_scale,
            )
            * _dust_attenuation(parameters.dust_av)
            for age in AGE_GYR
        ]
    )
    rows = []
    for _, center, width in BANDS:
        transmission = np.exp(
            -0.5 * ((WAVELENGTH_ANGSTROM - center) / width) ** 2
        )
        rows.append(
            np.trapezoid(
                spectra * transmission[:, None],
                WAVELENGTH_ANGSTROM,
                axis=0,
            )
            / np.trapezoid(transmission, WAVELENGTH_ANGSTROM)
        )
    return np.asarray(rows)


NOMINAL_RAW_MATRIX = _raw_band_matrix(NOMINAL_POPULATION)
TEMPLATE_NORMALIZATION = float(np.max(NOMINAL_RAW_MATRIX[:, 4]))
NOMINAL_BAND_MATRIX = NOMINAL_RAW_MATRIX / TEMPLATE_NORMALIZATION
MISMATCH_BAND_MATRICES = tuple(
    _raw_band_matrix(parameters) / TEMPLATE_NORMALIZATION
    for parameters in MISMATCH_POPULATIONS
)
BASELINE_FLUX = NOMINAL_BAND_MATRIX @ BASELINE_SFH


def _scenario_matrices(
    scenario: Scenario,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    noise = np.maximum(BASELINE_FLUX / scenario.snr, 1.0e-14)
    nominal = (NOMINAL_BAND_MATRIX / noise[:, None])[scenario.band_mask]
    alternatives = tuple(
        (matrix / noise[:, None])[scenario.band_mask]
        for matrix in MISMATCH_BAND_MATRICES
    )
    errors = tuple(matrix - nominal for matrix in alternatives)
    return nominal, errors, noise[scenario.band_mask]


def _response_basis(matrix: np.ndarray, candidate: Candidate) -> tuple[np.ndarray, int]:
    response = matrix @ candidate.basis
    left, singular_values, _ = np.linalg.svd(response, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=float), 0
    rank = int(np.sum(singular_values > singular_values[0] * 1.0e-10))
    return left[:, :rank], rank


def _objective_and_subgradient(
    reduced_weight: np.ndarray,
    response_basis: np.ndarray,
    nominal: np.ndarray,
    errors: tuple[np.ndarray, ...],
    robust: bool,
) -> tuple[float, np.ndarray, dict[str, float | int | np.ndarray]]:
    weight = response_basis @ reduced_weight
    residual = nominal.T @ weight - TARGET
    residual_norm = float(np.linalg.norm(residual))
    state_bias = STATE_RADIUS * residual_norm
    if residual_norm > 0.0:
        bias_gradient = (
            STATE_RADIUS
            * response_basis.T
            @ (nominal @ residual)
            / residual_norm
        )
    else:
        bias_gradient = np.zeros_like(reduced_weight)

    mismatch_bias = 0.0
    active_mismatch = -1
    if robust:
        mismatch_norms = np.array(
            [np.linalg.norm(error.T @ weight) for error in errors]
        )
        active_mismatch = int(np.argmax(mismatch_norms))
        active_norm = float(mismatch_norms[active_mismatch])
        mismatch_bias = STATE_RADIUS * active_norm
        if active_norm > 0.0:
            active_error = errors[active_mismatch]
            bias_gradient += (
                STATE_RADIUS
                * response_basis.T
                @ (active_error @ (active_error.T @ weight))
                / active_norm
            )

    total_bias = state_bias + mismatch_bias
    stability = float(np.linalg.norm(reduced_weight))
    objective = total_bias**2 + stability**2
    gradient = 2.0 * total_bias * bias_gradient + 2.0 * reduced_weight
    return (
        float(objective),
        gradient,
        {
            "weight": weight,
            "state_bias": float(state_bias),
            "mismatch_bias": float(mismatch_bias),
            "total_bias": float(total_bias),
            "stability": stability,
            "risk": float(np.sqrt(objective)),
            "active_mismatch": active_mismatch,
        },
    )


def _make_feasible(weight: np.ndarray) -> np.ndarray:
    result = np.asarray(weight, dtype=float).copy()
    norm_value = np.linalg.norm(result)
    if norm_value > KAPPA:
        result *= KAPPA / norm_value
    return result


def _solve(
    response_basis: np.ndarray,
    nominal: np.ndarray,
    errors: tuple[np.ndarray, ...],
    robust: bool,
    method: str,
) -> dict:
    dimension = response_basis.shape[1]
    objective = lambda value: _objective_and_subgradient(
        value, response_basis, nominal, errors, robust
    )[0]
    if robust:
        def epigraph_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
            reduced_weight = value[:dimension]
            mismatch_epigraph = float(value[-1])
            weight = response_basis @ reduced_weight
            residual = nominal.T @ weight - TARGET
            residual_norm = float(np.linalg.norm(residual))
            state_bias = STATE_RADIUS * residual_norm
            total_bias = state_bias + mismatch_epigraph
            objective_value = total_bias**2 + float(reduced_weight @ reduced_weight)
            if residual_norm > 0.0:
                state_gradient = (
                    STATE_RADIUS
                    * response_basis.T
                    @ (nominal @ residual)
                    / residual_norm
                )
            else:
                state_gradient = np.zeros(dimension, dtype=float)
            gradient = np.concatenate(
                (
                    2.0 * total_bias * state_gradient + 2.0 * reduced_weight,
                    np.array([2.0 * total_bias]),
                )
            )
            return float(objective_value), gradient

        def epigraph_constraints(value: np.ndarray) -> np.ndarray:
            reduced_weight = value[:dimension]
            mismatch_epigraph = float(value[-1])
            weight = response_basis @ reduced_weight
            return np.array(
                [
                    KAPPA**2 - float(reduced_weight @ reduced_weight),
                    mismatch_epigraph,
                    *(
                        mismatch_epigraph
                        - STATE_RADIUS * np.linalg.norm(error.T @ weight)
                        for error in errors
                    ),
                ]
            )

        def epigraph_constraint_jacobian(value: np.ndarray) -> np.ndarray:
            reduced_weight = value[:dimension]
            weight = response_basis @ reduced_weight
            rows = [
                np.concatenate((-2.0 * reduced_weight, np.array([0.0]))),
                np.concatenate((np.zeros(dimension), np.array([1.0]))),
            ]
            for error in errors:
                projection = error.T @ weight
                projection_norm = float(np.linalg.norm(projection))
                if projection_norm > 0.0:
                    gradient = (
                        STATE_RADIUS
                        * response_basis.T
                        @ (error @ projection)
                        / projection_norm
                    )
                else:
                    gradient = np.zeros(dimension, dtype=float)
                rows.append(np.concatenate((-gradient, np.array([1.0]))))
            return np.asarray(rows)

        use_analytic_derivatives = method == "SLSQP"
        if method not in ("SLSQP", "SLSQP_FD"):
            raise ValueError(f"Unknown robust method: {method}")
        initial_value = np.zeros(dimension + 1, dtype=float)
        if not use_analytic_derivatives:
            direction = np.arange(1.0, dimension + 1.0)
            initial_value[:dimension] = 1.0e-3 * direction / np.linalg.norm(direction)
            initial_weight = response_basis @ initial_value[:dimension]
            initial_value[-1] = 1.01 * max(
                STATE_RADIUS * np.linalg.norm(error.T @ initial_weight)
                for error in errors
            )
        result = minimize(
            lambda value: epigraph_objective(value)[0],
            initial_value,
            jac=(
                (lambda value: epigraph_objective(value)[1])
                if use_analytic_derivatives
                else None
            ),
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": epigraph_constraints,
                "jac": (
                    epigraph_constraint_jacobian
                    if use_analytic_derivatives
                    else None
                ),
            },
            options={
                "ftol": 1.0e-14 if use_analytic_derivatives else 1.0e-12,
                "maxiter": 10000,
                "disp": False,
            },
        )
        if not result.success and not use_analytic_derivatives:
            fallback = minimize(
                lambda value: epigraph_objective(value)[0],
                np.zeros(dimension + 1, dtype=float),
                method="SLSQP",
                constraints={
                    "type": "ineq",
                    "fun": epigraph_constraints,
                },
                options={"ftol": 1.0e-12, "maxiter": 10000, "disp": False},
            )
            if fallback.success or fallback.fun < result.fun:
                result = fallback
        iterations = int(result.nit)
        reduced_weight = _make_feasible(result.x[:dimension])
        value, _, components = _objective_and_subgradient(
            reduced_weight, response_basis, nominal, errors, robust=True
        )
        gradient = _minimum_norm_subgradient(
            reduced_weight, response_basis, nominal, errors
        )
        return {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": iterations,
            "reduced_weight": reduced_weight,
            "objective": value,
            "gradient": gradient,
            "components": components,
        }

    if method == "SLSQP":
        result = minimize(
            objective,
            np.zeros(dimension, dtype=float),
            jac=lambda value: _objective_and_subgradient(
                value, response_basis, nominal, errors, robust
            )[1],
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda value: KAPPA**2 - float(value @ value),
                "jac": lambda value: -2.0 * value,
            },
            options={"ftol": 1.0e-14, "maxiter": 10000, "disp": False},
        )
        iterations = int(result.nit)
    elif method == "POWELL":
        def radial_map(value: np.ndarray) -> np.ndarray:
            return KAPPA * value / np.sqrt(1.0 + float(value @ value))

        result = minimize(
            lambda value: objective(radial_map(value)),
            np.zeros(dimension, dtype=float),
            method="Powell",
            options={
                "xtol": 1.0e-10,
                "ftol": 1.0e-14,
                "maxiter": 10000,
            },
        )
        iterations = int(result.nfev)
        result.x = radial_map(result.x)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced_weight = _make_feasible(result.x)
    value, gradient, components = _objective_and_subgradient(
        reduced_weight, response_basis, nominal, errors, robust
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": iterations,
        "reduced_weight": reduced_weight,
        "objective": value,
        "gradient": gradient,
        "components": components,
    }


def _minimum_norm_subgradient(
    reduced_weight: np.ndarray,
    response_basis: np.ndarray,
    nominal: np.ndarray,
    errors: tuple[np.ndarray, ...],
) -> np.ndarray:
    weight = response_basis @ reduced_weight
    residual = nominal.T @ weight - TARGET
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > 0.0:
        state_gradient = (
            STATE_RADIUS
            * response_basis.T
            @ (nominal @ residual)
            / residual_norm
        )
    else:
        state_gradient = np.zeros_like(reduced_weight)

    mismatch_norms = np.array(
        [np.linalg.norm(error.T @ weight) for error in errors]
    )
    maximum_norm = float(np.max(mismatch_norms))
    active = np.flatnonzero(
        maximum_norm - mismatch_norms <= 1.0e-8 * max(1.0, maximum_norm)
    )
    mismatch_gradients = []
    for index in active:
        projection = errors[index].T @ weight
        norm_value = float(np.linalg.norm(projection))
        if norm_value > 0.0:
            mismatch_gradients.append(
                STATE_RADIUS
                * response_basis.T
                @ (errors[index] @ projection)
                / norm_value
            )
        else:
            mismatch_gradients.append(np.zeros_like(reduced_weight))
    gradient_matrix = np.column_stack(mismatch_gradients)
    total_bias = STATE_RADIUS * residual_norm + STATE_RADIUS * maximum_norm
    base = 2.0 * total_bias * state_gradient + 2.0 * reduced_weight
    scaled_gradients = 2.0 * total_bias * gradient_matrix
    if active.size == 1:
        return base + scaled_gradients[:, 0]

    simplex_start = np.full(active.size, 1.0 / active.size)
    combination = minimize(
        lambda coefficients: float(
            np.linalg.norm(base + scaled_gradients @ coefficients) ** 2
        ),
        simplex_start,
        jac=lambda coefficients: (
            2.0
            * scaled_gradients.T
            @ (base + scaled_gradients @ coefficients)
        ),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * active.size,
        constraints={
            "type": "eq",
            "fun": lambda coefficients: float(np.sum(coefficients) - 1.0),
            "jac": lambda coefficients: np.ones_like(coefficients),
        },
        options={"ftol": 1.0e-15, "maxiter": 1000, "disp": False},
    )
    if not combination.success:
        raise RuntimeError("Failed to construct a robust objective subgradient.")
    return base + scaled_gradients @ combination.x


def _risk_bracket(solutions: tuple[dict, ...]) -> tuple[float, float]:
    upper_objective = min(solution["objective"] for solution in solutions)
    tangent_bounds = []
    for solution in solutions:
        point = solution["reduced_weight"]
        gradient = solution["gradient"]
        tangent_bounds.append(
            solution["objective"]
            - float(gradient @ point)
            - KAPPA * np.linalg.norm(gradient)
        )
    scale = 1.0 + upper_objective
    lower_objective = max(
        0.0,
        max(tangent_bounds) - LOWER_BOUND_PADDING * scale,
    )
    upper_objective += LOWER_BOUND_PADDING * scale
    return float(np.sqrt(lower_objective)), float(np.sqrt(upper_objective))


def _status(lower_risk: float, upper_risk: float) -> str:
    if upper_risk <= EPSILON:
        return "feasible"
    if lower_risk > EPSILON:
        return "infeasible"
    return "unresolved"


def _actual_variant_risks(
    weight: np.ndarray,
    nominal: np.ndarray,
    errors: tuple[np.ndarray, ...],
) -> np.ndarray:
    stability = np.linalg.norm(weight)
    return np.array(
        [
            np.hypot(
                STATE_RADIUS
                * np.linalg.norm((nominal + error).T @ weight - TARGET),
                stability,
            )
            for error in errors
        ]
    )


def _run_candidate_grid() -> pd.DataFrame:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        nominal, errors, _ = _scenario_matrices(scenario)
        for candidate in CANDIDATES:
            response_basis, rank = _response_basis(nominal, candidate)
            nominal_slsqp = _solve(
                response_basis, nominal, errors, robust=False, method="SLSQP"
            )
            nominal_powell = _solve(
                response_basis, nominal, errors, robust=False, method="POWELL"
            )
            robust_slsqp = _solve(
                response_basis, nominal, errors, robust=True, method="SLSQP"
            )
            robust_finite_difference = _solve(
                response_basis, nominal, errors, robust=True, method="SLSQP_FD"
            )
            nominal_lower, nominal_upper = _risk_bracket(
                (nominal_slsqp, nominal_powell)
            )
            robust_lower, robust_upper = _risk_bracket(
                (robust_slsqp, robust_finite_difference)
            )

            nominal_weight = nominal_slsqp["components"]["weight"]
            robust_weight = robust_slsqp["components"]["weight"]
            nominal_actual = _actual_variant_risks(
                nominal_weight, nominal, errors
            )
            robust_actual = _actual_variant_risks(
                robust_weight, nominal, errors
            )
            nominal_worst_index = int(np.argmax(nominal_actual))
            robust_worst_index = int(np.argmax(robust_actual))

            rows.append(
                {
                    "scenario": scenario.name,
                    "coverage": scenario.coverage,
                    "snr": scenario.snr,
                    "band_count": int(np.sum(scenario.band_mask)),
                    "candidate": candidate.name,
                    "candidate_dimension": candidate.dimension,
                    "response_rank": rank,
                    "epsilon": EPSILON,
                    "kappa": KAPPA,
                    "nominal_risk": nominal_slsqp["components"]["risk"],
                    "nominal_lower_risk": nominal_lower,
                    "nominal_upper_risk": nominal_upper,
                    "nominal_bracket_width": nominal_upper - nominal_lower,
                    "nominal_status": _status(nominal_lower, nominal_upper),
                    "robust_risk": robust_slsqp["components"]["risk"],
                    "robust_lower_risk": robust_lower,
                    "robust_upper_risk": robust_upper,
                    "robust_bracket_width": robust_upper - robust_lower,
                    "robust_status": _status(robust_lower, robust_upper),
                    "nominal_solver_relative_difference": abs(
                        nominal_powell["components"]["risk"]
                        / nominal_slsqp["components"]["risk"]
                        - 1.0
                    ),
                    "robust_solver_relative_difference": abs(
                        robust_finite_difference["components"]["risk"]
                        / robust_slsqp["components"]["risk"]
                        - 1.0
                    ),
                    "nominal_weight": " ".join(
                        f"{value:.16g}" for value in nominal_weight
                    ),
                    "robust_weight": " ".join(
                        f"{value:.16g}" for value in robust_weight
                    ),
                    "nominal_stability": nominal_slsqp["components"]["stability"],
                    "robust_stability": robust_slsqp["components"]["stability"],
                    "nominal_selected_max_actual_risk": float(
                        nominal_actual[nominal_worst_index]
                    ),
                    "nominal_selected_worst_variant": MISMATCH_POPULATIONS[
                        nominal_worst_index
                    ].name,
                    "nominal_selected_is_false_certification": bool(
                        _status(nominal_lower, nominal_upper) == "feasible"
                        and nominal_actual[nominal_worst_index] > EPSILON
                    ),
                    "robust_selected_max_actual_risk": float(
                        robust_actual[robust_worst_index]
                    ),
                    "robust_selected_worst_variant": MISMATCH_POPULATIONS[
                        robust_worst_index
                    ].name,
                    "robust_outer_envelope_violation": bool(
                        robust_actual[robust_worst_index]
                        > robust_slsqp["components"]["risk"] + 1.0e-10
                    ),
                    "nominal_slsqp_success": nominal_slsqp["success"],
                    "nominal_powell_success": nominal_powell["success"],
                    "robust_slsqp_success": robust_slsqp["success"],
                    "robust_finite_difference_success": robust_finite_difference[
                        "success"
                    ],
                }
            )
    return pd.DataFrame(rows)


def _selected_row(group: pd.DataFrame, status_column: str) -> pd.Series | None:
    feasible = group.loc[group[status_column] == "feasible"].sort_values(
        "candidate_dimension", kind="stable"
    )
    if feasible.empty:
        return None
    return feasible.iloc[0]


def _scenario_selections(candidate_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        group = candidate_results.loc[
            candidate_results["scenario"] == scenario.name
        ]
        nominal = _selected_row(group, "nominal_status")
        robust = _selected_row(group, "robust_status")
        rows.append(
            {
                "scenario": scenario.name,
                "coverage": scenario.coverage,
                "snr": scenario.snr,
                "nominal_selected_candidate": None
                if nominal is None
                else nominal["candidate"],
                "nominal_selected_dimension": 0
                if nominal is None
                else int(nominal["candidate_dimension"]),
                "nominal_certified_risk": np.nan
                if nominal is None
                else float(nominal["nominal_risk"]),
                "nominal_max_actual_mismatch_risk": np.nan
                if nominal is None
                else float(nominal["nominal_selected_max_actual_risk"]),
                "nominal_false_certification": False
                if nominal is None
                else bool(nominal["nominal_selected_is_false_certification"]),
                "nominal_worst_variant": None
                if nominal is None
                else nominal["nominal_selected_worst_variant"],
                "robust_selected_candidate": None
                if robust is None
                else robust["candidate"],
                "robust_selected_dimension": 0
                if robust is None
                else int(robust["candidate_dimension"]),
                "robust_certified_outer_risk": np.nan
                if robust is None
                else float(robust["robust_risk"]),
                "robust_max_actual_mismatch_risk": np.nan
                if robust is None
                else float(robust["robust_selected_max_actual_risk"]),
                "robust_worst_variant": None
                if robust is None
                else robust["robust_selected_worst_variant"],
            }
        )
    return pd.DataFrame(rows)


def _parse_weight(text: str) -> np.ndarray:
    return np.fromstring(text, sep=" ")


def _run_selected_monte_carlo(
    candidate_results: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(MONTE_CARLO_SEED)
    rows = []
    indexed = candidate_results.set_index(["scenario", "candidate"])
    for selection in selections.itertuples(index=False):
        scenario = next(item for item in SCENARIOS if item.name == selection.scenario)
        nominal_matrix, errors, _ = _scenario_matrices(scenario)
        for certification in ("nominal", "robust"):
            candidate_name = getattr(
                selection, f"{certification}_selected_candidate"
            )
            if not isinstance(candidate_name, str):
                continue
            result = indexed.loc[(scenario.name, candidate_name)]
            weight = _parse_weight(result[f"{certification}_weight"])
            actual_risks = _actual_variant_risks(
                weight, nominal_matrix, errors
            )
            worst_index = int(np.argmax(actual_risks))
            residual = (nominal_matrix + errors[worst_index]).T @ weight - TARGET
            residual_norm = np.linalg.norm(residual)
            worst_state = (
                np.zeros_like(residual)
                if residual_norm == 0.0
                else STATE_RADIUS * residual / residual_norm
            )
            worst_bias = float(residual @ worst_state)
            noise_projection = np.linalg.norm(weight) * rng.standard_normal(
                MONTE_CARLO_REPETITIONS
            )
            errors_mc = worst_bias + noise_projection
            empirical_rmse = float(np.sqrt(np.mean(errors_mc**2)))
            exact_rmse = float(actual_risks[worst_index])
            rows.append(
                {
                    "scenario": scenario.name,
                    "coverage": scenario.coverage,
                    "snr": scenario.snr,
                    "certification": certification,
                    "selected_candidate": candidate_name,
                    "worst_variant": MISMATCH_POPULATIONS[worst_index].name,
                    "repetitions": MONTE_CARLO_REPETITIONS,
                    "exact_worst_case_rmse": exact_rmse,
                    "empirical_rmse": empirical_rmse,
                    "relative_difference": abs(empirical_rmse / exact_rmse - 1.0),
                }
            )
    return pd.DataFrame(rows)


def _build_summary(
    candidate_results: pd.DataFrame,
    selections: pd.DataFrame,
    monte_carlo: pd.DataFrame,
) -> dict:
    selection_records = []
    for record in selections.to_dict(orient="records"):
        selection_records.append(
            {
                key: (
                    None
                    if pd.isna(value)
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in record.items()
            }
        )
    return {
        "benchmark": "physically structured SED proxy with finite mismatch library",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "state": {
            "age_gyr": AGE_GYR.tolist(),
            "baseline_sfh": BASELINE_SFH.tolist(),
            "target_coefficients": TARGET.tolist(),
            "state_radius": STATE_RADIUS,
        },
        "selection_parameters": {
            "epsilon": EPSILON,
            "kappa": KAPPA,
            "snr_values": list(SNR_VALUES),
            "full_bands": [band[0] for band in BANDS],
            "optical_bands": [
                band[0] for band, keep in zip(BANDS, OPTICAL_MASK) if keep
            ],
        },
        "mismatch_library": [
            {
                "name": item.name,
                "metallicity_ratio": item.metallicity_ratio,
                "dust_av": item.dust_av,
                "nebular_scale": item.nebular_scale,
            }
            for item in MISMATCH_POPULATIONS
        ],
        "scenario_selections": selection_records,
        "headline_checks": {
            "scenario_count": int(len(selections)),
            "nominal_report_count": int(
                (selections["nominal_selected_dimension"] > 0).sum()
            ),
            "robust_report_count": int(
                (selections["robust_selected_dimension"] > 0).sum()
            ),
            "nominal_false_certification_count": int(
                selections["nominal_false_certification"].sum()
            ),
            "robust_outer_envelope_violations": int(
                candidate_results["robust_outer_envelope_violation"].sum()
            ),
            "unresolved_candidate_count": int(
                (candidate_results["nominal_status"] == "unresolved").sum()
                + (candidate_results["robust_status"] == "unresolved").sum()
            ),
            "maximum_nominal_solver_relative_difference": float(
                candidate_results["nominal_solver_relative_difference"].max()
            ),
            "maximum_robust_solver_relative_difference": float(
                candidate_results["robust_solver_relative_difference"].max()
            ),
            "maximum_risk_bracket_width": float(
                max(
                    candidate_results["nominal_bracket_width"].max(),
                    candidate_results["robust_bracket_width"].max(),
                )
            ),
            "maximum_monte_carlo_rmse_relative_difference": float(
                monte_carlo["relative_difference"].max()
            ),
        },
        "monte_carlo": {
            "repetitions_per_reported_procedure": MONTE_CARLO_REPETITIONS,
            "seed": MONTE_CARLO_SEED,
        },
        "scope_note": (
            "The robust guarantee covers the convex hull of the six declared "
            "one-factor mismatch operators. Joint extremes outside that set "
            "are not certified. The internal template generator is an SPS "
            "proxy and not a replacement for validation with an external "
            "production stellar-population library."
        ),
    }


def _validate(
    candidate_results: pd.DataFrame,
    selections: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    summary: dict,
) -> None:
    success_columns = (
        "nominal_slsqp_success",
        "nominal_powell_success",
        "robust_slsqp_success",
        "robust_finite_difference_success",
    )
    if not candidate_results[list(success_columns)].to_numpy().all():
        raise RuntimeError("At least one candidate solver failed.")
    checks = summary["headline_checks"]
    if checks["maximum_nominal_solver_relative_difference"] > 2.0e-3:
        raise RuntimeError("Nominal solvers disagree.")
    if checks["maximum_robust_solver_relative_difference"] > 2.0e-6:
        raise RuntimeError("Robust solvers disagree.")
    if checks["maximum_risk_bracket_width"] > 2.0e-4:
        raise RuntimeError("A risk bracket is too wide.")
    if checks["robust_outer_envelope_violations"] != 0:
        raise RuntimeError("The robust outer envelope failed to dominate actual risk.")
    if checks["unresolved_candidate_count"] != 0:
        raise RuntimeError("At least one candidate is numerically unresolved.")
    if checks["maximum_monte_carlo_rmse_relative_difference"] > 5.0e-3:
        raise RuntimeError("Monte Carlo and exact RMSE disagree.")

    expected = {
        "full_SNR15": ("C3", None),
        "full_SNR30": ("C3", "C6"),
        "full_SNR60": ("C3", "C6"),
        "optical_SNR15": ("C4", None),
        "optical_SNR30": ("C3", None),
        "optical_SNR60": ("C3", None),
    }
    for row in selections.itertuples(index=False):
        actual = (row.nominal_selected_candidate, row.robust_selected_candidate)
        if actual != expected[row.scenario]:
            raise RuntimeError(
                f"Unexpected selection for {row.scenario}: {actual}"
            )
    if checks["nominal_false_certification_count"] != len(SCENARIOS):
        raise RuntimeError("Not every nominal stress-test selection is false-certified.")


def _plot(
    candidate_results: pd.DataFrame,
    selections: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.labelsize": 9.7,
            "legend.fontsize": 7.7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.3,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))

    ax = axes[0, 0]
    template_spectra = []
    for age in (0.01, 0.10, 1.0, 12.0):
        spectrum = _population_spectrum(age, 1.0, 1.0) * _dust_attenuation(0.40)
        spectrum /= np.interp(5500.0, WAVELENGTH_ANGSTROM, spectrum)
        template_spectra.append(spectrum)
        ax.plot(
            WAVELENGTH_ANGSTROM / 1.0e4,
            spectrum,
            label=f"{age:g} Gyr",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.12, 2.4)
    ax.set_ylim(0.02, 80.0)
    ax.set_xlabel(r"Rest wavelength ($\mu$m)")
    ax.set_ylabel(r"Relative $f_\lambda$")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[0, 1]
    subset = candidate_results.loc[
        candidate_results["scenario"] == "full_SNR30"
    ].sort_values("candidate_dimension")
    ax.plot(
        subset["candidate_dimension"],
        subset["nominal_risk"],
        marker="o",
        color="#0072B2",
        label="Nominal risk",
    )
    ax.plot(
        subset["candidate_dimension"],
        subset["robust_risk"],
        marker="s",
        color="#D55E00",
        label="Robust outer risk",
    )
    ax.axhline(EPSILON, color="0.25", linestyle="--", linewidth=1.1, label=r"$\epsilon=0.047$")
    ax.scatter([3], [subset.loc[subset["candidate"] == "C3", "nominal_risk"].iloc[0]], s=105, facecolors="none", edgecolors="#0072B2", linewidths=1.5, zorder=5)
    ax.scatter([6], [subset.loc[subset["candidate"] == "C6", "robust_risk"].iloc[0]], s=105, facecolors="none", edgecolors="#D55E00", linewidths=1.5, zorder=5)
    ax.set_xticks(subset["candidate_dimension"])
    ax.set_xlabel("Candidate dimension")
    ax.set_ylabel("Certified risk")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=1.0)
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1, 0]
    markers = {
        ("full", "nominal"): ("o", "#0072B2", "Full, nominal"),
        ("full", "robust"): ("s", "#D55E00", "Full, robust"),
        ("optical", "nominal"): ("^", "#56B4E9", "Optical, nominal"),
        ("optical", "robust"): ("X", "#CC79A7", "Optical, robust"),
    }
    for (coverage, certification), (marker, color, label) in markers.items():
        group = selections.loc[selections["coverage"] == coverage].sort_values("snr")
        dimensions = group[f"{certification}_selected_dimension"].to_numpy()
        ax.plot(group["snr"], dimensions, marker=marker, color=color, label=label)
    ax.set_xticks(SNR_VALUES)
    ax.set_yticks([0, 2, 3, 4, 6, 8])
    ax.set_yticklabels(["failure", "2", "3", "4", "6", "8"])
    ax.set_xlabel("Per-band S/N")
    ax.set_ylabel("Selected dimension")
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(0.02, 0.96, "(c)", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1, 1]
    labels = [
        f"{row.coverage}\nS/N {row.snr}"
        for row in selections.itertuples(index=False)
    ]
    x = np.arange(len(labels), dtype=float)
    nominal_values = selections["nominal_max_actual_mismatch_risk"].to_numpy()
    robust_values = selections["robust_max_actual_mismatch_risk"].to_numpy()
    width = 0.34
    ax.bar(x - width / 2.0, nominal_values, width=width, color="#D55E00", label="Nominal-selected actual")
    ax.bar(x + width / 2.0, robust_values, width=width, color="#009E73", label="Robust-selected actual")
    ax.axhline(EPSILON, color="0.25", linestyle="--", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Worst library-variant risk")
    ax.grid(alpha=0.20, axis="y")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    ax.text(0.02, 0.96, "(d)", transform=ax.transAxes, va="top", fontweight="bold")

    fig.tight_layout(w_pad=1.5, h_pad=1.8)
    fig.savefig(OUTPUT_DIR / "fig_sed_proxy_benchmark.pdf", bbox_inches="tight")
    fig.savefig(
        OUTPUT_DIR / "fig_sed_proxy_benchmark.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    candidate_results = _run_candidate_grid()
    selections = _scenario_selections(candidate_results)
    monte_carlo = _run_selected_monte_carlo(candidate_results, selections)
    summary = _build_summary(candidate_results, selections, monte_carlo)
    _validate(candidate_results, selections, monte_carlo, summary)

    candidate_results.to_csv(
        OUTPUT_DIR / "sed_candidate_results.csv", index=False
    )
    selections.to_csv(OUTPUT_DIR / "sed_scenario_selections.csv", index=False)
    monte_carlo.to_csv(
        OUTPUT_DIR / "sed_selected_monte_carlo_results.csv", index=False
    )
    with (OUTPUT_DIR / "sed_proxy_benchmark_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    _plot(candidate_results, selections)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
