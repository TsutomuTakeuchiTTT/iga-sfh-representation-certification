#!/usr/bin/env python3
"""Validate the release-candidate repository and headline benchmark results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCRIPT_HASHES = {
    "benchmarks/01_diagonal_operator/iga_diagonal_benchmark.py": "770e37b69b17474717ed2fb0db4a320ff33c110dba76f0fe1cc1205aaa06ea0c",
    "benchmarks/02_nonnested_anisotropic/iga_nonnested_anisotropic_benchmark.py": "46d800e978b0d92fa9d1e9ab0a4ab3025947814cdfb75f995de84e53f0c7471d",
    "benchmarks/03_robust_postselection/iga_robust_postselection_benchmark.py": "12bd8f8e7461c2a76b6e231507cac2fa6cd766cf79832f97fb347e4e947020fc",
    "benchmarks/04_sed_proxy/iga_sed_proxy_benchmark.py": "abc8670106496dab882016f0bd71668fbe812fa5c91995c91ea01f4a78ff8a2f",
}

REQUIRED_ROOT_FILES = (
    "README.md",
    "REPRODUCIBILITY.md",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "requirements.txt",
    "environment.yml",
    "Dockerfile",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(value: float, expected: float, atol: float = 1e-12) -> bool:
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=atol)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    for relative in REQUIRED_ROOT_FILES:
        require((ROOT / relative).is_file(), f"Missing required root file: {relative}")

    for relative, expected in EXPECTED_SCRIPT_HASHES.items():
        path = ROOT / relative
        require(path.is_file(), f"Missing script: {relative}")
        require(sha256(path) == expected, f"Unexpected script hash: {relative}")

    json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))

    diagonal = load("benchmarks/01_diagonal_operator/diagonal_operator_benchmark_summary.json")
    require(diagonal["selected_dimensions"] == {"1": 6, "0.2": 7, "0.08": None}, "Benchmark 1 selections changed")
    require(not any(diagonal["monotonicity_violations"].values()), "Benchmark 1 monotonicity failed")
    require(diagonal["boundary_diagnostic"]["status"] == "unresolved", "Benchmark 1 boundary status changed")

    nonnested = load("benchmarks/02_nonnested_anisotropic/nonnested_anisotropic_benchmark_summary.json")
    expected_nonnested = {
        "full_anisotropic_directional": "B12",
        "anisotropic_no_discrepancy": "B23",
        "isotropic_no_discrepancy": "B12",
        "full_strict_stability": None,
    }
    require(nonnested["selected_candidates"] == expected_nonnested, "Benchmark 2 selections changed")
    require(nonnested["unresolved_candidate_count"] == 0, "Benchmark 2 unresolved count changed")

    robust = load("benchmarks/03_robust_postselection/robust_postselection_benchmark_summary.json")
    require(robust["selected_candidates"] == {"nominal": "V2", "robust": "V3"}, "Benchmark 3 selections changed")
    diagnostics = robust["selected_candidate_diagnostics"]
    require(diagnostics["nominal_selection_is_false_certification"], "Benchmark 3 false-certification test failed")
    require(diagnostics["robust_selection_is_valid_for_coupled_problem"], "Benchmark 3 robust-certification test failed")
    require(close(robust["coverage_results"]["same_data_naive"]["analytic_coverage"], 0.9025), "Benchmark 3 analytic coverage changed")

    sed = load("benchmarks/04_sed_proxy/sed_proxy_benchmark_summary.json")
    checks = sed["headline_checks"]
    require(checks["scenario_count"] == 6, "Benchmark 4 scenario count changed")
    require(checks["nominal_report_count"] == 6, "Benchmark 4 nominal report count changed")
    require(checks["robust_report_count"] == 2, "Benchmark 4 robust report count changed")
    require(checks["nominal_false_certification_count"] == 6, "Benchmark 4 false-certification count changed")
    require(checks["robust_outer_envelope_violations"] == 0, "Benchmark 4 outer-envelope violation detected")

    print("Repository validation passed.")


if __name__ == "__main__":
    main()
