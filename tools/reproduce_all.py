#!/usr/bin/env python3
"""Run all manuscript benchmarks and record a machine-readable summary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BENCHMARKS = (
    {
        "id": "01_diagonal_operator",
        "script": "iga_diagonal_benchmark.py",
        "summary": "diagonal_operator_benchmark_summary.json",
        "outputs": (
            "diagonal_operator_benchmark_results.csv",
            "diagonal_operator_benchmark_summary.json",
            "fig_diagonal_operator_benchmark.pdf",
            "fig_diagonal_operator_benchmark.png",
        ),
    },
    {
        "id": "02_nonnested_anisotropic",
        "script": "iga_nonnested_anisotropic_benchmark.py",
        "summary": "nonnested_anisotropic_benchmark_summary.json",
        "outputs": (
            "nonnested_anisotropic_benchmark_results.csv",
            "nonnested_anisotropic_benchmark_summary.json",
            "fig_nonnested_anisotropic_benchmark.pdf",
            "fig_nonnested_anisotropic_benchmark.png",
        ),
    },
    {
        "id": "03_robust_postselection",
        "script": "iga_robust_postselection_benchmark.py",
        "summary": "robust_postselection_benchmark_summary.json",
        "outputs": (
            "robustness_benchmark_results.csv",
            "postselection_coverage_results.csv",
            "robust_postselection_benchmark_summary.json",
            "fig_robust_postselection_benchmark.pdf",
            "fig_robust_postselection_benchmark.png",
        ),
    },
    {
        "id": "04_sed_proxy",
        "script": "iga_sed_proxy_benchmark.py",
        "summary": "sed_proxy_benchmark_summary.json",
        "outputs": (
            "sed_candidate_results.csv",
            "sed_scenario_selections.csv",
            "sed_selected_monte_carlo_results.csv",
            "sed_proxy_benchmark_summary.json",
            "fig_sed_proxy_benchmark.pdf",
            "fig_sed_proxy_benchmark.png",
        ),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated benchmark identifiers to run; the default is all four.",
    )
    parser.add_argument(
        "--record-existing",
        action="store_true",
        help="Validate and hash existing outputs without rerunning the scripts.",
    )
    parser.add_argument(
        "--output",
        default="reproduction_summary.json",
        help="Repository-relative path for the machine-readable run summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = {item.strip() for item in args.only.split(",") if item.strip()}
    selected = tuple(
        benchmark for benchmark in BENCHMARKS
        if not requested or benchmark["id"] in requested
    )
    known = {str(benchmark["id"]) for benchmark in BENCHMARKS}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown benchmark identifiers: {sorted(unknown)}")
    if not selected:
        raise ValueError("No benchmarks selected.")

    started = datetime.now(timezone.utc)
    records: list[dict[str, object]] = []
    failure = False

    for benchmark in selected:
        directory = ROOT / "benchmarks" / str(benchmark["id"])
        script = directory / str(benchmark["script"])
        if not script.is_file():
            raise FileNotFoundError(f"Missing benchmark script: {script}")

        if args.record_existing:
            print(f"Recording existing outputs for {benchmark['id']} ...", flush=True)
            elapsed = 0.0
            returncode = 0
            stdout = ""
            stderr = ""
        else:
            print(f"Running {benchmark['id']} ...", flush=True)
            start = time.perf_counter()
            completed = subprocess.run(
                [sys.executable, script.name],
                cwd=directory,
                text=True,
                capture_output=True,
                check=False,
            )
            elapsed = time.perf_counter() - start
            returncode = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
            (directory / "last_run.log").write_text(
                stdout + stderr,
                encoding="utf-8",
            )

        outputs: dict[str, dict[str, object]] = {}
        missing: list[str] = []
        for name in benchmark["outputs"]:
            path = directory / str(name)
            if path.is_file():
                outputs[str(name)] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            else:
                missing.append(str(name))

        summary_path = directory / str(benchmark["summary"])
        summary = None
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        passed = returncode == 0 and not missing and summary is not None
        failure = failure or not passed
        records.append(
            {
                "benchmark": benchmark["id"],
                "script": str(script.relative_to(ROOT)),
                "script_sha256": sha256(script),
                "elapsed_seconds": elapsed,
                "returncode": returncode,
                "passed": passed,
                "missing_outputs": missing,
                "outputs": outputs,
                "summary_file": str(summary_path.relative_to(ROOT)),
            }
        )
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {elapsed:.2f} s", flush=True)
        if returncode != 0:
            print(stdout, file=sys.stderr)
            print(stderr, file=sys.stderr)

    finished = datetime.now(timezone.utc)
    report = {
        "repository": "iga-sfh-representation-certification",
        "execution_mode": "record-existing" if args.record_existing else "execute",
        "selected_benchmarks": [str(item["id"]) for item in selected],
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "total_elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in records),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "matplotlib": package_version("matplotlib"),
        },
        "all_passed": not failure,
        "benchmarks": records,
    }
    report_path = ROOT / args.output
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {report_path.relative_to(ROOT)}")
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
