# v1.0.0: submission reproducibility release

This release archives the code and synchronized outputs for the manuscript **Inference of Galaxy Star Formation Histories from Spectral Energy Distribution: Target-Specific Representation Selection and Robust Certification**.

## Included benchmarks

1. Analytic diagonal-operator benchmark.
2. Non-nested anisotropic representation benchmark.
3. Operator-mismatch, nonlinear-remainder, and post-selection benchmark.
4. Physically structured, age-resolved SED proxy benchmark.

## Reproducibility status

- All four executable scripts complete successfully with the pinned dependencies.
- Every script performs internal validation and terminates with an error if a declared scientific or numerical check fails.
- The release includes machine-readable CSV and JSON outputs, vector PDF figures, PNG previews, fixed random seeds where Monte Carlo simulation is used, and a SHA-256 manifest.
- The four-benchmark verification run recorded in `reproduction_summary.json` passed in full.

## Scope limitation

The SED benchmark is a controlled, self-contained proxy. It is not a production stellar-population-synthesis implementation and does not establish external validity for observed galaxies or unlisted model mismatches.
