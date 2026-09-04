# Benchmark 3: Robustness and post-selection

This benchmark contains two separate modules:

1. nominal versus robust certification under structured operator mismatch and an explicit nonlinear remainder;
2. fixed-candidate, same-data selection, independent-selection, and calibrated post-selection coverage in a Gaussian two-candidate experiment.

## Run

```bash
python iga_robust_postselection_benchmark.py
```

## Generated files

- `robustness_benchmark_results.csv`
- `postselection_coverage_results.csv`
- `robust_postselection_benchmark_summary.json`
- `fig_robust_postselection_benchmark.pdf`
- `fig_robust_postselection_benchmark.png`

The script uses seed `20260826` and `1,000,000` Monte Carlo repetitions. It validates solver agreement, risk brackets, direct adversarial checks, robust-envelope inequalities, candidate selections, and analytic coverage calculations.
