# Benchmark 1: Analytic diagonal operator

This benchmark compares a closed-form trust-region solution, an independent constrained numerical optimizer, and Monte Carlo verification of the worst-case MSE identity.

## Run

```bash
python iga_diagonal_benchmark.py
```

## Generated files

- `diagonal_operator_benchmark_results.csv`
- `diagonal_operator_benchmark_summary.json`
- `fig_diagonal_operator_benchmark.pdf`
- `fig_diagonal_operator_benchmark.png`

The script uses seed `20260826` and `1,000,000` Monte Carlo realizations. It validates analytic and numerical agreement, nested-risk monotonicity, selected dimensions, Monte Carlo accuracy, and the unresolved boundary diagnostic before completing.
