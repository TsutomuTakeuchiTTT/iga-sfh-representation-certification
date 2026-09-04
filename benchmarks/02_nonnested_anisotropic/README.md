# Benchmark 2: Non-nested anisotropic candidates

This benchmark evaluates non-nested candidate representations under an anisotropic admissible state set and a direction-dependent model-discrepancy set. It also checks a shifted, non-centrally symmetric support-function calculation.

## Run

```bash
python iga_nonnested_anisotropic_benchmark.py
```

## Generated files

- `nonnested_anisotropic_benchmark_results.csv`
- `nonnested_anisotropic_benchmark_summary.json`
- `fig_nonnested_anisotropic_benchmark.pdf`
- `fig_nonnested_anisotropic_benchmark.png`

The script validates all expected selections, agreement between SLSQP and COBYLA, certified risk brackets, the strict-stability failure case, and direct adversarial support-function checks.
