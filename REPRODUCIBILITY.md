# Reproducibility record

## Verified release-candidate run

The four benchmark scripts were executed together on 2026-09-05 before assembly of this repository bundle.

### Environment

- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0
- pandas 2.2.3
- Matplotlib 3.10.8

The archived reference summaries for Benchmarks 1, 2, and 4 were generated with Python 3.12.13 and the same package versions. After excluding the software-version metadata, all fields in those three summary JSON files were reproduced exactly. Benchmark 3 was previously audited across Python 3.12.13 and 3.13.5. Its common output fields were reproduced exactly; the current script additionally stores analytic RMSE and standardized coverage diagnostics.

### Run times in the verified environment

| Benchmark | Wall-clock time | Exit status |
|---|---:|---:|
| 1. Analytic diagonal operator | 3.46 s | 0 |
| 2. Non-nested anisotropic candidates | 13.84 s | 0 |
| 3. Robustness and post-selection | 8.62 s | 0 |
| 4. Physically structured SED proxy | 14.74 s | 0 |
| **Total** | **40.65 s** | **all passed** |

Run times are descriptive and are not used as scientific acceptance criteria.

## Source-script SHA-256 hashes

```text
770e37b69b17474717ed2fb0db4a320ff33c110dba76f0fe1cc1205aaa06ea0c  benchmarks/01_diagonal_operator/iga_diagonal_benchmark.py
46d800e978b0d92fa9d1e9ab0a4ab3025947814cdfb75f995de84e53f0c7471d  benchmarks/02_nonnested_anisotropic/iga_nonnested_anisotropic_benchmark.py
12bd8f8e7461c2a76b6e231507cac2fa6cd766cf79832f97fb347e4e947020fc  benchmarks/03_robust_postselection/iga_robust_postselection_benchmark.py
abc8670106496dab882016f0bd71668fbe812fa5c91995c91ea01f4a78ff8a2f  benchmarks/04_sed_proxy/iga_sed_proxy_benchmark.py
```

The third hash agrees with the independently retained audit manifest dated 2026-08-27.

## Headline validation results

### Benchmark 1

- Maximum analytic-versus-numerical relative risk difference: `1.9984014443252818e-15`.
- Maximum Monte Carlo relative MSE difference: `6.874579115463408e-04`.
- Nested-risk monotonicity violations: zero.
- Boundary classification: `unresolved`.

### Benchmark 2

- Maximum SLSQP-versus-COBYLA relative difference: `9.7550856281714e-12`.
- Maximum certified risk-bracket width: `8.54092816471308e-08`.
- Maximum asymmetric support-function diagnostic error: `1.6653345369377348e-15`.
- Unresolved candidates: zero.

### Benchmark 3

- Nominal selection: `V2`.
- Robust selection: `V3`.
- Nominal-selected nominal risk: `0.4582109638733546`.
- Nominal-selected coupled risk: `0.4924223543611042`, above `epsilon = 0.48`.
- Robust-selected outer-envelope risk: `0.46441563211797676`.
- Naive same-data coverage: `0.902604`; analytic target: `0.9025`.
- Independent-selection and calibrated coverage: `0.950144`; analytic target: `0.95`.
- Robust outer-envelope violations: zero.

### Benchmark 4

- Scenarios: six.
- Nominal reports: six.
- Library-relative nominal false certifications: six.
- Robust reports: two.
- Robust outer-envelope violations: zero.
- Maximum nominal solver relative difference: `2.4376056728669937e-12`.
- Maximum robust solver relative difference: `4.5538968151959125e-09`.
- Maximum risk-bracket width: `2.1789436871061385e-06`.
- Maximum Monte Carlo relative RMSE difference: `1.4222990289497472e-03`.

## Interpretation of deterministic outputs

The fixed random seeds make the Monte Carlo records deterministic for a given NumPy random-number implementation and execution path. The scripts also check scientific tolerances rather than requiring byte-identical floating-point files across all possible systems. This distinction is important for optimizer traces and PDF metadata.

## Updating the record

Before creating a new release:

```bash
python tools/reproduce_all.py
python tools/create_manifest.py
```

Then inspect `reproduction_summary.json`, review `MANIFEST.sha256`, commit the synchronized outputs, and create a versioned GitHub release.
