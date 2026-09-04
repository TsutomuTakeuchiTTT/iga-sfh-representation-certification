# Benchmark 4: Physically structured SED proxy

This self-contained benchmark generates an age-dependent stellar-population proxy with continuum evolution, a smooth 4000-Angstrom break, metallicity-dependent blanketing, dust attenuation, and nebular emission lines.

The state contains formed-mass perturbations in eight age bins. Five nested candidate representations are tested for two wavelength coverages and three signal-to-noise ratios. A six-operator, one-factor-at-a-time mismatch library is used for nominal and robust certification.

## Run

```bash
python iga_sed_proxy_benchmark.py
```

## Generated files

- `sed_candidate_results.csv`
- `sed_scenario_selections.csv`
- `sed_selected_monte_carlo_results.csv`
- `sed_proxy_benchmark_summary.json`
- `fig_sed_proxy_benchmark.pdf`
- `fig_sed_proxy_benchmark.png`

The script uses seed `20260827` and `300,000` Monte Carlo repetitions per reported procedure. The robust guarantee covers the convex hull of the six declared mismatch operators; it does not cover unlisted joint extremes and does not replace validation with a production SPS library.
