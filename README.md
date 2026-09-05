# IGA SFH Representation Certification

[![Reproduce benchmarks](https://github.com/TsutomuTakeuchiTTT/iga-sfh-representation-certification/actions/workflows/reproduce.yml/badge.svg)](https://github.com/TsutomuTakeuchiTTT/iga-sfh-representation-certification/actions/workflows/reproduce.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22315863.svg)](https://doi.org/10.5281/zenodo.22315863)

Reproducibility code for the manuscript:

> **Inference of Galaxy Star Formation Histories from Spectral Energy Distribution: Target-Specific Representation Selection and Robust Certification**
>
> Tsutomu T. Takeuchi, Karin T. Sakuragi, Ryusei R. Kano, and Sena A. Matsui

The repository implements the four controlled benchmarks used to verify target-specific representation selection, numerical certification, robust risk envelopes, abstention, and post-selection coverage in integrated-light spectral energy distribution inverse problems.

## Scope

This repository is a reproducibility companion for a methodology paper. It is not a production stellar-population-synthesis library and it is not a general-purpose SED-fitting package. The fourth benchmark uses a self-contained, physically structured SED proxy so that the numerical experiment can be reproduced without an external SPS code. Its robust guarantee applies only to the declared uncertainty set.

## Repository contents

| Directory | Benchmark | Principal question |
|---|---|---|
| `benchmarks/01_diagonal_operator/` | Analytic diagonal operator | Does the implementation reproduce the closed-form candidate risks, nested selection, stability failure, and unresolved boundary case? |
| `benchmarks/02_nonnested_anisotropic/` | Non-nested anisotropic candidates | Can candidate geometry, anisotropic admissible states, and directional discrepancy alter the selected representation? |
| `benchmarks/03_robust_postselection/` | Operator mismatch and post-selection | Can nominal certification fail for the returned fixed estimator, and how does same-data selection affect coverage? |
| `benchmarks/04_sed_proxy/` | Age-resolved SED proxy | How do wavelength coverage, signal-to-noise ratio, and a finite mismatch library affect certification and abstention? |

Each benchmark directory contains one executable Python script and the synchronized CSV, JSON, PDF, and PNG outputs produced by that script.

## Quick start

Python 3.12 or 3.13 is recommended.

```bash
git clone https://github.com/TsutomuTakeuchiTTT/iga-sfh-representation-certification.git
cd iga-sfh-representation-certification
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/reproduce_all.py
```

The complete verified run takes about one minute on a contemporary desktop. Every benchmark performs internal validation and exits with a nonzero status if a declared numerical tolerance or expected selection fails.

A containerized run is also available:

```bash
docker build -t iga-sfh-certification .
docker run --rm iga-sfh-certification
```

## Principal reference results

| Benchmark | Reference result |
|---|---|
| 1 | At `epsilon = 0.55`, the selected dimensions are 6 for `kappa = 1.0`, 7 for `kappa = 0.20`, and none for `kappa = 0.08`. |
| 2 | The full anisotropic model selects `B12`; removing discrepancy selects `B23`; replacing the state set by an RMS-matched isotropic ball selects `B12`. |
| 3 | Nominal certification selects `V2`, whose coupled risk exceeds the tolerance; robust certification selects `V3`. Naive same-data 95 percent coverage is approximately 0.9025. |
| 4 | Nominal certification reports in all six scenarios and is falsely certified relative to the mismatch library in all six. Robust certification reports only for full wavelength coverage at S/N 30 and 60. |

The complete machine-readable results are stored in the benchmark summary JSON files. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the verified environment, run times, hashes, and cross-version checks.

## Reproducing one benchmark

For example:

```bash
python benchmarks/03_robust_postselection/iga_robust_postselection_benchmark.py
```

The script writes its synchronized outputs into the same benchmark directory. The same pattern applies to the other three directories.

## Numerical and file-level reproducibility

The scientific checks are based on numerical values, declared tolerances, solver agreement, adversarial checks, and Monte Carlo diagnostics. CSV and JSON results are the primary machine-readable records. PDF byte hashes can vary across operating systems or Matplotlib environments because of metadata and font handling even when the rendered figure is unchanged. The supplied manifest records the files in the submission release candidate.

## Citation

GitHub displays citation metadata from [`CITATION.cff`](CITATION.cff). The immutable `v1.0.0` software archive is assigned Zenodo DOI [`10.5281/zenodo.22315863`](https://doi.org/10.5281/zenodo.22315863). Until the manuscript receives a journal DOI, cite this software release. After publication, the preferred article citation can be added to `CITATION.cff` and the archived release metadata.

## Development disclosure

Generative AI assistance was used during portions of code drafting, debugging, documentation, and figure preparation. The mathematical formulation, numerical settings, executable scripts, generated outputs, and internal validation criteria were reviewed by the authors. Responsibility for the released code and scientific claims remains with the authors.

## License

The code is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).

## Contact

Tsutomu T. Takeuchi

Division of Particle and Astrophysical Science, Nagoya University
