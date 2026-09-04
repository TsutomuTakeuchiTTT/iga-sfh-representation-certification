# Publishing checklist for GitHub and Zenodo

## 1. Create the GitHub repository

Recommended repository name:

```text
iga-sfh-representation-certification
```

Recommended description:

```text
Reproducibility code for target-specific SFH representation selection and robust certification from integrated SEDs.
```

Recommended visibility: `Public`.

When creating the repository on GitHub, do not ask GitHub to add a README, license, or `.gitignore`; all three are already present in this bundle.

Recommended topics:

```text
astrophysics
galaxy-evolution
spectral-energy-distribution
star-formation-history
inverse-problems
robust-optimization
post-selection-inference
reproducible-research
python
```

## 2. Final local verification

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/reproduce_all.py
python tools/validate_repository.py
python tools/create_manifest.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Review the four summary JSON files, `reproduction_summary.json`, and `MANIFEST.sha256` before committing.

## 3. Push the release candidate

```bash
git init
git add .
git commit -m "Add submission reproducibility release"
git branch -M main
git remote add origin https://github.com/TsutomuTakeuchiTTT/iga-sfh-representation-certification.git
git push -u origin main
```

Confirm that the GitHub Actions reproduction workflow passes.

## 4. Connect the repository to Zenodo

1. Sign in to Zenodo using the GitHub account associated with the repository.
2. Open the GitHub integration page in Zenodo.
3. Enable archival for `iga-sfh-representation-certification`.
4. Confirm that `CITATION.cff`, `.zenodo.json`, the author list, and the BSD 3-Clause license are correct before creating the release.

## 5. Create the immutable submission release

Recommended tag:

```text
v1.0.0
```

Recommended release title:

```text
Submission release for the SFH representation-certification manuscript
```

Recommended release notes are provided in `RELEASE_NOTES_v1.0.0.md`.

After the GitHub release is published, Zenodo should archive that tagged snapshot and assign a version DOI. Record both the version DOI and the concept DOI shown by Zenodo.

## 6. Add the DOI without altering the archived snapshot

The first DOI is not known until Zenodo processes the first GitHub release. After the DOI has been assigned:

1. Add the Zenodo badge to the top-level `README.md` on the `main` branch.
2. Add the DOI to `CITATION.cff` on the `main` branch.
3. Add the DOI to the manuscript's Data Availability or Code Availability statement.
4. Do not move or retag `v1.0.0`; it should remain the immutable snapshot cited at submission.

A documentation-only follow-up commit does not require a new archival release unless a second immutable snapshot is desired.

## 7. Suggested manuscript statement

Replace `[ZENODO DOI]` and `[GITHUB URL]` after publication of the release:

> No observational data were used in this study. All numerical benchmark data were generated using the code described in the article. The source code, configuration metadata, machine-readable benchmark outputs, and figure-generation scripts required to reproduce the reported results are archived at Zenodo, [ZENODO DOI], and are also available at [GITHUB URL].

## 8. Version policy

- `v1.0.0`: exact code and outputs used for journal submission.
- `v1.0.1`: documentation or metadata corrections that do not alter numerical results.
- `v1.1.0`: revised manuscript code or additional diagnostics that preserve the methodology but change or extend outputs.
- `v2.0.0`: scientifically incompatible changes to the model, benchmark definitions, or public interface.

The manuscript should cite the version DOI corresponding to the code actually used for the reported results.
