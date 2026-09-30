# Delhi PM₂.₅ transient ADR project

This repository contains the code and supporting files for the completed Delhi PM₂.₅ reduced-order modelling study. The study develops a transient two-dimensional advection–diffusion–reaction (ADR) framework and reports numerical verification, a temporally separated internal assessment, and a conditional uncertainty analysis.

**Interpretation:** numerical verification means the implementation solves its stated discretised problem to the tested accuracy. It does not establish that the reduced-order model is a validated operational predictor or that its source-input scenarios estimate real-world policy effects.

## Repository contents

- `src/`: daily and transient ADR solvers.
- `scripts/`: data-processing, model calibration/evaluation, verification, diagnostics, and figure-generation scripts.
- `data/raw/`: original input files included in the repository snapshot.
- `data/processed/`: quality-controlled and derived tables, manifests, and templates.
- `outputs/`: archived diagnostic results. The `archive_diagnostics/` subfolder marks exploratory or non-converged runs.
- `docs/`: data provenance, reproducibility status, and review notes.

The manuscript is not included in this repository, as requested.

## Setup and verification

Use Python 3.11 or newer. Create an environment and install the packages listed in `requirements.txt`, then run:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/verify_solver.py --output-dir outputs/mms_verification
PYTHONPATH=. python scripts/verify_uniform_baseline.py --help
```

The MMS check was run during this packaging review. It returned observed refinement orders 0.9256, 0.9581, and 0.9771, with sparse residuals below 6×10⁻¹⁷.

## Reproducibility scope

The study and its reported analyses are complete. This repository is the project code archive and supports review of the solver, data-processing workflow, and numerical-verification checks. The current repository snapshot is not a one-command reproduction package for every manuscript table and figure: some original input files and the final run-level calibration and uncertainty-analysis outputs are not included. See `docs/reproducibility_status.md` for the specific inventory and status of each component.

## Data sources

CPCB CAAQMS observations: <https://airquality.cpcb.gov.in/ccr/#/repository/data>

ERA5: Hersbach et al. (2020), <https://doi.org/10.1002/qj.3803>

EDGAR: <https://edgar.jrc.ec.europa.eu/dataset_ap81>

NASA FIRMS: <https://firms.modaps.eosdis.nasa.gov/>

OpenStreetMap extract: supplied project file, with source notes in its original archive.
