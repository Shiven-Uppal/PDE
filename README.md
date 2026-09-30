# Delhi PM₂.₅ transient ADR project

This repository contains code and supporting data for a reduced-order two-dimensional advection–diffusion–reaction (ADR) model of PM₂.₅ on a Delhi-centred domain. The numerical solver was checked with a method of manufactured solutions (MMS), transient implementation checks, and sparse linear-system residual diagnostics.

**Interpretation:** numerical verification means the implementation solves its stated discretised problem to the tested accuracy. It does not establish that the reduced-order model is a validated operational predictor or that its source-input scenarios estimate real-world policy effects.

## Repository contents

- `src/`: daily and transient ADR solvers.
- `scripts/`: raw-data processing, model calibration/evaluation, verification, diagnostics, and figure-generation scripts.
- `data/raw/`: CPCB/DPCC station exports, NCR boundary exports, ERA5 files, FIRMS data, and the OSM GeoPackage supplied for the project.
- `data/interim/boundary_raw/`: retained copy of NCR station source exports.
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

## Reproducing data and model steps

The raw CPCB Delhi export files are in `data/raw/cpcb/delhi_raw/`; five NCR boundary stations are under `data/raw/cpcb/boundary_raw/`. Processed-panel scripts take explicit paths; run `python scripts/<script>.py --help` for options. The data periods and split logic are described in `docs/data_protocol.md`.

The bundle does **not** include all original inputs or final paper result files. In particular, the original station-coordinate sheet and the six EDGAR sector GeoTIFFs are absent; only processed station metadata and the derived EDGAR template are available. The raw ERA5 subset is present, but the uncertainty-analysis result files and final calibration table described in the manuscript were not present in the supplied ZIP. Therefore, this repository is a transparent working archive, not yet a fully verified, one-command reproduction of every manuscript table and figure. See `docs/reproducibility_status.md`.

## Data sources

CPCB CAAQMS observations: <https://airquality.cpcb.gov.in/ccr/#/repository/data>

ERA5: Hersbach et al. (2020), <https://doi.org/10.1002/qj.3803>

EDGAR: <https://edgar.jrc.ec.europa.eu/dataset_ap81>

NASA FIRMS: <https://firms.modaps.eosdis.nasa.gov/>

OpenStreetMap extract: supplied project file, with source notes in its original archive.
