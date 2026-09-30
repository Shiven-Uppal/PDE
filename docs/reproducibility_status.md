# Reproducibility and review status

## What was checked

- All Python sources compile.
- `scripts/verify_solver.py` ran successfully against the supplied processed EDGAR template and reproduced MMS orders 0.9256, 0.9581, and 0.9771.
- The extracted processed episode panel contains 17 calibration, 24 internal-validation, and 52 external-sealed episodes. It spans November 2022–February 2024, while the paper describes a January–February 2023 internal assessment and November 2023–February 2024 sealed external period. The panel’s actual split/episode counts should be reconciled against the manuscript and any final evaluation outputs before submission.
- The packaged script `calibrate_transient_adr.py` defaults to a very small differential-evolution search (`maxiter=4`, `popsize=3`). Do not describe that default as an exhaustive or fully converged search without checking the actual run configuration.

## Missing inputs or outputs in the supplied material

- The CPCB station-coordinate spreadsheet cited in the archive manifest is missing. The processed `station_metadata.csv` is included.
- The six raw EDGAR sector GeoTIFFs are missing. The processed 50×50 inventory template and its manifest are included. The manifest is sufficient to document the derived artifact but not to regenerate it.
- The raw ERA5 files present are only the subset included in the original bundle; confirm their coverage against the processed ERA5 manifests before claiming a complete raw-data archive.
- The paper’s final transient calibration selection and 500-draw uncertainty outputs (input samples, scenario outcomes, summaries, rank sensitivities, and final figures) are not in the supplied bundle.
- Two archived physical-parameter calibration results report `success: false` because maximum iterations were exceeded. They are preserved as exploratory records and clearly labelled; they must not be presented as final fitted parameters.
- One archived spatial-boundary diagnostic contains a `NaN` log-loss result. A separate fixed diagnostic is available, but it is labelled a one-episode code-path diagnostic, not a fitted/validated final model.

## Conclusion

The code archive supports solver verification and shows substantial data-processing work. It does not, as supplied, verify all quantitative results in the manuscript. Re-run and save the final calibration, temporal evaluation, and conditional uncertainty analysis from a locked configuration, reconcile the split counts/periods, and include the exact final outputs before calling the package fully reproducible.
