# Data protocol (from the project code and supplied reports)

## Delhi interior observations

Native CPCB/DPCC/IITM exports are retained in `data/raw/cpcb/delhi_raw/`. The project panel builder uses native 15-minute files, excludes macOS metadata entries and files whose timestamps are not actually at 15-minute intervals, aggregates to hourly means when at least three valid 15-minute observations are available, and forms daily means when at least 18 valid hourly means are available. The supplied processed quality report documents station-level coverage.

## Boundary observations

Five NCR stations are retained: Arya Nagar (Bahadurgarh), Murthal (Sonipat), Sanjay Nagar (Ghaziabad), Knowledge Park V (Greater Noida), and Vikas Sadan (Gurugram). Hourly and 15-minute exports can coexist and some filenames have numbered duplicates. The processed boundary manifest records selected hourly raw filenames and hashes. Consult that manifest when auditing source selection; do not concatenate all duplicates blindly.

## Time splits

The code defines winter windows in November 2022–February 2023 and November 2023–February 2024. Episode split labels and actual eligible episodes are in `data/processed/transient_episode_summary.csv`. The paper and the supplied processed episode panel need a final reconciliation of split dates/counts before submission.

## ERA5 and source templates

Processed ERA5 daily/hourly forcing is retained under `data/processed/`; raw NetCDF files bundled with the source archive are under `data/raw/era5/`. The EDGAR primary-PM₂.₅ template was built from six sector rasters according to `scripts/build_edgar_inventory_template.py`; those GeoTIFFs were not present in the supplied archive, so only the derived template, sector summary, and manifest are included.
