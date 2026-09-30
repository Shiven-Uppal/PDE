# Raw input data

Raw data files are retained in the local project archive. Large CPCB/DPCC CSV exports and the OSM GeoPackage are git-ignored to keep the GitHub repository within practical size limits; they can be shared separately or placed in Git LFS after the repository exists. Data acquisition and checksums are recorded in `data/processed/*manifest.csv` and `docs/file_manifest.csv`.

The supplied `data/raw/era5/` and `data/raw/firms/` files are small and included. EDGAR source rasters and the station-coordinate sheet were not included with the supplied files; see `../edgar/README.md` and `cpcb/coordinates/README.md`.
