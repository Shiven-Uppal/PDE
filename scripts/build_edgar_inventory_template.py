"""Convert independent EDGAR 1 km PM2.5 sector layers into the ADR grid.

The input is the *extracted* official EDGAR_2025_1km_PM2.5_2024 archive.
No CPCB concentration observations are read here.  Annual emissions are used
only to specify a fixed spatial prior; the daily magnitude is still profiled
from the calibration-period observations in ``calibrate_baseline.py``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import reproject
from shapely.geometry import box

CRS_UTM = "EPSG:32643"
CRS_WGS84 = "EPSG:4326"
X0_M, Y0_M, CELL_M, NX, NY = 682_750.0, 3_145_665.0, 1_000.0, 50, 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_polygon_wgs84():
    return gpd.GeoSeries([box(X0_M, Y0_M, X0_M + NX * CELL_M, Y0_M + NY * CELL_M)], crs=CRS_UTM).to_crs(CRS_WGS84).iloc[0]


def output_centres() -> tuple[np.ndarray, np.ndarray]:
    east = X0_M + (np.arange(NX) + 0.5) * CELL_M
    north = Y0_M + (np.arange(NY) + 0.5) * CELL_M
    return np.meshgrid(east, north)


def resample_to_adr_grid(input_tif: Path) -> np.ndarray:
    """Area-average annual PM2.5 emissions onto the fixed UTM 1 km ADR grid."""
    with rasterio.open(input_tif) as src:
        if src.count != 1:
            raise ValueError("Each EDGAR sector input must be a single-band PM2.5 GeoTIFF.")
        clipped, clipped_transform = mask(src, [domain_polygon_wgs84()], crop=True, filled=True, nodata=np.nan)
        source = clipped[0].astype(float)
        if src.nodata is not None:
            source[np.isclose(source, src.nodata)] = np.nan
        destination = np.full((NY, NX), np.nan, dtype=float)
        destination_transform = rasterio.transform.from_origin(X0_M, Y0_M + NY * CELL_M, CELL_M, CELL_M)
        reproject(
            source=source,
            destination=destination,
            src_transform=clipped_transform,
            src_crs=src.crs,
            dst_transform=destination_transform,
            dst_crs=CRS_UTM,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    destination = np.nan_to_num(destination, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(destination < 0) or destination.sum() <= 0:
        raise ValueError("EDGAR raster must contain positive, non-negative PM2.5 emissions in the ADR domain.")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edgar-tifs", required=True, nargs="+",
        help="All single-band EDGAR PM2.5 sector GeoTIFFs extracted from the official archive",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    input_tifs, output = [Path(item) for item in args.edgar_tifs], Path(args.output_dir)
    sector_names = [path.stem.removeprefix("PM2.5_2024_").lower() for path in input_tifs]
    if len(set(sector_names)) != len(sector_names):
        raise ValueError("Each EDGAR sector file must be supplied once only.")
    raw_layers = [resample_to_adr_grid(path) for path in input_tifs]
    raw = np.sum(raw_layers, axis=0)
    template = raw / raw.mean()
    east, north = output_centres()
    result = pd.DataFrame({
        "cell_id": np.arange(NX * NY),
        "i": np.tile(np.arange(NX), NY),
        "j": np.repeat(np.arange(NY), NX),
        "x_center_m": east.ravel(),
        "y_center_m": north.ravel(),
        "edgar_pm25_annual_raw": raw.ravel(),
        "inventory_template": template.ravel(),
    })
    for sector, layer in zip(sector_names, raw_layers):
        result[f"edgar_{sector}_annual_raw"] = layer.ravel()
        result[f"edgar_{sector}_template"] = layer.ravel() / raw.mean()
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "edgar_pm25_inventory_template.csv", index=False)
    pd.DataFrame([{
        "inventory": "EDGAR_2025_1km_PM2.5_2024",
        "source_files": ";".join(path.name for path in input_tifs),
        "sha256": ";".join(sha256(path) for path in input_tifs),
        "input_year": 2024,
        "input_resolution": "0.01 degree nominal (about 1 km)",
        "resampling": "area average into fixed UTM43N 1 km ADR grid",
        "domain_x0_m": X0_M, "domain_y0_m": Y0_M, "cell_m": CELL_M,
        "nx": NX, "ny": NY,
        "template_spatial_mean": template.mean(),
        "template_min": template.min(), "template_max": template.max(),
        "template_nonzero_cells": int((template > 0).sum()),
    }]).to_csv(output / "edgar_inventory_manifest.csv", index=False)
    sector_summary = pd.DataFrame({
        "sector": sector_names,
        "domain_annual_raw_sum": [float(layer.sum()) for layer in raw_layers],
    })
    sector_summary["domain_fraction"] = sector_summary.domain_annual_raw_sum / sector_summary.domain_annual_raw_sum.sum()
    sector_summary.to_csv(output / "edgar_sector_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    image = axes[0].imshow(template, origin="lower", cmap="magma")
    axes[0].set_title("EDGAR all-sector PM2.5 template")
    axes[0].set_xlabel("grid x (km)"); axes[0].set_ylabel("grid y (km)")
    fig.colorbar(image, ax=axes[0], label="relative annual emission intensity")
    axes[1].bar(sector_summary.sector.str.title(), sector_summary.domain_fraction, color="#2a6f97")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("domain inventory fraction")
    axes[1].set_title("EDGAR sector composition in ADR domain")
    axes[1].tick_params(axis="x", rotation=45)
    fig.savefig(output / "edgar_inventory_template.png", dpi=220)
    print(f"Summed {len(input_tifs)} sector layers; wrote {len(result)} ADR cells; nonzero cells: {(template > 0).sum()}")


if __name__ == "__main__":
    main()
