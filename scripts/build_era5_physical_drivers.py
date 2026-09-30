"""Merge ERA5 physical-driver NetCDF files and create an IST hourly table."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REQUIRED = ("blh", "t2m", "d2m", "sp")


def spatial_mean(series: xr.DataArray, time_name: str) -> pd.Series:
    """Mean an ERA5 field across all non-time dimensions, skipping expver NaNs."""
    dims = [dim for dim in series.dims if dim != time_name]
    return series.mean(dim=dims, skipna=True).to_series()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--start-ist", default="2022-11-01 00:00:00")
    parser.add_argument("--end-ist", default="2023-02-28 23:00:00")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.input_dir).glob("*.nc")):
        ds = xr.open_dataset(path, engine="h5netcdf")
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        available = [name for name in REQUIRED if name in ds.data_vars]
        if not available or time_name not in ds.coords:
            ds.close()
            continue
        item = pd.DataFrame({name: spatial_mean(ds[name], time_name) for name in available})
        item.index.name = "time_utc"
        item = item.reset_index()
        item["source_file"] = path.name
        rows.append(item)
        ds.close()

    if not rows:
        raise RuntimeError("No readable ERA5 files with physical-driver variables were found.")

    merged = pd.concat(rows, ignore_index=True)
    merged["time_utc"] = pd.to_datetime(merged["time_utc"], utc=True)
    # Duplicate variables can occur on boundary-date files; retain the first
    # non-missing value at each UTC timestamp.
    merged = merged.groupby("time_utc", as_index=False).agg(
        {**{name: "first" for name in REQUIRED}, "source_file": lambda x: ";".join(sorted(set(x)))}
    )
    merged["time_ist_exact"] = merged["time_utc"].dt.tz_convert("Asia/Kolkata")
    # ERA5 hourly timestamps occur at :30 IST.  The floor label is retained
    # solely as the corresponding CPCB hourly bin; exact IST is also saved.
    merged["hour_ist"] = merged["time_ist_exact"].dt.floor("h").dt.tz_localize(None)
    start, end = pd.Timestamp(args.start_ist), pd.Timestamp(args.end_ist)
    merged = merged[merged["hour_ist"].between(start, end)].copy()
    if merged[list(REQUIRED)].isna().any().any():
        missing = merged[list(REQUIRED)].isna().sum()
        raise RuntimeError(f"Missing physical-driver values after merge: {missing.to_dict()}")

    t_c = merged["t2m"] - 273.15
    td_c = merged["d2m"] - 273.15
    # Magnus formula; clipped only against numerical round-off.
    rh = 100.0 * np.exp((17.625 * td_c) / (243.04 + td_c) - (17.625 * t_c) / (243.04 + t_c))
    output = pd.DataFrame(
        {
            "hour_ist": merged["hour_ist"],
            "time_utc": merged["time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_ist_exact": merged["time_ist_exact"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "boundary_layer_height_m": merged["blh"],
            "temperature_2m_c": t_c,
            "dewpoint_2m_c": td_c,
            "relative_humidity_pct": np.clip(rh, 0.0, 100.0),
            "surface_pressure_pa": merged["sp"],
            "era5_source_files": merged["source_file"],
        }
    )
    output.to_csv(args.output_csv, index=False)
    print(
        {
            "output_rows": len(output),
            "first_hour_ist": str(output.hour_ist.min()),
            "last_hour_ist": str(output.hour_ist.max()),
            "missing_values": int(output.isna().sum().sum()),
        }
    )


if __name__ == "__main__":
    main()
