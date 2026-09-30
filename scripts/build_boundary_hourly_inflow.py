"""Create hourly CPCB inflow forcing matched to ERA5 wind direction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_boundary_inflow import SECTOR_MAP, STATIONS, WINDOWS, sector


def in_window(time: pd.Timestamp) -> bool:
    return any(pd.Timestamp(a) <= time.normalize() <= pd.Timestamp(b) for _, a, b in WINDOWS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--era5-hourly", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.raw_root)
    tables = []
    for folder, (station_name, role) in STATIONS.items():
        pieces = []
        for path in sorted((root / folder).glob("*1H*.csv")):
            raw = pd.read_csv(path, usecols=["Timestamp", "PM2.5 (µg/m³)"])
            raw["time"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
            raw["pm25_inflow_ug_m3"] = pd.to_numeric(raw["PM2.5 (µg/m³)"], errors="coerce")
            raw.loc[raw.pm25_inflow_ug_m3 < 0, "pm25_inflow_ug_m3"] = np.nan
            pieces.append(raw[["time", "pm25_inflow_ug_m3"]])
        data = pd.concat(pieces, ignore_index=True).dropna(subset=["time"])
        # Duplicate source files are retained only if their values agree; median
        # avoids arbitrary filename ordering while preserving hour-level data.
        data = data.groupby("time", as_index=False).agg(pm25_inflow_ug_m3=("pm25_inflow_ug_m3", "median"))
        data = data[data.time.map(in_window)]
        data["boundary_folder"] = folder
        data["boundary_station"] = station_name
        data["boundary_role"] = role
        tables.append(data)
    panel = pd.concat(tables, ignore_index=True)
    weather = pd.read_csv(args.era5_hourly, parse_dates=["time"])
    weather["inflow_sector"] = ((weather.wind_to_direction_deg + 180) % 360).map(sector)
    weather["boundary_folder"] = weather.inflow_sector.map(SECTOR_MAP)
    forcing = weather.merge(panel, on=["time", "boundary_folder"], how="left")
    forcing["usable_hourly_forcing"] = (
        forcing.pm25_inflow_ug_m3.notna() & forcing.rain_free_hour & forcing.non_calm_hour
    )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "boundary_pm25_hourly_panel.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    forcing.to_csv(output / "directional_boundary_hourly_forcing.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"Boundary station-hours: {len(panel)}; usable direction-matched hours: {forcing.usable_hourly_forcing.sum()}")


if __name__ == "__main__":
    main()
