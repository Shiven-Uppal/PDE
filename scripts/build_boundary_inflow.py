"""Create daily, direction-specific CPCB boundary PM2.5 forcing.

Only original 1-hour CPCB files are read; 15-minute duplicates in the archive
are intentionally not used. A daily value requires at least 18 valid hourly
PM2.5 observations. The raw file's station name is the authoritative name.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


STATIONS = {
    "Bahadurgarh": ("Arya Nagar, Bahadurgarh - HSPCB", "W"),
    "Ghaziabad": ("Sanjay Nagar, Ghaziabad - UPPCB", "E"),
    "Greater Noida": ("Knowledge Park V, Greater Noida - UPPCB", "SE"),
    "Gurugram": ("Vikas Sadan, Gurugram - HSPCB", "S"),
    "Sonipat": ("Murthal, Sonipat - HSPCB", "N_NW"),
}
WINDOWS = (("winter_2022_23", "2022-11-01", "2023-02-28"),
           ("winter_2023_24", "2023-11-01", "2024-02-29"))
SECTOR_MAP = {"N": "Sonipat", "NW": "Sonipat", "W": "Bahadurgarh",
              "E": "Ghaziabad", "SE": "Greater Noida", "S": "Gurugram"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sector(degrees: float) -> str:
    labels = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    return str(labels[int(((degrees + 22.5) % 360) // 45)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--era5-daily", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, output = Path(args.raw_root), Path(args.output_dir)
    all_daily, manifest = [], []
    for folder, (station_name, role) in STATIONS.items():
        files = sorted((root / folder).glob("*1H*.csv"))
        if not files:
            raise FileNotFoundError(f"No hourly files for {folder}")
        pieces = []
        for path in files:
            raw = pd.read_csv(path, usecols=["Timestamp", "PM2.5 (µg/m³)"])
            raw["time"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
            raw["pm25"] = pd.to_numeric(raw["PM2.5 (µg/m³)"], errors="coerce")
            raw.loc[raw.pm25 < 0, "pm25"] = np.nan
            pieces.append(raw[["time", "pm25"]])
            manifest.append({"boundary_station": station_name, "raw_filename": path.name,
                             "sha256": sha256(path), "rows": len(raw)})
        hourly = pd.concat(pieces, ignore_index=True).drop_duplicates("time")
        hourly["date"] = hourly.time.dt.normalize()
        daily = hourly.groupby("date", as_index=False).agg(pm25_inflow_ug_m3=("pm25", "mean"),
                                                              n_valid_hours=("pm25", "count"))
        daily = daily[daily.n_valid_hours >= 18].copy()
        daily["boundary_station"] = station_name
        daily["boundary_role"] = role
        all_daily.append(daily)
    panel = pd.concat(all_daily, ignore_index=True)
    panel["winter"] = pd.NA
    for name, start, end in WINDOWS:
        panel.loc[(panel.date >= start) & (panel.date <= end), "winter"] = name
    panel = panel.dropna(subset=["winter"]).sort_values(["date", "boundary_station"])

    weather = pd.read_csv(args.era5_daily, parse_dates=["date"])
    eligible = weather[weather.eligible_steady_state].copy()
    eligible["inflow_sector"] = ((eligible.wind_to_direction_deg + 180.0) % 360.0).map(sector)
    eligible["boundary_folder"] = eligible.inflow_sector.map(SECTOR_MAP)
    lookup = panel.copy()
    lookup["boundary_folder"] = lookup.boundary_station.map(
        {station: folder for folder, (station, _) in STATIONS.items()}
    )
    forcing = eligible.merge(lookup, on=["date", "boundary_folder"], how="left")
    forcing["usable_boundary_forcing"] = forcing.pm25_inflow_ug_m3.notna()

    output.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output / "boundary_pm25_daily_panel.csv", index=False, date_format="%Y-%m-%d")
    forcing.to_csv(output / "directional_boundary_forcing.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(manifest).to_csv(output / "boundary_raw_manifest.csv", index=False)
    print(f"Boundary station-days: {len(panel)} | eligible weather days: {len(forcing)} | usable forcing days: {forcing.usable_boundary_forcing.sum()}")


if __name__ == "__main__":
    main()
