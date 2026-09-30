"""Attach IST ERA5 physical drivers to the locked transient episode panel."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

DRIVERS = ["boundary_layer_height_m", "temperature_2m_c", "dewpoint_2m_c", "relative_humidity_pct", "surface_pressure_pa"]

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode-panel", required=True)
    p.add_argument("--driver-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--splits", nargs="+", default=["calibration", "internal_validation"])
    a = p.parse_args()
    panel = pd.read_csv(a.episode_panel, parse_dates=["time"])
    panel = panel[panel["split"].isin(a.splits)].copy()
    if panel.empty:
        raise RuntimeError(f"No panel rows found for requested splits: {a.splits}")
    drivers = pd.read_csv(a.driver_csv, parse_dates=["hour_ist"])
    if drivers["hour_ist"].duplicated().any():
        raise RuntimeError("ERA5 driver table has duplicate IST hourly labels.")
    merged = panel.merge(drivers[["hour_ist", *DRIVERS]], left_on="time", right_on="hour_ist", how="left", validate="many_to_one")
    missing = merged[DRIVERS].isna().any(axis=1)
    if missing.any():
        sample = merged.loc[missing, "time"].head(5).astype(str).tolist()
        raise RuntimeError(f"Missing physical drivers for {int(missing.sum())} panel rows; examples: {sample}")
    merged.drop(columns="hour_ist").to_csv(a.output_csv, index=False)
    print({"included_splits": a.splits, "episode_rows": len(merged), "driver_rows": len(drivers), "missing_driver_rows": 0, "output": str(Path(a.output_csv))})

if __name__ == "__main__":
    main()
