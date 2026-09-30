"""Build IST-aligned hourly domain-mean ERA5 forcing for the transient ADR model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

WINDOWS = (("2022-11-01", "2023-02-28"), ("2023-11-01", "2024-02-29"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def in_window(time: pd.Timestamp) -> bool:
    return any(pd.Timestamp(a) <= time.normalize() <= pd.Timestamp(b) for a, b in WINDOWS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    frames, manifest = [], []
    for item in args.inputs:
        path = Path(item)
        with h5py.File(path, "r") as ds:
            required = {"u10", "v10", "tp", "valid_time"}
            missing = required - set(ds.keys())
            if missing:
                raise ValueError(f"{path.name} lacks {sorted(missing)}")
            times_ist = pd.to_datetime(ds["valid_time"][:], unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
            # ERA5 UTC timestamps appear at :00 and consequently at :30 IST.
            # They are spatial hourly means; assign each to its containing CPCB
            # IST clock hour, whose records are labelled at :00.
            times = times_ist.floor("h")
            u, v, tp = (np.asarray(ds[key][:], dtype=float) for key in ("u10", "v10", "tp"))
            frames.append(pd.DataFrame({
                "time": times, "u10_m_s": np.nanmean(u, axis=(1, 2)),
                "v10_m_s": np.nanmean(v, axis=(1, 2)),
                "precipitation_mm": 1000 * np.nansum(tp, axis=(1, 2)) / tp.shape[1] / tp.shape[2],
            }))
            manifest.append({"raw_filename": path.name, "sha256": sha256(path), "n_hours": len(times)})
    hourly = pd.concat(frames, ignore_index=True).sort_values("time")
    if hourly.time.duplicated().any():
        raise ValueError("Duplicate ERA5 hourly timestamps.")
    hourly = hourly[hourly.time.map(in_window)].copy()
    hourly["wind_speed_m_s"] = np.hypot(hourly.u10_m_s, hourly.v10_m_s)
    hourly["wind_to_direction_deg"] = (np.degrees(np.arctan2(hourly.u10_m_s, hourly.v10_m_s)) + 360) % 360
    hourly["rain_free_hour"] = hourly.precipitation_mm <= 0.1
    hourly["non_calm_hour"] = hourly.wind_speed_m_s >= 0.5
    if len(hourly) != 241 * 24:
        raise ValueError(f"Expected 5784 study hours, found {len(hourly)}")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / "era5_hourly_forcing.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    pd.DataFrame(manifest).to_csv(output / "era5_hourly_raw_manifest.csv", index=False)
    print(f"Hourly forcing rows: {len(hourly)}; dry non-calm hours: {(hourly.rain_free_hour & hourly.non_calm_hour).sum()}")


if __name__ == "__main__":
    main()
