"""Build daily Delhi-NCR ERA5 forcing from original downloaded NetCDF files.

The script never changes raw ERA5 files. It vector-averages hourly u10/v10 over
the requested regional grid, aggregates these hourly values by Indian Standard
Time (to align with CPCB daily means), sums hourly total precipitation (metres)
to millimetres, and records provenance.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REQUIRED = {"u10", "v10", "tp", "valid_time", "latitude", "longitude"}
WINDOWS = (("2022-11-01", "2023-02-28"), ("2023-11-01", "2024-02-29"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_file(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    with h5py.File(path, "r") as dataset:
        missing = REQUIRED - set(dataset.keys())
        if missing:
            raise ValueError(f"{path.name} is missing {sorted(missing)}")
        times = pd.to_datetime(dataset["valid_time"][:], unit="s", utc=True)
        u = np.asarray(dataset["u10"][:], dtype=float)
        v = np.asarray(dataset["v10"][:], dtype=float)
        tp = np.asarray(dataset["tp"][:], dtype=float)
        if u.shape != v.shape or u.shape != tp.shape or u.shape[0] != len(times):
            raise ValueError(f"Unexpected variable dimensions in {path.name}")
        frame = pd.DataFrame({
            "time_utc": times,
            "u10_m_s": np.nanmean(u, axis=(1, 2)),
            "v10_m_s": np.nanmean(v, axis=(1, 2)),
            "precipitation_mm": 1000.0 * np.nansum(tp, axis=(1, 2)) / tp.shape[1] / tp.shape[2],
        })
        meta = {
            "raw_filename": path.name,
            "sha256": sha256(path),
            "first_time_utc": str(times.min()),
            "last_time_utc": str(times.max()),
            "n_hours": len(times),
            "n_lat": int(len(dataset["latitude"])),
            "n_lon": int(len(dataset["longitude"])),
            "u10_units": dataset["u10"].attrs.get("units", b"").decode() if isinstance(dataset["u10"].attrs.get("units", b""), bytes) else str(dataset["u10"].attrs.get("units", "")),
            "v10_units": dataset["v10"].attrs.get("units", b"").decode() if isinstance(dataset["v10"].attrs.get("units", b""), bytes) else str(dataset["v10"].attrs.get("units", "")),
            "tp_units": dataset["tp"].attrs.get("units", b"").decode() if isinstance(dataset["tp"].attrs.get("units", b""), bytes) else str(dataset["tp"].attrs.get("units", "")),
        }
    return frame, meta


def in_study_window(day: pd.Timestamp) -> bool:
    day = day.tz_localize(None) if day.tzinfo else day
    return any(pd.Timestamp(start) <= day <= pd.Timestamp(end) for start, end in WINDOWS)


def directional_resultant(frame: pd.DataFrame) -> float:
    """Return 0--1 directional consistency from non-calm hourly vectors."""
    speed = np.hypot(frame.u10_m_s, frame.v10_m_s)
    keep = speed >= 0.5
    if not keep.any():
        return float("nan")
    return float(np.hypot(
        np.mean(frame.loc[keep, "u10_m_s"] / speed[keep]),
        np.mean(frame.loc[keep, "v10_m_s"] / speed[keep]),
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="ERA5 .nc files")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    files = [Path(item) for item in args.inputs]
    frames, provenance = zip(*(read_file(path) for path in files))
    hourly = pd.concat(frames, ignore_index=True).sort_values("time_utc")
    if hourly.time_utc.duplicated().any():
        raise ValueError("Duplicate hourly ERA5 timestamps detected.")
    hourly["time_ist"] = hourly.time_utc.dt.tz_convert("Asia/Kolkata")
    hourly["date"] = hourly.time_ist.dt.tz_localize(None).dt.normalize()
    daily = hourly.groupby("date", as_index=False).agg(
        u10_m_s=("u10_m_s", "mean"),
        v10_m_s=("v10_m_s", "mean"),
        precipitation_mm=("precipitation_mm", "sum"),
        n_hours=("time_ist", "count"),
    )
    direction = hourly.groupby("date").apply(directional_resultant, include_groups=False)
    daily["hourly_directional_resultant"] = daily.date.map(direction)
    daily = daily[daily.date.map(in_study_window)].copy()
    daily["wind_speed_m_s"] = np.hypot(daily.u10_m_s, daily.v10_m_s)
    # Direction is meteorological *toward* direction, measured clockwise from north.
    daily["wind_to_direction_deg"] = (np.degrees(np.arctan2(daily.u10_m_s, daily.v10_m_s)) + 360.0) % 360.0
    # Locked before PM2.5 fitting: remove wet or directionally unstable days.
    daily["rain_free"] = daily.precipitation_mm <= 0.1
    daily["directionally_stable"] = (
        (daily.wind_speed_m_s >= 0.5) & (daily.hourly_directional_resultant >= 0.5)
    )
    daily["eligible_steady_state"] = daily.rain_free & daily.directionally_stable
    if len(daily) != 241 or not (daily.n_hours == 24).all():
        raise ValueError("Expected 241 complete study days with 24 hourly records each.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "era5_daily_forcing.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(provenance).to_csv(output / "era5_raw_manifest.csv", index=False)
    print(f"Daily forcing rows: {len(daily)} | raw files: {len(files)}")


if __name__ == "__main__":
    main()
