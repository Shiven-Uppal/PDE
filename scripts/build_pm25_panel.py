#!/usr/bin/env python3
"""Build a quality-controlled Delhi winter PM2.5 panel from CPCB 15-minute ZIP exports.

The script never changes raw files. It reads native 15-minute CSV records from
the supplied ZIP archives, retains two prespecified winter windows, and writes
reviewable metadata, hourly means, daily means, and coverage summaries.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd


COORDINATE_SOURCE = "https://airquality.cpcb.gov.in/ccr/#/continuous-stations"
COORDINATE_ACCESS_DATE = "2026-08-30"
WINDOWS = {
    "winter_2022_23": (pd.Timestamp("2022-11-01"), pd.Timestamp("2023-03-01")),
    "winter_2023_24": (pd.Timestamp("2023-11-01"), pd.Timestamp("2024-03-01")),
}
ALIASES = {
    "IHBAS, Dilshad Garden": "dilshad",
    "Burari Crossing": "burari",
    "Dwarka Sector 8": "dwarka sec 8",
    "IGI Airport (T3)": "igi airport",
    "Jahangirpuri": "jahangiri",
    "Dr. Karni Singh Shooting Range": "karni singh",
    "CRRI Mathura Road": "mathura road",
    "Major Dhyan Chand National Stadium": "major dhyan chand",
    "NSIT Dwarka": "nsit",
    "North Campus, DU": "north campus",
    "Okhla Phase II": "okhla phase 2",
    "Pusa, IITM": "pusa iitm",
    "Pusa, DPCC": "pusa dpcc",
    "R. K. Puram": "rk puram",
    "Sri Aurobindo Marg": "sri aurobindo",
}


def key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"(?<=[a-z])(?=\d)", " ", cleaned)


def find_window(ts: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=ts.index, dtype="string")
    for label, (start, end) in WINDOWS.items():
        result.loc[(ts >= start) & (ts < end)] = label
    return result


def agency_from_name(name: str) -> str:
    lower = name.lower()
    for agency in ("dpcc", "cpcb", "iitm"):
        if f"_-_{agency}_" in lower or f"_{agency}_" in lower:
            return agency.upper()
    raise ValueError(f"Could not infer agency from filename: {name}")


def read_15m_members(archive: Path, target_station_keys: set[str]) -> list[tuple[str, str, pd.DataFrame]]:
    records: list[tuple[str, str, pd.DataFrame]] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv") or "15m" not in member.lower() or "__macosx" in member.lower():
                continue
            station = Path(member).parent.name
            station_key = key(station)
            if station_key not in target_station_keys:
                continue
            with zf.open(member) as raw:
                frame = pd.read_csv(raw, usecols=lambda c: c in {"Timestamp", "PM2.5 (µg/m³)"})
            if set(frame.columns) != {"Timestamp", "PM2.5 (µg/m³)"}:
                continue
            frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], format="mixed", errors="coerce")
            # The portal sometimes labels an hourly export 15M. Keep only native 15-minute files.
            delta = frame["Timestamp"].sort_values().diff().dropna().mode()
            if delta.empty or delta.iloc[0] != pd.Timedelta(minutes=15):
                continue
            frame["pm25"] = pd.to_numeric(frame["PM2.5 (µg/m³)"], errors="coerce")
            frame = frame[["Timestamp", "pm25"]]
            records.append((station_key, agency_from_name(member), frame))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path, action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    coords = pd.read_csv(args.coordinates).rename(columns={"Latitude ": "latitude", "Longitude": "longitude", "Station Name": "station_name"})
    required = {"station_name", "latitude", "longitude"}
    if missing := required - set(coords.columns):
        raise ValueError(f"Missing coordinate columns: {sorted(missing)}")
    coords["station_key"] = coords["station_name"].map(lambda x: ALIASES.get(x, key(x))).map(key)
    if coords["station_key"].duplicated().any():
        raise ValueError("Coordinate sheet has duplicate station identities after normalisation.")

    all_records = []
    for archive in args.archive:
        all_records.extend(read_15m_members(archive, set(coords["station_key"])))
    by_station: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)
    for station, agency, frame in all_records:
        by_station[station].append((agency, frame))

    missing = sorted(set(coords["station_key"]) - set(by_station))
    if missing:
        raise ValueError(f"No native 15-minute raw records found for: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows, hourly_frames, daily_frames, qc_rows = [], [], [], []
    for meta in coords.sort_values("station_name").itertuples(index=False):
        station_records = by_station[meta.station_key]
        agencies = sorted({agency for agency, _ in station_records})
        if len(agencies) != 1:
            raise ValueError(f"Multiple agencies for {meta.station_name}: {agencies}")
        raw = pd.concat([frame.assign(source_agency=agency) for agency, frame in station_records], ignore_index=True)
        raw = raw.drop_duplicates(subset=["Timestamp"], keep="first").sort_values("Timestamp")
        raw["winter"] = find_window(raw["Timestamp"])
        raw = raw.loc[raw["winter"].notna()].copy()
        raw["is_valid_15m"] = raw["pm25"].notna() & raw["pm25"].ge(0)
        raw["hour"] = raw["Timestamp"].dt.floor("h")
        hourly = (raw.loc[raw["is_valid_15m"]]
                  .groupby(["winter", "hour"], as_index=False)
                  .agg(pm25_hourly=("pm25", "mean"), n_valid_15m=("pm25", "size")))
        hourly = hourly.loc[hourly["n_valid_15m"] >= 3].copy()
        hourly["station_id"] = re.sub(r"[^a-z0-9]+", "_", meta.station_name.lower()).strip("_")
        hourly["station_name"] = meta.station_name
        hourly["date"] = hourly["hour"].dt.floor("D")
        hourly_frames.append(hourly[["station_id", "station_name", "winter", "hour", "date", "pm25_hourly", "n_valid_15m"]])

        daily = (hourly.groupby(["winter", "date"], as_index=False)
                 .agg(pm25_daily=("pm25_hourly", "mean"), n_valid_hours=("pm25_hourly", "size")))
        daily = daily.loc[daily["n_valid_hours"] >= 18].copy()
        daily["station_id"] = hourly["station_id"].iloc[0]
        daily["station_name"] = meta.station_name
        daily_frames.append(daily[["station_id", "station_name", "winter", "date", "pm25_daily", "n_valid_hours"]])

        for label, (start, end) in WINDOWS.items():
            expected_15m = int((end - start) / pd.Timedelta(minutes=15))
            expected_days = int((end - start) / pd.Timedelta(days=1))
            valid_15m = int(raw.loc[(raw["winter"] == label) & raw["is_valid_15m"], "pm25"].count())
            valid_hours = int(hourly.loc[hourly["winter"] == label, "hour"].nunique())
            valid_days = int(daily.loc[daily["winter"] == label, "date"].nunique())
            qc_rows.append({"station_name": meta.station_name, "winter": label, "expected_15m": expected_15m,
                            "valid_15m": valid_15m, "coverage_15m": valid_15m / expected_15m,
                            "valid_hours": valid_hours, "coverage_hourly": valid_hours / (expected_days * 24),
                            "valid_days": valid_days, "coverage_daily": valid_days / expected_days,
                            "passes_daily_70pct": valid_days / expected_days >= 0.70})
        metadata_rows.append({"station_id": hourly["station_id"].iloc[0], "station_name": meta.station_name,
                              "latitude": float(meta.latitude), "longitude": float(meta.longitude),
                              "operating_agency": agencies[0], "coordinate_source_url": COORDINATE_SOURCE,
                              "coordinate_access_date": COORDINATE_ACCESS_DATE, "included_in_analysis": True})

    metadata = pd.DataFrame(metadata_rows).sort_values("station_name")
    hourly_out = pd.concat(hourly_frames, ignore_index=True).sort_values(["station_name", "hour"])
    daily_out = pd.concat(daily_frames, ignore_index=True).sort_values(["station_name", "date"])
    qc = pd.DataFrame(qc_rows).sort_values(["station_name", "winter"])
    metadata.to_csv(args.output_dir / "station_metadata.csv", index=False)
    hourly_out.to_csv(args.output_dir / "pm25_hourly_clean.csv", index=False)
    daily_out.to_csv(args.output_dir / "pm25_daily_panel.csv", index=False)
    qc.to_csv(args.output_dir / "data_quality_report.csv", index=False)
    print(f"Stations: {len(metadata)} | hourly rows: {len(hourly_out)} | daily rows: {len(daily_out)}")
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
