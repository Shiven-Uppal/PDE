#!/usr/bin/env python3
"""Build a reviewable station inventory from authoritative CPCB metadata.

The script intentionally does not guess coordinates, station roles, or
computational-domain membership. Those are research decisions that must be
recorded and reviewed before calibration begins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "city",
    "state",
    "metadata_source_url",
    "metadata_retrieved_on",
}

OUTPUT_COLUMNS = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "x_m",
    "y_m",
    "role",
    "domain_status",
    "inflow_sectors",
    "site_type",
    "metadata_source_url",
    "metadata_retrieved_on",
    "notes",
]


def validate_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    frame = frame.copy()
    frame["station_id"] = frame["station_id"].astype(str).str.strip()
    frame["station_name"] = frame["station_name"].astype(str).str.strip()
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")

    invalid = frame[
        frame["station_id"].eq("")
        | frame["station_name"].eq("")
        | frame["latitude"].isna()
        | frame["longitude"].isna()
        | ~frame["latitude"].between(-90, 90)
        | ~frame["longitude"].between(-180, 180)
    ]
    if not invalid.empty:
        raise ValueError(
            "Metadata has missing or invalid station identifiers, names, or coordinates: "
            + ", ".join(invalid["station_name"].fillna("<unnamed>").head(10))
        )

    if frame["station_id"].duplicated().any():
        duplicates = frame.loc[frame["station_id"].duplicated(), "station_id"].tolist()
        raise ValueError(f"Duplicate station IDs: {duplicates}")
    return frame


def build_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    inventory = pd.DataFrame(index=frame.index)
    for column in ["station_id", "station_name", "latitude", "longitude", "metadata_source_url", "metadata_retrieved_on"]:
        inventory[column] = frame[column]

    inventory["x_m"] = pd.NA
    inventory["y_m"] = pd.NA
    inventory["role"] = "unreviewed"
    inventory["domain_status"] = "unreviewed"
    inventory["inflow_sectors"] = ""
    inventory["site_type"] = frame.get("site_type", pd.Series("", index=frame.index)).fillna("")
    inventory["notes"] = "Review coordinates, domain membership, and role before use."
    return inventory[OUTPUT_COLUMNS].sort_values("station_name", kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = pd.read_csv(args.input)
    inventory = build_inventory(validate_metadata(metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output, index=False)
    print(f"Wrote {len(inventory)} review-ready station records to {args.output}")


if __name__ == "__main__":
    main()
