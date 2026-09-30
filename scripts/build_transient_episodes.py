"""Define deterministic contiguous hourly episodes for transient ADR fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MIN_STATIONS = 20
MIN_EPISODE_HOURS = 12
SPINUP_HOURS = 6
SPLITS = (
    ("calibration", "2022-11-01", "2022-12-31 23:00:00"),
    ("internal_validation", "2023-01-01", "2023-02-28 23:00:00"),
    ("external_sealed", "2023-11-01", "2024-02-29 23:00:00"),
)


def split_for(time: pd.Timestamp) -> str:
    for name, start, end in SPLITS:
        if pd.Timestamp(start) <= time <= pd.Timestamp(end):
            return name
    raise ValueError(f"Time outside declared study periods: {time}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    args = parser.parse_args()
    processed = Path(args.processed_dir)
    forcing = pd.read_csv(processed / "directional_boundary_hourly_forcing.csv", parse_dates=["time"])
    observations = pd.read_csv(processed / "pm25_hourly_clean.csv", parse_dates=["hour"])
    observed_count = observations.groupby("hour").station_id.nunique().rename("n_interior_stations").reset_index().rename(columns={"hour": "time"})
    panel = forcing.merge(observed_count, on="time", how="left").sort_values("time").reset_index(drop=True)
    panel["split"] = panel.time.map(split_for)
    panel["episode_eligible"] = panel.usable_hourly_forcing & (panel.n_interior_stations >= MIN_STATIONS)
    break_run = (
        panel.episode_eligible.ne(panel.episode_eligible.shift())
        | panel.time.diff().ne(pd.Timedelta(hours=1))
        | panel.split.ne(panel.split.shift())
    )
    panel["candidate_run"] = break_run.cumsum()
    runs = panel[panel.episode_eligible].groupby("candidate_run").agg(
        start_time=("time", "min"), end_time=("time", "max"), n_hours=("time", "size"), split=("split", "first"),
        min_interior_stations=("n_interior_stations", "min"),
    ).reset_index()
    runs = runs[runs.n_hours >= MIN_EPISODE_HOURS].copy().reset_index(drop=True)
    runs["episode_id"] = [f"{split}_{i:03d}" for split, block in runs.groupby("split", sort=False) for i in range(1, len(block) + 1)]
    # assign using candidate-run map, then score only after a six-hour spin-up
    run_map = runs.set_index("candidate_run").episode_id
    panel["episode_id"] = panel.candidate_run.map(run_map)
    panel["episode_hour_index"] = panel.groupby("episode_id", dropna=True).cumcount()
    panel["score_hour"] = panel.episode_id.notna() & (panel.episode_hour_index >= SPINUP_HOURS)
    runs["spinup_hours"] = SPINUP_HOURS
    runs["scored_hours"] = runs.n_hours - SPINUP_HOURS
    panel.to_csv(processed / "transient_hourly_episode_panel.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    runs.to_csv(processed / "transient_episode_summary.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    summary = runs.groupby("split").agg(episodes=("episode_id", "count"), total_hours=("n_hours", "sum"), scored_hours=("scored_hours", "sum"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
