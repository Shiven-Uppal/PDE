"""Fast, transparent one-episode ADR code-path diagnostic.

This is deliberately *not* a replacement for the 17-episode calibration.
It runs one locked calibration episode, prints progress at every simulated
hour, and evaluates a supplied candidate parameter triple.  Its role is to
check that the data, boundary forcing, source template and solver execute
together on the intended code path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize_scalar

from src.adr_solver import ADRConfig
from src.daily_adr import interpolate_to_station_coordinates, load_inventory_template
from src.transient_adr import advance_adr_implicit_many

LOG_OFFSET = 5.0
DT_S = 3600.0


def profile_q(background: np.ndarray, response: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    """Profile one non-negative constant source multiplier for this episode."""
    def loss(q: float) -> float:
        model = np.maximum(background + q * response, 0.0)
        return float(np.mean((np.log(model + LOG_OFFSET) - np.log(observed + LOG_OFFSET)) ** 2))

    upper, previous, current = 1e-8, loss(0.0), loss(1e-8)
    while current < previous and upper < 1.0:
        previous, upper = current, upper * 10.0
        current = loss(upper)
    fitted = minimize_scalar(loss, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    return float(fitted.x), float(fitted.fun)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-id", default="calibration_013")
    parser.add_argument("--parameters", nargs=3, type=float, required=True,
                        metavar=("D", "LAMBDA", "KAPPA"))
    args = parser.parse_args()
    d, lam, kappa = args.parameters
    if d <= 0 or lam <= 0 or not 0 < kappa <= 1:
        raise ValueError("Require D > 0, LAMBDA > 0, and 0 < KAPPA <= 1.")

    processed, output = Path(args.processed_dir), Path(args.output_dir)
    forcing = pd.read_csv(processed / "transient_hourly_episode_panel.csv", parse_dates=["time"])
    block = forcing[(forcing.split == "calibration") & (forcing.episode_id == args.episode_id)].copy()
    if block.empty:
        available = sorted(forcing.loc[forcing.split == "calibration", "episode_id"].dropna().unique())
        raise RuntimeError(f"Episode {args.episode_id!r} not found. Available: {available}")
    block = block.sort_values("time")
    observations = pd.read_csv(processed / "pm25_hourly_clean.csv", parse_dates=["hour"])
    stations = pd.read_csv(processed / "station_metadata.csv")
    east, north = Transformer.from_crs(4326, 32643, always_xy=True).transform(
        stations.longitude.to_numpy(), stations.latitude.to_numpy())
    station_map = stations.assign(easting_m=east, northing_m=north).set_index("station_id")
    template = load_inventory_template(processed / "edgar_pm25_inventory_template.csv")

    print(f"Diagnostic episode: {args.episode_id} ({block.time.iloc[0]} to {block.time.iloc[-1]})", flush=True)
    print(f"Candidate: D={d:g} m2/s, lambda={lam:.6g} 1/s, kappa={kappa:g}", flush=True)
    fields = np.stack((np.full((50, 50), float(block.iloc[0].pm25_inflow_ug_m3)), np.zeros((50, 50))))
    sources = np.stack((np.zeros((50, 50)), template))
    scored, max_residual = [], 0.0

    for hour_index, (_, row) in enumerate(block.iterrows(), start=1):
        config = ADRConfig(50_000.0, 50_000.0, 50, 50, d,
                           kappa * float(row.u10_m_s), kappa * float(row.v10_m_s), lam)
        fields, residual = advance_adr_implicit_many(
            config, fields, sources, np.array([float(row.pm25_inflow_ug_m3), 0.0]), DT_S)
        max_residual = max(max_residual, residual)
        print(f"  hour {hour_index:02d}/{len(block):02d}: {row.time}  residual={residual:.2e}", flush=True)
        if not bool(row.score_hour):
            continue
        observed = observations[observations.hour == row.time].merge(
            station_map[["easting_m", "northing_m"]], left_on="station_id", right_index=True)
        if len(observed) < 20:
            raise RuntimeError(f"Eligibility changed at {row.time}: only {len(observed)} stations.")
        item = observed[["hour", "station_id", "station_name", "pm25_hourly"]].copy()
        item["background_pm25"] = interpolate_to_station_coordinates(
            fields[0], config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
        item["unit_source_response"] = interpolate_to_station_coordinates(
            fields[1], config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
        item["relative_residual"] = residual
        scored.append(item)

    prediction = pd.concat(scored, ignore_index=True)
    prediction["date"] = prediction["hour"].dt.date

    # The spatial EDGAR template is fixed, but its amplitude is allowed to
    # vary by calendar day. This is a diagnostic source representation, not
    # a causal source-attribution estimate.
    daily_records = []
    for date, day in prediction.groupby("date", sort=True):
        q_day, loss_day = profile_q(
            day.background_pm25.to_numpy(),
            day.unit_source_response.to_numpy(),
            day.pm25_hourly.to_numpy(),
        )
        daily_records.append(
            {
                "date": str(date),
                "source_scale_q": q_day,
                "daily_log_mse": loss_day,
                "n_station_hours": int(len(day)),
            }
        )
    daily_q = pd.DataFrame(daily_records)
    prediction["date"] = prediction["date"].astype(str)
    prediction = prediction.merge(daily_q[["date", "source_scale_q"]], on="date", how="left")
    prediction["model_pm25"] = (
        prediction["background_pm25"]
        + prediction["source_scale_q"] * prediction["unit_source_response"]
    )
    loss = float(
        np.mean(
            (
                np.log(prediction["model_pm25"] + LOG_OFFSET)
                - np.log(prediction["pm25_hourly"] + LOG_OFFSET)
            ) ** 2
        )
    )
    q = float(daily_q["source_scale_q"].mean())
    rmse = float(np.sqrt(np.mean((prediction.model_pm25 - prediction.pm25_hourly) ** 2)))
    output.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output / "one_episode_station_hour_predictions.csv", index=False)
    daily_q.to_csv(output / "one_episode_daily_source_scales.csv", index=False)
    record = {
        "purpose": "one-episode code-path/sanity diagnostic with daily EDGAR source scaling; not multi-episode calibration",
        "episode_id": args.episode_id, "n_scored_station_hours": int(len(prediction)),
        "objective_log_mse": loss, "rmse_ug_m3": rmse,
        "mean_daily_source_scale_q": q, "n_daily_source_scales": int(len(daily_q)),
        "source_representation": "fixed EDGAR spatial template with one profiled non-negative source scale per calendar day",
        "diffusivity_m2_s": d, "removal_s_inv": lam, "wind_attenuation": kappa,
        "max_relative_linear_solve_residual": max_residual,
    }
    (output / "one_episode_diagnostic.json").write_text(json.dumps(record, indent=2))
    print("\nFINAL DIAGNOSTIC RECORD", flush=True)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
