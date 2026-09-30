"""No-leakage evaluation of daily EDGAR source scaling for the transient ADR model.

Calibration period (Nov-Dec 2022): profile one non-negative EDGAR multiplier
per calendar day.  Internal-validation period (Jan-Feb 2023): use the *median* multiplier
from calibration for all hours; no source scales or physical parameters are fit
to internal observations.
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
    """Profile the non-negative source multiplier under squared log error."""
    def loss(q: float) -> float:
        model = np.maximum(background + q * response, 0.0)
        return float(np.mean((np.log(model + LOG_OFFSET) - np.log(observed + LOG_OFFSET)) ** 2))

    upper, previous, current = 1e-8, loss(0.0), loss(1e-8)
    while current < previous and upper < 1.0:
        previous, upper = current, upper * 10.0
        current = loss(upper)
    fitted = minimize_scalar(loss, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    return float(fitted.x), float(fitted.fun)


def simulate_split(
    forcing: pd.DataFrame,
    observations: pd.DataFrame,
    station_map: pd.DataFrame,
    template: np.ndarray,
    split: str,
    d: float,
    lam: float,
    kappa: float,
) -> pd.DataFrame:
    """Return background and unit-source predictions for every scored station-hour."""
    rows = []
    episodes = list(forcing[forcing["split"] == split].dropna(subset=["episode_id"]).groupby("episode_id", sort=True))
    if not episodes:
        raise RuntimeError(f"No eligible {split} episodes found.")
    print(f"Running {len(episodes)} {split} episodes.", flush=True)
    for episode_number, (episode_id, block) in enumerate(episodes, start=1):
        block = block.sort_values("time")
        print(f"  [{episode_number}/{len(episodes)}] {episode_id}: {block.time.iloc[0]} to {block.time.iloc[-1]}", flush=True)
        fields = np.stack((np.full((50, 50), float(block.iloc[0].pm25_inflow_ug_m3)), np.zeros((50, 50))))
        sources = np.stack((np.zeros((50, 50)), template))
        for _, forcing_row in block.iterrows():
            config = ADRConfig(
                50_000.0, 50_000.0, 50, 50, d,
                kappa * float(forcing_row.u10_m_s),
                kappa * float(forcing_row.v10_m_s),
                lam,
            )
            fields, residual = advance_adr_implicit_many(
                config, fields, sources,
                np.array([float(forcing_row.pm25_inflow_ug_m3), 0.0]),
                DT_S,
            )
            if not bool(forcing_row.score_hour):
                continue
            observed = observations[observations.hour == forcing_row.time].merge(
                station_map[["easting_m", "northing_m"]],
                left_on="station_id", right_index=True,
            )
            if len(observed) < 20:
                raise RuntimeError(f"Eligibility changed at {forcing_row.time}: {len(observed)} stations.")
            item = observed[["hour", "station_id", "station_name", "pm25_hourly"]].copy()
            item["episode_id"] = episode_id
            item["background_pm25"] = interpolate_to_station_coordinates(
                fields[0], config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy()
            )
            item["unit_source_response"] = interpolate_to_station_coordinates(
                fields[1], config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy()
            )
            item["relative_residual"] = residual
            rows.append(item)
    return pd.concat(rows, ignore_index=True)


def metrics(prediction: pd.DataFrame) -> dict[str, float]:
    error = prediction["model_pm25"] - prediction["pm25_hourly"]
    return {
        "n_scored_station_hours": int(len(prediction)),
        "rmse_ug_m3": float(np.sqrt(np.mean(error**2))),
        "mae_ug_m3": float(np.mean(np.abs(error))),
        "objective_log_mse": float(np.mean((np.log(prediction["model_pm25"] + LOG_OFFSET) - np.log(prediction["pm25_hourly"] + LOG_OFFSET)) ** 2)),
        "station_hour_pearson_r": float(prediction[["model_pm25", "pm25_hourly"]].corr().iloc[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parameters", nargs=3, type=float, required=True, metavar=("D", "LAMBDA", "KAPPA"))
    args = parser.parse_args()
    d, lam, kappa = args.parameters
    if d <= 0 or lam <= 0 or not 0 < kappa <= 1:
        raise ValueError("Require D > 0, LAMBDA > 0, and 0 < KAPPA <= 1.")

    processed, output = Path(args.processed_dir), Path(args.output_dir)
    forcing = pd.read_csv(processed / "transient_hourly_episode_panel.csv", parse_dates=["time"])
    observations = pd.read_csv(processed / "pm25_hourly_clean.csv", parse_dates=["hour"])
    stations = pd.read_csv(processed / "station_metadata.csv")
    east, north = Transformer.from_crs(4326, 32643, always_xy=True).transform(
        stations.longitude.to_numpy(), stations.latitude.to_numpy()
    )
    station_map = stations.assign(easting_m=east, northing_m=north).set_index("station_id")
    template = load_inventory_template(processed / "edgar_pm25_inventory_template.csv")
    print(f"Frozen physical candidate: D={d:g}, lambda={lam:.6g}, kappa={kappa:g}", flush=True)

    calibration = simulate_split(forcing, observations, station_map, template, "calibration", d, lam, kappa)
    calibration["date"] = calibration["hour"].dt.date.astype(str)
    daily_q_rows = []
    for date, day in calibration.groupby("date", sort=True):
        q, loss = profile_q(day.background_pm25.to_numpy(), day.unit_source_response.to_numpy(), day.pm25_hourly.to_numpy())
        daily_q_rows.append({"date": date, "source_scale_q": q, "daily_log_mse": loss, "n_station_hours": int(len(day))})
    daily_q = pd.DataFrame(daily_q_rows)
    frozen_q = float(daily_q["source_scale_q"].median())
    calibration = calibration.merge(daily_q[["date", "source_scale_q"]], on="date", how="left")
    calibration["model_pm25"] = calibration["background_pm25"] + calibration["source_scale_q"] * calibration["unit_source_response"]

    # Write the completed calibration stage before beginning the separate holdout.
    output.mkdir(parents=True, exist_ok=True)
    daily_q.to_csv(output / "calibration_daily_source_scales.csv", index=False)
    calibration.to_csv(output / "calibration_daily_scaled_predictions.csv", index=False)

    print(f"Frozen calibration-median source scale for internal assessment: {frozen_q:.8g}", flush=True)
    internal = simulate_split(forcing, observations, station_map, template, "internal_validation", d, lam, kappa)
    internal["source_scale_q"] = frozen_q
    internal["model_pm25"] = internal["background_pm25"] + frozen_q * internal["unit_source_response"]

    internal.to_csv(output / "internal_frozen_source_predictions.csv", index=False)
    record = {
        "purpose": "no-leakage daily-source evaluation; conditional ADR assessment, not forecasting or policy-effect estimation",
        "physical_parameters_fixed_before_internal_assessment": {"diffusivity_m2_s": d, "removal_s_inv": lam, "wind_attenuation": kappa},
        "calibration": {**metrics(calibration), "n_daily_source_scales": int(len(daily_q))},
        "frozen_source_rule": {
            "rule": "median of calendar-day source scales profiled from Nov-Dec 2022 calibration observations only",
            "median_source_scale_q": frozen_q,
            "p05_source_scale_q": float(daily_q.source_scale_q.quantile(0.05)),
            "p95_source_scale_q": float(daily_q.source_scale_q.quantile(0.95)),
        },
        "internal_assessment": {
            **metrics(internal),
            "source_scale_q": frozen_q,
            "refit_on_internal_observations": False,
        },
    }
    (output / "no_leakage_daily_source_evaluation.json").write_text(json.dumps(record, indent=2))
    print("\nFINAL NO-LEAKAGE RECORD", flush=True)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
