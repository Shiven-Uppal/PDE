"""Calibrate shared transient ADR parameters on locked hourly episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import differential_evolution, minimize_scalar

from src.adr_solver import ADRConfig
from src.daily_adr import interpolate_to_station_coordinates, load_inventory_template
from src.transient_adr import advance_adr_implicit_many

LOG_OFFSET = 5.0
DT_S = 3600.0


def profile_q(background: np.ndarray, response: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    def loss(q: float) -> float:
        return float(np.mean((np.log(background + q * response + LOG_OFFSET) - np.log(observed + LOG_OFFSET)) ** 2))
    upper, previous, current = 1e-8, loss(0.0), loss(1e-8)
    while current < previous and upper < 1.0:
        previous, upper = current, upper * 10.0
        current = loss(upper)
    fitted = minimize_scalar(loss, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    return float(fitted.x), float(fitted.fun)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maxiter", type=int, default=4)
    parser.add_argument("--popsize", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--fixed-parameters", nargs=3, type=float, metavar=("D", "LAMBDA", "KAPPA"))
    args = parser.parse_args()
    processed, output = Path(args.processed_dir), Path(args.output_dir)
    forcing = pd.read_csv(processed / "transient_hourly_episode_panel.csv", parse_dates=["time"])
    forcing = forcing[(forcing.split == "calibration") & forcing.episode_id.notna()].copy()
    observations = pd.read_csv(processed / "pm25_hourly_clean.csv", parse_dates=["hour"])
    observations = observations[observations.date.between("2022-11-01", "2022-12-31")]
    stations = pd.read_csv(processed / "station_metadata.csv")
    east, north = Transformer.from_crs(4326, 32643, always_xy=True).transform(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    station_map = stations.assign(easting_m=east, northing_m=north).set_index("station_id")
    template = load_inventory_template(processed / "edgar_pm25_inventory_template.csv")
    episode_data = []
    for episode_id, block in forcing.groupby("episode_id", sort=True):
        block = block.sort_values("time").copy()
        hourly_observed = {}
        for time in block.time:
            observed = observations[observations.hour == time].merge(
                station_map[["easting_m", "northing_m"]], left_on="station_id", right_index=True
            )
            if len(observed) < 20:
                raise RuntimeError(f"Episode eligibility changed at {time}")
            hourly_observed[time] = observed
        episode_data.append((episode_id, block, hourly_observed))
    if not episode_data:
        raise RuntimeError("No calibration episodes.")

    trace = []
    def evaluate(z: np.ndarray, retain: bool = False):
        d, lam, kappa = float(np.exp(z[0])), float(np.exp(z[1])), float(z[2])
        all_loss, detail, episode_q = [], [], []
        for episode_id, block, hourly_observed in episode_data:
            first_boundary = float(block.iloc[0].pm25_inflow_ug_m3)
            fields = np.stack((np.full((50, 50), first_boundary), np.zeros((50, 50))))
            sources = np.stack((np.zeros((50, 50)), template))
            episode_rows = []
            max_residual = 0.0
            for _, row in block.iterrows():
                config = ADRConfig(50_000.0, 50_000.0, 50, 50, d,
                                   kappa * float(row.u10_m_s), kappa * float(row.v10_m_s), lam)
                fields, residual = advance_adr_implicit_many(
                    config, fields, sources, np.array([float(row.pm25_inflow_ug_m3), 0.0]), DT_S
                )
                max_residual = max(max_residual, residual)
                if not bool(row.score_hour):
                    continue
                observed = hourly_observed[row.time]
                background = interpolate_to_station_coordinates(fields[0], config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
                response = interpolate_to_station_coordinates(fields[1], config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
                item = observed[["hour", "station_id", "station_name", "pm25_hourly"]].copy()
                item["background_pm25"] = background
                item["unit_source_response"] = response
                item["episode_id"] = episode_id
                item["relative_residual"] = residual
                episode_rows.append(item)
            joined = pd.concat(episode_rows, ignore_index=True)
            q, loss = profile_q(joined.background_pm25.to_numpy(), joined.unit_source_response.to_numpy(), joined.pm25_hourly.to_numpy())
            all_loss.append(loss)
            episode_q.append({"episode_id": episode_id, "source_scale_q": q, "episode_log_mse": loss,
                              "n_station_hours": len(joined), "max_relative_residual": max_residual})
            if retain:
                joined["model_pm25"] = joined.background_pm25 + q * joined.unit_source_response
                joined["source_scale_q"] = q
                detail.append(joined)
        value = float(np.average(all_loss, weights=[item["n_station_hours"] for item in episode_q]))
        if not retain:
            trace.append({"diffusivity_m2_s": d, "removal_s_inv": lam, "wind_attenuation": kappa, "objective_log_mse": value})
            return value
        return value, pd.concat(detail, ignore_index=True), pd.DataFrame(episode_q)

    bounds = [(np.log(100.0), np.log(5000.0)), (np.log(1 / (96 * 3600)), np.log(1 / (4 * 3600))), (0.05, 1.0)]
    if args.fixed_parameters:
        d, lam, kappa = args.fixed_parameters
        result_x, success, message = np.array([np.log(d), np.log(lam), kappa]), True, "fixed candidate evaluation"
    else:
        result = differential_evolution(evaluate, bounds, seed=args.seed, maxiter=args.maxiter, popsize=args.popsize, polish=False, workers=1)
        result_x, success, message = result.x, bool(result.success), str(result.message)
    final_loss, predictions, q_table = evaluate(result_x, retain=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(output / "transient_optimisation_trace.csv", index=False)
    predictions.to_csv(output / "transient_calibration_station_hour_predictions.csv", index=False)
    q_table.to_csv(output / "transient_episode_source_scales.csv", index=False)
    record = {"split": "calibration", "n_episodes": len(episode_data), "n_scored_station_hours": len(predictions),
              "objective_log_mse": final_loss, "diffusivity_m2_s": float(np.exp(result_x[0])),
              "removal_s_inv": float(np.exp(result_x[1])), "wind_attenuation": float(result_x[2]),
              "source": "fixed EDGAR template with one profiled constant q per episode", "success": success, "message": message}
    (output / "transient_calibrated_parameters.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
