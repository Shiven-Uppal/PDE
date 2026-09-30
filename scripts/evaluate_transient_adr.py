"""Evaluate a fixed transient ADR candidate on one locked episode split."""

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

DT_S, LOG_OFFSET = 3600.0, 5.0


def profile_q(background: np.ndarray, response: np.ndarray, observed: np.ndarray) -> float:
    def loss(q: float) -> float:
        return float(np.mean((np.log(background + q * response + LOG_OFFSET) - np.log(observed + LOG_OFFSET)) ** 2))
    upper, previous, current = 1e-8, loss(0.0), loss(1e-8)
    while current < previous and upper < 1.0:
        previous, upper = current, upper * 10.0
        current = loss(upper)
    return float(minimize_scalar(loss, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10}).x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["internal_validation", "external_sealed"], required=True)
    parser.add_argument("--diffusivity", type=float, required=True)
    parser.add_argument("--removal", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    args = parser.parse_args()
    if args.split == "external_sealed":
        raise RuntimeError("External-sealed data may not be evaluated during model development.")
    processed, output = Path(args.processed_dir), Path(args.output_dir)
    forcing = pd.read_csv(processed / "transient_hourly_episode_panel.csv", parse_dates=["time"])
    forcing = forcing[(forcing.split == args.split) & forcing.episode_id.notna()].copy()
    observations = pd.read_csv(processed / "pm25_hourly_clean.csv", parse_dates=["hour"])
    stations = pd.read_csv(processed / "station_metadata.csv")
    east, north = Transformer.from_crs(4326, 32643, always_xy=True).transform(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    station_map = stations.assign(easting_m=east, northing_m=north).set_index("station_id")
    template = load_inventory_template(processed / "edgar_pm25_inventory_template.csv")
    predictions, q_rows = [], []
    for episode_id, block in forcing.groupby("episode_id", sort=True):
        block = block.sort_values("time")
        fields = np.stack((np.full((50, 50), float(block.iloc[0].pm25_inflow_ug_m3)), np.zeros((50, 50))))
        sources = np.stack((np.zeros((50, 50)), template))
        rows = []
        for _, row in block.iterrows():
            cfg = ADRConfig(50_000, 50_000, 50, 50, args.diffusivity, args.kappa * row.u10_m_s, args.kappa * row.v10_m_s, args.removal)
            fields, residual = advance_adr_implicit_many(cfg, fields, sources, np.array([row.pm25_inflow_ug_m3, 0.0]), DT_S)
            if not row.score_hour:
                continue
            observed = observations[observations.hour == row.time].merge(station_map[["easting_m", "northing_m"]], left_on="station_id", right_index=True)
            b = interpolate_to_station_coordinates(fields[0], cfg, observed.easting_m, observed.northing_m)
            r = interpolate_to_station_coordinates(fields[1], cfg, observed.easting_m, observed.northing_m)
            item = observed[["hour", "station_id", "station_name", "pm25_hourly"]].copy()
            item["background_pm25"] = b; item["unit_source_response"] = r; item["episode_id"] = episode_id; item["relative_residual"] = residual
            rows.append(item)
        item = pd.concat(rows, ignore_index=True)
        q = profile_q(item.background_pm25.to_numpy(), item.unit_source_response.to_numpy(), item.pm25_hourly.to_numpy())
        item["source_scale_q"] = q; item["model_pm25"] = item.background_pm25 + q * item.unit_source_response
        predictions.append(item); q_rows.append({"episode_id": episode_id, "source_scale_q": q, "n_station_hours": len(item)})
    pred = pd.concat(predictions, ignore_index=True)
    error = pred.model_pm25 - pred.pm25_hourly
    metrics = {"split": args.split, "n_episodes": int(pred.episode_id.nunique()), "n_station_hours": len(pred),
               "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))), "mean_bias": float(np.mean(error)),
               "pearson_correlation": float(pred.model_pm25.corr(pred.pm25_hourly)), "max_relative_residual": float(pred.relative_residual.max()),
               "diffusivity_m2_s": args.diffusivity, "removal_s_inv": args.removal, "wind_attenuation": args.kappa,
               "source_scale_note": "profiled per validation episode; this evaluates conditional spatial-temporal response, not source-magnitude forecasting"}
    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / "transient_validation_station_hour_predictions.csv", index=False)
    pd.DataFrame(q_rows).to_csv(output / "transient_validation_episode_source_scales.csv", index=False)
    (output / "transient_validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
