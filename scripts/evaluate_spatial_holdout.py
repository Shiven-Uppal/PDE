"""Evaluate a fixed ADR parameter set on an untouched daily spatial window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize_scalar

from src.daily_adr import DailyADRParameters, interpolate_to_station_coordinates, load_inventory_template, run_daily_adr_inventory


def profile_q(background: np.ndarray, response: np.ndarray, observed: np.ndarray) -> float:
    def loss(q: float) -> float:
        return float(np.mean((np.log(background + q * response + 5.0) - np.log(observed + 5.0)) ** 2))
    upper, previous, current = 1e-8, loss(0.0), loss(1e-8)
    while current < previous and upper < 1.0:
        previous, upper = current, upper * 10.0
        current = loss(upper)
    return float(minimize_scalar(loss, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10}).x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--diffusivity", type=float, required=True)
    parser.add_argument("--removal", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    args = parser.parse_args()
    processed, output = Path(args.processed_dir), Path(args.output_dir)
    forcing = pd.read_csv(processed / "directional_boundary_forcing.csv", parse_dates=["date"])
    forcing = forcing[(forcing.date >= args.start) & (forcing.date <= args.end) & forcing.usable_boundary_forcing]
    observations = pd.read_csv(processed / "pm25_daily_panel.csv", parse_dates=["date"])
    stations = pd.read_csv(processed / "station_metadata.csv")
    east, north = Transformer.from_crs(4326, 32643, always_xy=True).transform(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    station_map = stations.assign(easting_m=east, northing_m=north).set_index("station_id")
    inventory = load_inventory_template(processed / "edgar_pm25_inventory_template.csv")
    parameters0 = DailyADRParameters(args.diffusivity, args.removal, args.kappa, 0.0, 1.0)
    parameters1 = DailyADRParameters(args.diffusivity, args.removal, args.kappa, 1.0, 1.0)
    rows = []
    for _, force in forcing.iterrows():
        observed = observations[(observations.date == force.date)].merge(station_map[["easting_m", "northing_m"]], left_on="station_id", right_index=True)
        if len(observed) < 20:
            continue
        base, config, r0 = run_daily_adr_inventory(force.to_dict(), inventory, parameters0)
        unit, _, r1 = run_daily_adr_inventory(force.to_dict(), inventory, parameters1)
        b = interpolate_to_station_coordinates(base, config, observed.easting_m, observed.northing_m)
        response = interpolate_to_station_coordinates(unit - base, config, observed.easting_m, observed.northing_m)
        q = profile_q(b, response, observed.pm25_daily.to_numpy())
        item = observed[["date", "station_id", "station_name", "pm25_daily"]].copy()
        item["model_pm25"] = b + q * response
        item["source_scale_q"] = q
        item["relative_residual"] = max(r0, r1)
        rows.append(item)
    pred = pd.concat(rows, ignore_index=True)
    error = pred.model_pm25 - pred.pm25_daily
    metrics = {"window": [args.start, args.end], "n_station_days": len(pred), "n_days": int(pred.date.nunique()),
               "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
               "mean_bias": float(np.mean(error)), "pearson_correlation": float(pred.model_pm25.corr(pred.pm25_daily)),
               "max_relative_residual": float(pred.relative_residual.max()), "diffusivity_m2_s": args.diffusivity,
               "removal_s_inv": args.removal, "wind_attenuation": args.kappa,
               "source_template": "EDGAR all-sector PM2.5 inventory"}
    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / "holdout_station_day_predictions.csv", index=False)
    (output / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
