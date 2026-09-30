"""Fit a regularised, low-dimensional EDGAR sector-ratio ADR model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import minimize, minimize_scalar

from src.daily_adr import (
    DailyADRParameters,
    combine_inventory_sector_templates,
    interpolate_to_station_coordinates,
    load_inventory_sector_templates,
    run_daily_adr_inventory,
)

LOG_OFFSET = 5.0
ADJUSTABLE = ("industry", "residential", "transport")


def profile_q(background: np.ndarray, response: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    def loss(q: float) -> float:
        return float(np.mean((np.log(background + q * response + LOG_OFFSET) - np.log(observed + LOG_OFFSET)) ** 2))
    upper, previous, current = 1e-8, loss(0.0), loss(1e-8)
    while current < previous and upper < 1.0:
        previous, upper = current, upper * 10.0
        current = loss(upper)
    result = minimize_scalar(loss, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    return float(result.x), float(result.fun)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--penalty", type=float, required=True)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--initial-diffusivity", type=float, default=2731.9038610353705)
    parser.add_argument("--initial-removal", type=float, default=6.5781318640610365e-06)
    parser.add_argument("--initial-kappa", type=float, default=0.06553313888203532)
    args = parser.parse_args()
    processed, output = Path(args.processed_dir), Path(args.output_dir)
    forcing = pd.read_csv(processed / "directional_boundary_forcing.csv", parse_dates=["date"])
    forcing = forcing[(forcing.date >= args.start) & (forcing.date <= args.end) & forcing.usable_boundary_forcing].copy()
    observations = pd.read_csv(processed / "pm25_daily_panel.csv", parse_dates=["date"])
    observations = observations[(observations.date >= args.start) & (observations.date <= args.end)]
    stations = pd.read_csv(processed / "station_metadata.csv")
    east, north = Transformer.from_crs(4326, 32643, always_xy=True).transform(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    station_map = stations.assign(easting_m=east, northing_m=north).set_index("station_id")
    sectors = load_inventory_sector_templates(processed / "edgar_pm25_inventory_template.csv")
    day_data = []
    for _, force in forcing.iterrows():
        observed = observations[observations.date == force.date].merge(station_map[["easting_m", "northing_m"]], left_on="station_id", right_index=True)
        if len(observed) >= 20:
            day_data.append((force.to_dict(), observed))
    if not day_data:
        raise RuntimeError("No usable station-days in the requested window.")

    # z = (log D, log lambda, kappa, log industry, log residential, log transport)
    x0 = np.array([np.log(args.initial_diffusivity), np.log(args.initial_removal), args.initial_kappa, 0.0, 0.0, 0.0])
    bounds = [
        (np.log(100.0), np.log(5000.0)),
        (np.log(1 / (96 * 3600)), np.log(1 / (6 * 3600))),
        (0.05, 1.0),
        *( (np.log(0.25), np.log(4.0)), ) * 3,
    ]
    trace = []

    def objective(z: np.ndarray, retain: bool = False):
        multipliers = dict(zip(ADJUSTABLE, np.exp(z[3:])))
        template = combine_inventory_sector_templates(sectors, multipliers)
        p0 = DailyADRParameters(float(np.exp(z[0])), float(np.exp(z[1])), float(z[2]), 0.0, 1.0)
        p1 = DailyADRParameters(p0.diffusivity_m2_s, p0.removal_s_inv, p0.wind_attenuation, 1.0, 1.0)
        errors, detail = [], []
        for force, observed in day_data:
            base, config, r0 = run_daily_adr_inventory(force, template, p0)
            unit, _, r1 = run_daily_adr_inventory(force, template, p1)
            b = interpolate_to_station_coordinates(base, config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
            response = interpolate_to_station_coordinates(unit - base, config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
            q, _ = profile_q(b, response, observed.pm25_daily.to_numpy())
            errors.extend((np.log(b + q * response + LOG_OFFSET) - np.log(observed.pm25_daily.to_numpy() + LOG_OFFSET)) ** 2)
            if retain:
                item = observed[["date", "station_id", "station_name", "pm25_daily"]].copy()
                item["model_pm25"] = b + q * response
                item["source_scale_q"] = q
                item["relative_residual"] = max(r0, r1)
                detail.append(item)
        log_mse = float(np.mean(errors))
        penalty = float(args.penalty * np.sum(z[3:] ** 2))
        if retain:
            return log_mse + penalty, log_mse, pd.concat(detail, ignore_index=True), multipliers
        value = log_mse + penalty
        trace.append({"objective": value, "log_mse": log_mse, "penalty_term": penalty,
                      "diffusivity_m2_s": p0.diffusivity_m2_s, "removal_s_inv": p0.removal_s_inv,
                      "wind_attenuation": p0.wind_attenuation, **multipliers})
        return value

    # The PDE response is evaluated through a profiled daily source scale.  A
    # visible finite-difference step is therefore required for the sector-ratio
    # directions; SciPy's machine-scale default can incorrectly look flat.
    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-8, "eps": 0.02},
    )
    value, log_mse, predictions, multipliers = objective(result.x, retain=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(output / "sector_weight_optimisation_trace.csv", index=False)
    predictions.to_csv(output / "sector_weight_station_day_predictions.csv", index=False)
    record = {
        "window": [args.start, args.end], "n_weather_boundary_days": len(day_data),
        "n_station_days": len(predictions), "penalty_eta": args.penalty,
        "objective_penalised": value, "objective_log_mse": log_mse,
        "diffusivity_m2_s": float(np.exp(result.x[0])), "removal_s_inv": float(np.exp(result.x[1])),
        "wind_attenuation": float(result.x[2]), "sector_multipliers": multipliers,
        "method": "L-BFGS-B", "maxiter": args.maxiter, "success": bool(result.success), "message": str(result.message),
    }
    (output / "sector_weight_parameters.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
