"""Calibrate shared ADR transport parameters on Nov-Dec 2022 station-days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.optimize import differential_evolution, minimize_scalar

from src.daily_adr import DailyADRParameters, interpolate_to_station_coordinates, load_inventory_template, run_daily_adr_inventory


CAL_START, CAL_END = "2022-11-01", "2022-12-31"
LOG_OFFSET = 5.0
INITIAL_BOUNDS = [(np.log(100.0), np.log(2000.0)), (np.log(1 / (96 * 3600)), np.log(1 / (6 * 3600))), (0.20, 1.00)]
EXPANDED_BOUNDS = [(np.log(100.0), np.log(5000.0)), (np.log(1 / (96 * 3600)), np.log(1 / (6 * 3600))), (0.05, 1.00)]


def profile_daily_source(background: np.ndarray, response: np.ndarray, observed: np.ndarray) -> tuple[float, float]:
    """Profile q >= 0 under the pre-registered logarithmic loss."""
    def loss(q: float) -> float:
        return float(np.mean((np.log(background + q * response + LOG_OFFSET) - np.log(observed + LOG_OFFSET)) ** 2))

    upper = 1e-8
    previous = loss(0.0)
    current = loss(upper)
    while current < previous and upper < 1.0:
        previous, upper = current, upper * 10.0
        current = loss(upper)
    result = minimize_scalar(loss, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-10})
    return float(result.x), float(result.fun)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--fixed-parameters", nargs=3, type=float, metavar=("D", "LAMBDA", "KAPPA"))
    parser.add_argument("--expanded-bounds", action="store_true")
    args = parser.parse_args()
    processed, output = Path(args.processed_dir), Path(args.output_dir)
    forcing = pd.read_csv(processed / "directional_boundary_forcing.csv", parse_dates=["date"])
    forcing = forcing[(forcing.date >= CAL_START) & (forcing.date <= CAL_END) & forcing.usable_boundary_forcing].copy()
    observations = pd.read_csv(processed / "pm25_daily_panel.csv", parse_dates=["date"])
    observations = observations[(observations.date >= CAL_START) & (observations.date <= CAL_END)]
    stations = pd.read_csv(processed / "station_metadata.csv")
    transformer = Transformer.from_crs(4326, 32643, always_xy=True)
    east, north = transformer.transform(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    station_map = stations.assign(easting_m=east, northing_m=north).set_index("station_id")
    inventory = load_inventory_template(processed / "edgar_pm25_inventory_template.csv")
    day_data: list[tuple[dict[str, object], pd.DataFrame]] = []
    for _, force in forcing.iterrows():
        observed = observations[observations.date == force.date].merge(station_map[["easting_m", "northing_m"]], left_on="station_id", right_index=True)
        if len(observed) >= 20:
            day_data.append((force.to_dict(), observed))
    if not day_data:
        raise RuntimeError("No usable calibration station-days.")

    trace: list[dict[str, float]] = []
    def objective(z: np.ndarray, retain: bool = False):
        parameters0 = DailyADRParameters(float(np.exp(z[0])), float(np.exp(z[1])), float(z[2]), 0.0, 1.0)
        losses, details = [], []
        for force, observed in day_data:
            background, config, residual0 = run_daily_adr_inventory(force, inventory, parameters0)
            parameters1 = DailyADRParameters(parameters0.diffusivity_m2_s, parameters0.removal_s_inv, parameters0.wind_attenuation, 1.0, 1.0)
            unit_field, _, residual1 = run_daily_adr_inventory(force, inventory, parameters1)
            b = interpolate_to_station_coordinates(background, config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
            response = interpolate_to_station_coordinates(unit_field - background, config, observed.easting_m.to_numpy(), observed.northing_m.to_numpy())
            q, loss = profile_daily_source(b, response, observed.pm25_daily.to_numpy())
            losses.extend((np.log(b + q * response + LOG_OFFSET) - np.log(observed.pm25_daily.to_numpy() + LOG_OFFSET)) ** 2)
            if retain:
                item = observed[["date", "station_id", "station_name", "pm25_daily"]].copy()
                item["model_pm25"] = b + q * response
                item["source_scale_q"] = q
                item["relative_residual"] = max(residual0, residual1)
                details.append(item)
        value = float(np.mean(losses))
        if not retain:
            trace.append({"diffusivity_m2_s": parameters0.diffusivity_m2_s, "removal_s_inv": parameters0.removal_s_inv, "wind_attenuation": parameters0.wind_attenuation, "objective": value})
            return value
        return value, pd.concat(details, ignore_index=True)

    bounds = EXPANDED_BOUNDS if args.expanded_bounds else INITIAL_BOUNDS
    if args.fixed_parameters:
        d, lam, kappa = args.fixed_parameters
        result_x = np.array([np.log(d), np.log(lam), kappa])
        result_success, result_message = True, "fixed candidate evaluation"
    else:
        result = differential_evolution(objective, bounds, seed=args.seed, maxiter=args.maxiter, popsize=args.popsize, polish=not args.no_polish, workers=1)
        result_x, result_success, result_message = result.x, bool(result.success), str(result.message)
    final_loss, predictions = objective(result_x, retain=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace).to_csv(output / "calibration_optimisation_trace.csv", index=False)
    predictions.to_csv(output / "calibration_station_day_predictions.csv", index=False)
    parameter_record = {
        "calibration_window": [CAL_START, CAL_END], "source_template": "EDGAR all-sector PM2.5 inventory",
        "n_weather_boundary_days": len(day_data), "n_station_days": len(predictions),
        "objective_log_mse": final_loss, "diffusivity_m2_s": float(np.exp(result_x[0])),
        "removal_s_inv": float(np.exp(result_x[1])), "wind_attenuation": float(result_x[2]),
        "optimiser": "scipy.differential_evolution", "seed": args.seed,
        "bounds_phase": "expanded" if args.expanded_bounds else "initial",
        "success": result_success, "message": result_message,
    }
    (output / "calibrated_parameters_edgar_inventory.json").write_text(json.dumps(parameter_record, indent=2))
    print(json.dumps(parameter_record, indent=2))


if __name__ == "__main__":
    main()
