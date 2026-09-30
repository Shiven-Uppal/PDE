"""Run a pre-specified conditional uncertainty experiment for the transient ADR solver.

This script deliberately uses one pre-selected calibration episode only.  It does
not estimate forecast performance or a real-world policy effect.  Each Latin-
hypercube draw changes transport/removal and two model inputs (inflow and source
scale), then propagates the resulting field for a fixed 24-hour horizon.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.special import ndtri
from scipy.stats import qmc, spearmanr

from src.adr_solver import ADRConfig
from src.daily_adr import interpolate_to_station_coordinates, load_inventory_template
from src.transient_adr import advance_adr_implicit_many


DT_S = 3600.0


def _log_uniform(q: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.exp(np.log(lower) + q * (np.log(upper) - np.log(lower)))


def _lognormal(q: np.ndarray, median: float, log_sd: float) -> np.ndarray:
    # Clip guards against round-off at 0 or 1 before inverse-normal transform.
    return median * np.exp(log_sd * ndtri(np.clip(q, 1e-12, 1 - 1e-12)))


def _select_reference_episode(forcing: pd.DataFrame, summary: pd.DataFrame, horizon: int) -> pd.DataFrame:
    candidates = summary[summary.split.eq("calibration")].sort_values(
        ["n_hours", "episode_id"], ascending=[False, True]
    )
    for episode_id in candidates.episode_id:
        block = forcing[(forcing.split == "calibration") & (forcing.episode_id == episode_id)].sort_values("time")
        if len(block) >= horizon:
            return block.iloc[:horizon].copy()
    raise RuntimeError("No calibration episode has the required 24-hour horizon.")


def _draw_samples(spec: dict) -> pd.DataFrame:
    p = spec["sampling"]
    lhs = qmc.LatinHypercube(d=5, seed=20260902).random(p["n_samples"])
    return pd.DataFrame({
        "sample_id": np.arange(1, p["n_samples"] + 1),
        "diffusivity_m2_s": _log_uniform(lhs[:, 0], p["diffusivity_m2_s"]["lower"], p["diffusivity_m2_s"]["upper"]),
        "removal_s_inv": _log_uniform(lhs[:, 1], p["removal_s_inv"]["lower"], p["removal_s_inv"]["upper"]),
        "wind_attenuation": p["wind_attenuation"]["lower"] + lhs[:, 2] * (p["wind_attenuation"]["upper"] - p["wind_attenuation"]["lower"]),
        "inflow_multiplier": _lognormal(lhs[:, 3], p["inflow_multiplier"]["median"], p["inflow_multiplier"]["log_sd"]),
        "source_multiplier": _lognormal(lhs[:, 4], p["source_multiplier"]["median"], p["source_multiplier"]["log_sd"]),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--source-scales", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    processed, output = Path(args.processed_dir), Path(args.output_dir)
    spec = json.loads(Path(args.spec).read_text())
    horizon = int(spec["simulation_horizon_hours"])
    forcing = pd.read_csv(processed / "transient_hourly_episode_panel.csv", parse_dates=["time"])
    summary = pd.read_csv(processed / "transient_episode_summary.csv")
    episode = _select_reference_episode(forcing, summary, horizon)
    q_ref = float(pd.read_csv(args.source_scales).source_scale_q.median())
    template = load_inventory_template(processed / "edgar_pm25_inventory_template.csv")
    samples = _draw_samples(spec)

    stations = pd.read_csv(processed / "station_metadata.csv")
    east, north = Transformer.from_crs(4326, 32643, always_xy=True).transform(
        stations.longitude.to_numpy(), stations.latitude.to_numpy()
    )
    station_xy = stations.assign(easting_m=east, northing_m=north)
    scenario_factors = np.array(spec["conditional_scenarios"]["source_multipliers"], dtype=float)
    n_scenarios = len(scenario_factors)
    outcomes, station_outcomes, max_residual = [], [], 0.0

    # The linear PDE permits superposition: solve the background and one unit
    # local-source response together, then form all four source scalings without
    # changing the governing equation or running a separate solver per scenario.
    for idx, draw in samples.iterrows():
        fields = np.stack((np.full((50, 50), float(episode.iloc[0].pm25_inflow_ug_m3) * draw.inflow_multiplier), np.zeros((50, 50))))
        sources = np.stack((np.zeros((50, 50)), template))
        mean_accumulator = np.zeros(n_scenarios)
        for _, hour in episode.iterrows():
            cfg = ADRConfig(50_000.0, 50_000.0, 50, 50, float(draw.diffusivity_m2_s),
                            float(draw.wind_attenuation * hour.u10_m_s),
                            float(draw.wind_attenuation * hour.v10_m_s), float(draw.removal_s_inv))
            fields, residual = advance_adr_implicit_many(
                cfg, fields, sources, np.array([float(hour.pm25_inflow_ug_m3) * draw.inflow_multiplier, 0.0]), DT_S
            )
            max_residual = max(max_residual, residual)
            field_stack = fields[0][None, :, :] + (q_ref * draw.source_multiplier * scenario_factors)[:, None, None] * fields[1][None, :, :]
            mean_accumulator += field_stack.mean(axis=(1, 2))
        final_stack = fields[0][None, :, :] + (q_ref * draw.source_multiplier * scenario_factors)[:, None, None] * fields[1][None, :, :]
        final_station = np.vstack([
            interpolate_to_station_coordinates(final_stack[s], cfg, station_xy.easting_m, station_xy.northing_m)
            for s in range(n_scenarios)
        ])
        for s, factor in enumerate(scenario_factors):
            outcomes.append({**draw.to_dict(), "scenario_source_factor": factor,
                             "final_domain_mean_ug_m3": float(final_stack[s].mean()),
                             "horizon_mean_domain_ug_m3": float(mean_accumulator[s] / horizon),
                             "final_domain_max_ug_m3": float(final_stack[s].max())})
            station_outcomes.extend({"sample_id": int(draw.sample_id), "scenario_source_factor": factor,
                                     "station_id": row.station_id, "station_name": row.station_name,
                                     "final_station_sample_ug_m3": float(final_station[s, j])}
                                    for j, (_, row) in enumerate(station_xy.iterrows()))
        if (idx + 1) % 50 == 0:
            print(f"Completed {idx + 1}/{len(samples)} conditional draws", flush=True)

    results = pd.DataFrame(outcomes)
    station_results = pd.DataFrame(station_outcomes)
    baseline = results[results.scenario_source_factor.eq(1.0)].set_index("sample_id")
    summaries = []
    for factor, block in results.groupby("scenario_source_factor", sort=False):
        response = 100 * (1 - block.set_index("sample_id").final_domain_mean_ug_m3 / baseline.final_domain_mean_ug_m3)
        summaries.append({"scenario_source_factor": factor, "n_samples": len(block),
                          "final_domain_mean_median": float(block.final_domain_mean_ug_m3.median()),
                          "final_domain_mean_p05": float(block.final_domain_mean_ug_m3.quantile(.05)),
                          "final_domain_mean_p95": float(block.final_domain_mean_ug_m3.quantile(.95)),
                          "horizon_domain_mean_median": float(block.horizon_mean_domain_ug_m3.median()),
                          "relative_final_response_median_percent": float(response.median()),
                          "relative_final_response_p05_percent": float(response.quantile(.05)),
                          "relative_final_response_p95_percent": float(response.quantile(.95))})
    summary_df = pd.DataFrame(summaries)
    sensitivity = []
    for parameter in ["diffusivity_m2_s", "removal_s_inv", "wind_attenuation", "inflow_multiplier", "source_multiplier"]:
        rho, pvalue = spearmanr(baseline[parameter], baseline.final_domain_mean_ug_m3)
        sensitivity.append({"parameter": parameter, "spearman_rho_with_baseline_final_domain_mean": float(rho), "two_sided_p_value": float(pvalue), "absolute_rho": float(abs(rho))})
    sensitivity_df = pd.DataFrame(sensitivity).sort_values("absolute_rho", ascending=False)

    output.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output / "lhs_input_samples.csv", index=False)
    results.to_csv(output / "conditional_scenario_outcomes.csv", index=False)
    station_results.to_csv(output / "conditional_station_samples.csv", index=False)
    summary_df.to_csv(output / "conditional_scenario_summary.csv", index=False)
    sensitivity_df.to_csv(output / "rank_sensitivity.csv", index=False)
    manifest = {"analysis_type": spec["analysis_type"], "selected_episode_id": str(episode.iloc[0].episode_id),
                "episode_start_ist": str(episode.iloc[0].time), "episode_end_ist": str(episode.iloc[-1].time),
                "horizon_hours": horizon, "n_lhs_samples": int(len(samples)), "reference_q": q_ref,
                "maximum_linear_solve_relative_residual": max_residual,
                "external_test_status": spec["external_test_status"],
                "interpretation": spec["conditional_scenarios"]["interpretation"]}
    (output / "conditional_uncertainty_manifest.json").write_text(json.dumps(manifest, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([results.loc[results.scenario_source_factor.eq(f), "final_domain_mean_ug_m3"] for f in scenario_factors], tick_labels=[f"{f:.1f}" for f in scenario_factors], showfliers=False)
    ax.set_xlabel("Nominal all-local-primary-source input factor")
    ax.set_ylabel("Conditional final domain-mean PM2.5 (ug/m3)")
    ax.set_title("Conditional 24-hour ADR response across 500 LHS draws")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "conditional_uncertainty_distribution.png", dpi=200); plt.close(fig)
    print(json.dumps(manifest, indent=2))
    print(summary_df.to_string(index=False))
    print(sensitivity_df.to_string(index=False))


if __name__ == "__main__":
    main()
