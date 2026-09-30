"""Daily-data interface for the verified steady ADR solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from src.adr_solver import ADRConfig, solve_adr


DOMAIN_X0_M = 682_750.0
DOMAIN_Y0_M = 3_145_665.0
DOMAIN_LENGTH_M = 50_000.0
GRID_N = 50


@dataclass(frozen=True)
class DailyADRParameters:
    diffusivity_m2_s: float
    removal_s_inv: float
    wind_attenuation: float
    total_local_source_ug_m3_s: float
    traffic_share: float

    def __post_init__(self) -> None:
        if self.diffusivity_m2_s <= 0 or self.removal_s_inv < 0:
            raise ValueError("Diffusivity must be positive and removal non-negative.")
        if not 0 < self.wind_attenuation <= 1:
            raise ValueError("Wind attenuation must lie in (0, 1].")
        if self.total_local_source_ug_m3_s < 0:
            raise ValueError("Total local source must be non-negative.")
        if not 0 <= self.traffic_share <= 1:
            raise ValueError("Traffic share must lie in [0, 1].")


def load_component_templates(path: str | Path, nx: int = GRID_N, ny: int = GRID_N) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(path).sort_values("cell_id")
    if len(table) != nx * ny:
        raise ValueError("Template table does not match the requested grid.")
    traffic = table.traffic_template.to_numpy(float).reshape(ny, nx)
    dust = table.dust_template.to_numpy(float).reshape(ny, nx)
    if np.any(traffic < 0) or np.any(dust < 0):
        raise ValueError("Source templates must be non-negative.")
    return traffic, dust


def load_inventory_template(path: str | Path, nx: int = GRID_N, ny: int = GRID_N) -> np.ndarray:
    """Load a fixed, externally constructed all-sector emission template."""
    table = pd.read_csv(path).sort_values("cell_id")
    if len(table) != nx * ny or "inventory_template" not in table:
        raise ValueError("Inventory table does not match the requested grid or lacks inventory_template.")
    template = table.inventory_template.to_numpy(float).reshape(ny, nx)
    if np.any(template < 0) or not np.isclose(template.mean(), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("Inventory template must be non-negative and have spatial mean one.")
    return template


def load_inventory_sector_templates(path: str | Path, nx: int = GRID_N, ny: int = GRID_N) -> dict[str, np.ndarray]:
    """Load EDGAR sector fields expressed relative to the all-sector mean."""
    table = pd.read_csv(path).sort_values("cell_id")
    sectors = ("agriculture", "energy", "industry", "international", "residential", "transport")
    if len(table) != nx * ny:
        raise ValueError("Inventory table does not match the requested grid.")
    result = {}
    for sector in sectors:
        column = f"edgar_{sector}_template"
        if column not in table:
            raise ValueError(f"Inventory table lacks {column}.")
        values = table[column].to_numpy(float).reshape(ny, nx)
        if np.any(values < 0):
            raise ValueError("Sector templates must be non-negative.")
        result[sector] = values
    return result


def combine_inventory_sector_templates(
    sectors: Mapping[str, np.ndarray], multipliers: Mapping[str, float] | None = None
) -> np.ndarray:
    """Combine EDGAR sector maps, then normalise their spatial mean to one.

    A multiplier changes a sector's *relative* contribution only.  The daily
    nuisance scale remains responsible for the total local loading.
    """
    multipliers = multipliers or {}
    combined = np.zeros((GRID_N, GRID_N), dtype=float)
    for sector, template in sectors.items():
        factor = float(multipliers.get(sector, 1.0))
        if factor <= 0:
            raise ValueError("Inventory sector multipliers must be positive.")
        combined += factor * template
    if combined.mean() <= 0:
        raise ValueError("Combined inventory template must have positive spatial mean.")
    return combined / combined.mean()


def run_daily_adr(
    forcing: Mapping[str, object],
    traffic_template: np.ndarray,
    dust_template: np.ndarray,
    parameters: DailyADRParameters,
) -> tuple[np.ndarray, ADRConfig, float]:
    """Solve one valid day; returns field, applied solver configuration, residual."""
    if not bool(forcing["usable_boundary_forcing"]):
        raise ValueError("This day has no direction-matched observed boundary forcing.")
    if traffic_template.shape != dust_template.shape or traffic_template.shape != (GRID_N, GRID_N):
        raise ValueError("Component templates must both be 50 by 50.")
    alpha = parameters.traffic_share
    source = parameters.total_local_source_ug_m3_s * (alpha * traffic_template + (1 - alpha) * dust_template)
    config = ADRConfig(
        lx_m=DOMAIN_LENGTH_M, ly_m=DOMAIN_LENGTH_M, nx=GRID_N, ny=GRID_N,
        diffusivity_m2_s=parameters.diffusivity_m2_s,
        u_m_s=parameters.wind_attenuation * float(forcing["u10_m_s"]),
        v_m_s=parameters.wind_attenuation * float(forcing["v10_m_s"]),
        removal_s_inv=parameters.removal_s_inv,
    )
    field, residual = solve_adr(config, source, float(forcing["pm25_inflow_ug_m3"]), "inflow_outflow")
    return field, config, residual


def run_daily_adr_inventory(
    forcing: Mapping[str, object], inventory_template: np.ndarray, parameters: DailyADRParameters
) -> tuple[np.ndarray, ADRConfig, float]:
    """Run the ADR model using an independent all-sector emissions template.

    ``traffic_share`` is intentionally ignored: the source geometry has already
    been fixed externally by the EDGAR inventory before calibration.
    """
    frozen = DailyADRParameters(
        parameters.diffusivity_m2_s, parameters.removal_s_inv,
        parameters.wind_attenuation, parameters.total_local_source_ug_m3_s, 1.0,
    )
    return run_daily_adr(forcing, inventory_template, inventory_template, frozen)


def interpolate_to_station_coordinates(
    field: np.ndarray, config: ADRConfig, easting_m: np.ndarray, northing_m: np.ndarray
) -> np.ndarray:
    """Bilinearly interpolate a field to UTM-43N station coordinates."""
    if field.shape != (config.ny, config.nx):
        raise ValueError("Field shape does not match configuration.")
    interpolator = RegularGridInterpolator((config.y_m, config.x_m), field, bounds_error=True)
    points = np.column_stack((np.asarray(northing_m) - DOMAIN_Y0_M, np.asarray(easting_m) - DOMAIN_X0_M))
    return interpolator(points)
