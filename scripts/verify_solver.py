#!/usr/bin/env python3
"""Run a method-of-manufactured-solutions convergence test for the ADR solver."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.adr_solver import ADRConfig, solve_adr


def exact(x: np.ndarray, y: np.ndarray, lx: float, ly: float) -> np.ndarray:
    return 2.0 + np.sin(np.pi * x / lx) * np.sin(np.pi * y / ly)


def forcing(x: np.ndarray, y: np.ndarray, cfg: ADRConfig) -> np.ndarray:
    sx = np.sin(np.pi * x / cfg.lx_m)
    sy = np.sin(np.pi * y / cfg.ly_m)
    cx = np.cos(np.pi * x / cfg.lx_m)
    cy = np.cos(np.pi * y / cfg.ly_m)
    c = 2.0 + sx * sy
    c_x = (np.pi / cfg.lx_m) * cx * sy
    c_y = (np.pi / cfg.ly_m) * sx * cy
    laplacian = -((np.pi / cfg.lx_m) ** 2 + (np.pi / cfg.ly_m) ** 2) * sx * sy
    return -cfg.diffusivity_m2_s * laplacian + cfg.u_m_s * c_x + cfg.v_m_s * c_y + cfg.removal_s_inv * c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (20, 40, 80, 160):
        cfg = ADRConfig(50_000.0, 50_000.0, n, n, 300.0, 1.2, -0.7, 1.0 / 86_400.0)
        xx, yy = np.meshgrid(cfg.x_m, cfg.y_m)
        solution, residual = solve_adr(cfg, forcing(xx, yy, cfg), lambda x, y: exact(x, y, cfg.lx_m, cfg.ly_m), "dirichlet_all")
        error_l2 = float(np.sqrt(np.mean((solution - exact(xx, yy, cfg.lx_m, cfg.ly_m)) ** 2)))
        rows.append({"n": n, "h_m": cfg.hx_m, "l2_error": error_l2, "relative_residual_inf": residual,
                     "pe_x": cfg.cell_peclet_x, "pe_y": cfg.cell_peclet_y})
    report = pd.DataFrame(rows)
    report["observed_order"] = np.nan
    for k in range(1, len(report)):
        report.loc[k, "observed_order"] = np.log(report.loc[k - 1, "l2_error"] / report.loc[k, "l2_error"]) / np.log(2.0)
    report.to_csv(args.output_dir / "mms_convergence.csv", index=False)
    final_order = report["observed_order"].iloc[-1]
    if not (final_order > 0.8 and report["relative_residual_inf"].max() < 1e-10):
        raise RuntimeError("MMS verification did not meet convergence/residual criteria.")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.6g}"))


if __name__ == "__main__":
    main()
