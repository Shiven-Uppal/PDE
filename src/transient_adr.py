"""Implicit-Euler transient 2-D advection--diffusion--reaction solver.

For each hour, solves
    (C^(n+1)-C^n)/dt + u C_x + v C_y = D Delta C - lambda C + S,
using first-order upwind advection and centred diffusion.  Diffusion,
advection, and removal are implicit, so the time step is not subject to an
explicit diffusion CFL restriction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, eye, lil_matrix
from scipy.sparse.linalg import spsolve

from src.adr_solver import ADRConfig, _face_is_inflow, _index


def assemble_adr_operator(
    config: ADRConfig, boundary_value: float | Callable[[float, float], float]
) -> tuple[csr_matrix, np.ndarray]:
    """Return A and b such that dC/dt + A C = S + b for inflow/outflow BCs."""
    d, u, v, lam = (config.diffusivity_m2_s, config.u_m_s, config.v_m_s, config.removal_s_inv)
    hx, hy, nx, ny = config.hx_m, config.hy_m, config.nx, config.ny
    matrix = lil_matrix((nx * ny, nx * ny), dtype=float)
    rhs_boundary = np.zeros(nx * ny, dtype=float)

    def bvalue(x: float, y: float) -> float:
        return float(boundary_value(x, y) if callable(boundary_value) else boundary_value)

    def add_diffusion(row: int, i: int, j: int, di: int, dj: int, face: str) -> float:
        ni, nj = i + di, j + dj
        coef = d / (hx * hx if di else hy * hy)
        if 0 <= ni < nx and 0 <= nj < ny:
            matrix[row, _index(ni, nj, nx)] -= coef
            return coef
        if _face_is_inflow(face, u, v):
            x = 0.0 if face == "west" else config.lx_m if face == "east" else (i + 0.5) * hx
            y = 0.0 if face == "south" else config.ly_m if face == "north" else (j + 0.5) * hy
            rhs_boundary[row] += coef * bvalue(x, y)
            return coef
        return 0.0  # zero normal gradient on outflow

    for j in range(ny):
        for i in range(nx):
            row = _index(i, j, nx)
            diagonal = lam
            diagonal += add_diffusion(row, i, j, -1, 0, "west")
            diagonal += add_diffusion(row, i, j, 1, 0, "east")
            diagonal += add_diffusion(row, i, j, 0, -1, "south")
            diagonal += add_diffusion(row, i, j, 0, 1, "north")
            if u >= 0:
                diagonal += u / hx
                if i > 0:
                    matrix[row, _index(i - 1, j, nx)] -= u / hx
                else:
                    rhs_boundary[row] += (u / hx) * bvalue(0.0, (j + 0.5) * hy)
            else:
                diagonal += -u / hx
                if i < nx - 1:
                    matrix[row, _index(i + 1, j, nx)] += u / hx
                else:
                    rhs_boundary[row] += (-u / hx) * bvalue(config.lx_m, (j + 0.5) * hy)
            if v >= 0:
                diagonal += v / hy
                if j > 0:
                    matrix[row, _index(i, j - 1, nx)] -= v / hy
                else:
                    rhs_boundary[row] += (v / hy) * bvalue((i + 0.5) * hx, 0.0)
            else:
                diagonal += -v / hy
                if j < ny - 1:
                    matrix[row, _index(i, j + 1, nx)] += v / hy
                else:
                    rhs_boundary[row] += (-v / hy) * bvalue((i + 0.5) * hx, config.ly_m)
            matrix[row, row] += diagonal
    return matrix.tocsr(), rhs_boundary


def advance_adr_implicit(
    config: ADRConfig,
    concentration_previous: np.ndarray,
    source_ug_m3_s: np.ndarray,
    boundary_value: float | Callable[[float, float], float],
    dt_s: float = 3600.0,
) -> tuple[np.ndarray, float]:
    """Advance one implicit-Euler step and return field and linear residual."""
    if concentration_previous.shape != (config.ny, config.nx) or source_ug_m3_s.shape != (config.ny, config.nx):
        raise ValueError("Previous concentration and source must both have shape (ny, nx).")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")
    fields, residual = advance_adr_implicit_many(
        config, concentration_previous[None, :, :], source_ug_m3_s[None, :, :], boundary_value, dt_s
    )
    return fields[0], residual


def advance_adr_implicit_many(
    config: ADRConfig,
    concentration_previous: np.ndarray,
    source_ug_m3_s: np.ndarray,
    boundary_value: float | np.ndarray | Callable[[float, float], float],
    dt_s: float = 3600.0,
) -> tuple[np.ndarray, float]:
    """Advance one operator for several superposed fields simultaneously."""
    if concentration_previous.ndim != 3 or source_ug_m3_s.shape != concentration_previous.shape:
        raise ValueError("Previous concentration and source must have shape (n_fields, ny, nx).")
    if concentration_previous.shape[1:] != (config.ny, config.nx):
        raise ValueError("Field shape does not match configuration.")
    if dt_s <= 0:
        raise ValueError("dt_s must be positive.")
    n_fields = concentration_previous.shape[0]
    if isinstance(boundary_value, np.ndarray):
        if boundary_value.shape != (n_fields,):
            raise ValueError("Vector boundary values must have one value per field.")
        operator, boundary_unit = assemble_adr_operator(config, 1.0)
        boundary_rhs = boundary_unit[:, None] * boundary_value[None, :]
    else:
        operator, boundary_unit = assemble_adr_operator(config, boundary_value)
        boundary_rhs = np.repeat(boundary_unit[:, None], n_fields, axis=1)
    lhs = eye(config.nx * config.ny, format="csr") + dt_s * operator
    rhs = concentration_previous.reshape(n_fields, -1).T + dt_s * (
        source_ug_m3_s.reshape(n_fields, -1).T + boundary_rhs
    )
    flat = np.asarray(spsolve(lhs, rhs))
    if flat.ndim == 1:
        flat = flat[:, None]
    residual = lhs @ flat - rhs
    relative_residual = float(np.linalg.norm(residual, ord=np.inf) / max(1.0, np.linalg.norm(rhs, ord=np.inf)))
    return np.asarray(flat).T.reshape(n_fields, config.ny, config.nx), relative_residual
