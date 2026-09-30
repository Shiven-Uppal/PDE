"""Sparse finite-difference solver for a steady two-dimensional ADR equation.

Solves
    -D ΔC + u C_x + v C_y + lambda C = S
on a rectangular cell-centre grid. Advection uses first-order upwinding and
diffusion uses centred differences. Physical boundaries use Dirichlet inflow
and zero-normal-gradient outflow. A separate all-Dirichlet option is provided
only for the method-of-manufactured-solutions verification test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


BoundaryMode = Literal["inflow_outflow", "dirichlet_all"]


@dataclass(frozen=True)
class ADRConfig:
    lx_m: float
    ly_m: float
    nx: int
    ny: int
    diffusivity_m2_s: float
    u_m_s: float
    v_m_s: float
    removal_s_inv: float

    def __post_init__(self) -> None:
        if self.lx_m <= 0 or self.ly_m <= 0 or self.nx < 2 or self.ny < 2:
            raise ValueError("Domain lengths must be positive and nx, ny must be at least 2.")
        if self.diffusivity_m2_s <= 0 or self.removal_s_inv < 0:
            raise ValueError("Diffusivity must be positive and removal must be non-negative.")

    @property
    def hx_m(self) -> float:
        return self.lx_m / self.nx

    @property
    def hy_m(self) -> float:
        return self.ly_m / self.ny

    @property
    def x_m(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.hx_m

    @property
    def y_m(self) -> np.ndarray:
        return (np.arange(self.ny) + 0.5) * self.hy_m

    @property
    def cell_peclet_x(self) -> float:
        return abs(self.u_m_s) * self.hx_m / self.diffusivity_m2_s

    @property
    def cell_peclet_y(self) -> float:
        return abs(self.v_m_s) * self.hy_m / self.diffusivity_m2_s


def _index(i: int, j: int, nx: int) -> int:
    return j * nx + i


def _face_is_inflow(face: str, u: float, v: float) -> bool:
    return ((face == "west" and u > 0) or (face == "east" and u < 0) or
            (face == "south" and v > 0) or (face == "north" and v < 0))


def solve_adr(
    config: ADRConfig,
    source_ug_m3_s: np.ndarray,
    boundary_value: float | Callable[[float, float], float],
    boundary_mode: BoundaryMode = "inflow_outflow",
) -> tuple[np.ndarray, float]:
    """Return concentration field [ny, nx] and relative infinity residual.

    `source_ug_m3_s` is a cell-centred source term. For all-Dirichlet
    verification, `boundary_value(x, y)` gives the exact boundary value.
    """
    if source_ug_m3_s.shape != (config.ny, config.nx):
        raise ValueError("Source shape must be (ny, nx).")
    if boundary_mode not in {"inflow_outflow", "dirichlet_all"}:
        raise ValueError("Unknown boundary mode.")

    d, u, v, lam = (config.diffusivity_m2_s, config.u_m_s,
                     config.v_m_s, config.removal_s_inv)
    hx, hy, nx, ny = config.hx_m, config.hy_m, config.nx, config.ny
    matrix = lil_matrix((nx * ny, nx * ny), dtype=float)
    rhs = np.asarray(source_ug_m3_s, dtype=float).reshape(-1).copy()

    def bvalue(x: float, y: float) -> float:
        return float(boundary_value(x, y) if callable(boundary_value) else boundary_value)

    def add_diffusion(row: int, i: int, j: int, di: int, dj: int, face: str) -> float:
        """Add one diffusion face and return its diagonal contribution."""
        ni, nj = i + di, j + dj
        coef = d / (hx * hx if di else hy * hy)
        if 0 <= ni < nx and 0 <= nj < ny:
            matrix[row, _index(ni, nj, nx)] -= coef
            return coef
        if boundary_mode == "dirichlet_all" or _face_is_inflow(face, u, v):
            x = 0.0 if face == "west" else config.lx_m if face == "east" else (i + 0.5) * hx
            y = 0.0 if face == "south" else config.ly_m if face == "north" else (j + 0.5) * hy
            rhs[row] += coef * bvalue(x, y)
            return coef
        # Zero normal gradient: the boundary face makes no diffusive contribution.
        return 0.0

    for j in range(ny):
        for i in range(nx):
            row = _index(i, j, nx)
            diagonal = lam
            diagonal += add_diffusion(row, i, j, -1, 0, "west")
            diagonal += add_diffusion(row, i, j, 1, 0, "east")
            diagonal += add_diffusion(row, i, j, 0, -1, "south")
            diagonal += add_diffusion(row, i, j, 0, 1, "north")

            # Upwind x-advection.
            if u >= 0:
                diagonal += u / hx
                if i > 0:
                    matrix[row, _index(i - 1, j, nx)] -= u / hx
                else:
                    rhs[row] += (u / hx) * bvalue(0.0, (j + 0.5) * hy)
            else:
                diagonal += -u / hx
                if i < nx - 1:
                    matrix[row, _index(i + 1, j, nx)] += u / hx
                else:
                    rhs[row] += (-u / hx) * bvalue(config.lx_m, (j + 0.5) * hy)

            # Upwind y-advection.
            if v >= 0:
                diagonal += v / hy
                if j > 0:
                    matrix[row, _index(i, j - 1, nx)] -= v / hy
                else:
                    rhs[row] += (v / hy) * bvalue((i + 0.5) * hx, 0.0)
            else:
                diagonal += -v / hy
                if j < ny - 1:
                    matrix[row, _index(i, j + 1, nx)] += v / hy
                else:
                    rhs[row] += (-v / hy) * bvalue((i + 0.5) * hx, config.ly_m)
            matrix[row, row] += diagonal

    matrix = matrix.tocsr()
    flat_solution = spsolve(matrix, rhs)
    residual = matrix @ flat_solution - rhs
    relative_residual = float(np.linalg.norm(residual, ord=np.inf) / max(1.0, np.linalg.norm(rhs, ord=np.inf)))
    return flat_solution.reshape(ny, nx), relative_residual
