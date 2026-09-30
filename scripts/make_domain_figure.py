"""Create Figure 1: ADR domain, independent interior and boundary stations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from src.daily_adr import DOMAIN_LENGTH_M, DOMAIN_X0_M, DOMAIN_Y0_M


BOUNDARY_STATIONS = pd.DataFrame([
    ("Sonipat", "Sector 14, Sonipat - HSPCB", 29.0272, 77.0621, "N / NW"),
    ("Bahadurgarh", "Sector 6, Bahadurgarh - HSPCB", 28.6701, 76.9254, "W"),
    ("Ghaziabad", "Sanjay Nagar, Ghaziabad - UPPCB", 28.685382, 77.453839, "E"),
    ("Greater Noida", "Knowledge Park V, Greater Noida - UPPCB", 28.557054, 77.453663, "SE"),
    ("Gurugram", "Vikas Sadan, Gurugram - HSPCB", 28.4501238, 77.0263051, "S"),
], columns=["short_name", "station_name", "latitude", "longitude", "direction"])


def project(table: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs(4326, 32643, always_xy=True)
    x, y = transformer.transform(table.longitude.to_numpy(), table.latitude.to_numpy())
    return table.assign(x_km=(x - DOMAIN_X0_M) / 1000, y_km=(y - DOMAIN_Y0_M) / 1000)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stations = project(pd.read_csv(root / "data/processed/station_metadata.csv"))
    boundary = project(BOUNDARY_STATIONS)
    out = root / "outputs/figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.3, 7.4))
    # Display the full 1 km finite-difference grid without making it visually dominant.
    for coordinate in np.arange(0, 51, 1):
        ax.axhline(coordinate, color="#d7dce2", lw=.22, zorder=0)
        ax.axvline(coordinate, color="#d7dce2", lw=.22, zorder=0)
    ax.add_patch(plt.Rectangle((0, 0), 50, 50, fill=False, lw=1.6, ec="#203864", zorder=1))
    ax.scatter(stations.x_km, stations.y_km, s=24, c="#0c6b9d", edgecolor="white", linewidth=.35,
               label="Interior CPCB stations (n=35)", zorder=3)
    ax.scatter(boundary.x_km, boundary.y_km, s=84, marker="^", c="#bd3b24", edgecolor="white", linewidth=.55,
               label="Directional NCR boundary stations (n=5)", zorder=4, clip_on=False)
    for _, row in boundary.iterrows():
        ax.annotate(f"{row.short_name}\n({row.direction})", (row.x_km, row.y_km), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="#7b2013", weight="semibold")
    ax.annotate("50 km × 50 km\nADR computational domain", (25, 25), ha="center", va="center", fontsize=10,
                color="#203864", bbox={"boxstyle": "round,pad=.35", "fc": "white", "ec": "#203864", "alpha": .9})
    ax.set(xlim=(-10, 65), ylim=(-18, 72), xlabel="Easting relative to ADR domain origin (km)",
           ylabel="Northing relative to ADR domain origin (km)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", frameon=True, framealpha=.96, fontsize=8)
    ax.set_title("Figure 1. Delhi-centred ADR domain and independent monitoring geometry", weight="bold", pad=12)
    ax.text(.5, -.13, "Grid: 50 × 50 cells (Δx = Δy = 1 km). Boundary stations supply direction-matched inflow only;\n"
            "they are excluded from interior station assessment.", transform=ax.transAxes, ha="center", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(out / "figure_1_domain_and_station_geometry.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_1_domain_and_station_geometry.pdf", bbox_inches="tight")
    boundary.to_csv(out / "figure_1_boundary_station_coordinates.csv", index=False)


if __name__ == "__main__":
    main()
