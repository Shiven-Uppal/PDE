"""Create Supplementary Figure S1 from the locked calibration_013 forcing."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/directional_boundary_hourly_forcing.csv"
OUTPUT = ROOT / "outputs/figures/supplementary_figure_s1_episode_forcing.png"


def main() -> None:
    forcing = pd.read_csv(INPUT, parse_dates=["time"])
    episode = forcing.loc[
        (forcing["time"] >= "2022-12-03 14:00:00")
        & (forcing["time"] <= "2022-12-04 13:00:00")
    ].copy()
    if len(episode) != 24 or not episode["usable_hourly_forcing"].all():
        raise ValueError("calibration_013 must contain 24 usable forcing hours")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True, constrained_layout=True)

    axes[0].plot(episode["time"], episode["pm25_inflow_ug_m3"], color="#8b1e3f", marker="o", ms=3.5)
    axes[0].set_ylabel("Boundary PM$_{2.5}$\n($\\mu$g m$^{-3}$)")
    axes[0].set_title("Supplementary Figure S1. Forcing during the fixed 24-hour conditional episode")
    axes[0].grid(alpha=0.25)

    axes[1].plot(episode["time"], episode["wind_speed_m_s"], color="#1f77b4", marker="o", ms=3.5)
    axes[1].set_ylabel("ERA5 wind speed\n(m s$^{-1}$)")
    axes[1].grid(alpha=0.25)
    direction = axes[1].twinx()
    direction.plot(episode["time"], episode["wind_to_direction_deg"], color="#d17a22", lw=1.7, alpha=0.9)
    direction.set_ylabel("Wind-to direction ($^\\circ$)", color="#d17a22")
    direction.tick_params(axis="y", colors="#d17a22")
    direction.set_ylim(0, 360)

    sector_order = {"N": 0, "NW": 1, "W": 2, "SE": 3, "S": 4}
    episode["sector_code"] = episode["inflow_sector"].map(sector_order)
    axes[2].step(episode["time"], episode["sector_code"], where="mid", color="#3d7a3a", lw=2)
    axes[2].scatter(episode["time"], episode["sector_code"], color="#3d7a3a", s=20, zorder=3)
    axes[2].set_yticks(list(sector_order.values()), list(sector_order.keys()))
    axes[2].set_ylim(-0.5, 4.5)
    axes[2].set_ylabel("Selected inflow sector")
    axes[2].set_xlabel("Indian Standard Time")
    axes[2].grid(axis="x", alpha=0.25)
    axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=4))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))

    fig.text(
        0.5,
        -0.015,
        "All hours meet the fixed precipitation, wind-speed, boundary-data, and interior-station eligibility rules. "
        "The panel shows prescribed inputs, not forecasts.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
