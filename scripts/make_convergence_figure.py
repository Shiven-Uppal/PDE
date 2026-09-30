"""Create Figure 2 from the manufactured-solution convergence record."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    table = pd.read_csv(root / "outputs/verification/mms_convergence.csv")
    out = root / "outputs/figures"; out.mkdir(parents=True, exist_ok=True)
    h_km, error = table.h_m.to_numpy() / 1000, table.l2_error.to_numpy()
    reference = error[0] * (h_km / h_km[0])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.loglog(h_km, error, "o-", lw=2, ms=6.5, color="#0c6b9d", label="Computed $L^2$ error")
    ax.loglog(h_km, reference, "--", lw=1.5, color="#8b8b8b", label="First-order reference")
    for h, e, n in zip(h_km, error, table.n):
        ax.annotate(f"N={n}", (h, e), xytext=(5, 5), textcoords="offset points", fontsize=8)
    orders = table.observed_order.dropna().to_numpy()
    ax.text(.04, .08, "Observed refinement orders:\n" + ", ".join(f"{x:.3f}" for x in orders),
            transform=ax.transAxes, fontsize=9, bbox={"boxstyle":"round,pad=.35", "fc":"white", "ec":"#0c6b9d", "alpha":.95})
    ax.set(xlabel="Grid spacing $h$ (km)", ylabel="$L^2$ error", title="Figure 2. Manufactured-solution grid convergence", xlim=(.25, 3.1))
    ax.grid(True, which="both", alpha=.25); ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(out / "figure_2_manufactured_solution_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_2_manufactured_solution_convergence.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
