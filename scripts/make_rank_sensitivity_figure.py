from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RHO_COLUMN = "spearman_rho_with_baseline_final_domain_mean"

LABELS = {
    "inflow_multiplier": "Inflow multiplier",
    "removal_s_inv": r"Removal rate $\lambda$ (s$^{-1}$)",
    "source_multiplier": "Source multiplier",
    "wind_attenuation": r"Wind attenuation $\kappa$",
    "diffusivity_m2_s": r"Diffusivity $D$ (m$^2$ s$^{-1}$)",
}


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input_path = (
        repo_root
        / "outputs"
        / "conditional_uncertainty"
        / "rank_sensitivity.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find the sensitivity results file: {input_path}"
        )

    data = pd.read_csv(input_path)

    required_columns = {"parameter", RHO_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "The sensitivity results file is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    # Keep the coefficient's sign, but rank inputs by absolute magnitude.
    data["absolute_rho"] = data[RHO_COLUMN].abs()
    data = data.sort_values("absolute_rho", ascending=False)

    labels = [
        LABELS.get(parameter, parameter.replace("_", " "))
        for parameter in data["parameter"]
    ]
    correlations = data[RHO_COLUMN]
    colors = [
        "#bd3b24" if value < 0 else "#0c6b9d"
        for value in correlations
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(labels, correlations, color=colors)
    ax.invert_yaxis()  # Put the strongest absolute association at the top.
    ax.axvline(0, color="#333333", linewidth=0.8)

    ax.set_xlabel(
        "Spearman rank correlation with final conditional domain mean"
    )
    ax.set_title(
        "Figure 5. Conditional-model sensitivity to declared uncertain inputs"
    )
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    ax.text(
        0.5,
        -0.18,
        "Ranked by absolute correlation. Conditional-model sensitivity; "
        "not causal source attribution.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
    )

    fig.tight_layout()

    output_dir = repo_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / "figure_5_rank_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "figure_5_rank_sensitivity.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
