import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def add_policy_year_break(ax, break_year_start, break_year_end):
    ax.axvspan(
        break_year_start, break_year_end, color="white", zorder=2, edgecolor=None
    )
    return ax


def create_combined_did_charts(
    final_dataset,
    control_states,
    state,
    start_year,
    end_year,
    policy_year,
    control_label,
    save_as,
):
    final_dataset["opioid_YEAR"] = pd.to_numeric(
        final_dataset["opioid_YEAR"], errors="coerce"
    )
    final_dataset = final_dataset.dropna(subset=["opioid_YEAR"])

    final_dataset = final_dataset[
        (final_dataset["opioid_YEAR"] >= start_year)
        & (final_dataset["opioid_YEAR"] <= end_year)
    ]

    duke_colors = {
        "blue": "#00539B",
        "gray": "#666666",
        "orange": "#E89923",
        "background": "#F3F2F1",
    }

    state_data = final_dataset[final_dataset["State"] == state]

    state_aggregated = (
        state_data.groupby("opioid_YEAR")
        .agg(
            total_population=("pop_Population", "sum"),
            total_morphine=("opioid_morphine_equivalent_g", "sum"),
            total_deaths=("mort_overdose_deaths", "sum"),
        )
        .reset_index()
    )
    state_aggregated["morphine_per_capita"] = (
        state_aggregated["total_morphine"] / state_aggregated["total_population"]
    )
    state_aggregated["deaths_per_capita"] = (
        state_aggregated["total_deaths"] / state_aggregated["total_population"]
    )

    morphine_model = smf.ols(
        "morphine_per_capita ~ opioid_YEAR", data=state_aggregated
    ).fit()
    morphine_predictions = morphine_model.get_prediction(
        state_aggregated
    ).summary_frame()
    state_aggregated["morphine_predicted"] = morphine_predictions["mean"]
    state_aggregated["morphine_ci_low"] = morphine_predictions["mean_ci_lower"]
    state_aggregated["morphine_ci_high"] = morphine_predictions["mean_ci_upper"]

    deaths_model = smf.ols(
        "deaths_per_capita ~ opioid_YEAR", data=state_aggregated
    ).fit()
    deaths_predictions = deaths_model.get_prediction(state_aggregated).summary_frame()
    state_aggregated["deaths_predicted"] = deaths_predictions["mean"]
    state_aggregated["deaths_ci_low"] = deaths_predictions["mean_ci_lower"]
    state_aggregated["deaths_ci_high"] = deaths_predictions["mean_ci_upper"]

    control_data = final_dataset[final_dataset["State"].isin(control_states)]
    control_aggregated = (
        control_data.groupby("opioid_YEAR")
        .agg(
            total_population=("pop_Population", "sum"),
            total_morphine=("opioid_morphine_equivalent_g", "sum"),
            total_deaths=("mort_overdose_deaths", "sum"),
        )
        .reset_index()
    )
    control_aggregated["morphine_per_capita"] = (
        control_aggregated["total_morphine"] / control_aggregated["total_population"]
    )
    control_aggregated["deaths_per_capita"] = (
        control_aggregated["total_deaths"] / control_aggregated["total_population"]
    )

    control_morphine_model = smf.ols(
        "morphine_per_capita ~ opioid_YEAR", data=control_aggregated
    ).fit()
    control_morphine_predictions = control_morphine_model.get_prediction(
        control_aggregated
    ).summary_frame()
    control_aggregated["morphine_predicted"] = control_morphine_predictions["mean"]
    control_aggregated["morphine_ci_low"] = control_morphine_predictions[
        "mean_ci_lower"
    ]
    control_aggregated["morphine_ci_high"] = control_morphine_predictions[
        "mean_ci_upper"
    ]

    control_deaths_model = smf.ols(
        "deaths_per_capita ~ opioid_YEAR", data=control_aggregated
    ).fit()
    control_deaths_predictions = control_deaths_model.get_prediction(
        control_aggregated
    ).summary_frame()
    control_aggregated["deaths_predicted"] = control_deaths_predictions["mean"]
    control_aggregated["deaths_ci_low"] = control_deaths_predictions["mean_ci_lower"]
    control_aggregated["deaths_ci_high"] = control_deaths_predictions["mean_ci_upper"]

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), facecolor=duke_colors["background"])

    # Chart 1: Morphine per capita
    axes[0].plot(
        state_aggregated["opioid_YEAR"],
        state_aggregated["morphine_predicted"],
        color=duke_colors["blue"],
        label=f"{state} (Treated)",
        linewidth=2,
    )
    axes[0].fill_between(
        state_aggregated["opioid_YEAR"],
        state_aggregated["morphine_ci_low"],
        state_aggregated["morphine_ci_high"],
        color=duke_colors["blue"],
        alpha=0.3,
    )
    axes[0].plot(
        control_aggregated["opioid_YEAR"],
        control_aggregated["morphine_predicted"],
        color=duke_colors["orange"],
        label=control_label,
        linewidth=2,
    )
    axes[0].fill_between(
        control_aggregated["opioid_YEAR"],
        control_aggregated["morphine_ci_low"],
        control_aggregated["morphine_ci_high"],
        color=duke_colors["orange"],
        alpha=0.3,
    )
    axes[0] = add_policy_year_break(axes[0], policy_year - 1, policy_year)
    axes[0].axvline(
        x=policy_year,
        color=duke_colors["gray"],
        linestyle="--",
        linewidth=1.5,
        label="Policy Implementation",
    )
    axes[0].set_title(
        f"Opioid (MME) per Capita: {state} vs Control States",
        fontsize=12,
        weight="bold",
    )
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.8)
    axes[0].set_xlim([start_year, end_year])
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Opioid (MME) per Capita")

    # Chart 2: Overdose deaths per capita
    axes[1].plot(
        state_aggregated["opioid_YEAR"],
        state_aggregated["deaths_predicted"],
        color=duke_colors["blue"],
        label=f"{state} (Treated)",
        linewidth=2,
    )
    axes[1].fill_between(
        state_aggregated["opioid_YEAR"],
        state_aggregated["deaths_ci_low"],
        state_aggregated["deaths_ci_high"],
        color=duke_colors["blue"],
        alpha=0.3,
    )
    axes[1].plot(
        control_aggregated["opioid_YEAR"],
        control_aggregated["deaths_predicted"],
        color=duke_colors["orange"],
        label=control_label,
        linewidth=2,
    )
    axes[1].fill_between(
        control_aggregated["opioid_YEAR"],
        control_aggregated["deaths_ci_low"],
        control_aggregated["deaths_ci_high"],
        color=duke_colors["orange"],
        alpha=0.3,
    )
    axes[1] = add_policy_year_break(axes[1], policy_year - 1, policy_year)
    axes[1].axvline(
        x=policy_year,
        color=duke_colors["gray"],
        linestyle="--",
        linewidth=1.5,
        label="Policy Implementation",
    )
    axes[1].set_title(
        f"Overdose Deaths per Capita: {state} vs Control States",
        fontsize=12,
        weight="bold",
    )
    axes[1].legend(loc="center right", fontsize=8, framealpha=0.8)
    axes[1].set_xlim([start_year, end_year])
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Overdose Deaths per Capita")

    plt.tight_layout()
    plt.savefig(f"{save_as}.png")
    plt.close()


def main():
    final_dataset = pd.read_csv("/Users/alina/Desktop/final_dataset.csv")

    create_combined_did_charts(
        final_dataset,
        control_states=["AL", "OK", "GA"],
        state="FL",
        start_year=2007,
        end_year=2013,
        policy_year=2010,
        control_label="AL, OK, GA",
        save_as="florida_did_charts",
    )

    create_combined_did_charts(
        final_dataset,
        control_states=["ME", "OR", "CO"],
        state="WA",
        start_year=2009,
        end_year=2015,
        policy_year=2012,
        control_label="ME, OR, CO",
        save_as="washington_did_charts",
    )


if __name__ == "__main__":
    main()
