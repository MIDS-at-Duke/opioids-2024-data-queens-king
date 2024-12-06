import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import statsmodels.api as sm
from patsy import dmatrix


def add_policy_year_break(ax, break_year_start, break_year_end):
    """Add a shaded vertical area to highlight the policy year range."""
    ax.axvspan(
        break_year_start,
        break_year_end,
        color="white",
        alpha=1.0,
        zorder=2,
        edgecolor=None,
    )
    return ax


def prepare_data(data, states, start_year, end_year):
    """Filter and aggregate the dataset for the given states and years."""
    filtered_data = data[
        (data["opioid_YEAR"] >= start_year)
        & (data["opioid_YEAR"] <= end_year)
        & (data["State"].isin(states))
    ]
    aggregated = (
        filtered_data.groupby("opioid_YEAR")
        .agg(
            total_population=("pop_Population", "sum"),
            total_morphine=("opioid_morphine_equivalent_g", "sum"),
            total_deaths=("mort_overdose_deaths", "sum"),
        )
        .reset_index()
    )
    aggregated["log_population"] = np.log(aggregated["total_population"])
    aggregated["morphine_per_capita"] = (
        aggregated["total_morphine"] / aggregated["total_population"]
    )
    aggregated["deaths_per_capita"] = (
        aggregated["total_deaths"] / aggregated["total_population"]
    )
    return aggregated
    

def add_predictions(data, y_var, degree=3, df=4):
    """Fit a spline model and add predictions with confidence intervals."""
    spline_basis = dmatrix(
        f"bs(opioid_YEAR, degree={degree}, df={df})", data=data, return_type="dataframe"
    )
    model = sm.OLS(
        data[y_var],
        sm.add_constant(pd.concat([spline_basis, data["log_population"]], axis=1)),
    ).fit()
    predictions = model.get_prediction(
        sm.add_constant(pd.concat([spline_basis, data["log_population"]], axis=1))
    ).summary_frame()
    data[f"{y_var}_predicted"] = predictions["mean"]
    data[f"{y_var}_ci_low"] = predictions["mean_ci_lower"]
    data[f"{y_var}_ci_high"] = predictions["mean_ci_upper"]
    return data


def create_did_charts(
    final_dataset,
    treated_state,
    treated_state_full_name,
    control_states,
    start_year,
    end_year,
    policy_year_start,
    policy_year_end,
    save_as,
):
    """Generate DID charts for treated state vs control states."""
    treated_data = prepare_data(final_dataset, [treated_state], start_year, end_year)
    control_data = prepare_data(final_dataset, control_states, start_year, end_year)

    treated_data = add_predictions(treated_data, "morphine_per_capita")
    treated_data = add_predictions(treated_data, "deaths_per_capita")
    control_data = add_predictions(control_data, "morphine_per_capita")
    control_data = add_predictions(control_data, "deaths_per_capita")

    duke_colors = {
        "blue": "#00539B",
        "gray": "#666666",
        "orange": "#E89923",
        "background": "#F3F2F1",
    }

    fig = plt.figure(figsize=(12, 6), facecolor=duke_colors["background"])
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    # Chart 1: Morphine per capita
    ax1 = fig.add_subplot(spec[0])
    ax1.plot(
        treated_data["opioid_YEAR"],
        treated_data["morphine_per_capita_predicted"],
        color=duke_colors["blue"],
        label=f"{treated_state_full_name} Trendline",
        linewidth=2,
    )
    ax1.fill_between(
        treated_data["opioid_YEAR"],
        treated_data["morphine_per_capita_ci_low"],
        treated_data["morphine_per_capita_ci_high"],
        color=duke_colors["blue"],
        alpha=0.3,
    )
    ax1.plot(
        control_data["opioid_YEAR"],
        control_data["morphine_per_capita_predicted"],
        color=duke_colors["orange"],
        label=f"{', '.join(control_states)} Trendline",
        linewidth=2,
    )
    ax1.fill_between(
        control_data["opioid_YEAR"],
        control_data["morphine_per_capita_ci_low"],
        control_data["morphine_per_capita_ci_high"],
        color=duke_colors["orange"],
        alpha=0.3,
    )
    ax1 = add_policy_year_break(ax1, policy_year_start, policy_year_end)
    ax1.axvline(
        x=policy_year_end,
        color=duke_colors["gray"],
        linestyle="--",
        linewidth=1.5,
        label="Policy Implementation",
    )
    ax1.set_title(
        f"Opioid (MME) per Capita: {treated_state_full_name} vs Control States",
        fontsize=12,
        weight="bold",
    )
    ax1.set_xlabel("Year", fontsize=10)
    ax1.set_ylabel("Opioid (MME) per Capita", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.8)

    # Chart 2: Overdose deaths per capita
    ax2 = fig.add_subplot(spec[1])
    ax2.plot(
        treated_data["opioid_YEAR"],
        treated_data["deaths_per_capita_predicted"],
        color=duke_colors["blue"],
        label=f"{treated_state_full_name} Trendline",
        linewidth=2,
    )
    ax2.fill_between(
        treated_data["opioid_YEAR"],
        treated_data["deaths_per_capita_ci_low"],
        treated_data["deaths_per_capita_ci_high"],
        color=duke_colors["blue"],
        alpha=0.3,
    )
    ax2.plot(
        control_data["opioid_YEAR"],
        control_data["deaths_per_capita_predicted"],
        color=duke_colors["orange"],
        label=f"{', '.join(control_states)} Trendline",
        linewidth=2,
    )
    ax2.fill_between(
        control_data["opioid_YEAR"],
        control_data["deaths_per_capita_ci_low"],
        control_data["deaths_per_capita_ci_high"],
        color=duke_colors["orange"],
        alpha=0.3,
    )
    ax2 = add_policy_year_break(ax2, policy_year_start, policy_year_end)
    ax2.axvline(
        x=policy_year_end,
        color=duke_colors["gray"],
        linestyle="--",
        linewidth=1.5,
        label="Policy Implementation",
    )
    ax2.set_title(
        f"Overdose Deaths per Capita: {treated_state_full_name} vs Control States",
        fontsize=12,
        weight="bold",
    )
    ax2.set_xlabel("Year", fontsize=10)
    ax2.set_ylabel("Overdose Deaths per Capita", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{save_as}.png")
    plt.close()


def main():
    """Main function to generate DID charts."""
    file_path = "./data/final_dataset.csv"
    final_dataset = pd.read_csv(file_path)

    create_did_charts(
        final_dataset=final_dataset,
        treated_state="FL",
        treated_state_full_name="Florida",
        control_states=["AL", "OK", "GA"],
        start_year=2007,
        end_year=2013,
        policy_year_start=2009,
        policy_year_end=2010,
        save_as="florida_did_charts",
    )

    create_did_charts(
        final_dataset=final_dataset,
        treated_state="WA",
        treated_state_full_name="Washington",
        control_states=["ME", "OR", "CO"],
        start_year=2009,
        end_year=2015,
        policy_year_start=2011,
        policy_year_end=2012,
        save_as="washington_did_charts",
    )


if __name__ == "__main__":
    main()
