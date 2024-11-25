import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.set_option("mode.copy_on_write", True)

def create_charts_wa_with_gap_and_line():
    final_dataset=pd.read_csv("final_dataset.csv")
    # Filter data for WA only
    wa_data = final_dataset[final_dataset['State'] == 'WA']
    
    # Ensure 'opioid_YEAR' is numeric and drop invalid rows
    wa_data['opioid_YEAR'] = pd.to_numeric(wa_data['opioid_YEAR'], errors='coerce')
    wa_data = wa_data.dropna(subset=['opioid_YEAR'])
    
    # Filter for years between 2006 and 2015
    wa_data = wa_data[(wa_data['opioid_YEAR'] >= 2009) & (wa_data['opioid_YEAR'] <= 2015)]
    
    # Adjust the x-axis to use 2012 as the 0 point
    wa_data['relative_year'] = wa_data['opioid_YEAR'] - 2012
    
    # Group by relative year and aggregate the required metrics
    aggregated_data = wa_data.groupby('relative_year').agg(
        total_population=('pop_Population', 'sum'),
        total_morphine=('opioid_morphine_equivalent_g', 'sum'),
        total_deaths=('mort_overdose_deaths', 'sum')
    ).reset_index()
    
    # Compute per capita metrics
    aggregated_data['morphine_per_capita'] = aggregated_data['total_morphine'] / aggregated_data['total_population']
    aggregated_data['deaths_per_capita'] = aggregated_data['total_deaths'] / aggregated_data['total_population']
    
    morphine_model = smf.ols(
        "morphine_per_capita ~ relative_year", data=aggregated_data
    ).fit()
    morphine_predictions = morphine_model.get_prediction(aggregated_data).summary_frame()
    aggregated_data["morphine_predicted"] = morphine_predictions["mean"]
    aggregated_data["morphine_ci_low"] = morphine_predictions["mean_ci_lower"]
    aggregated_data["morphine_ci_high"] = morphine_predictions["mean_ci_upper"]

    deaths_model = smf.ols("deaths_per_capita ~ relative_year", data=aggregated_data).fit()
    deaths_predictions = deaths_model.get_prediction(aggregated_data).summary_frame()
    aggregated_data["deaths_predicted"] = deaths_predictions["mean"]
    aggregated_data["deaths_ci_low"] = deaths_predictions["mean_ci_lower"]
    aggregated_data["deaths_ci_high"] = deaths_predictions["mean_ci_upper"]
    
    
    # Split data to create a gap at 0
    before_gap = aggregated_data[aggregated_data['relative_year'] < 0]
    after_gap = aggregated_data[aggregated_data['relative_year'] >= 0]

    # Define Duke color palette
    duke_colors = {
        'blue': '#00539B',    # Duke Blue
        'gray': '#666666',    # Graphite
        'orange': '#E89923',  # Persimmon
        'green': '#339898',   # Eno
        'background': '#F3F2F1',  # Whisper Gray
        'highlight': '#D9E8F6'  # Light Blue for shaded area
    }
    
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor=duke_colors['background'])
    fig.suptitle('In 2012, WA Implemented Regulation for Opioids.\nThe figure shows the trend.', 
                 fontsize=20, weight='bold', color=duke_colors['blue'])
    
    # Chart 1: Morphine per capita
    axes[0].plot(
        before_gap['relative_year'], 
        before_gap['morphine_per_capita'], 
        marker='o', 
        color=duke_colors['blue'], 
        label='Morphine g per Capita (Before)',
        linewidth=2
    )
    axes[0].fill_between(
        before_gap["relative_year"],
        before_gap["morphine_ci_low"],
        before_gap["morphine_ci_high"],
        color=duke_colors["blue"],
        alpha=0.3,
    )
    axes[0].plot(
        after_gap['relative_year'], 
        after_gap['morphine_per_capita'], 
        marker='o', 
        color=duke_colors['blue'], 
        linestyle='--',
        label='Morphine g per Capita (After)',
        linewidth=2
    )
    axes[0].fill_between(
        after_gap["relative_year"],
        after_gap["morphine_ci_low"],
        after_gap["morphine_ci_high"],
        color=duke_colors["blue"],
        alpha=0.3,
    )
    axes[0].axvline(x=0, color=duke_colors['gray'], linestyle='--', linewidth=1.5, label='Year 2012')
    axes[0].set_title('Washington: Morphine g per Capita', fontsize=16, weight='bold', color=duke_colors['blue'])
    axes[0].set_xlabel('Years Relative to 2012', fontsize=14, color=duke_colors['gray'])
    axes[0].set_ylabel('Morphine g per Capita', fontsize=14, color=duke_colors['gray'])
    axes[0].grid(color=duke_colors['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    axes[0].legend(fontsize=12, loc='upper left', frameon=True, edgecolor=duke_colors['gray'])
    
    # Chart 2: Overdose deaths per capita
    axes[1].plot(
        before_gap['relative_year'], 
        before_gap['deaths_per_capita'], 
        marker='o', 
        color=duke_colors['orange'], 
        label='Overdose Deaths per Capita (Before)',
        linewidth=2
    )
    axes[1].fill_between(
        before_gap["relative_year"],
        before_gap["deaths_ci_low"],
        before_gap["deaths_ci_high"],
        color=duke_colors["orange"],
        alpha=0.3,
    )
    axes[1].plot(
        after_gap['relative_year'], 
        after_gap['deaths_per_capita'], 
        marker='o', 
        color=duke_colors['orange'], 
        linestyle='--',
        label='Overdose Deaths per Capita (After)',
        linewidth=2
    )
    axes[1].fill_between(
        after_gap["relative_year"],
        after_gap["deaths_ci_low"],
        after_gap["deaths_ci_high"],
        color=duke_colors["orange"],
        alpha=0.3,
    )
    axes[1].axvline(x=0, color=duke_colors['gray'], linestyle='--', linewidth=1.5, label='Year 2012')
    axes[1].set_title('Washington: Overdose Deaths per Capita', fontsize=16, weight='bold', color=duke_colors['blue'])
    axes[1].set_xlabel('Years Relative to 2012', fontsize=14, color=duke_colors['gray'])
    axes[1].set_ylabel('Overdose Deaths per Capita', fontsize=14, color=duke_colors['gray'])
    axes[1].grid(color=duke_colors['gray'], linestyle='--', linewidth=0.5, alpha=0.7)
    axes[1].legend(fontsize=12, loc='upper left', frameon=True, edgecolor=duke_colors['gray'])
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# Call the updated function
create_charts_wa_with_gap_and_line()
