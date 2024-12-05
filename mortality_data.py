# Import Required Libraries
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("mode.copy_on_write", True)


def load_and_combine_data(folder_path, start_year=2003, end_year=2015):
    """Load and combine yearly mortality data files into a single DataFrame."""
    return pd.concat(
        [
            pd.read_csv(
                f"{folder_path}/Underlying Cause of Death, {year}.txt",
                sep="\t",
                dtype=str,
            )
            for year in range(start_year, end_year + 1)
        ],
        ignore_index=True,
    )


def clean_and_standardize_data(raw_data):
    """Clean and standardize the raw data by filtering, renaming, and formatting."""
    # Drop unnecessary columns
    raw_data.drop(
        columns=["Notes", "Year Code", "Drug/Alcohol Induced Cause Code"],
        inplace=True,
    )

    # Rename columns for consistency
    cleaned_data = raw_data.rename(
        columns={
            "County": "county",
            "County Code": "county_code",
            "Year": "year",
            "Drug/Alcohol Induced Cause": "cause",
            "Deaths": "deaths",
        }
    )

    # Filter for main cause: unintentional drug overdose
    filtered_data = cleaned_data[
        cleaned_data["cause"] == "Drug poisonings (overdose) Unintentional (X40-X44)"
    ]

    # Extract state information from county names
    filtered_data["state"] = filtered_data["county"].str.split(", ").str[-1]

    # Filter for specific states of interest
    filtered_data = filtered_data[
        filtered_data["state"].isin(["FL", "WA", "OR", "GA", "OK", "AL", "CO", "ME"])
    ]

    # Standardize and clean various columns
    filtered_data["county_code"] = filtered_data["county_code"].astype(str).str.zfill(5)
    filtered_data["county"] = filtered_data["county"].str.strip().str.title()
    filtered_data["state"] = filtered_data["state"].str.strip().str.upper()
    filtered_data["year"] = pd.to_numeric(
        filtered_data["year"], errors="coerce"
    ).astype("int")
    filtered_data["deaths"] = pd.to_numeric(
        filtered_data["deaths"], errors="coerce"
    ).astype(int)

    return filtered_data


def aggregate_data(filtered_data):
    """Aggregate annual deaths by county and year."""
    return filtered_data.groupby(
        ["county_code", "county", "state", "year"], as_index=False
    )["deaths"].sum()


def save_to_parquet(data, output_path):
    """Save the processed data to a Parquet file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)
    print(f"Data saved to {output_path}")


def main():
    """Orchestrate the data loading, cleaning, aggregation, and saving process."""
    # Define input folder and output path
    folder_path = "~/Desktop/US_VitalStatistics/"
    output_path = "data/mortality_data.parquet"

    print("Loading data...")
    raw_data = load_and_combine_data(folder_path)

    print("Cleaning and standardizing data...")
    filtered_data = clean_and_standardize_data(raw_data)

    print("Aggregating data...")
    grouped_data = aggregate_data(filtered_data)

    print("Saving data to Parquet...")
    save_to_parquet(grouped_data, output_path)


if __name__ == "__main__":
    main()
