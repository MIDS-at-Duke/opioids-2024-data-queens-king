import pandas as pd
import warnings

pd.set_option("mode.copy_on_write", True)


def read_and_filter_data(file_path, chunk_size, states_to_keep, columns_to_keep):
    """Read data in chunks, filter for relevant states, and keep specified columns."""
    chunk_list = []
    for chunk in pd.read_csv(
        file_path,
        delimiter="\t",
        chunksize=chunk_size,
        compression="zip",
        usecols=columns_to_keep,
    ):
        filtered_chunk = chunk[chunk["BUYER_STATE"].isin(states_to_keep)]
        chunk_list.append(filtered_chunk)
    return pd.concat(chunk_list, ignore_index=True)


def preprocess_data(df, drug_names):
    """Preprocess the data: drop NAs, convert dates, filter drugs, and calculate morphine equivalent."""
    df = df.dropna()
    df["date"] = pd.to_datetime(
        df["TRANSACTION_DATE"], format="%Y-%m-%d", errors="coerce"
    )
    df["year"] = df["date"].dt.year
    df["DRUG_NAME"] = df["DRUG_NAME"].isin(drug_names)
    df["MME_Conversion_Factor"] = pd.to_numeric(
        df["MME_Conversion_Factor"], errors="coerce"
    )
    df["morphine_equivalent_g"] = df["CALC_BASE_WT_IN_GM"] * df["MME_Conversion_Factor"]
    return df


def group_data(df):
    """Group data by year, state, and county to sum morphine equivalent shipments."""
    return (
        df.groupby(["year", "BUYER_STATE", "BUYER_COUNTY"])
        .morphine_equivalent_g.sum()
        .reset_index()
    )


def validate_counties(df, state, max_counties):
    """Validate the number of unique counties for a given state."""
    county_count = len(df[df["BUYER_STATE"] == state]["BUYER_COUNTY"].unique())
    assert county_count <= max_counties, f"{state} exceeds the maximum county count."
    return county_count


def main():
    # Settings and configurations
    warnings.filterwarnings("ignore", category=FutureWarning)
    file_path = "~/Desktop/arcos_all_washpost.zip"
    chunk_size = 100000
    states_to_keep = ["FL", "WA", "OR", "GA", "OK", "AL", "CO", "ME"]
    columns_to_keep = [
        "BUYER_STATE",
        "BUYER_COUNTY",
        "DRUG_NAME",
        "TRANSACTION_DATE",
        "CALC_BASE_WT_IN_GM",
        "MME_Conversion_Factor",
    ]
    drug_names = ["OXYCODONE", "HYDROCODONE"]

    # Read and filter data
    df = read_and_filter_data(file_path, chunk_size, states_to_keep, columns_to_keep)

    # Preprocess data
    df = preprocess_data(df, drug_names)

    # Group data
    df_grouped = group_data(df)

    # Validate counties
    florida_and_constants = df[df["BUYER_STATE"].isin(["FL", "AL", "GA", "OK"])]
    florida_count = validate_counties(florida_and_constants, "FL", 67)
    georgia_count = validate_counties(florida_and_constants, "GA", 159)
    oklahoma_count = validate_counties(florida_and_constants, "OK", 77)
    alabama_count = validate_counties(florida_and_constants, "AL", 67)

    washington_and_constants = df[df["BUYER_STATE"].isin(["WA", "CO", "OR", "ME"])]
    washington_count = validate_counties(washington_and_constants, "WA", 39)
    colorado_count = validate_counties(washington_and_constants, "CO", 64)
    oregon_count = validate_counties(washington_and_constants, "OR", 36)
    maine_count = validate_counties(washington_and_constants, "ME", 16)

    # Print summary
    print(
        f"Florida has {florida_count} counties, Georgia has {georgia_count} counties, "
        f"Oklahoma has {oklahoma_count} counties, Alabama has {alabama_count} counties"
    )
    print(
        f"Washington has {washington_count} counties, Colorado has {colorado_count} counties, "
        f"Oregon has {oregon_count} counties, Maine has {maine_count} counties"
    )

    # Save to Parquet
    df_grouped.to_parquet(
        "data/opioid_shipment_WA_FL_andconstants.parquet", index=False
    )


if __name__ == "__main__":
    main()
