import pandas as pd
import numpy as np
import warnings

pd.set_option("mode.copy_on_write", True)


# Poverty Data in 2021
def get_poverty_data(filepath="USDA_data/Poverty.csv"):
    poverty_data=pd.read_csv(filepath)
    # columns name : ['FIPS*', 'Name', 'RUC Code', 'All people in poverty (2021) Percent', 'Children ages 0-17 in poverty (2021) Percent']
    poverty_data.rename(columns={'FIPS*': 'FIPS'}, inplace=True)
    poverty_data["YR"]="2021" # add a column as Year
    poverty_data=poverty_data[['FIPS', 'All people in poverty (2021) Percent','Children ages 0-17 in poverty (2021) Percent']]
    
    return poverty_data


# Population Data
def get_population_data(filepath="USDA_data/Population.csv"):
    population=pd.read_csv("USDA_data/Population.csv")
    population.rename(columns={'FIPS*': 'FIPS'}, inplace=True)
    population=population[["FIPS", "County name","Pop. 2010"]]
    population["YR"]=2010
    
    return population


#Unemployment data
def get_unemployment_data(filepath="USDA_data/Unemployment.csv"):
    unemployment_all=pd.read_csv(filepath)
    unemployment_all=unemployment_all[unemployment_all["State"].isin(["FL", "WA", "OR", "GA", "OK", "AL", "CO", "ME"])]
    unemployment_all = unemployment_all.iloc[:, :-2]
    unemployment_all_melt = unemployment_all.melt(
        id_vars=['FIPS_Code', 'State', 'Area_Name'],
        var_name='Metric',
        value_name='value'
    )
    unemployment_all_melt['YR'] = unemployment_all_melt['Metric'].str[-4:]
    unemployment_all_melt['Metric'] = unemployment_all_melt['Metric'].str[:-5]

    unemployment_all_pivot = unemployment_all_melt.pivot(
        index=['FIPS_Code', 'State', 'Area_Name', 'YR'],  # Identifiers for rows
        columns='Metric',                              # Column headers
        values='value'                                 # Values to populate
    )

    unemployment_all_pivot.reset_index(inplace=True)
    unemployment_all_data = unemployment_all_pivot[["FIPS_Code", "State", "Area_Name", "YR", "Civilian_labor_force", "Unemployment_rate"]]
    unemployment_all_data.rename(columns={'FIPS_Code': 'FIPS'}, inplace=True)
    unemployment_all_data.rename(columns={'Area_Name': 'Name'}, inplace=True)
    return unemployment_all_data


# Med-Income Data
def get_income_data(filepath="USDA_data/Unemployment.csv"):
    pre_income=pd.read_csv(filepath)
    pre_income=pre_income[pre_income["State"].isin(["FL", "WA", "OR", "GA", "OK", "AL", "CO", "ME"])]
    med_income_data=pre_income[["FIPS_Code","Median_Household_Income_2021", "Med_HH_Income_Percent_of_State_Total_2021"]]
    med_income_data.rename(columns={'FIPS_Code': 'FIPS'}, inplace=True)
    return med_income_data


def get_USDA_data():
    poverty_data=get_poverty_data(filepath="USDA_data/Poverty.csv")
    population2010_data=get_population_data(filepath="USDA_data/Population.csv")
    unemployment_all_data=get_unemployment_data(filepath="USDA_data/Unemployment.csv")
    med_income_data=get_income_data(filepath="USDA_data/Unemployment.csv")
    unemployment_all_data['YR'] = unemployment_all_data['YR'].astype('str')
    unemployment_all_data=unemployment_all_data.drop(columns=['Name', 'State'])
    
    # unemployment_all_data.to_parquet("data/USDA_education_unemployment.parquet", index=False)
    # population2021_data.to_parquet("data/USDA_population2021.parquet", index=False)
    # poverty_data.to_parquet("data/USDA_poverty2021.parquet", index=False)
    # med_income_data.to_parquet("data/USDA_medIncome2021.parquet", index=False)
    print(unemployment_all_data.head(1))
    print(population2010_data.head(1))
    print(poverty_data.head(1))
    print(med_income_data.head(1))
    print("save unemployment_all_data, population2021_data, poverty_data, med_income_data into data folder")

    return True

if __name__ == "__main__":
    get_USDA_data()