import pandas as pd

# Load data
df = pd.read_parquet("data/yellow_tripdata_2023-01.parquet")

# Looking for its shape
print("Shape")
print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")

# What columns exists and what type are they?
print("Column and Types")
print(df.dtypes)

# Looking for rows in dataframe
print(df.head(3).to_string)

# Basic stats on numeric rows of dataset
print(df[['fare_amount', 'trip_distance', 'passenger_count', 'tip_amount']].describe())

# Checking for useless data
print(df.isnull().sum)

# Exploring zones
zones = pd.read_csv("data/taxi_zone_lookup.csv")
print(zones.head(20))
print(f"Total zones: {len(zones)}")
print(f"Borough: {zones["Borough"].unique()}")