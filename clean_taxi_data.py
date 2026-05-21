from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, unix_timestamp, round, hour, dayofweek, when)

# Starting a spark session (entery point to everything in spark)
spark = SparkSession.builder \
    .appName("NYC Taxi Data Cleaning") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

# Only give warning not the extensive logs information
spark.sparkContext.setLogLevel("WARN")

print("Spark started successfully")

# Reading RAW data from HDFS
raw_df = spark.read.parquet("hdfs://namenode:9000/taxi/raw/2023/01/")

print("Raw data loaded successfully.")
print(f"Rows: {raw_df.count():,}")
print(f"Columns: {raw_df.columns}")

# Cleaning data (removing bad / useless rows)
cleaned_df = raw_df \
    .filter(col("fare_amount") > 0) \
    .filter(col("fare_amount") < 500) \
    .filter(col("trip_distance") > 0) \
    .filter(col("trip_distance") < 100) \
    .filter(col("passenger_count") > 0) \
    .filter(col("passenger_count") <= 6) \
    .filter(col("tip_amount") >= 0) \
    .filter(col("total_amount") > 0) \
    .filter(col("tpep_pickup_datetime").isNotNull()) \
    .filter(col("tpep_dropoff_datetime").isNotNull()) \
    .filter(col("PULocationID").isNotNull()) \
    .filter(col("DOLocationID").isNotNull())

print("Bad rows removed successfully.")
print(f"Cleaned row count: {cleaned_df.count():,}")

# Performing feature engineering
featured_df = cleaned_df \
    .withColumn(
        "trip_duration_mins",
        round(
            (unix_timestamp("tpep_dropoff_datetime") -
             unix_timestamp("tpep_pickup_datetime")) / 60,
             2
        )
    ) \
    .withColumn(
        # What hour of day did the pickup happen?
        "hour_of_day",
        hour("tpep_pickup_datetime")
    ) \
    .withColumn(
        # Speed in mph: distance / (duration in hours)
        "speed_mph",
        when(
            col("trip_duration_mins") > 0,
            round(col("trip_distance") / (col("trip_duration_mins") / 60), 2)
        ).otherwise(0)
    )

# Remove rows with impossible derived values
featured_df = featured_df \
    .filter(col("trip_duration_mins") > 0) \
    .filter(col("trip_duration_mins") < 180) \
    .filter(col("speed_mph") < 100)

print("Removed rows with impossible derived values")
print(f"Final row count: {featured_df.count():,}")

# Previewing the cleaned data
print("Sample of clean data:")
featured_df.select(
    "tpep_pickup_datetime",
    "trip_distance",
    "fare_amount",
    "trip_duration_mins",
    "hour_of_day",
    "speed_mph"
).show(5, truncate=False)

# Writing cleaned data back to HDFS
output_path = "hdfs://namenode:9000/taxi/processed/2023/01/"

featured_df.write \
    .mode("overwrite") \
    .partitionBy("hour_of_day") \
    .parquet(output_path)

print(f"Clean data written to HDFS at {output_path}")
print("Partitioned by hour_of_day")

# Verify the output
verify_df = spark.read.parquet(output_path)
print(f"Verification read:")
print(f"Rows in output: {verify_df.count():,}")
print(f"Columns in output: {verify_df.columns}")

spark.stop()
print("Done. Spark session closed.")