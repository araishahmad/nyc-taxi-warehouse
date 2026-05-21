from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, count, avg

# Starting spark session
spark = SparkSession.builder \
    .appName("Export for Superset") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Loading clean trips and zones data
trips = spark.read.parquet("hdfs://namenode:9000/taxi/processed/2023/01/")
zones = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://namenode:9000/taxi/lookup/taxi_zone_lookup.csv")

# Dataset 1 — trips by hour of day
print("Exporting trips by hour...")
trips.groupBy("hour_of_day") \
    .agg(
        count("*").alias("trip_count"),
        round(avg("fare_amount"), 2).alias("avg_fare"),
        round(avg("trip_distance"), 2).alias("avg_distance"),
        round(avg("trip_duration_mins"), 2).alias("avg_duration")
    ) \
    .orderBy("hour_of_day") \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv("/tmp/superset_exports/trips_by_hour")

# Dataset 2 — top pickup zones (joined with zone names)
print("Exporting top pickup zones...")
trips.join(zones, trips.PULocationID == zones.LocationID) \
    .groupBy("Zone", "Borough") \
    .agg(
        count("*").alias("pickup_count"),
        round(avg("fare_amount"), 2).alias("avg_fare")
    ) \
    .orderBy("pickup_count", ascending=False) \
    .limit(50) \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv("/tmp/superset_exports/top_pickup_zones")

# Dataset 3 — borough summary
print("Exporting borough summary...")
trips.join(zones, trips.PULocationID == zones.LocationID) \
    .groupBy("Borough") \
    .agg(
        count("*").alias("trip_count"),
        round(avg("fare_amount"), 2).alias("avg_fare"),
        round(avg("trip_distance"), 2).alias("avg_distance")
    ) \
    .orderBy("trip_count", ascending=False) \
    .coalesce(1) \
    .write.mode("overwrite") \
    .option("header", "true") \
    .csv("/tmp/superset_exports/borough_summary")

print("All exports done.")
spark.stop()