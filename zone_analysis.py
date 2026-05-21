from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Zone Analysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Read clean trips from HDFS
trips = spark.read.parquet("hdfs://namenode:9000/taxi/processed/2023/01/")

# Read zone lookup from HDFS
zones = spark.read.option("header", "true") \
    .option("inferSchema", "true") \
    .csv("hdfs://namenode:9000/taxi/lookup/taxi_zone_lookup.csv")

# Join trips with zones on pickup location
joined = trips.join(zones, trips.PULocationID == zones.LocationID)

# Top 10 pickup locations
print("=== TOP 10 PICKUP LOCATIONS ===")
joined.groupBy("Zone", "Borough") \
    .count() \
    .orderBy("count", ascending=False) \
    .show(10, truncate=False)

# Trips per borough
print("=== TRIPS PER BOROUGH ===")
joined.groupBy("Borough") \
    .count() \
    .orderBy("count", ascending=False) \
    .show()

spark.stop()