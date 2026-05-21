from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# Starting spark session (entry point to everything in spark)
spark = SparkSession.builder \
    .appName("Taxi Fare Prediction") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

# Only show warnings not the extensive info logs
spark.sparkContext.setLogLevel("WARN")

print("Spark started successfully")

# Loading clean data from HDFS
df = spark.read.parquet("hdfs://namenode:9000/taxi/processed/2023/01/")
print(f"Loaded rows: {df.count()}")

# Selecting only columns needed for the model and dropping null rows
model_df = df.select(
    "fare_amount",
    "trip_distance",
    "hour_of_day",
    "PULocationID",
    "DOLocationID",
    "trip_duration_mins",
    "passenger_count"
).dropna()

print(f"Rows after dropping nulls: {model_df.count()}")

# Splitting data — 80% for training, 20% for testing
# seed=42 makes split reproducible (same split every run)
train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training rows: {train_df.count()}")
print(f"Test rows: {test_df.count()}")

# Combining all feature columns into one vector column
# MLlib requires all inputs packed into a single column like [2.5, 17, 132, 90, 8.4, 2]
assembler = VectorAssembler(
    inputCols=[
        "trip_distance",
        "hour_of_day",
        "PULocationID",
        "DOLocationID",
        "trip_duration_mins",
        "passenger_count"
    ],
    outputCol="features"
)

# Defining the linear regression model
lr = LinearRegression(
    featuresCol="features",     # input vector column
    labelCol="fare_amount",     # what we are predicting
    maxIter=10,                 # number of training iterations
    regParam=0.1,               # prevents overfitting
    elasticNetParam=0.8         # mix of L1 and L2 regularization
)

# Building pipeline — runs assembler then model in sequence
pipeline = Pipeline(stages=[assembler, lr])

# Training the model on 80% of data
print("\nTraining model...")
model = pipeline.fit(train_df)
print("Training complete")

# Running predictions on the 20% test data
predictions = model.transform(test_df)

# Evaluating model performance
evaluator = RegressionEvaluator(
    labelCol="fare_amount",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator.evaluate(predictions)
r2   = evaluator.setMetricName("r2").evaluate(predictions)
mae  = evaluator.setMetricName("mae").evaluate(predictions)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"RMSE (avg dollar error):    ${rmse:.2f}")
print(f"MAE  (mean abs error):      ${mae:.2f}")
print(f"R2   (variance explained):  {r2:.4f}")

# Showing sample predictions vs actual fares
print("\n=== SAMPLE PREDICTIONS vs ACTUAL ===")
predictions.select(
    "trip_distance",
    "hour_of_day",
    "trip_duration_mins",
    "fare_amount",
    "prediction"
).show(10, truncate=False)

# Extracting what the model learned — coefficients per feature
lr_model = model.stages[-1]

print("\n=== WHAT THE MODEL LEARNED ===")
print(f"Base fare (intercept): ${lr_model.intercept:.2f}")

feature_names = [
    "trip_distance",
    "hour_of_day",
    "PULocationID",
    "DOLocationID",
    "trip_duration_mins",
    "passenger_count"
]

print("\nFeature coefficients (how much each unit changes the fare):")
for name, coef in zip(feature_names, lr_model.coefficients):
    print(f"  {name:25s}: {coef:.4f}")

# Saving trained model to HDFS for future use
model_path = "hdfs://namenode:9000/taxi/models/fare_prediction_lr"
model.write().overwrite().save(model_path)
print(f"\nModel saved to HDFS at: {model_path}")

spark.stop()
print("Done. Spark session closed.")