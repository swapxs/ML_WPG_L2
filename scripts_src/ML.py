from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession as ss
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler as va
from pyspark.ml.regression import LinearRegression as lreg
from pyspark.ml.evaluation import RegressionEvaluator as re

# ------------------------------------------------------------------------------
# 1. Initialize SparkSession (with Delta support)
# ------------------------------------------------------------------------------
builder = (
    ss.builder
        .appName("WindPowerPrediction")
        .master("spark://spark-master:7077")  # Or use your Spark master URL
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)
sprk = configure_spark_with_delta_pip(builder).getOrCreate()
sprk.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------------------
# 2. Read the Delta table
# ------------------------------------------------------------------------------
fp = "/data/delta_output"  # Path where your Kafka subscriber wrote Delta data
df = sprk.read.format("delta").load(fp)

# The schema includes:
# signal_date, signal_ts, create_date, create_ts, signals (map<string, string>)

# ------------------------------------------------------------------------------
# 3. Extract numeric columns from `signals` MapType
# ------------------------------------------------------------------------------
df = df.withColumn("active_power", col("signals")["LV ActivePower (kW)"].cast("double")) \
        .withColumn("wind_speed", col("signals")["Wind Speed (m/s)"].cast("double")) \
        .withColumn("theoretical_curve", col("signals")["Theoretical_Power_Curve (KWh)"].cast("double")) \
        .withColumn("wind_direction", col("signals")["Wind Direction (°)"].cast("double"))

# Filter out any rows with nulls in the columns we need
df = df.na.drop(subset=["active_power", "wind_speed", "theoretical_curve", "wind_direction"])

# ------------------------------------------------------------------------------
# 4. Build features & label, then split into train/test
# ------------------------------------------------------------------------------
feature_cols = ["wind_speed", "theoretical_curve", "wind_direction"]
assembler = va(inputCols=feature_cols, outputCol="features")

# We’ll consider predicting `active_power`:
assembled = assembler.transform(df).select("signal_date", "signal_ts", "features", "active_power")

# Train/test split (e.g., 80/20)
train_data, test_data = assembled.randomSplit([0.8, 0.2], seed=42)

# ------------------------------------------------------------------------------
# 5. Train a simple Regression Model (e.g., LinearRegression)
# ------------------------------------------------------------------------------
lr = lreg(featuresCol="features", labelCol="active_power", maxIter=50)
model = lr.fit(train_data)

# ------------------------------------------------------------------------------
# 6. Evaluate on the test set
# ------------------------------------------------------------------------------
predictions = model.transform(test_data)

evaluator = re(
    labelCol="active_power",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
r2 = re(
    labelCol="active_power",
    predictionCol="prediction",
    metricName="r2"
).evaluate(predictions)

print(f"Test RMSE: {rmse}")
print(f"Test R^2: {r2}")

# ------------------------------------------------------------------------------
# 7. Predicting Active Power for a single day (example)
# ------------------------------------------------------------------------------
# Suppose we want to generate predictions for a specific day in the dataset or new data.
# We'll filter an existing day (e.g., 2018-01-02) as a small example.
# NOTE: The sample dataset might only have 01-01-2018; adapt as needed.
day_to_predict = "2018-01-01"
future_df = assembled.filter(col("signal_date") == day_to_predict)

# Generate predictions
future_preds = model.transform(future_df).select("signal_date", "signal_ts", "prediction")
future_preds.show(30, truncate=False)

# Stop the sprk session
sprk.stop()
