from delta import configure_spark_with_delta_pip as csp
from pyspark.sql import SparkSession as ss
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler as va
from pyspark.ml.regression import LinearRegression as lreg
from pyspark.ml.evaluation import RegressionEvaluator as re

# For plotting
import matplotlib
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Initialize SparkSession (with Delta support)
# ------------------------------------------------------------------------------
scb = (
    ss.builder
        .appName("Prediction") \
        .master("spark://spark-master:7077") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.memory", "10g")\
        .config("spark.cores.max", "4") \
)
sprk = csp(scb).getOrCreate()
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
df = (
    df.withColumn("active_power", col("signals")["LV ActivePower (kW)"].cast("double"))
      .withColumn("wind_speed", col("signals")["Wind Speed (m/s)"].cast("double"))
      .withColumn("theoretical_curve", col("signals")["Theoretical_Power_Curve (KWh)"].cast("double"))
      .withColumn("wind_direction", col("signals")["Wind Direction (°)"].cast("double"))
)

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
pre = model.transform(test_data)

eval_rmse = re(
    labelCol="active_power",
    predictionCol="prediction",
    metricName="rmse"
)

eval_r2 = re(
    labelCol="active_power",
    predictionCol="prediction",
    metricName="r2"
)

rmse = eval_rmse.evaluate(pre)
r2 = eval_r2.evaluate(pre)

print(f"Test RMSE: {rmse}")
print(f"Test R^2: {r2}")

# ------------------------------------------------------------------------------
# 6a. Plot (Test Set) - Actual vs. Predicted
# ------------------------------------------------------------------------------
# Convert Spark DataFrame -> Pandas for plotting
img = (
    pre
    .select("signal_ts", "active_power", "prediction")
    .orderBy("signal_ts")  # ensure ascending timestamps
    .toPandas()
)

plt.figure(figsize=(10, 5))
plt.plot(img["signal_ts"], img["active_power"], label="Actual")
plt.plot(img["signal_ts"], img["prediction"], label="Predicted")
plt.xlabel("Timestamp")
plt.ylabel("Active Power (kW)")
plt.title("Test Set: Actual vs. Predicted Active Power")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# 6b. Scatter plot: Test Set - Actual vs. Predicted
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.scatter(img["active_power"], img["prediction"], alpha=0.6)

# Plot a reference line y = x
min_val = min(img["active_power"].min(), img["prediction"].min())
max_val = max(img["active_power"].max(), img["prediction"].max())
plt.plot([min_val, max_val], [min_val, max_val], label="y = x")

plt.xlabel("Actual Active Power (kW)")
plt.ylabel("Predicted Active Power (kW)")
plt.title("Test Set Scatter Plot: Actual vs. Predicted")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# 7. Predicting Active Power for a single day (example)
# ------------------------------------------------------------------------------
# Suppose we want to generate predictions for a specific day in the dataset.
# NOTE: The sample dataset might only have 2018-01-01. Adapt as needed.
date = "2018-02-11"
future_df = assembled.filter(col("signal_date") == date)

# Generate predictions
future_preds = model.transform(future_df).select("signal_date", "signal_ts", "active_power", "prediction")
future_preds.show(30, truncate=False)

# ------------------------------------------------------------------------------
# 7a. Plot single-day predictions
# ------------------------------------------------------------------------------
pdf_future = future_preds.orderBy("signal_ts").toPandas()

plt.figure(figsize=(10, 5))
plt.plot(pdf_future["signal_ts"], pdf_future["active_power"], label="Actual")
plt.plot(pdf_future["signal_ts"], pdf_future["prediction"], label="Predicted")
plt.xlabel("Timestamp")
plt.ylabel("Active Power (kW)")
plt.title(f"Predictions for day: {date}")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# 8. Stop the Spark session
# ------------------------------------------------------------------------------
sprk.stop()
