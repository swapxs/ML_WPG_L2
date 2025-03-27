from delta import configure_spark_with_delta_pip as csp
from pyspark.sql import SparkSession as ss
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler as va
from pyspark.ml.regression import LinearRegression as lreg
from pyspark.ml.evaluation import RegressionEvaluator as re
import matplotlib.pyplot as plt

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

fp = "/data/delta_output"

df = sprk.read.format("delta").load(fp)

df.count()

df = (
    df.withColumn("active_power", col("signals")["LV ActivePower (kW)"].cast("double"))
      .withColumn("wind_speed", col("signals")["Wind Speed (m/s)"].cast("double"))
      .withColumn("theoretical_curve", col("signals")["Theoretical_Power_Curve (KWh)"].cast("double"))
      .withColumn("wind_direction", col("signals")["Wind Direction (°)"].cast("double"))
)

df = df.na.drop(subset=["active_power", "wind_speed", "theoretical_curve", "wind_direction"])

df.show()
df.count()

feature_cols = ["wind_speed", "theoretical_curve", "wind_direction"]

assembler = va(inputCols=feature_cols, outputCol="features")

assembled = assembler.transform(df).select("signal_date", "signal_ts", "features", "active_power")

train_data, test_data = assembled.randomSplit([0.8, 0.2], seed=42)

train_data.count()

test_data.count()

lr = lreg(featuresCol="features", labelCol="active_power", maxIter=50)

model = lr.fit(train_data)

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

print(rmse, r2)

img = (
    pre
    .select("signal_ts", "active_power", "prediction")
    .orderBy("signal_ts")  # ensure ascending timestamps
    .toPandas()
)

plt.figure(figsize=(15, 5))
plt.plot(img["signal_ts"], img["active_power"], label="Actual")

plt.plot(img["signal_ts"], img["prediction"], label="Predicted")

plt.xlabel("Timestamp")
plt.ylabel("Active Power (kW)")
plt.title("Test Set: Actual vs. Predicted Active Power")

plt.legend()
plt.show()

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


# Prediction Part
date = "2018-02-11"

future_df = assembled.filter(col("signal_date") == date)

future_preds = model.transform(future_df).select("signal_date", "signal_ts", "active_power", "prediction")

future_preds.show(30, truncate=False)

img = future_preds.orderBy("signal_ts").toPandas()

plt.figure(figsize=(12, 5))
plt.plot(img["signal_ts"], img["active_power"], label="Actual")
plt.plot(img["signal_ts"], img["prediction"], label="Predicted")
plt.xlabel("Timestamp")
plt.ylabel("Active Power (kW)")
plt.title(f"Predictions for day: {date}")
plt.legend()
plt.show()

sprk.stop()
