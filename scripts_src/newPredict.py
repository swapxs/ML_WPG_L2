import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    to_timestamp,
    regexp_replace
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    # ------------------------------------------------------------------------------
    # 1) Initialize Spark Session
    # ------------------------------------------------------------------------------
    spark = (SparkSession.builder
             .appName("WindPowerCSVLinearRegression")
             .master("local[*]")  # or 'spark://spark-master:7077' inside Docker
             .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")

    # ------------------------------------------------------------------------------
    # 2) Read the CSV file
    # ------------------------------------------------------------------------------
    # Ensure the CSV path is correct or mounted in Docker at /data/dataset.csv
    csv_path = "/data/dataset.csv"
    
    # Because the CSV date/time format is not a standard one, we can pre-process it.
    # Example format in the sample: "01 01 2018 00:00"
    # We'll treat it as "dd MM yyyy HH:mm".
    # We'll read it in as strings first, then parse carefully.

    df_raw = (spark.read
              .option("header", True)
              .option("inferSchema", True)
              .csv(csv_path))

    # Preview
    print("Original CSV data sample:")
    df_raw.show(5, truncate=False)
    df_raw.printSchema()

    # ------------------------------------------------------------------------------
    # 3) Clean up columns & parse the date/time
    # ------------------------------------------------------------------------------
    # The CSV columns are something like:
    #   Date/Time, LV ActivePower (kW), Wind Speed (m/s), Theoretical_Power_Curve (KWh), Wind Direction (°)
    # We'll rename them to simpler column names:
    df = (df_raw
          .withColumnRenamed("LV ActivePower (kW)", "active_power")
          .withColumnRenamed("Wind Speed (m/s)", "wind_speed")
          .withColumnRenamed("Theoretical_Power_Curve (KWh)", "theoretical_power")
          .withColumnRenamed("Wind Direction (°)", "wind_direction")
          )

    # Convert the Date/Time string to a timestamp using the format "dd MM yyyy HH:mm".
    # First we might want to unify the spacing or ensure the columns are zero-padded, etc.
    # The sample data seems consistent. We'll parse it with to_timestamp and that pattern:
    # "01 01 2018 00:00" => day=01, month=01, year=2018, hour=00, minute=00
    df = df.withColumn(
        "date_time_parsed",
        to_timestamp(col("Date/Time"), "dd MM yyyy HH:mm")
    )

    # Drop rows with any nulls in the columns we need.
    df = df.dropna(subset=["active_power", "wind_speed", "theoretical_power", "wind_direction", "date_time_parsed"])

    print("Cleaned dataframe:")
    df.show(5, truncate=False)

    # ------------------------------------------------------------------------------
    # 4) Generate some exploratory visualizations
    #    We can collect a small portion of the data to the driver and plot it.
    # ------------------------------------------------------------------------------
    # For example, let's create a scatter plot of active_power vs wind_speed.
    sample_data = df.sample(fraction=0.1, seed=42).toPandas()  # 10% sample
    plt.figure()
    plt.scatter(sample_data["wind_speed"], sample_data["active_power"])
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Active Power (kW)")
    plt.title("Active Power vs. Wind Speed (Sample)")
    plt.show()

    # ------------------------------------------------------------------------------
    # 5) Assemble features for the linear regression
    #    We'll predict 'active_power' from [wind_speed, theoretical_power, wind_direction].
    # ------------------------------------------------------------------------------
    feature_cols = ["wind_speed", "theoretical_power", "wind_direction"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    df_assembled = assembler.transform(df).select("date_time_parsed", "features", col("active_power").alias("label"))

    # ------------------------------------------------------------------------------
    # 6) Split data into train/test sets and train the regression
    # ------------------------------------------------------------------------------
    train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=42)
    lr = LinearRegression(featuresCol="features", labelCol="label", maxIter=50)
    model = lr.fit(train_data)

    # ------------------------------------------------------------------------------
    # 7) Evaluate on the test set
    # ------------------------------------------------------------------------------
    test_predictions = model.transform(test_data)
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(test_predictions)
    r2 = evaluator_r2.evaluate(test_predictions)

    print("Test set evaluation metrics:")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R^2  = {r2:.2f}")

    # ------------------------------------------------------------------------------
    # 8) Plot Predictions vs. Actual for the test set
    # ------------------------------------------------------------------------------
    # We'll collect them to a pandas DataFrame for plotting.
    pd_preds = test_predictions.select("date_time_parsed", "prediction", "label").orderBy("date_time_parsed").toPandas()

    plt.figure()
    plt.plot(pd_preds["date_time_parsed"], pd_preds["label"], label="Actual", marker="o")
    plt.plot(pd_preds["date_time_parsed"], pd_preds["prediction"], label="Predicted", marker="x")
    plt.xlabel("Timestamp")
    plt.ylabel("Active Power (kW)")
    plt.title("Comparison of Actual vs. Predicted Active Power (Test Set)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------
    # 9) Show an example prediction for a “single day”
    #    For instance, filter the date_time_parsed for 2018-01-01 in the data.
    # ------------------------------------------------------------------------------
    # (This example depends on whether your dataset actually has that date range.)
    single_day_df = df_assembled.filter("date_time_parsed >= '2018-01-01' AND date_time_parsed < '2018-01-02'")
    single_day_pred = model.transform(single_day_df).orderBy("date_time_parsed")

    print("Predictions for 2018-01-01:")
    single_day_pred.select("date_time_parsed", "prediction", "label").show(50, truncate=False)

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()
