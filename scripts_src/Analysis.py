################################################################################
# TASK 3 - Using Delta Lake via configure_spark_with_delta_pip
################################################################################

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    countDistinct,
    hour,
    avg,
    when,
    lit,
    col,
    explode,
    create_map,
    broadcast
)
from pyspark.sql.types import StructType, sf, srt

# ------------------------------------------------------------------------------
# 1) Initialize SparkSession with Delta Lake
# ------------------------------------------------------------------------------
builder = (
    SparkSession.builder
        .appName("Task3-Analysis")
        .master("spark://spark-master:7077") 
        # The critical configs to enable Delta Lake
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # If you also need Kafka in this job, you can append:
        # .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5")
)

# This call modifies the builder so that we can use Delta in Python.
spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------------------
# 2) Read from the Delta table created in Task 2
# ------------------------------------------------------------------------------
delta_path = "/data/delta_output"  # Adjust if needed
df = spark.read.format("delta").load(delta_path)

print("Delta table schema:")
df.printSchema()

print("Sample rows from Delta table:")
df.show(5, truncate=False)

# The schema is typically:
# root
#  |-- signal_date: date (nullable = true)
#  |-- signal_ts: timestamp (nullable = true)
#  |-- create_date: date (nullable = true)
#  |-- create_ts: timestamp (nullable = true)
#  |-- signals: map (key: string, value: string) (nullable = true)

# ------------------------------------------------------------------------------
# 3) Calculate number of distinct `signal_ts` per day
# ------------------------------------------------------------------------------
daily_distinct_ts = (
    df.groupBy("signal_date")
      .agg(countDistinct("signal_ts").alias("distinct_ts_count"))
)

print("Distinct signal_ts counts per day:")
daily_distinct_ts.show()

# ------------------------------------------------------------------------------
# 4) Calculate average value of all signals per hour
# ------------------------------------------------------------------------------
# We have these signals stored in df["signals"], a MapType. Extract and cast them:

lv_active_power_col = col("signals")["LV ActivePower (kW)"].cast("double")
wind_speed_col      = col("signals")["Wind Speed (m/s)"].cast("double")
theoretical_col     = col("signals")["Theoretical_Power_Curve (KWh)"].cast("double")
wind_dir_col        = col("signals")["Wind Direction (°)"].cast("double")

# Group by date + hour of signal_ts, then compute averages:
grouped = (
    df.groupBy(
        "signal_date",
        hour("signal_ts").alias("hour_of_day")
    )
    .agg(
        avg(lv_active_power_col).alias("avg_active_power"),
        avg(wind_speed_col).alias("avg_wind_speed"),
        avg(theoretical_col).alias("avg_theoretical_power_curve"),
        avg(wind_dir_col).alias("avg_wind_direction")
    )
)

print("Hourly average signals:")
grouped.show(10, truncate=False)

# ------------------------------------------------------------------------------
# 5) Add generation_indicator column
# ------------------------------------------------------------------------------
# Conditions:
#   a) LV ActivePower<200 => "Low"
#   b) 200 <= LV ActivePower < 600 => "Medium"
#   c) 600 <= LV ActivePower < 1000 => "High"
#   d) >= 1000 => "Exceptional"

gen_indicator_df = (
    grouped.withColumn(
        "generation_indicator",
        when(col("avg_active_power") < 200, "Low")
        .when((col("avg_active_power") >= 200) & (col("avg_active_power") < 600), "Medium")
        .when((col("avg_active_power") >= 600) & (col("avg_active_power") < 1000), "High")
        .otherwise("Exceptional")
    )
)

print("Added generation_indicator column:")
gen_indicator_df.show(10, truncate=False)

# ------------------------------------------------------------------------------
# 6) Create a new DataFrame with JSON-based signal mapping and do broadcast join
# ------------------------------------------------------------------------------
# Usually, you'd have a small JSON or structured data that says:
# [
#   { "sig_name": "LV ActivePower (kW)", "sig_mapping_name": "active_power_average" },
#   { "sig_name": "Wind Speed (m/s)", "sig_mapping_name": "wind_speed_average" },
#   { "sig_name": "Theoretical_Power_Curve (KWh)", "sig_mapping_name": "theo_power_curve_average" },
#   { "sig_name": "Wind Direction (°)", "sig_mapping_name": "wind_direction_average" }
# ]
# We'll create that as a small Spark DataFrame:

mapping_data = [
    ("LV ActivePower (kW)", "active_power_average"),
    ("Wind Speed (m/s)", "wind_speed_average"),
    ("Theoretical_Power_Curve (KWh)", "theo_power_curve_average"),
    ("Wind Direction (°)", "wind_direction_average")
]

mapping_schema = st([
    sf("sig_name", srt(), True),
    sf("sig_mapping_name", srt(), True)
])

mapping_df = spark.createDataFrame(mapping_data, mapping_schema)

# We want to pivot the 'gen_indicator_df' so that the four average columns become rows.
long_format_df = (
    gen_indicator_df
    .select(
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       create_map(
         lit("LV ActivePower (kW)"), col("avg_active_power"),
         lit("Wind Speed (m/s)"), col("avg_wind_speed"),
         lit("Theoretical_Power_Curve (KWh)"), col("avg_theoretical_power_curve"),
         lit("Wind Direction (°)"), col("avg_wind_direction")
       ).alias("metrics")
    )
    # 'explode' transforms the map into key-value pairs
    .select(
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       explode(col("metrics")).alias("sig_name", "value")
    )
)

# Now do a broadcast join with mapping_df to get the user-friendly name
joined_df = (
    long_format_df
    .join(broadcast(mapping_df), on="sig_name", how="left")
    .select(
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       "sig_name",
       "sig_mapping_name",
       "value"
    )
)

print("Final joined data (signal name mapped to user-friendly):")
joined_df.show(20, truncate=False)

# (Optional) you can write out joined_df to a new Delta table, Parquet, etc.

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------
spark.stop()
