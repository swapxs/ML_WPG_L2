from delta import configure_spark_with_delta_pip as cdp
from pyspark.sql import SparkSession as ss
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

from pyspark.sql.types import (
    StructType as st,
    StructField as sf,
    StringType as srt,
)

scb = (
    ss.builder
        .appName("Analysis") \
        .master("spark://spark-master:7077") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.executor.cores", "4") \
        .config("spark.executor.memory", "10g")\
        .config("spark.cores.max", "4") \
)

sprk = cdp(scb).getOrCreate()

fp = "/data/delta_output"
df = sprk.read.format("delta").load(fp)
df.printSchema()
df.show(5)

dts = (
    df.groupBy("signal_date")
      .agg(countDistinct("signal_ts").alias("distinct_ts_count"))
)

dts.show(5)

lv_active_power_col = col("signals")["LV ActivePower (kW)"].cast("double")
wind_speed_col      = col("signals")["Wind Speed (m/s)"].cast("double")
theoretical_col     = col("signals")["Theoretical_Power_Curve (KWh)"].cast("double")
wind_dir_col        = col("signals")["Wind Direction (°)"].cast("double")

grouped = (
    df.groupBy( "signal_date", hour("signal_ts").alias("hour_of_day"))
    .agg(
        avg(lv_active_power_col).alias("avg_active_power"),
        avg(wind_speed_col).alias("avg_wind_speed"),
        avg(theoretical_col).alias("avg_theoretical_power_curve"),
        avg(wind_dir_col).alias("avg_wind_direction")
    )
)

grouped.show(5)

gen_indicator = (
    grouped.withColumn(
        "generation_indicator",
        when(col("avg_active_power") < 200, "Low")
        .when((col("avg_active_power") >= 200) & (col("avg_active_power") < 600), "Medium")
        .when((col("avg_active_power") >= 600) & (col("avg_active_power") < 1000), "High")
        .otherwise("Exceptional")
    )
)

gen_indicator.show(5)

data = [
    ("LV ActivePower (kW)", "active_power_average"),
    ("Wind Speed (m/s)", "wind_speed_average"),
    ("Theoretical_Power_Curve (KWh)", "theo_power_curve_average"),
    ("Wind Direction (°)", "wind_direction_average")
]

schema = st([
    sf("sig_name", srt(), True),
    sf("sig_mapping_name", srt(), True)
])

new_df = sprk.createDataFrame(data, schema)

formatted_df = (
    gen_indicator
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
    .select(
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       explode(col("metrics")).alias("sig_name", "value")
    )
)

joined_df = (
    formatted_df
    .join(broadcast(new_df), on="sig_name", how="left")
    .select(
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       "sig_name",
       "sig_mapping_name",
       "value"
    )
)

joined_df.show()

sprk.stop()
