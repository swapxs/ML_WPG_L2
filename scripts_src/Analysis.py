# Import the necessary function from Delta Lake library to configure Spark Session easily
from delta import configure_spark_with_delta_pip as cdp
# Import the SparkSession class and alias it as ss
from pyspark.sql import SparkSession as ss
# Import necessary Spark SQL functions for data manipulation and aggregation
from pyspark.sql.functions import (
    countDistinct,  # Counts distinct values in a column
    hour,           # Extracts the hour from a timestamp column
    avg,            # Calculates the average of a column
    when,           # Conditional logic (like CASE WHEN in SQL)
    lit,            # Creates a literal value column
    col,            # Selects a column by name
    explode,        # Transforms elements of an array or map into multiple rows
    create_map,     # Creates a map column from key-value pairs
    broadcast       # Hints Spark to broadcast a smaller DataFrame for joins
)
# Import types for defining DataFrame schemas
from pyspark.sql.types import (
    StructType as st,    # Defines the structure of a DataFrame
    StructField as sf,   # Defines a single field within a StructType
    StringType as srt,   # Defines a string data type for a field
)

# Start building the SparkSession configuration
scb = (
    ss.builder
        .appName("Analysis")                                # Set the application name
        .master("spark://spark-master:7077")                # Set the Spark master URL
        # Configure Spark SQL extensions to enable Delta Lake capabilities
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        # Configure the Spark catalog to use the Delta Catalog for managing Delta tables
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Set resource allocation for Spark executors
        .config("spark.executor.cores", "4")                # Number of cores per executor
        .config("spark.executor.memory", "10g")             # Amount of memory per executor
        .config("spark.cores.max", "4")                     # Maximum total cores for the application
)

# Configure the SparkSession builder with Delta Lake support using the imported helper function
# and then create or get the SparkSession instance
sprk = cdp(scb).getOrCreate()

# Define the file path to the Delta table created in the previous step
fp = "/data/delta_output"
# Read the data from the Delta table into a DataFrame
df = sprk.read.format("delta").load(fp)
# Print the schema (column names and data types) of the loaded DataFrame
df.printSchema()
# Show the first 5 rows of the DataFrame to inspect the data
df.show(5)

# Calculate the number of distinct timestamps for each signal date
dts = (
    df.groupBy("signal_date")                         # Group the DataFrame by the 'signal_date' column
      .agg(countDistinct("signal_ts").alias("distinct_ts_count")) # Aggregate by counting distinct 'signal_ts' values, naming the result 'distinct_ts_count'
)

# Show the first 5 rows of the distinct timestamp counts per date
dts.show(5)

# Extract values from the 'signals' map column and cast them to double type for calculations.
# This makes accessing these values easier in subsequent steps.
lv_active_power_col = col("signals")["LV ActivePower (kW)"].cast("double") # Get value for key 'LV ActivePower (kW)' and cast to double
wind_speed_col      = col("signals")["Wind Speed (m/s)"].cast("double")      # Get value for key 'Wind Speed (m/s)' and cast to double
theoretical_col     = col("signals")["Theoretical_Power_Curve (KWh)"].cast("double") # Get value for key 'Theoretical_Power_Curve (KWh)' and cast to double
wind_dir_col        = col("signals")["Wind Direction (°)"].cast("double")          # Get value for key 'Wind Direction (°)' and cast to double

# Group the data by date and hour, then calculate average values for the key metrics within each group
grouped = (
    # Group by both the signal date and the hour extracted from the signal timestamp
    df.groupBy( "signal_date", hour("signal_ts").alias("hour_of_day"))
    # Calculate averages for the extracted numeric signal columns
    .agg(
        avg(lv_active_power_col).alias("avg_active_power"),              # Average active power
        avg(wind_speed_col).alias("avg_wind_speed"),                    # Average wind speed
        avg(theoretical_col).alias("avg_theoretical_power_curve"),     # Average theoretical power
        avg(wind_dir_col).alias("avg_wind_direction")                  # Average wind direction
    )
)

# Show the first 5 rows of the aggregated hourly averages
grouped.show(5)

# Add a 'generation_indicator' column based on the average active power
gen_indicator = (
    grouped.withColumn( # Add a new column or replace an existing one
        "generation_indicator", # Name of the new column
        # Use 'when' conditions (similar to CASE WHEN) to categorize the power generation
        when(col("avg_active_power") < 200, "Low")  # If avg power < 200, label as "Low"
        .when((col("avg_active_power") >= 200) & (col("avg_active_power") < 600), "Medium") # If between 200 and 600, label as "Medium"
        .when((col("avg_active_power") >= 600) & (col("avg_active_power") < 1000), "High")   # If between 600 and 1000, label as "High"
        .otherwise("Exceptional") # For all other cases (>= 1000), label as "Exceptional"
    )
)

# Show the first 5 rows of the DataFrame with the added generation indicator
gen_indicator.show(5)

# --- Prepare data for unpivoting/exploding the aggregated metrics ---

# Define data for a small mapping DataFrame. This will map the original signal names
# to potentially different, more user-friendly names or IDs if needed later (here, it's used for joining).
data = [
    ("LV ActivePower (kW)", "active_power_average"),
    ("Wind Speed (m/s)", "wind_speed_average"),
    ("Theoretical_Power_Curve (KWh)", "theo_power_curve_average"),
    ("Wind Direction (°)", "wind_direction_average")
]

# Define the schema for the mapping DataFrame
schema = st([
    sf("sig_name", srt(), True),         # Original signal name (will be used as join key)
    sf("sig_mapping_name", srt(), True)  # New mapped name/ID for the signal
])

# Create the small mapping DataFrame in Spark from the defined data and schema
new_df = sprk.createDataFrame(data, schema)

# --- Transform the aggregated data into a 'long' format ---

# Reshape the 'gen_indicator' DataFrame: create a map of metrics, then explode it.
formatted_df = (
    gen_indicator
    .select( # Select specific columns and create a new map column
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       # Create a map where keys are signal names (literals) and values are the corresponding average columns
       create_map(
         lit("LV ActivePower (kW)"), col("avg_active_power"),                # Key-Value pair 1
         lit("Wind Speed (m/s)"), col("avg_wind_speed"),                    # Key-Value pair 2
         lit("Theoretical_Power_Curve (KWh)"), col("avg_theoretical_power_curve"), # Key-Value pair 3
         lit("Wind Direction (°)"), col("avg_wind_direction")                # Key-Value pair 4
       ).alias("metrics") # Name the new map column 'metrics'
    )
    .select( # Select columns again, this time applying 'explode'
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       # Explode the 'metrics' map. This creates new rows for each key-value pair in the map.
       # The key goes into the 'sig_name' column, and the value goes into the 'value' column.
       explode(col("metrics")).alias("sig_name", "value")
    )
)

# --- Join the formatted data with the mapping DataFrame ---

# Join the 'formatted_df' (long format data) with the 'new_df' (mapping data)
joined_df = (
    formatted_df
    # Perform a left join: keep all rows from 'formatted_df' and add matching columns from 'new_df'
    # Use 'broadcast' hint as 'new_df' is small, suggesting Spark to send it entirely to each executor for efficiency.
    .join(broadcast(new_df), on="sig_name", how="left") # Join based on the 'sig_name' column
    .select( # Select the desired final columns
       "signal_date",
       "hour_of_day",
       "generation_indicator",
       "sig_name",           # Original signal name from the exploded map
       "sig_mapping_name",   # Mapped signal name from the joined 'new_df'
       "value"               # The actual average value for that signal, date, and hour
    )
)

# Show the first 20 rows (default) of the final joined and reshaped DataFrame
joined_df.show()

# Stop the SparkSession to release resources
sprk.stop()
