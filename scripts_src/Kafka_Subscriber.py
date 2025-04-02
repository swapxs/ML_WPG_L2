# Import the SparkSession class and alias it as ss
from pyspark.sql import SparkSession as ss
# Import StructType and StructField for defining DataFrame schemas
from pyspark.sql.types import (
    StructType as st,
    StructField as sf,
    StringType as srt,
    DoubleType as dt,
    LongType as lt
)
# Import necessary Spark SQL functions for data processing and manipulation
from pyspark.sql.functions import (
    from_json,       # Parses a JSON string column into a struct
    col,             # Selects a column
    to_date,         # Converts a timestamp string to a date
    to_timestamp,    # Converts a string to a timestamp
    current_date,    # Gets the current date
    current_timestamp,# Gets the current timestamp
    lit,             # Creates a literal value column
    map_from_arrays, # Creates a map column from two array columns (keys and values)
    array,           # Creates an array column
)

# Build a SparkSession instance configured for Kafka and Delta Lake
sprk = ss.builder \
    .appName("KafkaSubscriber") \
    .master("spark://spark-master:7077") \
    .config(
        "spark.jars.packages",
        # Include packages for Kafka SQL connector and Delta Lake integration
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5,"
        "io.delta:delta-spark_2.12:3.3.0"
    ) \
    .config("spark.jars.repositories", "https://repos.spark-packages.org") \
    # Enable Delta Lake SQL extensions for the Spark session
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    # Configure the Spark catalog to use the Delta Catalog implementation
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "10g")\
    .config("spark.cores.max", "4") \
    .getOrCreate() # Get or create the SparkSession

# Define the schema expected for the JSON messages coming from Kafka
json_schema = st([
    sf("Date/Time", srt(), True),                  # String field for date/time
    sf("LV ActivePower (kW)", dt(), True),        # Double field for active power
    sf("Wind Speed (m/s)", dt(), True),            # Double field for wind speed
    sf("Theoretical_Power_Curve (KWh)", dt(), True), # Double field for theoretical power
    sf("Wind Direction (°)", dt(), True),          # Double field for wind direction
    sf("row_id", lt(), True)                       # Long field for a row identifier
])

# Define the Kafka stream reader
kafka_df = sprk.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "xenon-topic") \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", "false") \
    .load() # Load data from the specified Kafka topic

# Process the raw Kafka data
df = kafka_df.select(
    # Parse the 'value' column (which is binary) by first casting it to string
    # and then applying the predefined json_schema
    from_json(
        col("value").cast("string"), # Cast the Kafka message value to a string
        json_schema                 # Apply the schema to parse the JSON
    ).alias("jsonData")             # Alias the resulting struct column as "jsonData"
).select("jsonData.*")              # Flatten the struct to get individual columns

# Transform the parsed data into the final desired structure
final_df = df.select(
    # Convert the 'Date/Time' string first to a timestamp, then extract the date
    to_date(
        to_timestamp(col("Date/Time"), "dd MM yyyy HH:mm"), # Convert string to timestamp with specified format
        "yyyy-MM-dd"                                       # Format the timestamp as a date string
    ).alias("signal_date"),                               # Alias the column as 'signal_date'
    # Convert the 'Date/Time' string to a proper timestamp type
    to_timestamp(col("Date/Time"), "dd MM yyyy HH:mm").alias("signal_ts"), # Alias as 'signal_ts'
    # Add the current date when the record is processed
    current_date().alias("create_date"),                  # Alias as 'create_date'
    # Add the current timestamp when the record is processed
    current_timestamp().alias("create_ts"),              # Alias as 'create_ts'
    # Create a map column containing the signal names as keys and their values as strings
    map_from_arrays(
        # Array of literal string keys (signal names)
        array(
            lit("LV ActivePower (kW)"),
            lit("Wind Speed (m/s)"),
            lit("Theoretical_Power_Curve (KWh)"),
            lit("Wind Direction (°)")
        ),
        # Array of corresponding signal values, cast to string
        array(
            col("LV ActivePower (kW)").cast("string"),
            col("Wind Speed (m/s)").cast("string"),
            col("Theoretical_Power_Curve (KWh)").cast("string"),
            col("Wind Direction (°)").cast("string")
        )
    ).alias("signals") # Alias the resulting map column as 'signals'
)

# Print the schema of the transformed DataFrame to the console
final_df.printSchema()

# Define and start the stream writer to save the data to a Delta table
final_df.writeStream \
    .format("delta") \
    .option("checkpointLocation", "/tmp/delta_kafka_subscriber_checkpoint") \
    .outputMode("append") \
    .start("/data/delta_output") # Start writing to the Delta table at the specified path

# === Batch Read Section (Typically for testing or subsequent processing) ===
# Note: In a continuous streaming application, the Spark session might be kept running
# using `query.awaitTermination()` after starting the stream. This section reads
# data written by the stream *up to this point*.

# Read the data from the Delta table as a batch DataFrame
df = sprk.read.format("delta").load("/data/delta_output")
# Show the first 5 rows of the Delta table contents
df.show(5)

# Stop the SparkSession
# Note: This will stop the Spark application, including the background streaming query.
sprk.stop()
