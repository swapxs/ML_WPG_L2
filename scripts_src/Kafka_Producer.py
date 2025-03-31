# Import the SparkSession class from the pyspark.sql module and alias it as ss
from pyspark.sql import SparkSession as ss
# Import necessary Spark SQL functions for data manipulation
from pyspark.sql.functions import to_json, struct, col

# Build a SparkSession instance
sprk = ss.builder \
    .appName("KafkaProducer") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .config("spark.jars.repositories", "https://repos.spark-packages.org") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "10g")\
    .config("spark.cores.max", "4") \
    .getOrCreate()

# Define the file path for the CSV dataset to be read initially (likely for schema inference)
fp = "/data/dataset.csv"

# Read the CSV file into a batch DataFrame
# header=True specifies that the first row is the header
# inferSchema=True tells Spark to automatically determine column data types
df = sprk.read.csv(fp, header=True, inferSchema=True)

# Display the first 2 rows of the batch DataFrame to the console
df.show(2)

# Print the schema (column names and inferred data types) of the batch DataFrame
df.printSchema()

# Define a streaming DataFrame (ds) by reading CSV files from a directory
ds = sprk.readStream \
    .option("header", True) \
    .schema(df.schema) \
    .csv("/data/") # Monitor the /data/ directory for new CSV files

# Transform the streaming DataFrame:
# 1. Select all columns using a list comprehension: [col(c) for c in ds.columns]
# 2. Combine these columns into a single struct (complex type)
# 3. Convert the struct into a JSON string using to_json
# 4. Alias the resulting JSON string column as "value" (required format for Kafka sink)
out = ds.select(to_json(struct([col(c) for c in ds.columns])).alias("value"))

# Configure and start the streaming query to write data to Kafka
out.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("topic", "xenon-topic") \
    .option("checkpointLocation", "/tmp/kafka_checkpoint") \
    .start()

# Stop the SparkSession
# Note: This will likely terminate the streaming application almost immediately after starting it.
# In a real-world scenario, you'd typically use query.awaitTermination() to keep the application running.
sprk.stop()
