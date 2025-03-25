from pyspark.sql import SparkSession as ss, Row as r
from pyspark.sql.functions import to_json, struct, col, lit
from pyspark.sql.types import LongType as lt

# ------------------------------------------------------------------------------
# 1. Initialize Spark Session
# ------------------------------------------------------------------------------
sprk = ss.builder \
    .appName("KafkaProducer") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .config("spark.jars.repositories", "https://repos.spark-packages.org") \
    .getOrCreate()

sprk.sparkContext.setLogLevel("WARN")

# ------------------------------------------------------------------------------
# 2. Load the Static CSV DataFrame and add a sequential index (row_id)
# ------------------------------------------------------------------------------
fp = "/data/dataset.csv"
df = sprk.read.csv(fp, header=True, inferSchema=True).cache()

print("Preview of static CSV DataFrame:")
df.show(5)

# Create an RDD with index and convert back to a DataFrame with a new column "row_id"
def add_idx_row(row, index):
    row_dict = row.asDict()
    row_dict["row_id"] = index
    return r(**row_dict)

rdd = df.rdd.zipWithIndex().map(lambda x: add_idx_row(x[0], x[1]))
new_schema = df.schema.add("row_id", lt())
df_w_idx = sprk.createDataFrame(rdd, schema=new_schema)

print("Static CSV Input DF Schema:")
df_w_idx.printSchema()

# ------------------------------------------------------------------------------
# 3. Create the Streaming Rate DataFrame and use its built-in "value" as row_id
# ------------------------------------------------------------------------------
new_df = sprk.readStream \
    .format("rate") \
    .option("rowsPerSecond", 10) \
    .load()

# Rename "value" to "row_id" to serve as the join key.
new_df = new_df.withColumnRenamed("value", "row_id")

# ------------------------------------------------------------------------------
# 4. Limit the streaming DataFrame to match the static DataFrame size
# ------------------------------------------------------------------------------
c = df_w_idx.count()
print("Static CSV row count:", c)

# Filter the stream so that only rows with row_id < c are processed.
new_df = new_df.filter(col("row_id") < lit(c))

# ------------------------------------------------------------------------------
# 5. Join the streaming and static DataFrames on "row_id"
# ------------------------------------------------------------------------------
jt = "left_outer"  # or "inner" if preferred
joined_df = new_df.join(df_w_idx, on="row_id", how=jt).drop("timestamp")

print("Joined Streaming DF Schema:")
joined_df.printSchema()

# ------------------------------------------------------------------------------
# 6. Convert the joined DataFrame to JSON for output
# ------------------------------------------------------------------------------
out = joined_df.select(
    to_json(struct([col(x) for x in joined_df.columns])).alias("value")
)

print("Kafka Output DF Schema (JSON):")
out.printSchema()

# ------------------------------------------------------------------------------
# 7. Write the output to Kafka using a trigger that runs once (bounded mode)
# ------------------------------------------------------------------------------
# Adjust Kafka settings as needed. If running inside Docker, use the internal hostname.

q = out.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("topic", "xenon-topic") \
    .option("checkpointLocation", "/tmp/kafka_checkpoint") \
    .outputMode("append") \
    .trigger(once=True) \
    .start()

q.awaitTermination()

sprk.stop()
