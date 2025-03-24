from pyspark.sql import SparkSession as ss, Row as r
from pyspark.sql.functions import to_json, struct, col, lit
from pyspark.sql.types import LongType as lt

sprk = ss.builder \
    .appName("KafkaProducer") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .config("spark.jars.repositories", "https://repos.spark-packages.org") \
    .getOrCreate()

sprk.sparkContext.setLogLevel("WARN")
fp = "/data/dataset.csv"
df = sprk.read.csv(fp, header=True, inferSchema=True).cache()

print("Preview of static CSV DataFrame:")
df.show(5)

def add_idx_row(row, index):
    row_dict = row.asDict()
    row_dict["row_id"] = index
    return r(**row_dict)

rdd_w_idx = df.rdd.zipWithIndex().map(lambda x: add_idx_row(x[0], x[1]))
new_schema = df.schema.add("row_id", lt())
df_w_idx = sprk.createDataFrame(rdd_w_idx, schema=new_schema)

print("Static CSV Input DF Schema:")
df_w_idx.printSchema()

strmng_df = sprk.readStream \
    .format("rate") \
    .option("rowsPerSecond", 10) \
    .load()

strmng_df = strmng_df.withColumnRenamed("value", "row_id")

print("Static CSV row count:", df_w_idx.count())

strmng_df = strmng_df.filter(col("row_id") < lit(df_w_idx.count()))

new_df = strmng_df.join(df_w_idx, on="row_id", how="left_outer").drop("timestamp")

print("Joined Streaming DF Schema:")
new_df.printSchema()

out = new_df.select(
    to_json(struct([col(x) for x in new_df.columns])).alias("value")
)

print("Kafka Output DF Schema (JSON):")
out.printSchema()

query = out.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("topic", "xenon-topic") \
    .option("checkpointLocation", "/tmp/kafka_checkpoint") \
    .outputMode("append") \
    .trigger(once=True) \
    .start()

query.awaitTermination()
