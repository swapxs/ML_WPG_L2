from pyspark.sql import SparkSession as ss
from pyspark.sql.functions import to_json, struct, col

sprk = ss.builder \
    .appName("KafkaProducer") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5") \
    .config("spark.jars.repositories", "https://repos.spark-packages.org") \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "10g")\
    .config("spark.cores.max", "4") \
    .getOrCreate()

fp = "/data/dataset.csv"

df = sprk.read.csv(fp, header=True, inferSchema=True)

df.show(2)

df.printSchema()

ds = sprk.readStream \
    .option("header", True) \
    .schema(df.schema) \
    .csv("/data/")

out = ds.select(to_json(struct([col(c) for c in ds.columns])).alias("value"))

out.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("topic", "xenon-topic") \
    .option("checkpointLocation", "/tmp/kafka_checkpoint") \
    .start()

sprk.stop()
