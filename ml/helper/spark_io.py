import abc
import pathlib
import sys

from pyspark.sql import SparkSession, DataFrame


class Spark:
    def __init__(self):
        sys.path.insert(0, "/Users/chjin/bin/spark/python")
        sys.path.insert(0, "/Users/chjin/bin/spark/python/lib/py4j-0.10.7-src.zip")

        self.spark = (
            SparkSession.builder.appName("Python Spark SQL basic example")
            .master("local[4]")
            # .config("spark.executor.memory", "32g")
            # .config("spark.driver.memory", "16g")
            .config("spark.driver.maxResultSize", "0")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config(
                "spark.executor.extraJavaOptions",
                "-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps",
            )
            .getOrCreate()
        )


class SparkReader(Spark):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.datalake_url = self.config["io"]["datalake_url"]

    def read(self, stream_name, columns, source, extension, location):
        if source == "local":
            location == location
        elif source == "datalake":
            location = self.datalake_url + location

        if extension == "csv":
            df = self.spark.read.load(
                location, format="csv", sep=",", inferSchema="true", header="true"
            )
        elif extension == "parquet":
            df = self.spark.read.parquet(location)

        df = df.select(columns)
        return df


class SparkWriter(Spark):
    pass