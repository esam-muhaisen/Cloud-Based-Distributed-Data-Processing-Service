from pyspark.sql import SparkSession


def get_spark(cores=1):
    return (
        SparkSession.builder
        .master(f"local[{cores}]")
        .appName("FinalProject")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )