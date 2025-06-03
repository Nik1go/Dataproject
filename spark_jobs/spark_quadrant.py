import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, to_date, lit, year
from datetime import datetime

spark = SparkSession.builder \
    .appName("QuadrantAnalysis") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

try:
    # Chemin absolu recommandé
    df_raw = spark.read.csv("home/leoja/data/backup/High_Yield_Bond_SPREAD.csv", header=False, inferSchema=True) \
        .toDF("year", "month", "day", "value")

    df = df_raw.withColumn("date_str", concat_ws("-", col("year"), col("month"), col("day"))) \
        .withColumn("date", to_date(col("date_str"), "yyyy-MM-dd")) \
        .filter(col("value").isNotNull())

    today = datetime.today()
    start_year = today.year - 10

    filtered = df.filter(year(col("date")) >= start_year)

    if filtered.count() == 0:
        raise ValueError("Aucune donnée après filtrage")

    median_value = filtered.approxQuantile("value", [0.5], 0.01)[0]

    today_df = df.filter(col("date") == lit(today.date()))

    if today_df.count() > 0:
        today_val = today_df.first()["value"]
        result = today_val >= median_value
    else:
        result = False

    print(f"Résultat final : {result}")

except Exception as e:
    print(f"Erreur critique : {str(e)}")
    raise

finally:
    spark.stop()