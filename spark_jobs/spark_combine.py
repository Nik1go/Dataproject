from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, datediff
from pyspark.sql.window import Window
import pyspark.sql.functions as F
import argparse
import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("CountercyclicalStrategy") \
        .config("spark.sql.shuffle.partitions", "1") \
        .getOrCreate()

    # Charger les données avec gestion des valeurs manquantes
    df = spark.read.csv(args.input, header=True, inferSchema=True) \
        .withColumn("date", to_date(col("date"))) \
        .fillna(0, subset=["hy_spread", "long_spread"])

    # Calculer les médianes sur 10 ans (3650 jours) avec fenêtre glissante
    window_spec = Window.orderBy("date").rowsBetween(-3650, 0)

    df = df.withColumn("hy_median",
                       F.percentile_approx("hy_spread", 0.5).over(window_spec))

    df = df.withColumn("long_spread_median",
                       F.percentile_approx("long_spread", 0.5).over(window_spec))

    # Dernière observation complète
    latest = df.filter(
        (col("hy_spread").isNotNull()) &
        (col("long_spread").isNotNull())
    ).orderBy(col("date").desc()).first()

    # Logique de décision
    decision = "UNDEFINED"
    if latest and latest['hy_spread'] > latest['hy_median']:
        decision = "CROISSANCE"
        print(f"\nDECISION: Allocation au Portefeuille Croissance")
        print(f"  HY Spread actuel: {latest['hy_spread']:.2f}%")
        print(f"  Médiane 10 ans: {latest['hy_median']:.2f}%")
    elif latest:
        if latest['long_spread'] < latest['long_spread_median']:
            decision = "INFLATION"
            print(f"\nDECISION: Allocation au Portefeuille Inflation")
            print(f"  Long Spread actuel: {latest['long_spread']:.2f}%")
            print(f"  Médiane 10 ans: {latest['long_spread_median']:.2f}%")
            print("  Composition: 50% S&P 500, 40% or, 10% small-cap value")
        else:
            decision = "RALENTISSEMENT"
            print(f"\nDECISION: Allocation au Portefeuille Ralentissement")

    # Sauvegarder les résultats
    result_data = [(datetime.datetime.now().isoformat(), decision)]
    result_df = spark.createDataFrame(result_data, ["timestamp", "decision"])
    result_df.write.mode("overwrite").csv(f"{args.output}/strategy_decision")

    spark.stop()


if __name__ == "__main__":
    main()