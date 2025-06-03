#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, lag, year, month
from pyspark.sql.window import Window
import argparse
import os

# Configuration Java pour Spark
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Chemin du fichier Parquet des indicateurs")
    parser.add_argument('--output', required=True, help="Répertoire de sortie")
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("CountercyclicalStrategyMonthly") \
        .getOrCreate()

    # 1. Chargement des données mensuelles
    df = spark.read.parquet(args.input)

    # 2. Calcul des médianes sur 10 ans (120 mois)
    window_spec = Window.orderBy("date").rowsBetween(-120, 0)

    df = df.withColumn("hy_median",
                       expr(
                           "percentile_approx(High_Yield_Bond_SPREAD, 0.5) OVER (ORDER BY date ROWS BETWEEN 120 PRECEDING AND CURRENT ROW)"))

    df = df.withColumn("long_spread_median",
                       expr(
                           "percentile_approx(`10-2Year_Treasury_Yield_Bond`, 0.5) OVER (ORDER BY date ROWS BETWEEN 120 PRECEDING AND CURRENT ROW)"))

    # 3. Calcul de la décision
    df = df.withColumn("decision",
                       when(col("High_Yield_Bond_SPREAD") > col("hy_median"), "CROISSANCE")
                       .when((col("High_Yield_Bond_SPREAD") <= col("hy_median")) &
                             (col("10-2Year_Treasury_Yield_Bond") < col("long_spread_median")), "INFLATION")
                       .otherwise("RALENTISSEMENT"))

    # 4. Détection des changements d'état
    window_spec_prev = Window.orderBy("date")
    df = df.withColumn("prev_decision", lag("decision", 1).over(window_spec_prev))
    df = df.withColumn("state_change",
                       when(col("decision") != col("prev_decision"), 1).otherwise(0))

    # 5. Ajout d'information temporelle
    df = df.withColumn("year", year("date"))
    df = df.withColumn("month", month("date"))

    # 6. Sauvegarde des résultats
    # Historique complet
    df.write.mode("overwrite") \
        .option("header", "true") \
        .parquet(args.output + "/full_history")

    # Dernière décision
    latest = df.orderBy(col("date").desc()).first()
    result_data = [(latest["date"], latest["decision"])]
    result_df = spark.createDataFrame(result_data, ["date", "decision"])
    result_df.write.mode("overwrite").parquet(args.output + "/latest_decision")

    # 7. Affichage des résultats
    print("\n" + "=" * 50)
    print("DERNIÈRE DÉCISION MENSUELLE:")
    print(f"Date: {latest['date']}")
    print(f"HY Spread: {latest['High_Yield_Bond_SPREAD']:.2f}% (Médiane: {latest['hy_median']:.2f}%)")
    print(f"10-2Y Spread: {latest['10-2Year_Treasury_Yield_Bond']:.2f}% (Médiane: {latest['long_spread_median']:.2f}%)")
    print(f"Décision: {latest['decision']}")
    print("=" * 50)

    # Afficher les changements récents
    changes = df.filter(col("state_change") == 1).orderBy(col("date").desc()).limit(5)
    print("\nDerniers changements d'état:")
    changes.select("date", "decision").show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()