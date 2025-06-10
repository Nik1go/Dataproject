import sys
import os
import shutil
import glob
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    lag,
    avg,
    stddev_samp,
    when,
    expr,
    to_date,
    greatest,
)
""" compute_quadrants.py

Objectif :
Ce script Spark permet de déterminer dans quel quadrant économique (Q1 à Q4) se trouve chaque mois,
en analysant des indicateurs macroéconomiques comme l’inflation, le chômage, ou les spreads de crédit.

Définition des quadrants :
- Q1 : Croissance & faible inflation (environnement favorable)
- Q2 : Croissance & forte inflation
- Q3 : Récession & forte inflation (stagflation)
- Q4 : Récession & faible inflation

Étapes réalisées :
1. Lecture des indicateurs macroéconomiques à partir d’un fichier `.parquet`
2. Calcul :
   - des deltas (variations mensuelles),
   - des z-scores (anomalies par rapport à l'historique),
   - des scores de position absolue (niveau par rapport à la médiane)
   - des scores de variation (delta par rapport à l'écart type)
3. Attribution de scores à chaque quadrant en fonction de la direction de chaque indicateur :
   (ex: inflation élevée → Q2 ou Q3 / chômage faible → Q1 ou Q2, etc.)
4. Quadrant final attribué = celui avec le score total le plus élevé.
5. Export final dans un unique fichier Parquet et un fichier CSV unique, utilisables dans les étapes suivantes du pipeline.

Usage :
Le script s'utilise avec 3 arguments :
```bash
spark-submit compute_quadrants.py <input_indicators.parquet> <output_quadrants.parquet> <output_quadrants.csv> """


def write_single_parquet(df, output_path: str):
    tmp_dir = output_path + "_tmp_parquet"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    df.coalesce(1).write.mode("overwrite").parquet(tmp_dir)

    part_files = glob.glob(os.path.join(tmp_dir, "part-*.parquet"))
    if not part_files:
        raise RuntimeError(f"Aucun part-*.parquet trouvé dans {tmp_dir}")
    single_part = part_files[0]

    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(single_part, output_path)
    shutil.rmtree(tmp_dir)

def write_single_csv(df, output_path: str):

    tmp_dir = output_path + "_tmp_csv"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_dir)
    part_files = glob.glob(os.path.join(tmp_dir, "part-*.csv"))
    if not part_files:
        raise RuntimeError(f"Aucun part-*.csv trouvé dans {tmp_dir}")
    single_part = part_files[0]
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(single_part, output_path)
    shutil.rmtree(tmp_dir)


def main(indicators_parquet_path: str, output_parquet_path: str, output_csv_path: str, full_df=None):

    spark = (
        SparkSession.builder
        .appName("ComputeEconomicQuadrants")
        .getOrCreate()
    )

    df_ind = (
        spark.read.parquet(indicators_parquet_path)
        .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
        .orderBy("date")
    )

    existing_df = None
    last_date = None

    if Path(output_parquet_path).is_file():
        existing_df = spark.read.parquet(output_parquet_path)
        max_row = existing_df.agg({"date": "max"}).collect()[0]
        last_date = max_row[0]  # type: ignore
        new_ind = df_ind.filter(col("date") > last_date)
    else:
        new_ind = df_ind

    if new_ind.rdd.isEmpty():
        print(f"Aucune nouvelle donnée après {last_date}. Pas de réécriture.")
        spark.stop()
        return

    window_lag = Window.orderBy("date")
    window_roll = Window.orderBy("date").rowsBetween(-120, -1)
    working_df = df_ind
    indicator_cols = [
        "INFLATION",
        "UNEMPLOYMENT",
        "CONSUMER_SENTIMENT",
        "High_Yield_Bond_SPREAD",
        "10-2Year_Treasury_Yield_Bond",
        "TAUX_FED",
    ]

    last_two = full_df.orderBy("date", ascending=False).limit(2).orderBy("date")

    for ind in indicator_cols:
        prev_val = lag(col(ind), 1).over(window_lag)
        working_df = (
            working_df
            .withColumn(f"{ind}_delta", (col(ind) - prev_val) / prev_val)
            .withColumn(f"{ind}_delta_mean", avg(f"{ind}_delta").over(window_roll))
            .withColumn(f"{ind}_delta_std", stddev_samp(f"{ind}_delta").over(window_roll))
            .withColumn(
                f"{ind}_zscore",
                (col(f"{ind}_delta") - col(f"{ind}_delta_mean")) / col(f"{ind}_delta_std")
            )
        )

    median_std = {}
    for ind in indicator_cols:
        median_val = working_df.select(expr(f"percentile_approx(`{ind}`, 0.5)")).first()[0]
        std_all = working_df.agg(stddev_samp(col(ind))).first()[0]
        median_std[ind] = (median_val, std_all)

    def pos_score(col_name: str, med: float, std_all: float):
        return (
            when(col(col_name) > med + 1.5 * std_all, 2)
            .when(col(col_name) > med + 0.5 * std_all, 1)
            .when(col(col_name) < med - 1.5 * std_all, -2)
            .when(col(col_name) < med - 0.5 * std_all, -1)
            .otherwise(0)
        )

    def var_score(z_col_name: str):
        return (
            when(col(z_col_name) > 2, 2)
            .when(col(z_col_name) > 1, 1)
            .when(col(z_col_name) < -2, -2)
            .when(col(z_col_name) < -1, -1)
            .otherwise(0)
        )

    new_df = working_df.filter(col("date") > last_date) if last_date is not None else working_df

    for ind in indicator_cols:
        med, std_all = median_std[ind]
        new_df = (
            new_df
            .withColumn(f"{ind}_pos_score", pos_score(ind, med, std_all))
            .withColumn(f"{ind}_var_score", var_score(f"{ind}_zscore"))
            .withColumn(f"{ind}_combined", col(f"{ind}_pos_score") + col(f"{ind}_var_score"))
        )

    new_df = (
        new_df
        .withColumn("score_Q1", expr("0"))
        .withColumn("score_Q2", expr("0"))
        .withColumn("score_Q3", expr("0"))
        .withColumn("score_Q4", expr("0"))
    )

    def add_points(df, combined_col: str, pos_quads: list, neg_quads: list):
        exprs = {}
        for q in pos_quads:
            exprs[q] = when(col(combined_col) > 0, 1).otherwise(0)
        for q in neg_quads:
            exprs[q] = when(col(combined_col) < 0, 1).otherwise(0)
        return df, exprs

    mappings = {
        "INFLATION_combined":               (["score_Q2", "score_Q3"], ["score_Q1", "score_Q4"]),
        "UNEMPLOYMENT_combined":            (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
        "CONSUMER_SENTIMENT_combined":      (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
        "High_Yield_Bond_SPREAD_combined":  (["score_Q3", "score_Q2"], ["score_Q1", "score_Q4"]),
        "10-2Year_Treasury_Yield_Bond_combined": (["score_Q4"], ["score_Q3"]),
        "TAUX_FED_combined":                (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
    }

    for combined_col, (pos_quads, neg_quads) in mappings.items():
        new_df, exprs = add_points(new_df, combined_col, pos_quads, neg_quads)
        for q, expr_cond in exprs.items():
            new_df = new_df.withColumn(q, col(q) + expr_cond)

    new_df = (
        new_df
        .withColumn(
            "max_score",
            greatest(col("score_Q1"), col("score_Q2"), col("score_Q3"), col("score_Q4"))
        )
        .withColumn(
            "assigned_quadrant",
            when(col("score_Q1") == col("max_score"), 1)
            .when(col("score_Q2") == col("max_score"), 2)
            .when(col("score_Q3") == col("max_score"), 3)
            .otherwise(4)
        )
    )

    final_cols = [
        "date",
        *indicator_cols,
        *[f"{ind}_delta" for ind in indicator_cols],
        *[f"{ind}_zscore" for ind in indicator_cols],
        *[f"{ind}_pos_score" for ind in indicator_cols],
        *[f"{ind}_var_score" for ind in indicator_cols],
        *[f"{ind}_combined" for ind in indicator_cols],
        "score_Q1", "score_Q2", "score_Q3", "score_Q4", "assigned_quadrant"
    ]
    new_df = new_df.select(*final_cols)

    if existing_df is not None:
        merged_df = existing_df.select(*final_cols).unionByName(new_df)
    else:
        merged_df = new_df

    write_single_parquet(merged_df, output_parquet_path)
    print(f"✔ Written single-file Parquet → {output_parquet_path}")
    write_single_csv(merged_df, output_csv_path)
    print(f"✔ Written single-file CSV   → {output_csv_path}")

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: compute_quadrants.py <input_indicators.parquet> <output_quadrants.parquet> <output_quadrants.csv>")
        sys.exit(1)

    indicators_path = sys.argv[1]
    output_parquet = sys.argv[2]
    output_csv = sys.argv[3]
    main(indicators_path, output_parquet, output_csv)
