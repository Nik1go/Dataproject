#!/usr/bin/env python3
# compute_quadrants_incremental.py

import sys
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


def main(indicators_parquet_path: str, output_parquet_path: str):
    """
    Si output_parquet_path existe, lit l'ancien résultat (anciens mois calculés),
    détermine la date max, puis ne calcule les nouvelles lignes d'indicateurs qu'après cette date.
    Enfin, réécrit tout (anciens + nouveaux) en overwrite.
    """

    spark = (
        SparkSession.builder
        .appName("ComputeEconomicQuadrantsIncremental")
        .getOrCreate()
    )

    # --- 1. Chargement des indicateurs et conversion de la date ---
    df_ind = spark.read.parquet(indicators_parquet_path) \
        .withColumn("date", to_date(col("date"), "yyyy-MM-dd")) \
        .orderBy("date")

    # 2. Détecter si output_parquet_path existe déjà
    output_path = Path(output_parquet_path)
    existing_df = None
    last_date = None

    if output_path.exists():
        # __Spark__ ne peut pas lire un Path Python directement, donc on fait spark.read
        existing_df = spark.read.parquet(output_parquet_path)
        # La colonne date a déjà été convertie en DateType, donc :
        max_row = existing_df.agg({"date": "max"}).collect()[0]
        last_date = max_row[0]  # type: ignore
        # On garde uniquement les mois strictement supérieurs à last_date
        new_ind = df_ind.filter(col("date") > last_date)
    else:
        # Pas d'ancienne version : on calcule tous les mois
        new_ind = df_ind

    # Si aucune nouvelle ligne à traiter, on sort immédiatement
    if new_ind.rdd.isEmpty():
        print(f"Aucun nouvel indicateur à traiter après {last_date}. Sortie sans rien faire.")
        spark.stop()
        return

    # --- 3. Définition des fenêtres sur la totalité de df_ind (pour rolling) ---
    window_lag = Window.orderBy("date")
    window_roll = Window.orderBy("date").rowsBetween(-120, -1)

    # On va travailler sur l’ensemble df_ind pour calculer δ et zscore,
    # mais on ne gardera – dans la suite – que les lignes > last_date. 
    # Pour ne pas dégrader les stats mois par mois, on calcule d'abord sur df_ind complet :
    working_df = df_ind

    indicator_cols = [
        "INFLATION",
        "UNEMPLOYMENT",
        "CONSUMER_SENTIMENT",
        "High_Yield_Bond_SPREAD",
        "10-2Year_Treasury_Yield_Bond",
    ]

    # --- 4. Calcul des deltas et z-scores (sur tout l’historique) ---
    for ind in indicator_cols:
        prev_val = lag(col(ind), 1).over(window_lag)
        working_df = working_df.withColumn(
            f"{ind}_delta", (col(ind) - prev_val) / prev_val
        ).withColumn(
            f"{ind}_delta_mean", avg(f"{ind}_delta").over(window_roll)
        ).withColumn(
            f"{ind}_delta_std", stddev_samp(f"{ind}_delta").over(window_roll)
        ).withColumn(
            f"{ind}_zscore",
            (col(f"{ind}_delta") - col(f"{ind}_delta_mean")) / col(f"{ind}_delta_std"),
        )

    # --- 5. Calcul des médianes & écarts-type long terme (sur tout l’historique) ---
    median_std = {}
    for ind in indicator_cols:
        # On entoure toujours le nom de colonne entre backticks dans expr()
        median_val = working_df.select(expr(f"percentile_approx(`{ind}`, 0.5)")).first()[0]
        std_all = working_df.agg(stddev_samp(col(ind))).first()[0]
        median_std[ind] = (median_val, std_all)

    # --- 6. Fonctions utilitaires pour position‐score et variation‐score ---
    def pos_score(col_name: str, med: float, std_all: float):
        return when(col(col_name) > med + 1.5 * std_all, 2) \
               .when(col(col_name) > med + 0.5 * std_all, 1) \
               .when(col(col_name) < med - 1.5 * std_all, -2) \
               .when(col(col_name) < med - 0.5 * std_all, -1) \
               .otherwise(0)

    def var_score(z_col_name: str):
        return when(col(z_col_name) > 2, 2) \
               .when(col(z_col_name) > 1, 1) \
               .when(col(z_col_name) < -2, -2) \
               .when(col(z_col_name) < -1, -1) \
               .otherwise(0)

    # --- 7. On commence par ne garder, dans un DataFrame intermédiaire, que les lignes nouvelles ---
    new_df = working_df.filter(col("date") > last_date) if last_date is not None else working_df

    # --- 8. Sur new_df, ajouter les colonnes pos_score et var_score ---
    for ind in indicator_cols:
        med, std_all = median_std[ind]
        new_df = new_df.withColumn(f"{ind}_pos_score", pos_score(ind, med, std_all)) \
                       .withColumn(f"{ind}_var_score", var_score(f"{ind}_zscore")) \
                       .withColumn(f"{ind}_combined", col(f"{ind}_pos_score") + col(f"{ind}_var_score"))

    # --- 9. Initialiser et ajouter les scores Q1…Q4 (pour seulement ces nouvelles dates) ---
    new_df = new_df.withColumn("score_Q1", expr("0")) \
                   .withColumn("score_Q2", expr("0")) \
                   .withColumn("score_Q3", expr("0")) \
                   .withColumn("score_Q4", expr("0"))

    def add_points(df, combined_col: str, pos_quads: list, neg_quads: list):
        exprs = {}
        for q in pos_quads:
            exprs[q] = when(col(combined_col) > 0, 1).otherwise(0)
        for q in neg_quads:
            exprs[q] = when(col(combined_col) < 0, 1).otherwise(0)
        return df, exprs

    mappings = {
        "INFLATION_combined":   (["score_Q2", "score_Q3"], ["score_Q1", "score_Q4"]),
        "UNEMPLOYMENT_combined":(["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
        "CONSUMER_SENTIMENT_combined": (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
        "High_Yield_Bond_SPREAD_combined": (["score_Q3", "score_Q2"], ["score_Q1", "score_Q4"]),
        "10-2Year_Treasury_Yield_Bond_combined": (["score_Q4"], ["score_Q3"])
    }

    for combined_col, (pos_quads, neg_quads) in mappings.items():
        new_df, exprs = add_points(new_df, combined_col, pos_quads, neg_quads)
        for q, expr_cond in exprs.items():
            new_df = new_df.withColumn(q, col(q) + expr_cond)

    # --- 10. Calculer assigned_quadrant pour new_df ---
    new_df = new_df.withColumn(
        "max_score",
        greatest(col("score_Q1"), col("score_Q2"), col("score_Q3"), col("score_Q4"))
    ).withColumn(
        "assigned_quadrant",
        when(col("score_Q1") == col("max_score"), 1)
        .when(col("score_Q2") == col("max_score"), 2)
        .when(col("score_Q3") == col("max_score"), 3)
        .otherwise(4)
    )

    # Sélectionner les colonnes finales dans new_df
    final_cols = [
        "date",
        *indicator_cols,
        *[f"{ind}_delta" for ind in indicator_cols],
        *[f"{ind}_zscore" for ind in indicator_cols],
        *[f"{ind}_pos_score" for ind in indicator_cols],
        *[f"{ind}_var_score" for ind in indicator_cols],
        "score_Q1", "score_Q2", "score_Q3", "score_Q4", "assigned_quadrant"
    ]
    new_df = new_df.select(*final_cols)

    # --- 11. Concaténer avec existing_df si présent ---
    if existing_df is not None:
        # On garde aussi les mêmes colonnes, dans le même ordre
        merged = existing_df.select(*final_cols).unionByName(new_df)
    else:
        merged = new_df

    # --- 12. Écrire (overwrite) merged dans output_parquet_path ---
    merged.write.mode("overwrite").parquet(output_parquet_path)

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compute_quadrants_incremental.py <input_indicators.parquet> <output_quadrants.parquet>")
        sys.exit(1)

    indicators_path = sys.argv[1]
    output_path     = sys.argv[2]
    main(indicators_path, output_path)
