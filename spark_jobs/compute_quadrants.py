#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def single_file_parquet_write(df, output_file_path: str):
    """
    Écrit le DataFrame `df` en un SEUL fichier Parquet nommé `output_file_path`.
    - On écrit d'abord dans un répertoire temporaire (suffixe "_tmp_parquet")
      pour obtenir le part-00000-*.parquet unique (coalesce(1)).
    - On reprend ce fichier, on le déplace/renomme en `output_file_path`.
    - On supprime ensuite le dossier temporaire.
    """
    tmp_dir = output_file_path + "_tmp_parquet"
    # 1) Si tmp_dir existe, on le supprime
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    # 2) On coalesce en 1 partition et on écrit dans tmp_dir
    df.coalesce(1).write.mode("overwrite").parquet(tmp_dir)

    # 3) Rechercher le seul fichier *.parquet dans tmp_dir
    list_parquets = glob.glob(os.path.join(tmp_dir, "part-*.parquet"))
    if len(list_parquets) != 1:
        raise RuntimeError(f"Expected exactly one part-*.parquet in {tmp_dir}, found {list_parquets}")

    part_file = list_parquets[0]  # chemin complet vers part-00000-xxxxx.parquet
    # 4) Supprimer l’ancien fichier de sortie s’il existe
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)

    # 5) Déplacer/renommer ce part_file vers output_file_path
    shutil.move(part_file, output_file_path)

    # 6) Supprimer le dossier temporaire (reste _SUCCESS, éventuellement dossiers _common_metadata, etc.)
    shutil.rmtree(tmp_dir)


def single_file_csv_write(df, output_file_path: str):
    """
    Écrit le DataFrame `df` en un SEUL fichier CSV nommé `output_file_path`.
    - On écrit d'abord dans un répertoire temporaire (suffixe "_tmp_csv")
      pour obtenir le part-00000-*.csv unique (coalesce(1) + header).
    - On reprend ce fichier, on le déplace/renomme en `output_file_path`.
    - On supprime ensuite le dossier temporaire.
    """
    tmp_dir = output_file_path + "_tmp_csv"
    # 1) Si tmp_dir existe, on le supprime
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    # 2) On coalesce en 1 partition et on écrit dans tmp_dir (avec header)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_dir)

    # 3) Rechercher le seul fichier *.csv dans tmp_dir
    list_csvs = glob.glob(os.path.join(tmp_dir, "part-*.csv"))
    if len(list_csvs) != 1:
        raise RuntimeError(f"Expected exactly one part-*.csv in {tmp_dir}, found {list_csvs}")

    part_file = list_csvs[0]  # chemin complet vers part-00000-xxxxx.csv
    # 4) Supprimer l’ancien fichier de sortie s’il existe
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)

    # 5) Déplacer/renommer ce part_file vers output_file_path
    shutil.move(part_file, output_file_path)

    # 6) Supprimer le dossier temporaire
    shutil.rmtree(tmp_dir)


def main(indicators_parquet_path: str, output_parquet_file: str):
    """
    Compute quadrants (incrémental) et écrit deux fichiers "single‐file" :
    - <output_parquet_file>         (exemple: /home/leoja/airflow/data/score_quadrant.parquet)
    - <output_parquet_file>.csv     (exemple: /home/leoja/airflow/data/score_quadrant.csv)

    1) Si <output_parquet_file> existe déjà :
       • on lit l’ancien DataFrame et on récupère la date max (last_date).
       • on calcule uniquement les mois > last_date (new_ind).
       • on ne change PAS les données antérieures.
    2) Sinon (première exécution) :
       • on calcule tout depuis 0.
    3) Calcul complet (delta, z-score, pos/var/combined, répartition Q1→Q4).
    4) On fusionne avec l’ancien DataFrame (incrémental).
    5) Écriture "single‐file" Parquet + CSV.
    """
    spark = (
        SparkSession.builder
        .appName("ComputeEconomicQuadrantsSingleFile")
        .getOrCreate()
    )

    # ────────────────────────────────────
    # 1) Lecture des indicateurs (tout l’historique)
    # ────────────────────────────────────
    df_ind = (
        spark.read.parquet(indicators_parquet_path)
             .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
             .orderBy("date")
    )

    # ────────────────────────────────────
    # 2) Vérifier si <output_parquet_file> existe déjà
    # ────────────────────────────────────
    out_parquet = Path(output_parquet_file)
    existing_df = None
    last_date = None

    if out_parquet.exists():
        # Si le fichier unique Parquet est déjà là → on le lit
        existing_df = spark.read.parquet(str(output_parquet_file))
        # On récupère la date max
        row = existing_df.agg({"date": "max"}).collect()[0]
        last_date = row[0]
        # On filtre df_ind pour ne garder que les nouvelles dates
        new_ind = df_ind.filter(col("date") > last_date)
    else:
        new_ind = df_ind

    # ────────────────────────────────────
    # 3) Si pas de nouvelles lignes ET existing_df n’est pas None :
    #    → on ne recalcule rien, merged_df = existing_df
    # ────────────────────────────────────
    if existing_df is not None and new_ind.rdd.isEmpty():
        merged_df = existing_df
    else:
        # ────────────────────────────────────
        # 4) Calcul des deltas & z-scores sur tout l’historique df_ind
        # ────────────────────────────────────
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

        # 4.1) Pour chaque indicateur, on calcule : delta, moyenne mobile, stddev mobile, z-score
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

        # 4.2) Calcul des médianes et écarts-type globaux
        median_std = {}
        for ind in indicator_cols:
            méd = working_df.select(expr(f"percentile_approx(`{ind}`, 0.5)")).first()[0]
            std_all = working_df.agg(stddev_samp(col(ind))).first()[0]
            median_std[ind] = (méd, std_all)

        # ────────────────────────────────────
        # 5) Ne garder que les nouvelles lignes (dates > last_date) pour le scoring
        # ────────────────────────────────────
        if last_date is not None:
            score_base_df = working_df.filter(col("date") > last_date)
        else:
            score_base_df = working_df

        # ────────────────────────────────────
        # 6) Calcul des scores pos_score, var_score, combined pour chaque indicateur
        # ────────────────────────────────────
        def pos_score(col_name: str, méd: float, std_all: float):
            return (
                when(col(col_name) > méd + 1.5 * std_all, 2)
                .when(col(col_name) > méd + 0.5 * std_all, 1)
                .when(col(col_name) < méd - 1.5 * std_all, -2)
                .when(col(col_name) < méd - 0.5 * std_all, -1)
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

        for ind in indicator_cols:
            méd, std_all = median_std[ind]
            score_base_df = (
                score_base_df
                .withColumn(f"{ind}_pos_score", pos_score(ind, méd, std_all))
                .withColumn(f"{ind}_var_score", var_score(f"{ind}_zscore"))
                .withColumn(f"{ind}_combined", col(f"{ind}_pos_score") + col(f"{ind}_var_score"))
            )

        # ────────────────────────────────────
        # 7) Initialiser les compteurs Q1, Q2, Q3, Q4 à 0
        # ────────────────────────────────────
        score_base_df = (
            score_base_df
            .withColumn("score_Q1", expr("0"))
            .withColumn("score_Q2", expr("0"))
            .withColumn("score_Q3", expr("0"))
            .withColumn("score_Q4", expr("0"))
        )

        # ────────────────────────────────────
        # 8) Répartition des points selon chaque indicateur & logique métier
        # ────────────────────────────────────
        def add_points(df, combined_col: str, pos_quads: list, neg_quads: list):
            exprs = {}
            for q in pos_quads:
                exprs[q] = when(col(combined_col) > 0, 1).otherwise(0)
            for q in neg_quads:
                exprs[q] = when(col(combined_col) < 0, 1).otherwise(0)
            return df, exprs

        mappings = {
            "INFLATION_combined":                    (["score_Q2", "score_Q3"], ["score_Q1", "score_Q4"]),
            "UNEMPLOYMENT_combined":                 (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
            "CONSUMER_SENTIMENT_combined":           (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
            "High_Yield_Bond_SPREAD_combined":       (["score_Q3", "score_Q2"], ["score_Q1", "score_Q4"]),
            "10-2Year_Treasury_Yield_Bond_combined": (["score_Q4"],               ["score_Q3"]),
            "TAUX_FED_combined":                     (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
        }

        for combined_col, (pos_quads, neg_quads) in mappings.items():
            score_base_df, exprs = add_points(score_base_df, combined_col, pos_quads, neg_quads)
            for q, expr_cond in exprs.items():
                score_base_df = score_base_df.withColumn(q, col(q) + expr_cond)

        # ────────────────────────────────────
        # 9) Déterminer le quadrant final (celui qui a la plus grosse valeur)
        # ────────────────────────────────────
        score_base_df = (
            score_base_df
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

        # ────────────────────────────────────
        # 10) Projection des colonnes finales (bonne ordre)
        # ────────────────────────────────────
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
        new_scores_df = score_base_df.select(*final_cols)

        # ────────────────────────────────────
        # 11) On fusionne (union) avec l’ancien existing_df si besoin
        # ────────────────────────────────────
        if existing_df is not None:
            merged_df = existing_df.select(*final_cols).unionByName(new_scores_df)
        else:
            merged_df = new_scores_df

    # ────────────────────────────────────
    # 12) À CE STADE, merged_df contient la totalité des dates + scores (anciennes+nouvelles)
    # ────────────────────────────────────

    # 13) Suppression des anciens fichiers, s’ils sont encore présents
    #     (on supprime uniquement <output_parquet_file> et <output_parquet_file>.csv si présents)
    if out_parquet.exists():
        os.remove(str(out_parquet))

    out_csv_file = str(Path(output_parquet_file).with_suffix(".csv"))
    if Path(out_csv_file).exists():
        os.remove(out_csv_file)

    # ────────────────────────────────────
    # 14) Écriture “single‐file” Parquet
    # ────────────────────────────────────
    try:
        single_file_parquet_write(merged_df, str(output_parquet_file))
        print(f"> Généré un fichier unique Parquet : {output_parquet_file}")
    except Exception as e:
        print(f"✗ Erreur lors de l’écriture Parquet single‐file : {e}")
        raise

    # ────────────────────────────────────
    # 15) Écriture “single‐file” CSV
    # ────────────────────────────────────
    try:
        single_file_csv_write(merged_df, out_csv_file)
        print(f"> Généré un fichier unique CSV     : {out_csv_file}")
    except Exception as e:
        print(f"✗ Erreur lors de l’écriture CSV single‐file        : {e}")
        raise

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compute_quadrants.py <input_indicators.parquet> <output_score_quadrant.parquet>")
        sys.exit(1)

    indicators_path = sys.argv[1]
    output_path     = sys.argv[2]
    main(indicators_path, output_path)
