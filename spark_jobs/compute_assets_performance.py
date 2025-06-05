#!/usr/bin/env python3
# compute_quadrants.py

import sys
import os
import glob
import shutil
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
    1) Si output_parquet_path existe déjà (fichier OU dossier), on le supprime complètement.
    2) On lit Indicators.parquet, calcule les scores (incrémental si possible).
    3) On écrit le résultat dans un unique fichier Parquet : output_parquet_path.
    4) On écrit aussi un fichier CSV identique, même dossier, nommé <même_chemin_avec_extension_.csv>.
    """

    spark = (
        SparkSession.builder
        .appName("ComputeEconomicQuadrants")
        .getOrCreate()
    )

    # ─── 1. LECTURE DES INDICATEURS ───────────────────────────────────────────────
    df_ind = (
        spark.read.parquet(indicators_parquet_path)
             .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
             .orderBy("date")
    )

    # ─── 2. DÉTECTION & SUPPRESSION DE l’ANCIEN OUTPUT ────────────────────────────
    out_path = Path(output_parquet_path)

    # Si le chemin existe (que ce soit un fichier ou un dossier), on le supprime
    if out_path.exists():
        if out_path.is_dir():
            # s’il s’agit d’un dossier quadrants.parquet/, on le supprime entièrement
            shutil.rmtree(output_parquet_path)
        else:
            # s’il s’agit d’un fichier quadrants.parquet, on le supprime
            out_path.unlink()

    # On va stocker ici l’ancien DataFrame si on voulait faire une logique incrémentale
    existing_df = None
    last_date = None

    # ─── 3. DÉTECTION DU MODE (FULL/PARTIAL) ─────────────────────────────────────
    #    En mode FULL : on recalcule tout depuis 0
    #    En mode PARTIAL : on va uniquement traiter les dates > last_date
    #    Pour simplifier, on considère FULL si le fichier n’existait pas.
    mode_full = True

    # Si le fichier quadrants.parquet avait existé avant suppression, on l’aurait déjà récupéré ici.
    # Mais on vient juste de supprimer. Donc à ce stade, on est forcément en FULL.
    # (Si vous voulez vraiment faire PARTIAL, il faudrait garder une sauvegarde avant suppression.)

    # ─── 4. CALCUL DU DATAFRAME « WORKING » (toutes les dates) POUR DELTA & ZSCORES ─
    #     Cette étape se fait sur tout l’historique, même si on fait incrémental ensuite.
    window_lag  = Window.orderBy("date")
    window_roll = Window.orderBy("date").rowsBetween(-120, -1)
    working_df  = df_ind

    indicator_cols = [
        "INFLATION",
        "UNEMPLOYMENT",
        "CONSUMER_SENTIMENT",
        "High_Yield_Bond_SPREAD",
        "10-2Year_Treasury_Yield_Bond",
        "TAUX_FED",
    ]

    # 4.1 Calcul des deltas + moyenne mobile + écart-type mobile + z-score
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

    # 4.2 Calcul des médianes & écarts-type long terme sur la valeur brute
    median_std = {}
    for ind in indicator_cols:
        méd = working_df.select(expr(f"percentile_approx(`{ind}`, 0.5)")).first()[0]
        std_all = working_df.agg(stddev_samp(col(ind))).first()[0]
        median_std[ind] = (méd, std_all)

    # ─── 5. FILTRAGE EN MODE INCRÉMENTAL (si on voulait PARTIAL) ─────────────────
    #     Ici, on force FULL (mode_full = True), donc new_df = working_df.
    new_df = working_df

    # ─── 6. CALCUL DES SCORES (pos_score, var_score, combined) ───────────────────
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
        new_df = (
            new_df
            .withColumn(f"{ind}_pos_score", pos_score(ind, méd, std_all))
            .withColumn(f"{ind}_var_score", var_score(f"{ind}_zscore"))
            .withColumn(f"{ind}_combined", col(f"{ind}_pos_score") + col(f"{ind}_var_score"))
        )

    # ─── 7. INITIALISATION DES SCORES DES QUADRANTS ───────────────────────────────
    new_df = (
        new_df
        .withColumn("score_Q1", expr("0"))
        .withColumn("score_Q2", expr("0"))
        .withColumn("score_Q3", expr("0"))
        .withColumn("score_Q4", expr("0"))
    )

    # ─── 8. RÈGLES MÉTIER POUR AJOUTER DES POINTS DANS CHAQUE QUADRANT ───────────
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
        "10-2Year_Treasury_Yield_Bond_combined": (["score_Q4"], ["score_Q3"]),
        "TAUX_FED_combined":                     (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
    }

    for combined_col, (pos_quads, neg_quads) in mappings.items():
        new_df, exprs = add_points(new_df, combined_col, pos_quads, neg_quads)
        for q, expr_cond in exprs.items():
            new_df = new_df.withColumn(q, col(q) + expr_cond)

    # ─── 9. DÉTERMINER LE QUADRANT ASSIGNÉ (celui qui a le score max) ────────────
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

    # ─── 10. SÉLECTION DES COLONNES FINALES DANS LE BON ORDRE ────────────────────
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
    final_df = new_df.select(*final_cols)

    # ─── 11. ÉCRITURE DU PARQUET UNIQUE ───────────────────────────────────────────
    # 11.1 Écrire INTO un dossier temporaire "quadrants.parquet_tmp" (coalesce(1))
    tmp_parquet_dir = output_parquet_path + "_tmp"
    if os.path.isdir(tmp_parquet_dir):
        shutil.rmtree(tmp_parquet_dir)

    final_df.coalesce(1).write.mode("overwrite").parquet(tmp_parquet_dir)

    # 11.2 Repérer le fichier part-*.parquet
    part_parquet_pattern = os.path.join(tmp_parquet_dir, "part-*.parquet")
    part_list = glob.glob(part_parquet_pattern)
    if not part_list:
        spark.stop()
        raise RuntimeError(f"Aucun part-*.parquet trouvé dans {tmp_parquet_dir}")
    unique_part_parquet = part_list[0]

    # 11.3 Déplacer/renommer ce part-*.parquet en output_parquet_path
    #      (supprime tout ce qui existait déjà sous ce nom)
    if os.path.exists(output_parquet_path):
        if os.path.isdir(output_parquet_path):
            shutil.rmtree(output_parquet_path)
        else:
            os.remove(output_parquet_path)
    shutil.move(unique_part_parquet, output_parquet_path)

    # 11.4 Supprimer le dossier temporaire
    shutil.rmtree(tmp_parquet_dir)
    print(f"> Parquet unique écrit dans : {output_parquet_path}")

    # ─── 12. ÉCRITURE DU CSV UNIQUE (mêmes données) ─────────────────────────────
    output_csv_path = str(Path(output_parquet_path).with_suffix(".csv"))
    tmp_csv_dir = output_csv_path + "_tmp"
    if os.path.isdir(tmp_csv_dir):
        shutil.rmtree(tmp_csv_dir)

    # 12.1 Écrire final_df en CSV (header=True) dans tmp_csv_dir
    final_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_csv_dir)

    # 12.2 Repérer part-*.csv
    part_csv_pattern = os.path.join(tmp_csv_dir, "part-*.csv")
    csv_list = glob.glob(part_csv_pattern)
    if not csv_list:
        spark.stop()
        raise RuntimeError(f"Aucun part-*.csv trouvé dans {tmp_csv_dir}")
    unique_part_csv = csv_list[0]

    # 12.3 Supprimer l’ancien quadrants.csv s’il existe
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    # 12.4 Déplacer (rename) part-*.csv → quadrants.csv
    shutil.move(unique_part_csv, output_csv_path)
    shutil.rmtree(tmp_csv_dir)
    print(f"> CSV unique écrit dans : {output_csv_path}")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compute_quadrants.py <input_indicators.parquet> <output_quadrants.parquet>")
        sys.exit(1)

    indicators_path = sys.argv[1]
    output_path     = sys.argv[2]
    main(indicators_path, output_path)
