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


def write_single_parquet(df, output_path: str):
    """
    Écrit la DataFrame `df` en un UNIQUE fichier Parquet (coalesce(1)) :
    1. On écrit d'abord dans `output_path + "_tmp_parquet"`
    2. On repère le part-*.parquet généré.
    3. On supprime l'ancien `output_path` (s'il existe) et on déplace le part-*.parquet
       en nom exact `output_path`.
    4. On supprime le dossier temporaire.
    """
    tmp_dir = output_path + "_tmp_parquet"

    # 1) S'assurer qu'il n'existe pas de dossier temporaire ancien
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # 2) Coalesce à 1 partition et écrire dans tmp_dir
    df.coalesce(1).write.mode("overwrite").parquet(tmp_dir)

    # 3) Repérer le seul fichier part-XXXX.parquet à l'intérieur
    part_files = glob.glob(os.path.join(tmp_dir, "part-*.parquet"))
    if not part_files:
        raise RuntimeError(f"Aucun part-*.parquet trouvé dans {tmp_dir}")

    single_part = part_files[0]

    # 4) Supprimer l'ancien fichier de sortie s'il existe, puis déplacer le nouveau
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(single_part, output_path)

    # 5) Supprimer le dossier temporaire
    shutil.rmtree(tmp_dir)


def write_single_csv(df, output_path: str):
    """
    Écrit la DataFrame `df` en un UNIQUE fichier CSV (coalesce(1)) avec header :
    1. On écrit d'abord dans `output_path + "_tmp_csv"`
    2. On repère le part-*.csv généré.
    3. On supprime l'ancien `output_path` (s'il existe) et on déplace le part-*.csv
       en nom exact `output_path`.
    4. On supprime le dossier temporaire.
    """
    tmp_dir = output_path + "_tmp_csv"

    # 1) S'assurer qu'il n'existe pas de dossier temporaire ancien
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # 2) Coalesce à 1 partition et écrire dans tmp_dir (avec header)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_dir)

    # 3) Repérer le seul fichier part-XXXX.csv à l'intérieur
    part_files = glob.glob(os.path.join(tmp_dir, "part-*.csv"))
    if not part_files:
        raise RuntimeError(f"Aucun part-*.csv trouvé dans {tmp_dir}")

    single_part = part_files[0]

    # 4) Supprimer l'ancien fichier de sortie s'il existe, puis déplacer le nouveau
    if os.path.exists(output_path):
        os.remove(output_path)
    shutil.move(single_part, output_path)

    # 5) Supprimer le dossier temporaire
    shutil.rmtree(tmp_dir)


def main(indicators_parquet_path: str, output_parquet_path: str, output_csv_path: str, full_df=None):
    """
    - Si `output_parquet_path` existe déjà, on le lit, on récupère le max(date),
      puis on ne calcule que les mois > last_date (mode incrémental).
    - Sinon, on refait tout l'historique (mode full).
    - Ensuite, on concatène l'ancien DataFrame avec les nouvelles lignes.
    - Enfin, on écrit :
         • Un seul fichier Parquet → `output_parquet_path`
         • Un seul fichier CSV   → `output_csv_path`
    """
    spark = (
        SparkSession.builder
        .appName("ComputeEconomicQuadrants")
        .getOrCreate()
    )

    # --- 1. Charger Indicators + convertir la colonne "date" en DateType ---
    df_ind = (
        spark.read.parquet(indicators_parquet_path)
        .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
        .orderBy("date")
    )

    # 2. Détecter si le fichier Parquet de sortie existe déjà
    existing_df = None
    last_date = None

    if Path(output_parquet_path).is_file():
        # Lire l'ancien quadrants.parquet
        existing_df = spark.read.parquet(output_parquet_path)
        # Récupérer la date max
        max_row = existing_df.agg({"date": "max"}).collect()[0]
        last_date = max_row[0]  # type: ignore
        # Filtrer seulement les nouvelles lignes > last_date
        new_ind = df_ind.filter(col("date") > last_date)
    else:
        # Pas de fichier existant → full refresh
        new_ind = df_ind

    # Si aucune nouvelle ligne à traiter, on sort
    if new_ind.rdd.isEmpty():
        print(f"Aucune nouvelle donnée après {last_date}. Pas de réécriture.")
        spark.stop()
        return

    # --- 3. Définir les fenêtres pour delta / z-score sur l’historique complet ---
    window_lag = Window.orderBy("date")
    window_roll = Window.orderBy("date").rowsBetween(-120, -1)

    working_df = df_ind  # on va calculer deltas et z-scores sur tout l’historique

    indicator_cols = [
        "INFLATION",
        "UNEMPLOYMENT",
        "CONSUMER_SENTIMENT",
        "High_Yield_Bond_SPREAD",
        "10-2Year_Treasury_Yield_Bond",
        "TAUX_FED",
    ]

    last_two = full_df.orderBy("date", ascending=False).limit(2).orderBy("date")

    # --- 4. Calcul des deltas et z-scores sur tout l’historique ---
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

    # --- 5. Calcul des médianes & écart‐type long‐terme sur les valeurs brutes ---
    median_std = {}
    for ind in indicator_cols:
        median_val = working_df.select(expr(f"percentile_approx(`{ind}`, 0.5)")).first()[0]
        std_all = working_df.agg(stddev_samp(col(ind))).first()[0]
        median_std[ind] = (median_val, std_all)

    # 6. Fonctions utilitaires pour pos_score et var_score
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

    # --- 7. Ne conserver que les nouvelles lignes dans new_df ---
    new_df = working_df.filter(col("date") > last_date) if last_date is not None else working_df

    # --- 8. Pour chaque indicateur, ajouter pos_score, var_score, combined ---
    for ind in indicator_cols:
        med, std_all = median_std[ind]
        new_df = (
            new_df
            .withColumn(f"{ind}_pos_score", pos_score(ind, med, std_all))
            .withColumn(f"{ind}_var_score", var_score(f"{ind}_zscore"))
            .withColumn(f"{ind}_combined", col(f"{ind}_pos_score") + col(f"{ind}_var_score"))
        )

    # 9. Répartition des points dans les quadrants Q1..Q4
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

    # 10. Calculer assigned_quadrant (celui qui a le plus gros score)
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

    # 11. Sélectionner toutes les colonnes finales
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

    # 12. Concaténer avec existing_df si on était en mode incrémental
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
