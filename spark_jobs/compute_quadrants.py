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
from pyspark.sql.utils import AnalysisException


def main(indicators_parquet_path: str, output_parquet_path: str):
    """
    Si quadrants.parquet existe, on lit l’ancien résultat, on détermine la date max déjà calculée,
    puis on ne recalcul que les mois supérieurs à cette date. On concatène anciens + nouveaux
    et on réécrit en overwrite. Sinon, on fait un full refresh depuis 2005 (toutes les données).
    """

    spark = (
        SparkSession.builder
        .appName("ComputeEconomicQuadrantsIncremental")
        .getOrCreate()
    )

    # 1) Chargement des indicateurs historiques
    df_ind = (
        spark.read
            .parquet(indicators_parquet_path)
            .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
            .orderBy("date")
    )

    # 2) Vérifier si output_parquet_path existe déjà
    output_path = Path(output_parquet_path)
    existing_df = None
    last_date = None

    if output_path.exists():
        # On charge l'ancien résultat
        existing_df = spark.read.parquet(output_parquet_path)
        max_row = existing_df.agg({"date": "max"}).collect()[0]
        last_date = max_row[0]  # On récupère la date max déjà traitée
        # On ne garde que les mois strictement supérieurs
        df_to_process = df_ind.filter(col("date") > last_date)
    else:
        # Full refresh : on calcule tout depuis 2005
        df_to_process = df_ind

    # Si pas de nouvelles données à traiter, on quitte
    if df_to_process.rdd.isEmpty():
        print(f"Aucun nouvel indicateur à traiter après {last_date}. Sortie sans rien faire.")
        spark.stop()
        return

    # 3) Fenêtres pour rolling (120 mois) et lag (1 mois)
    window_lag = Window.orderBy("date")
    window_roll = Window.orderBy("date").rowsBetween(-120, -1)

    # On va calculer delta + zscore sur tout l’historique (df_ind),
    # mais n’extraire que les nouvelles lignes plus tard.
    working_df = df_ind

    indicator_cols = [
        "INFLATION",
        "UNEMPLOYMENT",
        "CONSUMER_SENTIMENT",
        "High_Yield_Bond_SPREAD",
        "10-2Year_Treasury_Yield_Bond",
        "TAUX_FED",
    ]

    # 4) Calcul des deltas et z-scores sur l’ensemble de df_ind
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
    # 5) Calcul des médianes & écarts‐type long terme
    median_std = {}
    for ind in indicator_cols:
        median_val = working_df.select(expr(f"percentile_approx(`{ind}`, 0.5)")).first()[0]
        std_all = working_df.agg(stddev_samp(col(ind))).first()[0]
        median_std[ind] = (median_val, std_all)

    # 6) Fonctions de scoring (position et variation)
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

    # 7) On ne conserve dans new_df que les données postérieures à last_date (ou tout si full)
    if last_date is not None:
        new_df = working_df.filter(col("date") > last_date)
    else:
        new_df = working_df

    # 8) Ajout des colonnes pos_score, var_score et combined pour chaque indicateur
    for ind in indicator_cols:
        med, std_all = median_std[ind]
        new_df = (
            new_df
            .withColumn(f"{ind}_pos_score", pos_score(ind, med, std_all))
            .withColumn(f"{ind}_var_score", var_score(f"{ind}_zscore"))
            .withColumn(f"{ind}_combined", col(f"{ind}_pos_score") + col(f"{ind}_var_score"))
        )

    # 9) Initialiser les colonnes de score par quadrant à 0
    new_df = (
        new_df
        .withColumn("score_Q1", expr("0"))
        .withColumn("score_Q2", expr("0"))
        .withColumn("score_Q3", expr("0"))
        .withColumn("score_Q4", expr("0"))
    )

    # 10) Fonction utilitaire pour ajouter 1 point selon le signe de combined
    def add_points(df, combined_col: str, pos_quads: list, neg_quads: list):
        for q in pos_quads:
            df = df.withColumn(q, col(q) + when(col(combined_col) > 0, 1).otherwise(0))
        for q in neg_quads:
            df = df.withColumn(q, col(q) + when(col(combined_col) < 0, 1).otherwise(0))
        return df

    # 11) Mapping des combinaisons (pos vs. neg) vers Q1…Q4
    mappings = {
        "INFLATION_combined":   (["score_Q2", "score_Q3"], ["score_Q1", "score_Q4"]),
        "UNEMPLOYMENT_combined":(["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
        "CONSUMER_SENTIMENT_combined": (["score_Q1", "score_Q2"], ["score_Q3", "score_Q4"]),
        "High_Yield_Bond_SPREAD_combined": (["score_Q2", "score_Q3"], ["score_Q1", "score_Q4"]),
        "10-2Year_Treasury_Yield_Bond_combined": (["score_Q2"], ["score_Q3"]),
        "TAUX_FED_combined": (["score_Q2"], ["score_Q3"]),
    }

    for combined_col, (pos_quads, neg_quads) in mappings.items():
        new_df = add_points(new_df, combined_col, pos_quads, neg_quads)

    # 12) Déterminer assigned_quadrant
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

    # 13) Colonnes finales à retenir
    final_cols = [
        "date",
        *indicator_cols,
        *[f"{ind}_delta" for ind in indicator_cols],
        *[f"{ind}_zscore" for ind in indicator_cols],
        *[f"{ind}_pos_score" for ind in indicator_cols],
        *[f"{ind}_var_score" for ind in indicator_cols],
        *[f"{ind}_combined" for ind in indicator_cols],
        "score_Q1", "score_Q2", "score_Q3", "score_Q4", "assigned_quadrant",
    ]
    new_df = new_df.select(*final_cols)

    # 14) Concaténer ancien et nouveau si nécessaire, puis écrire en overwrite
    if existing_df is not None:
        merged = existing_df.select(*final_cols).unionByName(new_df)
    else:
        merged = new_df

    merged.write.mode("overwrite").parquet(output_parquet_path)
    print(f"Écriture {'incrémentale' if existing_df else 'full refresh'} terminée dans {output_parquet_path}.")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compute_quadrants_incremental.py <input_indicators.parquet> <output_quadrants.parquet>")
        sys.exit(1)

    indicators_path = sys.argv[1]
    output_path     = sys.argv[2]
    main(indicators_path, output_path)
