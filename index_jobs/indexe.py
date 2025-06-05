#!/usr/bin/env python3
import sys
from pyspark.sql import SparkSession
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def indexer_parquet_dans_es(parquet_path, es_host, es_index, es_user, es_pass):
    # 1. On crée une session Spark pour lire le parquet
    spark = SparkSession.builder.appName("IndexerQuadrants").getOrCreate()
    df = spark.read.parquet(parquet_path)
    # On ramène tout en Pandas (taille raisonnable) ou on parcourt les DataFrame Spark en partitions
    pdf = df.toPandas()

    # 2. Connexion “HTTPS – pas de vérification de certificat”
    es = Elasticsearch(
        hosts=[ { "host": es_host, "port": 9200, "scheme": "https" } ],
        basic_auth=(es_user, es_pass),
        verify_certs=False
    )

    # 3. Vérifier si l’index existe, sinon le créer
    if not es.indices.exists(index=es_index):
        es.indices.create(index=es_index)

    # 4. Préparer les documents à indexer (exemple : chaque ligne devient un doc JSON)
    actions = []
    for _, row in pdf.iterrows():
        # row.to_dict() contient toutes les colonnes du DataFrame
        action = {
            "_index": es_index,
            # Optionnel : vous pouvez forcer un _id = str(row["date"]) ou laissez ES en générer un
            "_source": row.to_dict()
        }
        actions.append(action)

    # 5. Utiliser bulk() pour indexer en lot
    success, _ = bulk(es, actions, refresh=True)
    print(f"Indexation terminée, nombre de docs indexés : {success}")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: indexe.py <path_parquet> <es_host> <es_index> <es_user> <es_pass>")
        sys.exit(1)

    parquet_path = sys.argv[1]
    es_host      = sys.argv[2]
    es_index     = sys.argv[3]
    es_user      = sys.argv[4]
    es_pass      = sys.argv[5]

    indexer_parquet_dans_es(parquet_path, es_host, es_index, es_user, es_pass)
