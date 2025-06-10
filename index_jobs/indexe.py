import os, csv, json, requests
import shutil
""" indexe.py

Objectif :
Ce script permet d’indexer un fichier Parquet dans Elasticsearch.
Il est utilisé à la fin du pipeline pour stocker les résultats d’analyse (comme les quadrants économiques)
dans un moteur de recherche/visualisation (pour notre projet dans : Kibana).

Étapes réalisées :
1. Chargement du fichier `.parquet` avec Spark (permet de gérer de gros volumes).
2. Conversion en DataFrame Pandas (si la taille est raisonnable pour indexation rapide).
3. Connexion à Elasticsearch (avec authentification de base et désactivation des certificats SSL).
4. Création de l’index s’il n’existe pas encore.
5. Transformation ligne par ligne en documents JSON puis envoi groupé dans Elasticsearch.

Paramètres attendus :
Ce script s’utilise en ligne de commande avec 5 arguments :
```bash
python3 indexe.py <chemin_parquet> <host_ES> <index_name> <username> <password> """

ES_URL = "http://127.0.0.1:9200"
HEADERS = {
    "Accept":       "application/vnd.elasticsearch+json;compatible-with=8",
    "Content-Type": "application/vnd.elasticsearch+json;compatible-with=8",
}

DATA_DIR = os.path.expanduser("~/airflow/data")

mapping_q = {
  "properties": {
    "date": {"type":"date","format":"yyyy-MM-dd"},
    **{fld: {"type":"float"} for fld in [
       "INFLATION","UNEMPLOYMENT","CONSUMER_SENTIMENT",
       "High_Yield_Bond_SPREAD","10-2Year_Treasury_Yield_Bond","TAUX_FED"
    ]},
    **{f"{fld}_{suf}": {"type":"float"}
       for fld in ["INFLATION","UNEMPLOYMENT","CONSUMER_SENTIMENT",
                   "High_Yield_Bond_SPREAD","10-2Year_Treasury_Yield_Bond","TAUX_FED"]
       for suf in ["delta","zscore","pos_score","var_score","combined"]
    },
    **{f"score_Q{i}": {"type":"float"} for i in (1,2,3,4)},
    "assigned_quadrant": {"type":"keyword"}
  }
}

mapping_a = {
  "properties": {
    "asset_id":          {"type":"keyword"},
    "assigned_quadrant": {"type":"keyword"},
    "start_date":        {"type":"date","format":"yyyy-MM-dd"},
    "end_date":          {"type":"date","format":"yyyy-MM-dd"},
    "monthly_return":    {"type":"float"},
    "max_drawdown":      {"type":"float"},
    "sharpe_annualized": {"type":"float"},
    "sharpe_mensuel":    {"type":"float"}
  }
}
mapping_bt_timeseries = {
  "properties": {
    "year_month":        {"type":"date","format":"yyyy-MM-dd"},
    "SP500_ret":         {"type":"float"},
    "GOLD_OZ_USD_ret":   {"type":"float"},
    "portfolio_return":  {"type":"float"},
    "wealth":            {"type":"float"},
    "SP500_wealth":      {"type":"float"},
    "GOLD_wealth":       {"type":"float"}
  }
}

mapping_bt_stats = {
  "properties": {
    **{f"{lbl}_{metric}": {"type":"float"}
       for lbl in ("strategy","SP500","GOLD")
       for metric in ("vol_annual","sharpe_annual","max_drawdown","avg_year_return")}
  }
}

CSV_SPECS = {
    "quadrants":           ("quadrants.csv",          mapping_q),
    "assets_performance":  ("assets_performance_by_quadrant.csv", mapping_a),
    "backtest_timeseries": ("backtest_results/backtest_timeseries.csv",mapping_bt_timeseries),
    "backtest_stats":      ("backtest_results/backtest_stats.csv",     mapping_bt_stats),
}

def create_index(name, mapping):
    requests.delete(f"{ES_URL}/{name}", headers=HEADERS)
    body = {
      "settings": {"number_of_shards":1,"number_of_replicas":0},
      "mappings": mapping
    }
    resp = requests.put(f"{ES_URL}/{name}", headers=HEADERS, data=json.dumps(body))
    resp.raise_for_status()

def bulk_index(name, filename, mapping):
    path = os.path.join(DATA_DIR, filename)
    date_fields  = [k for k,v in mapping["properties"].items() if v.get("type")=="date"]
    float_fields = [k for k,v in mapping["properties"].items() if v.get("type")=="float"]
    lines = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            doc = {}
            for k,v in row.items():
                if k in date_fields:
                    doc[k] = v
                elif k in float_fields:
                    try: doc[k] = float(v)
                    except: doc[k] = None
                else:
                    doc[k] = v
            lines.append(json.dumps({"index":{"_index":name}}))
            lines.append(json.dumps(doc))
    payload = "\n".join(lines) + "\n"
    resp = requests.post(f"{ES_URL}/_bulk", headers=HEADERS, data=payload)
    resp.raise_for_status()
    result = resp.json()
    if result.get("errors"):
        for item in result["items"]:
            err = item.get("index",{}).get("error")
            if err:
                print("Erreur bulk:", err)
                break
    return len(result["items"])

def index_data():
    r = requests.get(ES_URL, headers=HEADERS, timeout=5); r.raise_for_status()
    print("✅ ES reachable:", r.json().get("tagline"))
    for idx,(fname,mapping) in CSV_SPECS.items():
        full = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(full):
            print(f"[!] Missing file: {full}")
            continue
        print(f"→ Index '{idx}'…")
        create_index(idx, mapping)
        count = bulk_index(idx, fname, mapping)
        print(f"   ✅ {count} docs indexed in '{idx}'")

if __name__=="__main__":
    index_data()
