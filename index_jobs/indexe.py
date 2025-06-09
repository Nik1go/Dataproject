import os, csv, json, requests

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
    "sset_id":           {"type":"keyword"},
    "assigned_quadrant": {"type":"keyword"},
    "start_date":        {"type":"date","format":"yyyy-MM-dd"},
    "end_date":          {"type":"date","format":"yyyy-MM-dd"},
    "monthly_return":    {"type":"float"},
    "max_drawdown":      {"type":"float"},
    "sharpe_annualized": {"type":"float"},
    "sharpe_mensuel":    {"type":"float"}
  }
}

CSV_SPECS = {
    "quadrants":          ("quadrants.csv",          mapping_q),
    "assets_performance": ("assets_performance.csv", mapping_a),
}

def create_index(name, mapping):
    requests.delete(f"{ES_URL}/{name}", headers=HEADERS)
    body = {"settings":{"number_of_shards":1,"number_of_replicas":0}, "mappings":mapping}
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
        # affiche la première erreur
        for item in result["items"]:
            err = item.get("index",{}).get("error")
            if err:
                print("Erreur bulk:", err)
                break
    return len(result["items"])

def index_data():
    # Test de connexion
    r = requests.get(ES_URL, headers=HEADERS, timeout=5); r.raise_for_status()
    print("✅ ES reachable:", r.json().get("tagline"))

    for idx,(fname,mapping) in CSV_SPECS.items():
        full = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(full):
            print(f"[!] Manquant : {full}")
            continue
        print(f"→ Index '{idx}'…")
        create_index(idx, mapping)
        count = bulk_index(idx, fname, mapping)
        print(f"   ✅ {count} docs indexés dans '{idx}'")

if __name__=="__main__":
    index_data()
