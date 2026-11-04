import json

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, ROOT
from features_pandas import build_features

meta = json.load(open(ROOT / "models" / "meta.json"))
features = [f for f in meta["features"] if f != "city_code"]

raw = pd.read_parquet(RAW_DIR / "dhaka.parquet")
mine = build_features(raw, meta["cities"])
spark = pd.read_parquet(PROCESSED_DIR / "features.parquet")
spark = spark[spark["city"] == "Dhaka"]

for d in (mine, spark):
    d["time"] = d["time"].astype("datetime64[ns]")
m = mine.merge(spark, on=["city", "time"], suffixes=("_mine", "_spark"))
print(f"Compared {len(m):,} Dhaka rows")

worst = []
for f in features:
    diff = (m[f + "_mine"] - m[f + "_spark"]).abs().max()
    worst.append((diff, f))
worst.sort(reverse=True)
print("Largest differences:", [(f, float(f"{d:.2e}")) for d, f in worst[:3]])
print("PASS" if worst[0][0] < 1e-4 else "MISMATCH: send me this output")