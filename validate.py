import pandas as pd

icu = pd.read_csv("data/ICUSTAYS.csv", low_memory=False)
icu.columns = [c.lower() for c in icu.columns]

print(60324 in icu["icustay_id"].values)
print(icu["icustay_id"].min(), "→", icu["icustay_id"].max())
print(f"Total stays: {len(icu)}")