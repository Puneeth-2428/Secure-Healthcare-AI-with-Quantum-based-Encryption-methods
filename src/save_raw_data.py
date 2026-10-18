import os
from ucimlrepo import fetch_ucirepo

os.makedirs("data/raw", exist_ok=True)
ds = fetch_ucirepo(id=45)                      # UCI Heart Disease
df = ds.data.features.copy()
df["target"] = ds.data.targets.iloc[:, 0]      # 0 = no disease, 1-4 = disease
df.to_csv("data/raw/heart_disease_uci.csv", index=False)
print(df.shape, "->", "data/raw/heart_disease_uci.csv")