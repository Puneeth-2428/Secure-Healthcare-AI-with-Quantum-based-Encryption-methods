import os
import numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

os.makedirs("results", exist_ok=True)
parts = [np.load(f"data/processed/hospital_{i}.npz") for i in range(3)]
X = np.vstack([p["X"] for p in parts])
y = np.concatenate([p["y"] for p in parts])
test = np.load("data/processed/test.npz")

model = LogisticRegression(max_iter=1000).fit(X, y)
proba = model.predict_proba(test["X"])[:, 1]
res = {"accuracy": accuracy_score(test["y"], proba > 0.5),
       "auc": roc_auc_score(test["y"], proba)}
print(res)
json.dump(res, open("results/baseline.json", "w"))