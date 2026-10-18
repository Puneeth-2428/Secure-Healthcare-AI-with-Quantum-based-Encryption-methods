import sys
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

path = sys.argv[1] if len(sys.argv) > 1 else "models/global_model_nm10.npz"
if not os.path.exists(path):
    print(f"Error: Model file '{path}' not found. Please verify the path or run training first.")
    sys.exit(1)

m = np.load(path)
t = np.load("data/processed/test.npz")
p = 1 / (1 + np.exp(-(t["X"].astype(float) @ m["w"] + float(m["b"]))))
print("model:", path, "| noise", float(m.get("noise_multiplier", 0.0)), "| epsilon", float(m.get("epsilon", 0.0)))
print("accuracy", round(accuracy_score(t["y"], p > 0.5), 3), "| AUC", round(roc_auc_score(t["y"], p), 3))
print("first 5 predicted risks:", np.round(p[:5], 2), "| true labels:", t["y"][:5])