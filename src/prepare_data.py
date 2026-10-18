import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

SEED, N_HOSPITALS, SOURCE = 42, 3, "heart"
os.makedirs("data/processed", exist_ok=True)

if SOURCE == "heart":
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=45)
    X = ds.data.features.copy()
    y = (ds.data.targets.iloc[:, 0] > 0).astype(int)
    
    # Impute missing values instead of discarding rows (preserves sample size)
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    y = y.to_numpy()
else:
    X, y = load_breast_cancer(return_X_y=True)

# Global held-out test set
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

# Fine-tuning: Simulate Non-IID data distribution using Dirichlet distribution
rng = np.random.default_rng(SEED)
min_size = 0
# Dirichlet concentration parameter (lower = more non-IID skew between hospitals)
alpha = 0.5 
proportions = rng.dirichlet(np.repeat(alpha, N_HOSPITALS))
proportions = (proportions * len(X_tr)).astype(int)
proportions[-1] = len(X_tr) - proportions[:-1].sum() # Ensure exact sum

idx = rng.permutation(len(X_tr))
splits = np.split(idx, np.cumsum(proportions)[:-1])

for i, part in enumerate(splits):
    np.savez(f"data/processed/hospital_{i}.npz",
             X=X_tr[part].astype("float32"), y=y_tr[part].astype("int64"))
    print(f"hospital_{i} (Non-IID): {len(part)} patients, {y_tr[part].mean():.2f} positive rate")

np.savez("data/processed/test.npz",
         X=X_te.astype("float32"), y=y_te.astype("int64"))
print("test:", len(X_te), "patients, features:", X_tr.shape[1])