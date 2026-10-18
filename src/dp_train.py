import os
import numpy as np, json
from sklearn.metrics import accuracy_score, roc_auc_score

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(42)
parts = [np.load(f"data/processed/hospital_{i}.npz") for i in range(3)]
X = np.vstack([p["X"] for p in parts]).astype(np.float64)
y = np.concatenate([p["y"] for p in parts]).astype(np.float64)
test = np.load("data/processed/test.npz")
Xt, yt = test["X"].astype(np.float64), test["y"]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_dp(X, y, epochs=60, lr=0.1, clip=1.0, noise_mult=1.0, batch=32):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for epoch in range(epochs):
        # Fine-tuning: Optional learning rate decay per epoch for stable convergence
        curr_lr = lr / (1.0 + 0.01 * epoch)
        for _ in range(n // batch):
            idx = rng.choice(n, batch, replace=False)
            Xb, yb = X[idx], y[idx]
            err = sigmoid(Xb @ w + b) - yb
            g = np.hstack([Xb * err[:, None], err[:, None]])   # per-example gradients
            norms = np.linalg.norm(g, axis=1, keepdims=True)
            g = g / np.maximum(1.0, norms / clip)               # clip each one
            noisy = g.sum(0) + rng.normal(0, noise_mult * clip, d + 1)
            step = noisy / batch
            w -= curr_lr * step[:-1]
            b -= curr_lr * step[-1]
    return w, b

results = {}
for nm in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    w, b = train_dp(X, y, noise_mult=nm)
    p = sigmoid(Xt @ w + b)
    results[str(nm)] = {"accuracy": float(accuracy_score(yt, p > 0.5)),
                        "auc": float(roc_auc_score(yt, p))}
    print(f"noise multiplier {nm}: {results[str(nm)]}")

json.dump(results, open("results/dp.json", "w"), indent=2)  