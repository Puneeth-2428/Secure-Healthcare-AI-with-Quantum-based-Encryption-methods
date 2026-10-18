import os
import numpy as np, json
from sklearn.metrics import accuracy_score, roc_auc_score

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(42)
parts = [np.load(f"data/processed/hospital_{i}.npz") for i in range(3)]
H = [(p["X"].astype(np.float64), p["y"].astype(np.float64)) for p in parts]
test = np.load("data/processed/test.npz")
Xt, yt = test["X"].astype(np.float64), test["y"]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def local_train(w, b, X, y, epochs, lr=0.1, clip=1.0, nm=0.0, batch=32):
    w = w.copy()
    n = len(X)
    bs = min(batch, n)
    for _ in range(epochs):
        for _ in range(max(1, n // bs)):
            idx = rng.choice(n, bs, replace=False)
            err = sigmoid(X[idx] @ w + b) - y[idx]
            g = np.hstack([X[idx] * err[:, None], err[:, None]])
            g = g / np.maximum(1.0, np.linalg.norm(g, axis=1, keepdims=True) / clip)
            step = (g.sum(0) + rng.normal(0, nm * clip, g.shape[1])) / bs
            w -= lr * step[:-1]
            b -= lr * step[-1]
    return w, b

def federated(nm, rounds=20, local_epochs=3):
    d = H[0][0].shape[1]
    w, b = np.zeros(d), 0.0
    sizes = np.array([len(x) for x, _ in H], float)
    for _ in range(rounds):
        ups = [local_train(w, b, x, y, local_epochs, nm=nm) for x, y in H]
        w = sum(s * u[0] for s, u in zip(sizes, ups)) / sizes.sum()
        b = sum(s * u[1] for s, u in zip(sizes, ups)) / sizes.sum()
    return w, b

results = {}
for nm in [0.0, 1.0, 2.0]:
    w, b = federated(nm)
    p = sigmoid(Xt @ w + b)
    results[str(nm)] = {"accuracy": float(accuracy_score(yt, p > 0.5)),
                        "auc": float(roc_auc_score(yt, p))}
    print(f"federated, noise multiplier {nm}: {results[str(nm)]}")

json.dump(results, open("results/federated.json", "w"), indent=2)