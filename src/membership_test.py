import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.makedirs("results", exist_ok=True)
rng = np.random.default_rng(7)
parts = [np.load(f"data/processed/hospital_{i}.npz") for i in range(3)]
test = np.load("data/processed/test.npz")
X = np.vstack([p["X"] for p in parts] + [test["X"]]).astype(np.float64)
y = np.concatenate([p["y"] for p in parts] + [test["y"]]).astype(np.float64)
N_MEM, TRIALS = 40, 50

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def losses(w, b, Xs, ys):
    p = np.clip(sigmoid(Xs @ w + b), 1e-9, 1 - 1e-9)
    return -(ys * np.log(p) + (1 - ys) * np.log(1 - p))

def train_dp(Xs, ys, nm, epochs=100, lr=0.1, clip=1.0, batch=8):
    n, d = Xs.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        for _ in range(max(1, n // batch)):
            idx = rng.choice(n, batch, replace=False)
            err = sigmoid(Xs[idx] @ w + b) - ys[idx]
            g = np.hstack([Xs[idx] * err[:, None], err[:, None]])
            g = g / np.maximum(1.0, np.linalg.norm(g, axis=1, keepdims=True) / clip)
            step = (g.sum(0) + rng.normal(0, nm * clip, d + 1)) / batch
            w -= lr * step[:-1]
            b -= lr * step[-1]
    return w, b

def fit(name, Xm, ym):
    if name == "non-private":
        m = LogisticRegression(C=1e3, max_iter=5000).fit(Xm, ym)
        return m.coef_[0], m.intercept_[0]
    return train_dp(Xm, ym, float(name.split()[-1]))

names = ["non-private", "dp noise 0.0", "dp noise 1.0", "dp noise 2.0", "dp noise 5.0", "dp noise 10.0"]
attack = {n: [] for n in names}
acc = {n: [] for n in names}
labels = np.array([1] * N_MEM + [0] * N_MEM)

for _ in range(TRIALS):
    idx = rng.permutation(len(X))
    mem, non = idx[:N_MEM], idx[N_MEM:2 * N_MEM]
    for n in names:
        w, b = fit(n, X[mem], y[mem])
        score = -np.concatenate([losses(w, b, X[mem], y[mem]),
                                 losses(w, b, X[non], y[non])])
        attack[n].append(roc_auc_score(labels, score))
        acc[n].append(np.mean((sigmoid(X[non] @ w + b) > 0.5) == y[non]))

results = {}
for n in names:
    a = np.array(attack[n])
    se = a.std(ddof=1) / np.sqrt(TRIALS)
    results[n] = {"attack_auc": float(a.mean()), "std_error": float(se),
                  "nonmember_accuracy": float(np.mean(acc[n]))}
    print(f"{n:14s} attack AUC {a.mean():.3f} +/- {se:.3f}   unseen-patient accuracy {np.mean(acc[n]):.3f}")

json.dump(results, open("results/membership.json", "w"), indent=2)