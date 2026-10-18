import os, json, time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from kyber_py.ml_kem import ML_KEM_768
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Ensure directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

rng = np.random.default_rng(int(os.environ.get("SEED", "42")))
SCALE, N_NODES = 2**20, 3
ROUNDS, LOCAL_EPOCHS, CLIP = 20, 3, 1.0
NM = float(os.environ.get("NM", "1.0"))

EPS = None
epsilon_path = "results/epsilon.json"
if os.path.exists(epsilon_path):
    try:
        EPS = json.load(open(epsilon_path))[str(NM)]["worst_hospital_epsilon"]
    except Exception:
        pass

parts = [np.load(f"data/processed/hospital_{i}.npz") for i in range(3)]
H = [(p["X"].astype(np.float64), p["y"].astype(np.float64)) for p in parts]
test = np.load("data/processed/test.npz")
Xt, yt = test["X"].astype(np.float64), test["y"]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def local_train(w, b, X, y, epochs, lr=0.1, clip=CLIP, nm=NM, batch=32):
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

# ---- secret sharing ----
def encode(v):
    return np.round(v * SCALE).astype(np.int64).view(np.uint64)

def decode(u):
    return u.view(np.int64).astype(np.float64) / SCALE

def share(vec):
    shares = [np.frombuffer(os.urandom(8 * len(vec)), dtype=np.uint64).copy()
              for _ in range(N_NODES - 1)]
    last = vec.copy()
    for s in shares:
        last = last - s
    return shares + [last]

# ---- post-quantum encrypted transport (ML-KEM-768 + AES-256-GCM) ----
def skey(shared):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=b"fl-share-v1").derive(shared)

def send(vec, ek, aad):
    shared, kem_ct = ML_KEM_768.encaps(ek)
    nonce = os.urandom(12)
    return kem_ct, nonce, AESGCM(skey(shared)).encrypt(nonce, vec.tobytes(), aad)

def receive(pkg, dk, aad):
    kem_ct, nonce, sealed = pkg
    shared = ML_KEM_768.decaps(dk, kem_ct)
    return np.frombuffer(AESGCM(skey(shared)).decrypt(nonce, sealed, aad),
                         dtype=np.uint64).copy()

node_keys = [ML_KEM_768.keygen() for _ in range(N_NODES)]

d = H[0][0].shape[1]
w, b = np.zeros(d), 0.0
sizes = np.array([len(x) for x, _ in H], float)
wts = sizes / sizes.sum()
bytes_sent, worst = 0, 0.0
t0 = time.time()

for r in range(ROUNDS):
    ups = [local_train(w, b, x, y, LOCAL_EPOCHS) for x, y in H]
    vecs = [np.append(u[0], u[1]) for u in ups]
    node_sums = [np.zeros(d + 1, dtype=np.uint64) for _ in range(N_NODES)]
    for h, (v, wt) in enumerate(zip(vecs, wts)):
        for node, s in enumerate(share(encode(v * wt))):
            aad = f"hospital_{h}|node_{node}|round_{r}".encode()
            pkg = send(s, node_keys[node][0], aad)
            bytes_sent += sum(len(p) for p in pkg)
            node_sums[node] = node_sums[node] + receive(pkg, node_keys[node][1], aad)
    total = np.zeros(d + 1, dtype=np.uint64)
    for s in node_sums:
        total = total + s
    agg = decode(total)
    plain = sum(wt * v for wt, v in zip(wts, vecs))
    worst = max(worst, float(np.abs(agg - plain).max()))
    w, b = agg[:-1], agg[-1]

p = sigmoid(Xt @ w + b)
res = {"accuracy": float(accuracy_score(yt, p > 0.5)),
       "auc": float(roc_auc_score(yt, p)),
       "max_diff_vs_plain_average": worst,
       "rounds": ROUNDS, "noise_multiplier": NM, "clip_norm": CLIP,
       "mlkem_variant": "ML-KEM-768", "validation_nodes": N_NODES,
       "bytes_transmitted": int(bytes_sent), "seconds": round(time.time() - t0, 1),
       "epsilon": EPS, "delta": 1e-5, "tee_enabled": False}
print(res)
json.dump(res, open(os.environ.get("OUT", "results/pipeline.json"), "w"), indent=2)
if os.environ.get("MODEL_OUT"):
    np.savez(os.environ["MODEL_OUT"], w=w, b=b, noise_multiplier=NM, clip_norm=CLIP,
             epsilon=EPS if EPS is not None else np.nan, delta=1e-5, rounds=ROUNDS)
    print("model saved to", os.environ["MODEL_OUT"])