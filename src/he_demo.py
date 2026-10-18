import time
import numpy as np
from phe import paillier

rng = np.random.default_rng(0)

for key_len in [2048, 3072]:
    print(f"\n--- Testing Paillier with Key Length: {key_len} bits ---")
    t0 = time.time()
    pub, priv = paillier.generate_paillier_keypair(n_length=key_len)
    print(f"keygen: {time.time() - t0:.1f} s")

    D = 15
    updates = [rng.normal(size=D) for _ in range(3)]
    sizes = np.array([100.0, 100.0, 97.0])
    wts = sizes / sizes.sum()

    t0 = time.time()
    enc = [[pub.encrypt(float(v)) for v in u] for u in updates]
    print(f"encrypt 3 updates: {time.time() - t0:.1f} s")

    t0 = time.time()
    agg = [sum(enc[h][j] * float(wts[h]) for h in range(3)) for j in range(D)]
    print(f"homomorphic aggregation: {time.time() - t0:.2f} s")

    result = np.array([priv.decrypt(c) for c in agg])
    plain = sum(w * u for w, u in zip(wts, updates))
    print("max difference vs plain average:", float(np.abs(result - plain).max()))