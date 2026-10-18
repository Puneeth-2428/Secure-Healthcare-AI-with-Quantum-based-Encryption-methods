import os, json, math
import numpy as np

os.makedirs("results", exist_ok=True)
ROUNDS, LOCAL_EPOCHS, BATCH, DELTA = 20, 3, 32, 1e-5
ORDERS = list(range(2, 257))

def log_binom(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def rdp_step(q, sigma, alpha):
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    terms = [log_binom(alpha, k) + k * math.log(q) + (alpha - k) * math.log1p(-q)
             + (k * k - k) / (2 * sigma ** 2) for k in range(alpha + 1)]
    m = max(terms)
    return (m + math.log(sum(math.exp(t - m) for t in terms))) / (alpha - 1)

def epsilon(q, sigma, steps):
    best = (float("inf"), None)
    for a in ORDERS:
        e = steps * rdp_step(q, sigma, a) + math.log(1 / DELTA) / (a - 1)
        if e < best[0]:
            best = (e, a)
    return best

sizes = [len(np.load(f"data/processed/hospital_{i}.npz")["y"]) for i in range(3)]
print("hospital sizes:", sizes, "| delta:", DELTA)

results = {}
# Fine-tuning: Expanded noise grid for precision privacy-utility trade-offs
for nm in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
    per_hospital = []
    for n in sizes:
        bs = min(BATCH, n)
        q = bs / n
        steps = ROUNDS * LOCAL_EPOCHS * max(1, n // bs)
        e, a = epsilon(q, nm, steps)
        per_hospital.append({"n": n, "q": round(q, 3), "steps": steps,
                             "epsilon": round(e, 2), "best_alpha": a})
    worst = max(h["epsilon"] for h in per_hospital)
    results[str(nm)] = {"worst_hospital_epsilon": worst, "hospitals": per_hospital}
    print(f"noise multiplier {nm}: epsilon <= {worst}  (delta {DELTA})")

json.dump(results, open("results/epsilon.json", "w"), indent=2)