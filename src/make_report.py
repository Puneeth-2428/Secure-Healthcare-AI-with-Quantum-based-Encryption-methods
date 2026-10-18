import glob, json, os
import numpy as np

os.makedirs("results", exist_ok=True)

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

def row(name, r):
    if not r:
        return f"| {name} | n/a | n/a |"
    return f"| {name} | {r.get('accuracy', 0):.3f} | {r.get('auc', 0):.3f} |"

base, dp, fed = load("results/baseline.json"), load("results/dp.json"), load("results/federated.json")
sec, mem, pipe = load("results/federated_secure.json"), load("results/membership.json"), load("results/pipeline.json")

sweep = {}
for f in sorted(glob.glob("results/sweep/nm*_seed*.json")):
    r = json.load(open(f))
    sweep.setdefault(r["noise_multiplier"], []).append(r)

out = ["# Results Summary", "",
       "Test set: 60 held-out patients (one patient = 1.7 accuracy points).", "",
       "## Single Runs with Fixed Seed (Reference)", "",
       "| Configuration | Accuracy | AUC |", "|---|---|---|",
       row("Centralized baseline (no privacy)", base),
       row("Centralized DP-SGD (noise 1.0, clip 1.0)", dp and dp.get("1.0")),
       row("Federated, 3 hospitals (noise 0.0)", fed and fed.get("0.0")),
       row("Federated + DP-SGD (noise 1.0)", fed and fed.get("1.0")),
       row("Federated + DP-SGD + secret sharing (noise 1.0)", sec and sec.get("1.0")), ""]

if sweep:
    n = len(next(iter(sweep.values())))
    out += [f"## Full Secure Pipeline: DP + Secret Sharing + ML-KEM-768 ({n} seeds per setting)", "",
            "| Noise multiplier | Epsilon (delta 1e-5) | Accuracy (mean +/- std) | AUC (mean +/- std) |",
            "|---|---|---|---|"]
    for nm in sorted(sweep):
        a = np.array([r["accuracy"] for r in sweep[nm]])
        u = np.array([r["auc"] for r in sweep[nm]])
        out.append(f"| {nm} | {sweep[nm][0].get('epsilon', 'N/A')} | {a.mean():.3f} +/- {a.std(ddof=1):.3f} "
                   f"| {u.mean():.3f} +/- {u.std(ddof=1):.3f} |")
    worst = max((r.get("max_diff_vs_plain_average", 0.0) for v in sweep.values() for r in v), default=0.0)
    out += ["", "Seeds vary training randomness only; identical test set used throughout.",
            f"Secret-shared average vs plain average, largest difference across all runs: {worst:.2e}"]
if pipe:
    out.append(f"Transport per run: {pipe.get('bytes_transmitted', 0):,} bytes over {pipe.get('rounds', 0)} rounds, {pipe.get('seconds', 0)} s")
out.append("")

if mem:
    out += ["## Membership-Inference Attack (50 trials, 40 training patients)", "",
            "| Model | Attack AUC (0.5 = no leakage) | Std. error | Accuracy, unseen patients |",
            "|---|---|---|---|"]
    for k, v in mem.items():
        out.append(f"| {k} | {v['attack_auc']:.3f} | {v['std_error']:.3f} | {v['nonmember_accuracy']:.3f} |")
    out.append("")

open("results/RESULTS.md", "w").write("\n".join(out))
print("\n".join(out))