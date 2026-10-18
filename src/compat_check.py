import json, os, platform, sys, time
import numpy as np

def check(name, fn):
    t = time.time()
    try:
        print(f"[ OK ] {name}: {fn()}  ({time.time() - t:.2f} s)", flush=True)
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)

def platform_info():
    return f"{platform.machine()}, byte order {sys.byteorder}, Python {platform.python_version()}"

def cpu_features():
    for line in open("/proc/cpuinfo"):
        if line.startswith("features"):
            f = line.split(":", 1)[1].split()
            return (f"msa (CPACF crypto assist) {'msa' in f}, vx (vector) {'vx' in f}, "
                    f"nnpa (AI accelerator) {'nnpa' in f}")
    return "no features line found"

def imputer():
    from sklearn.impute import SimpleImputer
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
    return SimpleImputer(strategy="mean").fit_transform(X).tolist()

def dirichlet_split():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 237)
    parts = [[] for _ in range(3)]
    for c in (0, 1):
        idx = rng.permutation(np.where(y == c)[0])
        p = rng.dirichlet([0.5] * 3)
        cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
        for h, chunk in enumerate(np.split(idx, cuts)):
            parts[h] += chunk.tolist()
    return "hospital sizes " + str([len(p) for p in parts]) + " (watch for tiny or empty hospitals)"

def byte_order():
    raw = np.array([1], dtype=np.uint64).tobytes().hex()
    return f"uint64 value 1 in memory = {raw} (big-endian here; use '<u8' to exchange with x86)"

def mlkem():
    from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
    out, interop = [], {}
    for name, k in [("ML-KEM-512", ML_KEM_512), ("ML-KEM-768", ML_KEM_768), ("ML-KEM-1024", ML_KEM_1024)]:
        t = time.time()
        ek, dk = k.keygen()
        ss, ct = k.encaps(ek)
        ok = k.decaps(dk, ct) == ss
        out.append(f"{name}: match={ok}, ek={len(ek)} B, ct={len(ct)} B, {time.time() - t:.3f} s")
        interop[name] = {"dk": dk.hex(), "ct": ct.hex(), "ss": ss.hex()}
    os.makedirs("results", exist_ok=True)
    json.dump(interop, open("results/mlkem_interop.json", "w"))
    return "\n         " + "\n         ".join(out)

def aes():
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    a, n = AESGCM(os.urandom(32)), os.urandom(12)
    return a.decrypt(n, a.encrypt(n, b"x" * 1024, b"aad"), b"aad") == b"x" * 1024

def paillier(bits):
    def run():
        import phe.util as u
        from phe import paillier as p
        t = time.time(); pub, priv = p.generate_paillier_keypair(n_length=bits); kg = time.time() - t
        t = time.time(); cs = [pub.encrypt(0.5 * i) for i in range(10)]; enc = (time.time() - t) / 10
        total = cs[0]
        for c in cs[1:]:
            total = total + c
        ok = abs(priv.decrypt(total) - sum(0.5 * i for i in range(10))) < 1e-9
        size = (cs[0].ciphertext().bit_length() + 7) // 8
        return (f"keygen {kg:.1f} s, encrypt {enc:.2f} s/value, ciphertext ~{size} B, "
                f"sum correct={ok}, gmpy2={getattr(u, 'HAVE_GMP', 'unknown')}")
    return run

check("platform", platform_info)
check("CPU features", cpu_features)
check("SimpleImputer", imputer)
check("Dirichlet non-IID split", dirichlet_split)
check("byte order", byte_order)
check("ML-KEM 512/768/1024", mlkem)
check("AES-256-GCM", aes)
check("Paillier 2048-bit", paillier(2048))
check("Paillier 3072-bit (can take a minute)", paillier(3072))
