import os
import numpy as np
from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

for name, variant in [("ML-KEM-512", ML_KEM_512), ("ML-KEM-768", ML_KEM_768), ("ML-KEM-1024", ML_KEM_1024)]:
    print(f"\n--- Benchmarking {name} ---")
    ek, dk = variant.keygen()

    def session_key(shared):
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"fl-update-v1").derive(shared)

    update = np.random.default_rng(0).normal(size=14).astype(np.float64)
    shared_h, kem_ct = variant.encaps(ek)
    nonce = os.urandom(12)
    sealed = AESGCM(session_key(shared_h)).encrypt(nonce, update.tobytes(), b"hospital_0|round_1")

    shared_s = variant.decaps(dk, kem_ct)
    plain = AESGCM(session_key(shared_s)).decrypt(nonce, sealed, b"hospital_0|round_1")
    recovered = np.frombuffer(plain, dtype=np.float64)

    print(f"[{name}] Shared secrets match:", shared_h == shared_s)
    print(f"[{name}] Update recovered exactly:", np.array_equal(update, recovered))
    print(f"[{name}] KEM ciphertext bytes:", len(kem_ct), "| sealed update bytes:", len(sealed))