"""
CryptoForge Web Backend — FastAPI

Location: C:\\Users\\Krishna\\Downloads\\CryptoForge\\webapp\\backend\\main.py

This server imports YOUR REAL CryptoForge modules:
  - core.matrixhash.MatriXHash256     → real GF(2^8) matrix hash
  - core.sbox.AES_SBOX                → real AES S-box (constant, not a function)
  - core.sbox.build_power_map_sbox    → real power map builder
  - core.sbox.build_identity_sbox     → real identity builder
  - core.sbox.build_random_sbox       → real random builder
  - spectral.walsh_spectrum.spectral_fingerprint       → real Walsh fingerprint
  - spectral.walsh_spectrum.walsh_hadamard_spectrum     → real 256×256 Walsh transform

And loads YOUR REAL JSON result files from the project root.

Run locally:
  cd C:\\Users\\Krishna\\Downloads\\CryptoForge
  pip install fastapi uvicorn python-multipart
  python -m uvicorn webapp.backend.main:app --reload --port 8000

Test:
  http://localhost:8000/           → shows status + what modules loaded
  http://localhost:8000/api/health → quick health check
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# from matrixhash_c import MatriXHash256C

# ─────────────────────────────────────────────────────────────
# PATH SETUP
# main.py lives at CryptoForge/webapp/backend/main.py
# Project root is two levels up.
# ─────────────────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # CryptoForge/

# Override for Render deployment (where core/ is in the same dir as main.py)
CRYPTOFORGE_ROOT = Path(os.environ.get("CRYPTOFORGE_ROOT", str(_PROJECT_ROOT)))

if str(CRYPTOFORGE_ROOT) not in sys.path:
    sys.path.insert(0, str(CRYPTOFORGE_ROOT))

print(f"[CryptoForge API] Project root: {CRYPTOFORGE_ROOT}")
print(f"[CryptoForge API] core/ exists: {(CRYPTOFORGE_ROOT / 'core').exists()}")
print(f"[CryptoForge API] spectral/ exists: {(CRYPTOFORGE_ROOT / 'spectral').exists()}")

# ─────────────────────────────────────────────────────────────
# IMPORT REAL MODULES
# Exact signatures verified from past sessions:
#
#   MatriXHash256(config: dict = None, config_path: str = None)
#     .hash(message: bytes, num_rounds: int = None) -> bytes
#     .hexdigest(message: bytes, num_rounds: int = None) -> str
#     .describe() -> str
#
#   AES_SBOX: np.ndarray (256,) uint8 — a constant, not a function
#   build_power_map_sbox(d: int) -> np.ndarray
#   build_identity_sbox() -> np.ndarray
#   build_random_sbox(seed: int) -> np.ndarray
#
#   walsh_hadamard_spectrum(sbox: np.ndarray) -> np.ndarray (256,256) int32
#   spectral_fingerprint(sbox: np.ndarray, mixing_matrix=None) -> dict
#     Keys: nonlinearity, max_walsh_coefficient, spectral_flatness,
#           spectral_entropy, differential_uniformity, algebraic_degree,
#           differential_branch_number, linear_branch_number,
#           walsh_spectrum (256×256 ndarray), vector (ndarray)
# ─────────────────────────────────────────────────────────────
HASH_OK = False
SPECTRAL_OK = False

try:
    from core.matrixhash import MatriXHash256
    from core.sbox import (
        AES_SBOX,
        build_power_map_sbox,
        build_identity_sbox,
        build_random_sbox,
    )
    HASH_OK = True
    print("[CryptoForge API] ✓ core modules imported")
except ImportError as e:
    print(f"[CryptoForge API] ✗ core import failed: {e}")

try:
    from spectral.walsh_spectrum import (
        spectral_fingerprint,
        walsh_hadamard_spectrum,
    )
    SPECTRAL_OK = True
    print("[CryptoForge API] ✓ spectral modules imported")
except ImportError as e:
    print(f"[CryptoForge API] ✗ spectral import failed: {e}")

# Try fast C implementation
C_HASH_OK = False
try:
    from matrixhash_c import MatriXHash256C
    C_HASH_OK = True
    print("[CryptoForge API] ✓ C MatriXHash-256 loaded (fast mode)")
except (ImportError, FileNotFoundError, OSError) as e:
    print(f"[CryptoForge API] C library not available: {e}")


# ─────────────────────────────────────────────────────────────
# JSON LOADER — reads your 14+ verified result files
# ─────────────────────────────────────────────────────────────
def load_json(filename: str):
    """Load a JSON file from the project root. Returns None if not found."""
    fp = CRYPTOFORGE_ROOT / filename
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_spectral_fingerprints_file():
    """Load pre-computed spectral fingerprints from regression data dirs."""
    for name in [
        "regression_v2_data/spectral_fingerprints_v2.json",
        "regression_data/spectral_fingerprints.json",
    ]:
        data = load_json(name)
        if data:
            return data
    return {}


CACHED_FINGERPRINTS = load_spectral_fingerprints_file()
if CACHED_FINGERPRINTS:
    print(f"[CryptoForge API] ✓ Loaded {len(CACHED_FINGERPRINTS)} cached spectral fingerprints")

# List all JSON result files found
_JSON_FILES = sorted(f.name for f in CRYPTOFORGE_ROOT.glob("*.json"))
print(f"[CryptoForge API] Found {len(_JSON_FILES)} JSON result files")


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="CryptoForge API",
    description="Wraps real MatriXHash-256 + spectral analysis + verified results",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════
# ROOT + HEALTH
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name": "CryptoForge API",
        "version": "1.0.0",
        "project_root": str(CRYPTOFORGE_ROOT),
        "matrixhash_available": HASH_OK,
        "spectral_available": SPECTRAL_OK,
        "json_result_files": _JSON_FILES,
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "matrixhash": HASH_OK, "spectral": SPECTRAL_OK}


# ═══════════════════════════════════════════════════════════════
# POST /api/hash — REAL MatriXHash-256
# ═══════════════════════════════════════════════════════════════

@app.post("/api/hash")
async def hash_endpoint(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Hash text or file with the REAL MatriXHash-256 and SHA-256.

    MatriXHash-256 pipeline:
      core/gf_arithmetic.py  → GF(2^8) field with AES polynomial
      core/sbox.py           → AES S-box (NL=112, δ=4, deg=7)
      core/matrix_builders.py → 32×32 Cauchy MDS matrix (branch number 33)
      core/permutation.py     → ShiftRows 4×8
      core/matrixhash.py      → 8 rounds of INJECT→SUBSTITUTE→DIFFUSE→PERMUTE
                                 Merkle-Damgård + Davies-Meyer compression

    Returns both digests for side-by-side display.
    """
    if text is not None:
        data = text.encode("utf-8")
    elif file is not None:
        data = await file.read()
        if len(data) > 10_000_000:
            raise HTTPException(413, "File too large (max 10MB)")
    else:
        raise HTTPException(400, "Provide 'text' or 'file'")

    # SHA-256 (C implementation via hashlib — fast)
    sha256_hex = hashlib.sha256(data).hexdigest()

    # MatriXHash-256 (YOUR Python implementation — slow but real)
    mh_hex = None
    mh_time_ms = None
    if C_HASH_OK:
        try:
            t0 = time.perf_counter()
            mh_hex = MatriXHash256C().hexdigest(data)
            mh_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as e:
            print(f"[hash] C error: {e}")
    if mh_hex is None and HASH_OK:
        try:
            t0 = time.perf_counter()
            mh_hex = MatriXHash256().hexdigest(data)
            mh_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as e:
            print(f"[hash] Python error: {e}")

    return {
        "matrixhash256": mh_hex,
        "sha256": sha256_hex,
        "input_size_bytes": len(data),
        "matrixhash_time_ms": mh_time_ms,
        "matrixhash_native": mh_hex is not None,
    }


# ═══════════════════════════════════════════════════════════════
# POST /api/verify — Compare two files
# ═══════════════════════════════════════════════════════════════

@app.post("/api/verify")
async def verify_endpoint(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    """Hash two files with real MatriXHash-256 and compare."""
    data1 = await file1.read()
    data2 = await file2.read()

    if max(len(data1), len(data2)) > 10_000_000:
        raise HTTPException(413, "Files too large (max 10MB each)")

    sha1 = hashlib.sha256(data1).hexdigest()
    sha2 = hashlib.sha256(data2).hexdigest()

    mh1 = mh2 = None
    try:
        if C_HASH_OK:
            mh1 = MatriXHash256C().hexdigest(data1)
            mh2 = MatriXHash256C().hexdigest(data2)
        elif HASH_OK:
            mh1 = MatriXHash256().hexdigest(data1)
            mh2 = MatriXHash256().hexdigest(data2)
    except Exception as e:
        print(f"[verify] Error: {e}")

    return {
        "match_matrixhash": mh1 == mh2 if (mh1 and mh2) else None,
        "match_sha256": sha1 == sha2,
        "file1": {"name": file1.filename, "size": len(data1), "matrixhash256": mh1, "sha256": sha1},
        "file2": {"name": file2.filename, "size": len(data2), "matrixhash256": mh2, "sha256": sha2},
    }


# ═══════════════════════════════════════════════════════════════
# POST /api/spectral — REAL Walsh spectrum + fingerprint
# ═══════════════════════════════════════════════════════════════

def _build_sbox(name: str):
    """Build an S-box by name. Returns numpy uint8 array (256,)."""
    if name == "aes":
        return AES_SBOX                        # constant, already built
    elif name == "identity":
        return build_identity_sbox()
    elif name.startswith("power_d"):
        d = int(name.replace("power_d", ""))
        return build_power_map_sbox(d)
    elif name.startswith("random_s"):
        seed = int(name.replace("random_s", ""))
        return build_random_sbox(seed)
    else:
        return None


def _fingerprint_to_json(fp: dict) -> dict:
    """Convert spectral_fingerprint() output to JSON-safe dict.
    
    The raw output contains numpy arrays (walsh_spectrum, vector)
    which can't be serialized directly.
    """
    result = {}
    for key, val in fp.items():
        if key == "walsh_spectrum":
            # 256×256 int32 array — send as nested list for heatmap
            result["walsh_spectrum"] = val.tolist()
        elif key == "vector":
            result["vector"] = val.tolist()
        elif hasattr(val, "item"):
            # numpy scalar → Python scalar
            result[key] = val.item()
        else:
            result[key] = val
    return result


@app.post("/api/spectral")
async def spectral_endpoint(sbox_name: str = Form("aes")):
    """
    Compute Walsh spectrum + spectral fingerprint for a given S-box.

    Uses spectral/walsh_spectrum.py:
      spectral_fingerprint(sbox) → {nonlinearity, max_walsh_coefficient,
        spectral_flatness, spectral_entropy, differential_uniformity,
        algebraic_degree, walsh_spectrum (256×256), ...}
    
    Falls back to pre-computed data from spectral_fingerprints_v2.json.
    """
    # Try LIVE computation with your real spectral module
    if SPECTRAL_OK and HASH_OK:
        sbox = _build_sbox(sbox_name)
        if sbox is not None:
            try:
                t0 = time.perf_counter()
                fp = spectral_fingerprint(sbox)    # YOUR real function
                compute_ms = round((time.perf_counter() - t0) * 1000, 2)

                result = _fingerprint_to_json(fp)
                result["sbox"] = sbox_name
                result["live_computation"] = True
                result["compute_time_ms"] = compute_ms
                return result
            except Exception as e:
                print(f"[spectral] Live computation error: {e}")

    # Fall back to pre-computed fingerprints from your JSON files
    if sbox_name in CACHED_FINGERPRINTS:
        cached = CACHED_FINGERPRINTS[sbox_name]
        return {"sbox": sbox_name, "live_computation": False, **cached}

    raise HTTPException(404, f"S-box '{sbox_name}' not found and spectral module unavailable")


# ═══════════════════════════════════════════════════════════════
# GET endpoints — load YOUR REAL JSON result files
# ═══════════════════════════════════════════════════════════════

@app.get("/api/evolution")
async def evolution_endpoint():
    """Adversarial co-evolution results (v3) from evolution_adversarial_results.json."""
    data = load_json("evolution_adversarial_results.json")
    if data:
        return data
    raise HTTPException(404, "evolution_adversarial_results.json not found in project root")


@app.get("/api/regression")
async def regression_endpoint():
    """Spectral-neural regression v2 from spectral_neural_regression_v2.json."""
    data = load_json("spectral_neural_regression_v2.json")
    if data:
        return data
    raise HTTPException(404, "spectral_neural_regression_v2.json not found in project root")


@app.get("/api/cryptogenesis")
async def cryptogenesis_endpoint():
    """CryptoGenesis-lite results from cryptogenesis_results.json."""
    data = load_json("cryptogenesis_results.json")
    if data:
        return data
    raise HTTPException(404, "cryptogenesis_results.json not found in project root")


@app.get("/api/sweep")
async def sweep_endpoint():
    """Neural security margin sweep from sweep_results.json."""
    data = load_json("sweep_results.json")
    if data:
        return data
    raise HTTPException(404, "sweep_results.json not found")


@app.get("/api/nist")
async def nist_endpoint():
    """NIST SP 800-22 results from nist_results.json."""
    data = load_json("nist_results.json")
    if data:
        return data
    raise HTTPException(404, "nist_results.json not found")


@app.get("/api/statistical")
async def statistical_endpoint():
    """Full statistical test results from stat_results_full.json."""
    data = load_json("stat_results_full.json")
    if data:
        return data
    raise HTTPException(404, "stat_results_full.json not found")


@app.get("/api/robustness")
async def robustness_endpoint():
    """Architecture robustness results from robustness_results.json."""
    data = load_json("robustness_results.json")
    if data:
        return data
    raise HTTPException(404, "robustness_results.json not found")


@app.get("/api/sha256-benchmark")
async def sha256_benchmark_endpoint():
    """SHA-256 benchmark comparison from sha256_benchmark.json."""
    data = load_json("sha256_benchmark.json")
    if data:
        return data
    raise HTTPException(404, "sha256_benchmark.json not found")


@app.get("/api/rl-collision")
async def rl_collision_endpoint():
    """RL collision finder results from rl_collision_results.json."""
    data = load_json("rl_collision_results.json")
    if data:
        return data
    raise HTTPException(404, "rl_collision_results.json not found")


@app.get("/api/discrepancy")
async def discrepancy_endpoint():
    """Spectral-neural discrepancy results from discrepancy_results.json."""
    data = load_json("discrepancy_results.json")
    if data:
        return data
    raise HTTPException(404, "discrepancy_results.json not found")


@app.get("/api/pareto")
async def pareto_endpoint():
    """Pareto frontier results from pareto_results.json."""
    data = load_json("pareto_results.json")
    if data:
        return data
    raise HTTPException(404, "pareto_results.json not found")


@app.get("/api/all-results")
async def all_results_endpoint():
    """Load ALL JSON result files into a single response. For the frontend to grab everything at once."""
    results = {}
    for filename in _JSON_FILES:
        key = filename.replace(".json", "").replace("-", "_")
        results[key] = load_json(filename)
    return results


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)