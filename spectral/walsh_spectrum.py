"""
CryptoForge — Walsh-Hadamard Spectral Analysis Engine

The Walsh spectrum is the Fourier transform of a Boolean function over GF(2)^n.
For cryptographic S-boxes, it characterizes ALL linear approximation vulnerabilities
simultaneously. A flat Walsh spectrum means no linear shortcut exists — the S-box
resists linear cryptanalysis.

This module computes:
  1. Walsh spectrum of an 8-bit S-box (256×256 coefficient matrix)
  2. Spectral security fingerprint (8-dimensional summary vector)
  3. Differential spectrum (for differential cryptanalysis resistance)
  4. Algebraic degree of coordinate functions

The spectral fingerprint is the key innovation from the Spectral Forge concept:
it provides a cheap, continuous, fixed-dimensional security descriptor that
(we hypothesize) predicts neural cryptanalytic resistance.

Mathematical foundation:
  For S-box S: GF(2)^8 → GF(2)^8, the Walsh-Hadamard coefficient is:
    W_S(a, b) = Σ_{x=0}^{255} (-1)^{⟨b, S(x)⟩ ⊕ ⟨a, x⟩}
  
  where ⟨·,·⟩ is the inner product over GF(2) (parity of bitwise AND).
  
  This measures the correlation between the linear function ⟨a, x⟩ of the
  input and the linear function ⟨b, S(x)⟩ of the output. If W_S(a,b) is
  large, the linear approximation ⟨b, S(x)⟩ ≈ ⟨a, x⟩ holds with bias.
"""

import numpy as np
from core.sbox import AES_SBOX, differential_uniformity


# ---------------------------------------------------------------------------
# Core: Inner product over GF(2) / Parity
# ---------------------------------------------------------------------------

def _gf2_inner_product(a: int, x: int) -> int:
    """Compute ⟨a, x⟩ over GF(2) = parity of (a & x).
    
    Returns 0 or 1.
    """
    return bin(a & x).count('1') % 2


# Precompute parity lookup for 8-bit values (256 entries)
_PARITY_TABLE = np.array([bin(i).count('1') % 2 for i in range(256)], dtype=np.int8)


def _parity(x: int) -> int:
    """Fast parity via lookup table."""
    return _PARITY_TABLE[x & 0xFF]


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform of an S-box
# ---------------------------------------------------------------------------

def walsh_hadamard_spectrum(sbox: np.ndarray) -> np.ndarray:
    """Compute the full Walsh-Hadamard spectrum of an 8-bit S-box.
    
    For S: {0,...,255} → {0,...,255}, computes:
      W(a, b) = Σ_{x=0}^{255} (-1)^{⟨b, S(x)⟩ ⊕ ⟨a, x⟩}
    
    for all 256 values of a (input mask) and 256 values of b (output mask).
    
    Uses the Fast Walsh-Hadamard Transform (butterfly algorithm) for efficiency.
    Complexity: O(n · 2^n) per component function = O(256 × 8) = 2048 ops
    per output mask b, total O(256 × 2048) = O(524,288).
    
    Args:
        sbox: 256-element uint8 array (S-box lookup table)
    
    Returns:
        W: (256, 256) int32 array where W[a][b] is the Walsh coefficient
           for input mask a and output mask b.
    """
    n = 8  # bit width
    N = 256  # 2^n
    
    W = np.zeros((N, N), dtype=np.int32)
    
    for b in range(N):
        # For each output mask b, compute the component Boolean function:
        #   f_b(x) = ⟨b, S(x)⟩ = parity(b & S(x))
        # Then compute its Walsh-Hadamard transform.
        
        # Step 1: Build the ±1 truth table of f_b
        # f_values[x] = (-1)^{f_b(x)} = (-1)^{parity(b & S(x))}
        f_values = np.array([
            1 - 2 * _PARITY_TABLE[b & sbox[x]]
            for x in range(N)
        ], dtype=np.int32)
        
        # Step 2: Fast Walsh-Hadamard Transform (in-place butterfly)
        # After FWHT, f_values[a] = Σ_x f_values[x] · (-1)^{⟨a,x⟩}
        #                          = Σ_x (-1)^{f_b(x) ⊕ ⟨a,x⟩}
        #                          = W(a, b)
        h = 1
        while h < N:
            for i in range(0, N, h * 2):
                for j in range(i, i + h):
                    u = f_values[j]
                    v = f_values[j + h]
                    f_values[j] = u + v
                    f_values[j + h] = u - v
            h *= 2
        
        W[:, b] = f_values
    
    return W


# ---------------------------------------------------------------------------
# Spectral Security Fingerprint
# ---------------------------------------------------------------------------

def spectral_fingerprint(sbox: np.ndarray, mixing_matrix=None) -> dict:
    """Compute the spectral security fingerprint of an S-box.
    
    The fingerprint is an 8-dimensional vector summarizing the cryptographic
    quality of the S-box from its Walsh and differential spectra.
    
    Args:
        sbox: 256-element uint8 S-box
        mixing_matrix: optional GF(2^8) mixing matrix (for branch numbers)
    
    Returns:
        dict with all fingerprint components plus the raw Walsh spectrum
    """
    # Compute Walsh spectrum
    W = walsh_hadamard_spectrum(sbox)
    
    # ---------------------------------------------------------------
    # 1. Nonlinearity
    #    NL = (2^n - max_{b≠0} max_a |W(a,b)|) / 2
    #    Measures worst-case linear approximation bias.
    #    AES S-box: NL = 112. Maximum possible for 8-bit bijection: 120.
    # ---------------------------------------------------------------
    W_nonzero_b = W[:, 1:]  # exclude b=0 column
    max_walsh = int(np.abs(W_nonzero_b).max())
    nonlinearity = (256 - max_walsh) // 2
    
    # ---------------------------------------------------------------
    # 2. Maximum Walsh coefficient magnitude
    #    Lower is better. Directly measures worst linear approximation.
    # ---------------------------------------------------------------
    # (already computed as max_walsh)
    
    # ---------------------------------------------------------------
    # 3. Spectral flatness
    #    Ratio of geometric mean to arithmetic mean of |W(a,b)|²
    #    over nonzero (a,b). Value in [0, 1]. Higher = flatter = better.
    #    A perfectly flat spectrum (bent function) has flatness = 1.
    # ---------------------------------------------------------------
    # Use nonzero (a,b) pairs — exclude (0,0) which is always ±256
    W_sq = W_nonzero_b[1:, :].astype(np.float64) ** 2  # exclude a=0 too
    W_sq_flat = W_sq.flatten()
    # Replace zeros with small epsilon to avoid log(0)
    W_sq_nonzero = W_sq_flat[W_sq_flat > 0]
    if len(W_sq_nonzero) > 0:
        log_geo_mean = np.mean(np.log(W_sq_nonzero))
        geo_mean = np.exp(log_geo_mean)
        arith_mean = np.mean(W_sq_flat)
        spectral_flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0
    else:
        spectral_flatness = 0.0
    
    # ---------------------------------------------------------------
    # 4. Spectral entropy
    #    H = -Σ p(w) log₂ p(w) where p(w) is the normalized Walsh
    #    power distribution. Higher entropy = more uniform = better.
    # ---------------------------------------------------------------
    power_dist = W_sq_flat / W_sq_flat.sum() if W_sq_flat.sum() > 0 else W_sq_flat
    power_dist_nonzero = power_dist[power_dist > 0]
    spectral_entropy = float(-np.sum(power_dist_nonzero * np.log2(power_dist_nonzero)))
    
    # ---------------------------------------------------------------
    # 5. Differential uniformity
    #    δ = max_{Δ≠0, ∇} |{x : S(x⊕Δ) ⊕ S(x) = ∇}|
    #    AES: δ=4. Lower is better. Minimum for 8-bit bijection: 2 (APN).
    # ---------------------------------------------------------------
    diff_uniformity = differential_uniformity(sbox)
    
    # ---------------------------------------------------------------
    # 6. Algebraic degree
    #    Degree of the algebraic normal form (ANF) of the S-box's
    #    coordinate functions. Higher = harder to attack algebraically.
    #    AES: degree 7 (maximum for 8-bit).
    # ---------------------------------------------------------------
    alg_degree = _algebraic_degree(sbox)
    
    # ---------------------------------------------------------------
    # 7 & 8. Branch numbers (require mixing matrix)
    # ---------------------------------------------------------------
    diff_branch_number = None
    linear_branch_number = None
    
    if mixing_matrix is not None:
        from core.matrix_builders import differential_branch_number
        diff_branch_number = differential_branch_number(mixing_matrix, num_samples=5000)
        # Linear branch number = differential branch number of M^T
        M_T = np.array(mixing_matrix, dtype=int)  # transpose
        from core.gf_arithmetic import GF256
        M_transpose = GF256(M_T.T)
        linear_branch_number = differential_branch_number(M_transpose, num_samples=5000)
    
    # ---------------------------------------------------------------
    # Assemble fingerprint
    # ---------------------------------------------------------------
    fingerprint = {
        # Core spectral properties
        'nonlinearity': nonlinearity,
        'max_walsh_coefficient': max_walsh,
        'spectral_flatness': spectral_flatness,
        'spectral_entropy': spectral_entropy,
        
        # Differential properties
        'differential_uniformity': diff_uniformity,
        
        # Algebraic properties
        'algebraic_degree': alg_degree,
        
        # Branch numbers (if matrix provided)
        'differential_branch_number': diff_branch_number,
        'linear_branch_number': linear_branch_number,
        
        # Raw data for visualization
        'walsh_spectrum': W,
        
        # Summary vector (for regression / Bayesian optimization)
        'vector': _fingerprint_to_vector(
            nonlinearity, max_walsh, spectral_flatness, spectral_entropy,
            diff_uniformity, alg_degree, diff_branch_number, linear_branch_number
        ),
    }
    
    return fingerprint


def _fingerprint_to_vector(nonlinearity, max_walsh, flatness, entropy,
                            diff_unif, alg_deg, diff_bn, lin_bn) -> np.ndarray:
    """Convert fingerprint components to a fixed-size numeric vector.
    
    This vector is what the Bayesian optimizer and regression model use.
    All values normalized to roughly [0, 1] range.
    """
    vec = [
        nonlinearity / 120.0,          # max possible NL for 8-bit
        1.0 - max_walsh / 256.0,       # inverted: lower max_walsh = better
        flatness,                       # already in [0, 1]
        entropy / 16.0,                 # rough normalization
        1.0 - diff_unif / 256.0,       # inverted: lower δ = better
        alg_deg / 7.0,                 # max degree for 8-bit
    ]
    
    if diff_bn is not None:
        vec.append(diff_bn / 33.0)     # max for 32×32 MDS
    if lin_bn is not None:
        vec.append(lin_bn / 33.0)
    
    return np.array(vec, dtype=np.float64)


# ---------------------------------------------------------------------------
# Algebraic Degree via Möbius Transform
# ---------------------------------------------------------------------------

def _algebraic_degree(sbox: np.ndarray) -> int:
    """Compute the algebraic degree of an S-box.
    
    The algebraic degree is the maximum degree of the algebraic normal form
    (ANF) across all coordinate (output bit) functions.
    
    For an 8-bit S-box, the ANF of each coordinate function f_i(x) is:
      f_i(x) = ⊕_{u ∈ {0,1}^8} a_u · x^u
    
    where x^u = ∏_j x_j^{u_j} and a_u ∈ {0,1} are the ANF coefficients.
    The degree of f_i is max{wt(u) : a_u = 1}.
    
    The Möbius transform computes ANF coefficients from the truth table.
    """
    max_degree = 0
    
    for bit in range(8):
        # Extract coordinate function: f(x) = bit `bit` of S(x)
        truth_table = np.array([(sbox[x] >> bit) & 1 for x in range(256)], dtype=np.int32)
        
        # Möbius transform (in-place, over GF(2))
        anf = truth_table.copy()
        for i in range(8):
            step = 1 << i
            for j in range(256):
                if j & step:
                    anf[j] ^= anf[j ^ step]
        
        # Find maximum Hamming weight among nonzero ANF coefficients
        for u in range(256):
            if anf[u] & 1:  # coefficient is 1
                deg = bin(u).count('1')
                if deg > max_degree:
                    max_degree = deg
    
    return max_degree


# ---------------------------------------------------------------------------
# Differential Spectrum
# ---------------------------------------------------------------------------

def differential_spectrum(sbox: np.ndarray) -> np.ndarray:
    """Compute the full differential distribution table (DDT) of an S-box.
    
    DDT[Δ_in][Δ_out] = |{x : S(x ⊕ Δ_in) ⊕ S(x) = Δ_out}|
    
    For a random permutation, most entries are 0 or 2.
    For AES, max entry is 4 (differential uniformity).
    
    Returns:
        DDT: (256, 256) int32 array
    """
    N = 256
    DDT = np.zeros((N, N), dtype=np.int32)
    
    for delta_in in range(N):
        for x in range(N):
            delta_out = sbox[x ^ delta_in] ^ sbox[x]
            DDT[delta_in, delta_out] += 1
    
    return DDT


# ---------------------------------------------------------------------------
# Comparison across S-boxes
# ---------------------------------------------------------------------------

def compare_sboxes(sboxes: dict, mixing_matrix=None) -> dict:
    """Compute and compare spectral fingerprints for multiple S-boxes.
    
    Args:
        sboxes: dict mapping name → 256-element sbox array
        mixing_matrix: optional mixing matrix for branch numbers
    
    Returns:
        dict mapping name → fingerprint (without raw Walsh spectrum for compactness)
    """
    results = {}
    
    for name, sbox in sboxes.items():
        print(f"  Analyzing {name}...")
        fp = spectral_fingerprint(sbox, mixing_matrix)
        
        # Store without large arrays
        results[name] = {
            'nonlinearity': fp['nonlinearity'],
            'max_walsh_coefficient': fp['max_walsh_coefficient'],
            'spectral_flatness': fp['spectral_flatness'],
            'spectral_entropy': fp['spectral_entropy'],
            'differential_uniformity': fp['differential_uniformity'],
            'algebraic_degree': fp['algebraic_degree'],
            'differential_branch_number': fp['differential_branch_number'],
            'linear_branch_number': fp['linear_branch_number'],
            'vector': fp['vector'].tolist(),
        }
    
    return results


# ---------------------------------------------------------------------------
# Verification / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    from core.sbox import (AES_SBOX, build_power_map_sbox,
                           build_random_sbox, build_identity_sbox)
    from core.matrix_builders import build_cauchy_mds
    
    print("CryptoForge — Spectral Analysis Engine Verification")
    print("=" * 60)
    
    # -------------------------------------------
    # Test 1: Walsh spectrum of AES S-box
    # -------------------------------------------
    print("\n[1] Walsh-Hadamard Transform of AES S-box...")
    t0 = time.time()
    W = walsh_hadamard_spectrum(AES_SBOX)
    elapsed = time.time() - t0
    
    print(f"  Spectrum shape: {W.shape}  (expected: (256, 256))")
    print(f"  Computation time: {elapsed:.2f}s")
    print(f"  W(0,0) = {W[0,0]}  (expected: 256, since Σ_x (-1)^0 = 256)")
    print(f"  Max |W(a,b)| for b≠0: {np.abs(W[:, 1:]).max()}")
    print(f"  Expected for AES: 32 (giving NL = (256-32)/2 = 112)")
    
    assert W[0, 0] == 256, f"W(0,0) should be 256, got {W[0,0]}"
    assert np.abs(W[:, 1:]).max() == 32, f"Max Walsh should be 32 for AES S-box"
    
    # Parseval's theorem check: Σ_a W(a,b)² = 256² for each b
    for b in range(256):
        sum_sq = np.sum(W[:, b].astype(np.int64) ** 2)
        assert sum_sq == 256 * 256, f"Parseval failed for b={b}: sum={sum_sq}"
    print(f"  Parseval's theorem: VERIFIED for all 256 output masks")
    
    # -------------------------------------------
    # Test 2: Full spectral fingerprint of AES S-box
    # -------------------------------------------
    print("\n[2] Spectral fingerprint of AES S-box...")
    fp = spectral_fingerprint(AES_SBOX)
    
    print(f"  Nonlinearity:           {fp['nonlinearity']}  (expected: 112)")
    print(f"  Max Walsh coefficient:  {fp['max_walsh_coefficient']}  (expected: 32)")
    print(f"  Spectral flatness:      {fp['spectral_flatness']:.4f}")
    print(f"  Spectral entropy:       {fp['spectral_entropy']:.4f} bits")
    print(f"  Differential uniformity: {fp['differential_uniformity']}  (expected: 4)")
    print(f"  Algebraic degree:       {fp['algebraic_degree']}  (expected: 7)")
    print(f"  Fingerprint vector:     {fp['vector']}")
    
    assert fp['nonlinearity'] == 112, f"AES NL should be 112, got {fp['nonlinearity']}"
    assert fp['differential_uniformity'] == 4, f"AES δ should be 4"
    assert fp['algebraic_degree'] == 7, f"AES degree should be 7"
    
    # -------------------------------------------
    # Test 3: Compare multiple S-boxes
    # -------------------------------------------
    print("\n[3] Comparing S-box families...")
    
    sboxes = {
        'AES (x^{-1} + affine)': AES_SBOX,
        'Power map x^3 (cube)': build_power_map_sbox(3),
        'Power map x^127': build_power_map_sbox(127),
        'Power map x^254 (≈inverse)': build_power_map_sbox(254),
        'Random permutation': build_random_sbox(seed=42),
        'Identity (NO security)': build_identity_sbox(),
    }
    
    results = compare_sboxes(sboxes)
    
    # Print comparison table
    print(f"\n  {'S-box':<28s} {'NL':>4s} {'MaxW':>5s} {'Flat':>6s} "
          f"{'Entropy':>8s} {'δ':>3s} {'Deg':>4s}")
    print(f"  {'─'*28} {'─'*4} {'─'*5} {'─'*6} {'─'*8} {'─'*3} {'─'*4}")
    
    for name, r in results.items():
        print(f"  {name:<28s} {r['nonlinearity']:4d} {r['max_walsh_coefficient']:5d} "
              f"{r['spectral_flatness']:6.3f} {r['spectral_entropy']:8.2f} "
              f"{r['differential_uniformity']:3d} {r['algebraic_degree']:4d}")
    
    # -------------------------------------------
    # Test 4: Fingerprint with mixing matrix
    # -------------------------------------------
    print("\n[4] Fingerprint with Cauchy MDS matrix (branch numbers)...")
    M = build_cauchy_mds(32)
    fp_with_matrix = spectral_fingerprint(AES_SBOX, mixing_matrix=M)
    
    print(f"  Differential branch number: {fp_with_matrix['differential_branch_number']}")
    print(f"  Linear branch number:       {fp_with_matrix['linear_branch_number']}")
    print(f"  Full fingerprint vector ({len(fp_with_matrix['vector'])}D): "
          f"{np.round(fp_with_matrix['vector'], 3)}")
    
    # -------------------------------------------
    # Test 5: Differential Distribution Table
    # -------------------------------------------
    print("\n[5] Differential Distribution Table of AES S-box...")
    DDT = differential_spectrum(AES_SBOX)
    print(f"  DDT shape: {DDT.shape}")
    print(f"  DDT[0][0] = {DDT[0,0]}  (expected: 256, trivial case)")
    print(f"  Max DDT entry (Δ≠0): {DDT[1:, :].max()}  (expected: 4)")
    print(f"  DDT value distribution (Δ≠0):")
    unique, counts = np.unique(DDT[1:, :], return_counts=True)
    for val, cnt in zip(unique, counts):
        print(f"    DDT value {val:3d}: appears {cnt:6d} times")
    
    print(f"\n{'='*60}")
    print("ALL SPECTRAL ANALYSIS TESTS PASSED")
    print("=" * 60)