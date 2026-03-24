"""
CryptoForge — Matrix Construction for Linear Diffusion Layer

The mixing matrix M operates over GF(2^8)^n (n=32 for MatriXHash-256).
Each round computes: s ← M · s, where multiplication and addition are in GF(2^8).

This module provides three matrix families:

1. Cauchy MDS — Maximum Distance Separable. Every square submatrix is invertible.
   Branch number = n+1 (optimal). Parameterized by 2n field elements → O(n) search space.

2. Circulant — Each row is a cyclic shift of the first row.
   Parameterized by n elements. May or may not be MDS.

3. Random Invertible — Full-rank random matrix. No algebraic guarantees.
   O(n²) parameters. Baseline comparison only.

MDS property: A matrix M is MDS if and only if every square submatrix of M
is invertible (nonsingular). This is equivalent to the associated code having
maximum minimum distance, hence "Maximum Distance Separable."

For cryptographic diffusion, MDS matrices maximize the number of active S-boxes
in any differential or linear trail, making attacks exponentially harder.
"""

import numpy as np
from core.gf_arithmetic import GF256, gf, gf_array, gf_random_vector, gf_identity


# ---------------------------------------------------------------------------
# 1. Cauchy MDS Matrix
# ---------------------------------------------------------------------------

def build_cauchy_mds(n: int, x_start: int = 1, y_start: int = None) -> 'galois.FieldArray':
    """Construct an n×n Cauchy matrix over GF(2^8).
    
    A Cauchy matrix is defined as:
        M[i][j] = 1 / (x_i + y_j)
    
    where x = (x_0, ..., x_{n-1}) and y = (y_0, ..., y_{n-1}) are two sets
    of field elements such that x_i + y_j ≠ 0 for all i, j (i.e., no x_i equals
    any y_j, since in GF(2^k), a + b = 0 iff a = b).
    
    CRITICAL PROPERTY: Every square submatrix of a Cauchy matrix is invertible
    (over any field), which means the matrix is automatically MDS.
    
    For GF(2^8) with n ≤ 128, we can always find 2n distinct elements.
    For n=32, we need 64 distinct nonzero elements out of 255 available — easy.
    
    Args:
        n: matrix dimension (32 for MatriXHash-256)
        x_start: starting element for the x-set (default 1)
        y_start: starting element for the y-set (default n+1 to ensure disjointness)
    
    Returns:
        n×n GF(2^8) matrix with MDS property
    """
    if y_start is None:
        y_start = x_start + n
    
    # Need 2n distinct elements in GF(2^8)\{0}. Since |GF(2^8)*| = 255,
    # this works for n ≤ 127.
    if 2 * n > 255:
        raise ValueError(f"Cannot build {n}x{n} Cauchy MDS: need {2*n} distinct "
                         f"nonzero elements, only 255 available in GF(2^8)")
    
    # Generate x and y sets as consecutive field elements, ensuring disjointness
    # We pick x = [x_start, x_start+1, ..., x_start+n-1]
    #         y = [y_start, y_start+1, ..., y_start+n-1]
    # All values must be in [1, 255] and x ∩ y = ∅
    x_vals = list(range(x_start, x_start + n))
    y_vals = list(range(y_start, y_start + n))
    
    # Validate: all in range, all distinct, sets disjoint
    all_vals = x_vals + y_vals
    if any(v < 0 or v > 255 for v in all_vals):
        raise ValueError("Field elements must be in [0, 255]")
    if len(set(all_vals)) != 2 * n:
        raise ValueError("x and y sets must be mutually disjoint with all distinct elements")
    # Ensure x_i + y_j ≠ 0 (i.e., x_i ≠ y_j) — guaranteed by disjointness in GF(2^8)
    # since a + b = 0 iff a = b in characteristic 2.
    # Actually, in GF(2^8), a + b = a XOR b = 0 iff a = b.
    # Our x and y are disjoint integer sets, but we must check that no x_i equals y_j
    # as GF elements (they're just integers 0-255, so integer equality = field equality). ✓
    
    # Build the matrix
    x_gf = GF256(np.array(x_vals, dtype=int))
    y_gf = GF256(np.array(y_vals, dtype=int))
    
    M = GF256(np.zeros((n, n), dtype=int))
    for i in range(n):
        for j in range(n):
            # M[i][j] = 1 / (x_i + y_j) in GF(2^8)
            # Addition in GF(2^8) is XOR
            denom = x_gf[i] + y_gf[j]
            M[i, j] = GF256(1) / denom
    
    return M


def build_cauchy_mds_from_params(x_vals: list, y_vals: list) -> 'galois.FieldArray':
    """Build a Cauchy MDS matrix from explicit x and y parameter sets.
    
    This is the version the evolutionary search will use — it can optimize
    the specific field elements in x and y to find matrices with desired
    spectral properties while maintaining the MDS guarantee.
    
    Args:
        x_vals: list of n distinct GF(2^8) elements (ints 0-255)
        y_vals: list of n distinct GF(2^8) elements (ints 0-255), disjoint from x_vals
    
    Returns:
        n×n GF(2^8) Cauchy MDS matrix
    """
    n = len(x_vals)
    assert len(y_vals) == n, "x and y must have same length"
    assert len(set(x_vals) & set(y_vals)) == 0, "x and y must be disjoint"
    assert len(set(x_vals)) == n and len(set(y_vals)) == n, "elements must be distinct"
    
    x_gf = GF256(np.array(x_vals, dtype=int))
    y_gf = GF256(np.array(y_vals, dtype=int))
    
    M = GF256(np.zeros((n, n), dtype=int))
    for i in range(n):
        for j in range(n):
            M[i, j] = GF256(1) / (x_gf[i] + y_gf[j])
    
    return M


# ---------------------------------------------------------------------------
# 2. Circulant Matrix
# ---------------------------------------------------------------------------

def build_circulant(first_row: list) -> 'galois.FieldArray':
    """Construct a circulant matrix over GF(2^8).
    
    A circulant matrix is fully determined by its first row:
        M[i][j] = first_row[(j - i) mod n]
    
    Each row is a cyclic right-shift of the previous row.
    
    Properties:
    - Parameterized by n elements (vs. n² for general matrix)
    - Multiplication can be done in O(n log n) via NTT (not implemented here)
    - NOT guaranteed MDS — must verify separately
    - Used in some lightweight ciphers (e.g., related structures in PHOTON)
    
    Args:
        first_row: list of n GF(2^8) elements (ints 0-255)
    
    Returns:
        n×n GF(2^8) circulant matrix
    """
    n = len(first_row)
    row_gf = GF256(np.array(first_row, dtype=int))
    
    M = GF256(np.zeros((n, n), dtype=int))
    for i in range(n):
        for j in range(n):
            M[i, j] = row_gf[(j - i) % n]
    
    return M


# ---------------------------------------------------------------------------
# 3. Random Invertible Matrix
# ---------------------------------------------------------------------------

def build_random_invertible(n: int, seed: int = 42, max_attempts: int = 100) -> 'galois.FieldArray':
    """Construct a random invertible n×n matrix over GF(2^8).
    
    Strategy: generate random matrices until one is invertible.
    For GF(2^8), a random matrix is invertible with high probability
    (roughly ∏_{k=1}^{n} (1 - 2^{-k}) ≈ 0.29 for large n over GF(2),
    but much higher over GF(2^8) since the field is larger).
    
    Args:
        n: matrix dimension
        seed: random seed for reproducibility
        max_attempts: maximum tries before giving up
    
    Returns:
        n×n invertible GF(2^8) matrix
    """
    rng = np.random.RandomState(seed)
    
    for attempt in range(max_attempts):
        vals = rng.randint(0, 256, size=(n, n))
        M = GF256(vals)
        # Check invertibility by trying to compute the inverse
        try:
            M_inv = np.linalg.inv(M)
            # If we get here, it's invertible
            return M
        except (np.linalg.LinAlgError, Exception):
            continue
    
    raise RuntimeError(f"Failed to generate invertible {n}x{n} matrix "
                       f"after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# MDS Verification
# ---------------------------------------------------------------------------

def check_submatrix_invertible(M, rows: list, cols: list) -> bool:
    """Check if a specific square submatrix of M is invertible.
    
    Args:
        M: the full matrix (GF(2^8) array)
        rows: list of row indices
        cols: list of column indices (must be same length as rows)
    
    Returns:
        True if the submatrix M[rows, cols] is invertible
    """
    assert len(rows) == len(cols), "Must select square submatrix"
    sub = M[np.ix_(rows, cols)]
    try:
        _ = np.linalg.inv(sub)
        return True
    except (np.linalg.LinAlgError, Exception):
        return False


def verify_mds_exhaustive(M, max_size: int = None) -> bool:
    """Verify MDS property by checking ALL square submatrices.
    
    A matrix is MDS iff every square submatrix is invertible.
    
    For an n×n matrix, this requires checking C(n,k)² submatrices for each k=1..n.
    For n=32 this is astronomically expensive. Use verify_mds_sampled() instead.
    
    This function is only practical for small matrices (n ≤ 8).
    
    Args:
        M: n×n GF(2^8) matrix
        max_size: only check submatrices up to this size (default: all)
    """
    from itertools import combinations
    
    n = M.shape[0]
    if max_size is None:
        max_size = n
    
    for k in range(1, max_size + 1):
        for rows in combinations(range(n), k):
            for cols in combinations(range(n), k):
                if not check_submatrix_invertible(M, list(rows), list(cols)):
                    return False
    return True


def verify_mds_sampled(M, num_samples: int = 5000, seed: int = 42) -> dict:
    """Verify MDS property by random sampling of square submatrices.
    
    For large matrices (n=32), exhaustive checking is impossible.
    Instead, we randomly sample submatrices of various sizes and check invertibility.
    
    If ANY sampled submatrix is singular → NOT MDS (definitive).
    If ALL sampled submatrices are invertible → likely MDS (probabilistic).
    
    For Cauchy matrices, this should always pass (they're MDS by construction).
    For circulant/random matrices, failures indicate non-MDS.
    
    Args:
        M: n×n GF(2^8) matrix
        num_samples: total number of random submatrices to test
        seed: random seed
    
    Returns:
        dict with:
            'is_mds': bool (False if any failure found, True if all passed)
            'tested': total submatrices tested
            'failures': number of singular submatrices found
            'failure_sizes': list of sizes where failures occurred
    """
    n = M.shape[0]
    rng = np.random.RandomState(seed)
    
    failures = 0
    failure_sizes = []
    
    # Distribute samples across submatrix sizes
    # Test more small submatrices (they're cheaper) and some large ones
    sizes_to_test = []
    for k in range(1, n + 1):
        # More samples for small sizes, fewer for large
        count = max(1, num_samples // n)
        sizes_to_test.extend([k] * count)
    
    # Shuffle and truncate to num_samples
    rng.shuffle(sizes_to_test)
    sizes_to_test = sizes_to_test[:num_samples]
    
    for k in sizes_to_test:
        rows = sorted(rng.choice(n, size=k, replace=False).tolist())
        cols = sorted(rng.choice(n, size=k, replace=False).tolist())
        
        if not check_submatrix_invertible(M, rows, cols):
            failures += 1
            failure_sizes.append(k)
    
    return {
        'is_mds': failures == 0,
        'tested': len(sizes_to_test),
        'failures': failures,
        'failure_sizes': failure_sizes,
    }


# ---------------------------------------------------------------------------
# Branch Number Computation
# ---------------------------------------------------------------------------

def hamming_weight_gf(v) -> int:
    """Count the number of nonzero entries in a GF(2^8) vector.
    
    This is the "symbol weight" used in MDS/branch number calculations.
    NOT the bit-level Hamming weight — it counts nonzero bytes.
    """
    return int(np.count_nonzero(np.array(v, dtype=int)))


def differential_branch_number(M, exhaustive: bool = False, num_samples: int = 10000, seed: int = 42) -> int:
    """Compute (or estimate) the differential branch number of matrix M.
    
    The differential branch number is:
        B_d(M) = min over all nonzero x of: wt(x) + wt(M·x)
    
    where wt() counts nonzero GF(2^8) entries (symbol-level Hamming weight).
    
    For an n×n MDS matrix, B_d = n + 1 (the maximum possible).
    
    For n=32, exhaustive search over all 256^32 - 1 nonzero vectors is impossible.
    We estimate by random sampling (lower bound) or use the theoretical value for MDS.
    
    Args:
        M: n×n GF(2^8) matrix
        exhaustive: if True and n ≤ 4, check all nonzero vectors
        num_samples: number of random vectors to test (for estimation)
        seed: random seed
    
    Returns:
        Estimated (lower bound) or exact branch number
    """
    n = M.shape[0]
    
    if exhaustive and n <= 4:
        # Exhaustive: check all nonzero vectors
        min_bn = 2 * n  # upper bound
        for val in range(1, 256**n):
            # Convert integer to GF(2^8) vector
            x_ints = []
            v = val
            for _ in range(n):
                x_ints.append(v % 256)
                v //= 256
            x = GF256(np.array(x_ints, dtype=int))
            y = M @ x
            bn = hamming_weight_gf(x) + hamming_weight_gf(y)
            if bn < min_bn:
                min_bn = bn
        return min_bn
    
    # Sampling-based estimation (lower bound on the true minimum)
    rng = np.random.RandomState(seed)
    min_bn = 2 * n  # start with upper bound
    
    for _ in range(num_samples):
        # Generate random nonzero vector with varying weight
        # Include low-weight vectors (more likely to achieve low branch number)
        weight = rng.randint(1, n + 1)
        x_ints = np.zeros(n, dtype=int)
        positions = rng.choice(n, size=weight, replace=False)
        x_ints[positions] = rng.randint(1, 256, size=weight)
        
        x = GF256(x_ints)
        y = M @ x
        bn = hamming_weight_gf(x) + hamming_weight_gf(y)
        if bn < min_bn:
            min_bn = bn
    
    return min_bn


# ---------------------------------------------------------------------------
# Convenience: get matrix by type name
# ---------------------------------------------------------------------------

def build_matrix(n: int, matrix_type: str = 'cauchy_mds', seed: int = 42, **kwargs) -> 'galois.FieldArray':
    """Build a matrix by type name (for the config system).
    
    Args:
        n: matrix dimension
        matrix_type: one of 'cauchy_mds', 'circulant', 'random'
        seed: random seed (for circulant and random)
        **kwargs: additional parameters passed to the specific builder
    
    Returns:
        n×n GF(2^8) matrix
    """
    if matrix_type == 'cauchy_mds':
        return build_cauchy_mds(n, **kwargs)
    
    elif matrix_type == 'circulant':
        # Generate a random first row from seed
        rng = np.random.RandomState(seed)
        first_row = rng.randint(1, 256, size=n).tolist()  # nonzero entries
        return build_circulant(first_row)
    
    elif matrix_type == 'random':
        return build_random_invertible(n, seed=seed)
    
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}. "
                         f"Choose from: cauchy_mds, circulant, random")


# ---------------------------------------------------------------------------
# Verification / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CryptoForge — Matrix Builder Verification")
    print("=" * 55)
    
    # -------------------------------------------
    # Test 1: Small Cauchy MDS (4×4)
    # -------------------------------------------
    print("\n--- Test 1: 4×4 Cauchy MDS Matrix ---")
    M4 = build_cauchy_mds(4, x_start=1, y_start=5)
    print(f"Matrix:\n{M4}")
    
    # Exhaustive MDS check (feasible for small matrices with few submatrices)
    is_mds_4 = verify_mds_exhaustive(M4)
    print(f"MDS (exhaustive check of all submatrices): {is_mds_4}")
    
    # Branch number via sampling (exhaustive over 256^4 vectors is too slow)
    bn4 = differential_branch_number(M4, exhaustive=False, num_samples=20000)
    print(f"Differential branch number (sampled lower bound): {bn4}")
    print(f"Theoretical for 4×4 MDS: 5")
    
    # Invertibility check
    M4_inv = np.linalg.inv(M4)
    is_id = np.array_equal(M4 @ M4_inv, gf_identity(4))
    print(f"M × M^(-1) = I: {is_id}")
    
    # -------------------------------------------
    # Test 2: 32×32 Cauchy MDS (sampled)
    # -------------------------------------------
    print("\n--- Test 2: 32×32 Cauchy MDS Matrix ---")
    M32 = build_cauchy_mds(32)
    print(f"Shape: {M32.shape}")
    print(f"First row (first 8 entries): {M32[0, :8]}")
    
    result = verify_mds_sampled(M32, num_samples=3000)
    print(f"MDS (sampled, {result['tested']} submatrices): {result['is_mds']}")
    print(f"Failures: {result['failures']}")
    
    bn32 = differential_branch_number(M32, num_samples=5000)
    print(f"Differential branch number (sampled lower bound): {bn32}")
    print(f"Theoretical for 32×32 MDS: 33")
    
    # Invertibility check
    M32_inv = np.linalg.inv(M32)
    is_id_32 = np.array_equal(M32 @ M32_inv, gf_identity(32))
    print(f"32×32 Cauchy: M × M^(-1) = I: {is_id_32}")
    
    # -------------------------------------------
    # Test 3: Circulant matrix
    # -------------------------------------------
    print("\n--- Test 3: 4×4 Circulant Matrix ---")
    M_circ = build_circulant([2, 3, 1, 1])  # AES MixColumns-like
    print(f"Matrix:\n{M_circ}")
    
    is_mds_circ = verify_mds_exhaustive(M_circ)
    print(f"MDS (exhaustive): {is_mds_circ}")
    
    # -------------------------------------------
    # Test 4: Random invertible matrix
    # -------------------------------------------
    print("\n--- Test 4: 4×4 Random Invertible Matrix ---")
    M_rand = build_random_invertible(4, seed=42)
    print(f"Matrix:\n{M_rand}")
    
    M_rand_inv = np.linalg.inv(M_rand)
    is_id_rand = np.array_equal(M_rand @ M_rand_inv, gf_identity(4))
    print(f"M × M^(-1) = I: {is_id_rand}")
    
    is_mds_rand = verify_mds_exhaustive(M_rand)
    print(f"MDS (exhaustive): {is_mds_rand}")
    
    # -------------------------------------------
    # Summary
    # -------------------------------------------
    print("\n" + "=" * 55)
    print("SUMMARY")
    print(f"  4×4 Cauchy MDS:    MDS={is_mds_4},  BN≥{bn4} (theoretical: 5)")
    print(f"  32×32 Cauchy MDS:  MDS={result['is_mds']} (sampled), BN≥{bn32} (theoretical: 33)")
    print(f"  4×4 Circulant:     MDS={is_mds_circ}")
    print(f"  4×4 Random:        MDS={is_mds_rand}")