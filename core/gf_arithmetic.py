"""
CryptoForge — GF(2^8) Arithmetic Foundation

Field: GF(2^8) with AES irreducible polynomial x^8 + x^4 + x^3 + x + 1
       (0x11b in hex, which is 283 in decimal)

This module provides:
  - Field initialization
  - Element creation, addition (XOR), multiplication, inversion
  - Vector and matrix operations over GF(2^8)
  - Utility functions for converting between bytes and field elements

All arithmetic delegates to the `galois` library for correctness and performance.
"""

import numpy as np
import galois

# ---------------------------------------------------------------------------
# Field definition
# ---------------------------------------------------------------------------
# AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1
# In galois library notation, pass the integer whose binary representation
# gives the polynomial coefficients: 1_0001_1011 = 0x11B = 283
# galois wants the polynomial as an integer; it auto-detects the degree.
# ---------------------------------------------------------------------------

GF256 = galois.GF(2**8, irreducible_poly=0x11B)

# ---------------------------------------------------------------------------
# Element helpers
# ---------------------------------------------------------------------------

def gf(val: int) -> galois.FieldArray:
    """Create a single GF(2^8) element from an integer 0-255."""
    return GF256(val)


def gf_array(vals) -> galois.FieldArray:
    """Create a GF(2^8) array (vector or matrix) from a list/nested-list of ints."""
    return GF256(np.array(vals, dtype=int))


def gf_random_vector(n: int, nonzero: bool = False) -> galois.FieldArray:
    """Random vector of n elements in GF(2^8).
    
    Args:
        n: length of vector
        nonzero: if True, all elements are guaranteed nonzero (1-255)
    """
    if nonzero:
        vals = np.random.randint(1, 256, size=n)
    else:
        vals = np.random.randint(0, 256, size=n)
    return GF256(vals)


def gf_random_matrix(rows: int, cols: int) -> galois.FieldArray:
    """Random rows×cols matrix over GF(2^8)."""
    vals = np.random.randint(0, 256, size=(rows, cols))
    return GF256(vals)


def gf_identity(n: int) -> galois.FieldArray:
    """n×n identity matrix over GF(2^8)."""
    return GF256(np.eye(n, dtype=int))


# ---------------------------------------------------------------------------
# Arithmetic wrappers (for clarity — galois already overloads +, *, /, etc.)
# ---------------------------------------------------------------------------

def gf_add(a, b):
    """Addition in GF(2^8) = XOR. Works on scalars, vectors, matrices."""
    return a + b  # galois overloads this as XOR


def gf_mul(a, b):
    """Multiplication in GF(2^8). Works on scalars, vectors, matrices.
    
    Auto-wraps raw Python ints to GF256 field elements to prevent
    silent incorrect results (ordinary integer multiplication instead
    of field multiplication).
    """
    if isinstance(a, (int, np.integer)) and not isinstance(a, galois.FieldArray):
        a = GF256(int(a))
    if isinstance(b, (int, np.integer)) and not isinstance(b, galois.FieldArray):
        b = GF256(int(b))
    return a * b


def gf_inv(a):
    """Multiplicative inverse in GF(2^8). Raises error on 0."""
    return GF256(1) / a


def gf_mat_vec(M, v):
    """Matrix-vector multiply over GF(2^8).
    
    M: (n, n) GF(2^8) matrix
    v: (n,) GF(2^8) vector
    Returns: (n,) GF(2^8) vector = M @ v
    """
    return M @ v


def gf_mat_mul(A, B):
    """Matrix-matrix multiply over GF(2^8)."""
    return A @ B


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def bytes_to_gf_vector(data: bytes) -> galois.FieldArray:
    """Convert a bytes object to a GF(2^8) vector."""
    return GF256(np.frombuffer(data, dtype=np.uint8).astype(int))


def gf_vector_to_bytes(vec) -> bytes:
    """Convert a GF(2^8) vector back to bytes."""
    return bytes(np.array(vec, dtype=np.uint8))


def int_to_bits(val: int, width: int = 8) -> list:
    """Convert integer to list of bits (MSB first)."""
    return [(val >> (width - 1 - i)) & 1 for i in range(width)]


def bits_to_int(bits: list) -> int:
    """Convert list of bits (MSB first) to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | b
    return val


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_field():
    """Run basic self-tests on GF(2^8) arithmetic. Returns True if all pass."""
    passed = True
    
    # Test 1: Addition is XOR
    a, b = gf(0x57), gf(0x83)
    result = gf_add(a, b)
    expected = gf(0x57 ^ 0x83)
    if result != expected:
        print(f"FAIL: {a} + {b} = {result}, expected {expected}")
        passed = False
    
    # Test 2: Multiplicative identity
    a = gf(0xAB)
    if gf_mul(a, gf(1)) != a:
        print(f"FAIL: {a} * 1 != {a}")
        passed = False
    
    # Test 3: Multiplicative inverse for all nonzero elements
    fail_count = 0
    for i in range(1, 256):
        a = gf(i)
        a_inv = gf_inv(a)
        if gf_mul(a, a_inv) != gf(1):
            fail_count += 1
    if fail_count > 0:
        print(f"FAIL: {fail_count}/255 elements have incorrect inverse")
        passed = False
    
    # Test 4: Self-inverse of addition (a + a = 0 in GF(2^k))
    a = gf(0x57)
    if gf_add(a, a) != gf(0):
        print(f"FAIL: {a} + {a} != 0")
        passed = False
    
    # Test 5: Matrix-vector multiply dimension check
    M = gf_random_matrix(4, 4)
    v = gf_random_vector(4)
    result = gf_mat_vec(M, v)
    if result.shape != (4,):
        print(f"FAIL: mat-vec result shape {result.shape}, expected (4,)")
        passed = False
    
    # Test 6: Known AES multiplication (0x57 * 0x83 = 0xC1 in AES field)
    a, b = gf(0x57), gf(0x83)
    product = gf_mul(a, b)
    if product != gf(0xC1):
        print(f"FAIL: 0x57 * 0x83 = {product}, expected 0xC1")
        passed = False
    
    return passed


if __name__ == "__main__":
    print("CryptoForge — GF(2^8) Arithmetic Verification")
    print("=" * 50)
    print(f"Field: GF(2^8) with irreducible poly: {GF256.irreducible_poly}")
    print(f"Field order: {GF256.order}")
    print()
    
    if verify_field():
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED — check output above")
    
    # Demo: show some operations
    print()
    print("--- Demo Operations ---")
    a, b = gf(0x57), gf(0x83)
    print(f"a = GF(0x57) = {a}")
    print(f"b = GF(0x83) = {b}")
    print(f"a + b = {gf_add(a, b)}  (XOR: 0x{0x57 ^ 0x83:02X})")
    print(f"a * b = {gf_mul(a, b)}")
    print(f"a^(-1) = {gf_inv(a)}")
    print(f"a * a^(-1) = {gf_mul(a, gf_inv(a))}")
    
    print()
    print("--- 4x4 Matrix-Vector Multiply ---")
    M = gf_array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])
    v = gf_array([1, 1, 1, 1])
    print(f"M =\n{M}")
    print(f"v = {v}")
    print(f"M @ v = {gf_mat_vec(M, v)}")