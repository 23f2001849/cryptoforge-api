"""
CryptoForge — S-box Implementation

The AES S-box: multiplicative inverse in GF(2^8) followed by an affine transform.

S(x) = A * x^{-1} + c   (over GF(2), operating on the bit vector)

where:
  - x^{-1} is the multiplicative inverse in GF(2^8), with 0 mapped to 0
  - A is a fixed 8x8 binary matrix (the AES affine matrix)
  - c = 0x63

This module provides:
  - AES S-box lookup table (precomputed)
  - Inverse S-box
  - Verification against known AES test vectors
  - Alternative S-box constructors (power map, identity) for comparison
"""

import numpy as np
from core.gf_arithmetic import GF256, gf, gf_inv


# ---------------------------------------------------------------------------
# AES Affine Transform (operates on bit vectors over GF(2))
# ---------------------------------------------------------------------------

def _affine_transform(byte_val: int) -> int:
    """Apply the AES affine transformation to an 8-bit value.
    
    The transform is: b_i' = b_i XOR b_{(i+4)%8} XOR b_{(i+5)%8} 
                              XOR b_{(i+6)%8} XOR b_{(i+7)%8} XOR c_i
    where c = 0x63 = 01100011 in binary.
    """
    c = 0x63
    result = 0
    for i in range(8):
        # Extract the relevant bits
        bit = (byte_val >> i) & 1
        bit ^= (byte_val >> ((i + 4) % 8)) & 1
        bit ^= (byte_val >> ((i + 5) % 8)) & 1
        bit ^= (byte_val >> ((i + 6) % 8)) & 1
        bit ^= (byte_val >> ((i + 7) % 8)) & 1
        bit ^= (c >> i) & 1
        result |= (bit << i)
    return result


def _build_aes_sbox() -> np.ndarray:
    """Construct the full 256-entry AES S-box.
    
    S(0) = affine(0)  (since 0 has no inverse, we map 0 → 0 before affine)
    S(x) = affine(x^{-1}) for x != 0
    """
    sbox = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        if x == 0:
            inv_val = 0
        else:
            inv_val = int(gf_inv(gf(x)))
        sbox[x] = _affine_transform(inv_val)
    return sbox


def _build_inverse_sbox(sbox: np.ndarray) -> np.ndarray:
    """Build the inverse S-box from a forward S-box.
    
    inv_sbox[sbox[x]] = x for all x.
    Only works if the S-box is a bijection (all 256 outputs distinct).
    """
    inv_sbox = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        inv_sbox[sbox[x]] = x
    return inv_sbox


# ---------------------------------------------------------------------------
# Precomputed AES S-box and its inverse
# ---------------------------------------------------------------------------

AES_SBOX = _build_aes_sbox()
AES_INV_SBOX = _build_inverse_sbox(AES_SBOX)


# ---------------------------------------------------------------------------
# Alternative S-boxes (for the parameterized search space)
# ---------------------------------------------------------------------------

def build_power_map_sbox(exponent: int) -> np.ndarray:
    """S-box defined as S(x) = x^d in GF(2^8).
    
    No affine transform. The exponent d controls algebraic degree
    and differential properties. Common choices:
      d=3   (cube map — low degree, weak)
      d=127 (inverse-like — high degree)
      d=254 (equivalent to x^{-1} since x^{254} = x^{-1} in GF(2^8))
    """
    sbox = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        sbox[x] = int(gf(x) ** exponent)
    return sbox


def build_identity_sbox() -> np.ndarray:
    """Identity S-box: S(x) = x. No nonlinearity. For testing only."""
    return np.arange(256, dtype=np.uint8)


def build_random_sbox(seed: int = 42) -> np.ndarray:
    """Random bijective S-box (random permutation of 0-255)."""
    rng = np.random.RandomState(seed)
    sbox = np.arange(256, dtype=np.uint8)
    rng.shuffle(sbox)
    return sbox


# ---------------------------------------------------------------------------
# S-box application
# ---------------------------------------------------------------------------

def apply_sbox(state: np.ndarray, sbox: np.ndarray = AES_SBOX) -> np.ndarray:
    """Apply S-box byte-wise to a state vector.
    
    Args:
        state: GF(2^8) array (any shape). Will be converted to int for lookup.
        sbox: 256-entry lookup table.
    
    Returns:
        New GF(2^8) array with S-box applied to each byte.
    """
    int_state = np.array(state, dtype=int)
    substituted = sbox[int_state]
    return GF256(substituted)


def apply_inv_sbox(state, inv_sbox: np.ndarray = AES_INV_SBOX):
    """Apply inverse S-box byte-wise."""
    int_state = np.array(state, dtype=int)
    substituted = inv_sbox[int_state]
    return GF256(substituted)


# ---------------------------------------------------------------------------
# S-box quality metrics
# ---------------------------------------------------------------------------

def is_bijection(sbox: np.ndarray) -> bool:
    """Check that the S-box is a bijection (permutation of 0-255)."""
    return len(set(sbox)) == 256


def differential_uniformity(sbox: np.ndarray) -> int:
    """Compute the differential uniformity δ of the S-box.
    
    δ = max over all nonzero Δ_in and all Δ_out of:
        |{ x : S(x ⊕ Δ_in) ⊕ S(x) = Δ_out }|
    
    AES S-box has δ = 4. Lower is better. Minimum possible for 8-bit bijection is 2 (APN).
    """
    max_count = 0
    for delta_in in range(1, 256):  # skip 0
        # Compute the difference distribution for this input difference
        diff_counts = np.zeros(256, dtype=int)
        for x in range(256):
            delta_out = sbox[x ^ delta_in] ^ sbox[x]
            diff_counts[delta_out] += 1
        current_max = diff_counts.max()
        if current_max > max_count:
            max_count = current_max
    return int(max_count)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

# Known AES S-box values for spot-checking
_AES_SBOX_TEST_VECTORS = {
    0x00: 0x63,
    0x01: 0x7C,
    0x02: 0x77,
    0x03: 0x7B,
    0x10: 0xCA,
    0x53: 0xED,
    0xFF: 0x16,
}


def verify_sbox():
    """Run self-tests on the AES S-box. Returns True if all pass."""
    passed = True
    
    # Test 1: Known values
    for inp, expected_out in _AES_SBOX_TEST_VECTORS.items():
        actual = AES_SBOX[inp]
        if actual != expected_out:
            print(f"FAIL: S(0x{inp:02X}) = 0x{actual:02X}, expected 0x{expected_out:02X}")
            passed = False
    
    # Test 2: Bijection
    if not is_bijection(AES_SBOX):
        print("FAIL: AES S-box is not a bijection")
        passed = False
    
    # Test 3: Inverse S-box correctness
    for x in range(256):
        if AES_INV_SBOX[AES_SBOX[x]] != x:
            print(f"FAIL: inv_sbox(sbox(0x{x:02X})) != 0x{x:02X}")
            passed = False
            break
    
    # Test 4: S(0) = 0x63 (affine transform of 0)
    if AES_SBOX[0] != 0x63:
        print(f"FAIL: S(0x00) = 0x{AES_SBOX[0]:02X}, expected 0x63")
        passed = False
    
    return passed


if __name__ == "__main__":
    print("CryptoForge — S-box Verification")
    print("=" * 50)
    
    if verify_sbox():
        print("ALL S-BOX TESTS PASSED")
    else:
        print("SOME S-BOX TESTS FAILED")
    
    print()
    print("--- AES S-box (first 2 rows) ---")
    for row in range(2):
        entries = [f"{AES_SBOX[row * 16 + col]:02X}" for col in range(16)]
        print(f"  {row:X}x: {' '.join(entries)}")
    
    print()
    print(f"Bijection: {is_bijection(AES_SBOX)}")
    
    # Differential uniformity is expensive (255 * 256 iterations) but fast for 8-bit
    print("Computing differential uniformity (takes a moment)...")
    du = differential_uniformity(AES_SBOX)
    print(f"Differential uniformity: {du}  (AES expected: 4)")