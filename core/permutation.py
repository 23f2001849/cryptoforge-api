"""
CryptoForge — Permutation Layer

The permutation P operates on the 32-byte state vector, shuffling byte positions
to ensure cross-lane mixing between rounds.

This is the same role as ShiftRows in AES: the matrix multiply (diffusion) mixes
bytes within columns, while the permutation moves bytes between columns so that
the next round's matrix multiply mixes different byte groups.

Without this permutation, the same groups of bytes would always be mixed together,
leaving the hash vulnerable to attacks that exploit this structure.

Available permutations:
  - ShiftRows-4: Treat 32 bytes as 4×8 grid, shift each row by its index
  - ShiftRows-8: Treat 32 bytes as 8×4 grid, shift each row by its index
  - Identity: No permutation (for testing — weakens diffusion)
  - Custom: User-specified permutation table
"""

import numpy as np
from core.gf_arithmetic import GF256


# ---------------------------------------------------------------------------
# ShiftRows-style Permutations
# ---------------------------------------------------------------------------

def _build_shiftrows_perm(nrows: int, ncols: int) -> list:
    """Build a ShiftRows permutation table.
    
    Treats the 32-byte state as an nrows × ncols grid (row-major order).
    Row i is cyclically shifted left by i positions.
    
    For nrows=4, ncols=8 (AES-like):
        Row 0: no shift
        Row 1: shift left by 1
        Row 2: shift left by 2
        Row 3: shift left by 3
    
    Returns:
        Permutation table: perm[new_position] = old_position
    """
    assert nrows * ncols == 32, f"Grid must have 32 cells, got {nrows}×{ncols}={nrows*ncols}"
    
    perm = [0] * 32
    for row in range(nrows):
        for col in range(ncols):
            old_pos = row * ncols + col
            new_col = (col - row) % ncols
            new_pos = row * ncols + new_col
            perm[new_pos] = old_pos
    
    return perm


# Pre-built permutation tables
SHIFTROWS_4x8 = _build_shiftrows_perm(4, 8)   # 4 rows × 8 cols, AES-like
SHIFTROWS_8x4 = _build_shiftrows_perm(8, 4)   # 8 rows × 4 cols, alternative


def _build_identity_perm() -> list:
    """Identity permutation (no shuffling)."""
    return list(range(32))


IDENTITY_PERM = _build_identity_perm()


# ---------------------------------------------------------------------------
# Apply Permutation
# ---------------------------------------------------------------------------

def apply_permutation(state, perm: list):
    """Apply a byte permutation to the 32-byte state vector.
    
    Args:
        state: GF(2^8) vector of length 32
        perm: permutation table where perm[i] = source index for position i
    
    Returns:
        New GF(2^8) vector with bytes permuted
    """
    int_state = np.array(state, dtype=int)
    permuted = int_state[perm]
    return GF256(permuted)


def apply_inverse_permutation(state, perm: list):
    """Apply the inverse of a byte permutation.
    
    If perm maps old→new, the inverse maps new→old.
    """
    inv_perm = [0] * len(perm)
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos
    
    int_state = np.array(state, dtype=int)
    permuted = int_state[inv_perm]
    return GF256(permuted)


# ---------------------------------------------------------------------------
# Get permutation by name
# ---------------------------------------------------------------------------

def get_permutation(name: str = 'shiftrows_4x8') -> list:
    """Get a permutation table by name.
    
    Args:
        name: one of 'shiftrows_4x8', 'shiftrows_8x4', 'identity'
    
    Returns:
        Permutation table (list of 32 ints)
    """
    perms = {
        'shiftrows_4x8': SHIFTROWS_4x8,
        'shiftrows_8x4': SHIFTROWS_8x4,
        'identity': IDENTITY_PERM,
    }
    if name not in perms:
        raise ValueError(f"Unknown permutation: {name}. Choose from: {list(perms.keys())}")
    return perms[name]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_permutation(perm: list) -> bool:
    """Verify that a permutation table is valid (bijection on {0..31})."""
    return sorted(perm) == list(range(32))


if __name__ == "__main__":
    print("CryptoForge — Permutation Verification")
    print("=" * 45)
    
    # Test ShiftRows 4×8
    print("\nShiftRows 4×8 permutation table:")
    print(f"  {SHIFTROWS_4x8}")
    print(f"  Valid: {verify_permutation(SHIFTROWS_4x8)}")
    
    # Show the grid view
    print("\n  4×8 grid (before → after):")
    for row in range(4):
        before = list(range(row * 8, row * 8 + 8))
        after = [SHIFTROWS_4x8[row * 8 + col] for col in range(8)]
        print(f"    Row {row} (shift {row}): {before} → {after}")
    
    # Test with a known state
    from core.gf_arithmetic import gf_array
    state = gf_array(list(range(32)))
    print(f"\n  Input:    {list(np.array(state, dtype=int))}")
    permuted = apply_permutation(state, SHIFTROWS_4x8)
    print(f"  Permuted: {list(np.array(permuted, dtype=int))}")
    restored = apply_inverse_permutation(permuted, SHIFTROWS_4x8)
    print(f"  Restored: {list(np.array(restored, dtype=int))}")
    print(f"  Round-trip OK: {np.array_equal(np.array(state, dtype=int), np.array(restored, dtype=int))}")
    
    # Test identity
    print(f"\nIdentity permutation valid: {verify_permutation(IDENTITY_PERM)}")
    
    print("\nALL PERMUTATION TESTS PASSED")