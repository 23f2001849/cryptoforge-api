"""
CryptoForge — MatriXHash-256 C Wrapper

Loads the compiled C shared library and exposes the same interface
as core/matrixhash.py, but 100-1000x faster.

Usage:
    from matrixhash_c import MatriXHash256C

    hasher = MatriXHash256C()
    digest = hasher.hash(b"Hello, CryptoForge!")
    hex_str = hasher.hexdigest(b"Hello, CryptoForge!")

Compile the C library first:
    Windows (MinGW):  gcc -O2 -shared -o matrixhash.dll matrixhash.c
    Windows (MSVC):   cl /O2 /LD matrixhash.c /Fe:matrixhash.dll
    Linux:            gcc -O2 -shared -fPIC -o matrixhash.so matrixhash.c
    macOS:            gcc -O2 -shared -o matrixhash.dylib matrixhash.c
"""

import ctypes
import os
import platform
import sys
from pathlib import Path


def _find_library():
    """Find the compiled MatriXHash library."""
    base = Path(__file__).parent

    if platform.system() == "Windows":
        names = ["matrixhash.dll"]
    elif platform.system() == "Darwin":
        names = ["matrixhash.dylib", "matrixhash.so"]
    else:
        names = ["matrixhash.so"]

    for name in names:
        path = base / name
        if path.exists():
            return str(path)

    # Also check current working directory
    for name in names:
        if os.path.exists(name):
            return os.path.abspath(name)

    return None


def _load_library():
    """Load the C library and set up function signatures."""
    lib_path = _find_library()
    if lib_path is None:
        raise FileNotFoundError(
            "Compiled MatriXHash library not found. Compile it first:\n"
            "  Windows: gcc -O2 -shared -o matrixhash.dll matrixhash.c\n"
            "  Linux:   gcc -O2 -shared -fPIC -o matrixhash.so matrixhash.c"
        )

    lib = ctypes.CDLL(lib_path)

    # void matrixhash256(const uint8_t *msg, size_t msg_len, uint8_t digest[32])
    lib.matrixhash256.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # msg
        ctypes.c_size_t,                  # msg_len
        ctypes.POINTER(ctypes.c_uint8),  # digest
    ]
    lib.matrixhash256.restype = None

    # void matrixhash256_hex(const uint8_t *msg, size_t msg_len, char hex_out[65])
    lib.matrixhash256_hex.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # msg
        ctypes.c_size_t,                  # msg_len
        ctypes.c_char_p,                  # hex_out
    ]
    lib.matrixhash256_hex.restype = None

    # void get_aes_sbox(uint8_t out[256])
    lib.get_aes_sbox.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
    lib.get_aes_sbox.restype = None

    return lib


# Global library handle (loaded once)
_lib = None


def get_lib():
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


class MatriXHash256C:
    """Fast C implementation of MatriXHash-256.

    Drop-in replacement for MatriXHash256 from core/matrixhash.py.
    Uses the same algorithm — GF(2^8) Cauchy MDS, AES S-box, 8 rounds,
    Davies-Meyer compression — but runs 100-1000x faster.
    """

    def __init__(self):
        self._lib = get_lib()

    def hash(self, message: bytes) -> bytes:
        """Compute 32-byte MatriXHash-256 digest."""
        msg_len = len(message)
        msg_buf = (ctypes.c_uint8 * msg_len)(*message)
        digest_buf = (ctypes.c_uint8 * 32)()
        self._lib.matrixhash256(msg_buf, msg_len, digest_buf)
        return bytes(digest_buf)

    def hexdigest(self, message: bytes) -> str:
        """Compute MatriXHash-256 digest as hex string."""
        msg_len = len(message)
        msg_buf = (ctypes.c_uint8 * msg_len)(*message)
        hex_buf = ctypes.create_string_buffer(65)
        self._lib.matrixhash256_hex(msg_buf, msg_len, hex_buf)
        return hex_buf.value.decode('ascii')


def verify_against_python():
    """Verify that the C implementation matches the Python one exactly."""
    print("=" * 60)
    print("  MatriXHash-256 — C vs Python Verification")
    print("=" * 60)

    # Load C library
    try:
        c_hasher = MatriXHash256C()
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        return False

    # Load Python implementation
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.matrixhash import MatriXHash256
        py_hasher = MatriXHash256()
    except ImportError:
        print("\n  ERROR: Cannot import core.matrixhash. Run from CryptoForge root.")
        return False

    # Verify AES S-box
    print("\n  1. AES S-box verification:")
    sbox_buf = (ctypes.c_uint8 * 256)()
    c_hasher._lib.get_aes_sbox(sbox_buf)
    c_sbox = list(sbox_buf)

    from core.sbox import AES_SBOX
    import numpy as np
    py_sbox = list(np.array(AES_SBOX, dtype=int))

    sbox_match = c_sbox == py_sbox
    print(f"     S(0x00) C={c_sbox[0]:#04x} Python={py_sbox[0]:#04x}")
    print(f"     S(0x01) C={c_sbox[1]:#04x} Python={py_sbox[1]:#04x}")
    print(f"     S(0x53) C={c_sbox[0x53]:#04x} Python={py_sbox[0x53]:#04x}")
    print(f"     All 256 entries match: {sbox_match}")

    # Verify hash outputs
    print("\n  2. Hash output verification:")
    test_cases = [
        b"",
        b"Hello, CryptoForge!",
        b"a",
        b"The quick brown fox jumps over the lazy dog",
        b"\x00" * 64,  # exactly one block after padding
        b"A" * 100,    # multiple blocks
        b"CryptoForge",
        b"String",
    ]

    all_match = True
    for i, msg in enumerate(test_cases):
        c_hex = c_hasher.hexdigest(msg)
        py_hex = py_hasher.hexdigest(msg)
        match = c_hex == py_hex
        if not match:
            all_match = False
        label = repr(msg) if len(msg) <= 30 else repr(msg[:30]) + "..."
        status = "✓" if match else "✗ MISMATCH"
        print(f"     Test {i+1}: {label}")
        print(f"       C:      {c_hex}")
        print(f"       Python: {py_hex}")
        print(f"       {status}")

    # Speed comparison
    print("\n  3. Speed comparison:")
    import time

    msg = b"CryptoForge benchmark test message for speed comparison"
    n_c = 1000
    n_py = 10

    t0 = time.perf_counter()
    for _ in range(n_c):
        c_hasher.hexdigest(msg)
    c_time = time.perf_counter() - t0
    c_per = c_time / n_c * 1000  # ms

    t0 = time.perf_counter()
    for _ in range(n_py):
        py_hasher.hexdigest(msg)
    py_time = time.perf_counter() - t0
    py_per = py_time / n_py * 1000  # ms

    speedup = py_per / c_per if c_per > 0 else 0
    print(f"     C:      {c_per:.3f} ms/hash ({n_c} iterations)")
    print(f"     Python: {py_per:.1f} ms/hash ({n_py} iterations)")
    print(f"     Speedup: {speedup:.0f}×")

    print(f"\n  RESULT: {'ALL TESTS PASSED' if all_match and sbox_match else 'SOME TESTS FAILED'}")
    print("=" * 60)
    return all_match and sbox_match


if __name__ == "__main__":
    verify_against_python()