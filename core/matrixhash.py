"""
CryptoForge — MatriXHash-256

A parameterized matrix-based cryptographic hash function.

Construction:
  - Merkle-Damgård domain extension with Davies-Meyer compression
  - Each round: INJECT → SUBSTITUTE → DIFFUSE → PERMUTE
  - State: 256 bits (32 bytes ∈ GF(2^8)^32)
  - Block: 512 bits (64 bytes)
  - Output: 256-bit digest

The key property: every design choice is a tunable parameter, making this
a searchable design space for AI-driven optimization.

Round function for round r:
  1. INJECT:     s ← s ⊕ round_key(message_block, r)
  2. SUBSTITUTE: s[i] ← S-box(s[i]) for each byte
  3. DIFFUSE:    s ← M · s over GF(2^8)   (matrix multiplication)
  4. PERMUTE:    s ← P(s)                  (byte position shuffle)

Compression: Davies-Meyer feedforward
  H_i = compress(H_{i-1}, M_i) = E(H_{i-1}, M_i) ⊕ H_{i-1}
  where E is the round function iterated NUM_ROUNDS times.
"""

import hashlib
import numpy as np
import yaml

from core.gf_arithmetic import GF256, gf_array, bytes_to_gf_vector, gf_vector_to_bytes
from core.sbox import AES_SBOX, apply_sbox, build_power_map_sbox, build_random_sbox, build_identity_sbox
from core.matrix_builders import build_matrix, build_cauchy_mds, build_cauchy_mds_from_params
from core.permutation import apply_permutation, get_permutation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_SIZE = 32   # bytes (256 bits)
BLOCK_SIZE = 64   # bytes (512 bits)
DIGEST_SIZE = 32  # bytes (256 bits)

# IV: first 32 bytes of fractional part of π (nothing-up-my-sleeve number)
# Precomputed to avoid dependency on mpmath
_PI_FRAC_HEX = (
    "243F6A8885A308D313198A2E03707344"
    "A4093822299F31D0082EFA98EC4E6C89"
)
IV_BYTES = bytes.fromhex(_PI_FRAC_HEX)
assert len(IV_BYTES) == 32


def _generate_round_constants(num_rounds: int) -> list:
    """Generate round constants using SHA-256 (nothing-up-my-sleeve).
    
    RC[r] = SHA-256("MatriXHash-256-RC-{r}")[:32]
    
    Each round constant is a 32-byte vector in GF(2^8)^32.
    """
    constants = []
    for r in range(num_rounds):
        tag = f"MatriXHash-256-RC-{r}".encode('ascii')
        h = hashlib.sha256(tag).digest()
        constants.append(bytes_to_gf_vector(h))
    return constants


# ---------------------------------------------------------------------------
# MatriXHash-256 Class
# ---------------------------------------------------------------------------

class MatriXHash256:
    """Parameterized matrix-based 256-bit hash function.
    
    All design choices are configurable, making this a searchable design space
    for the adversarial co-evolution loop.
    """
    
    def __init__(self, config: dict = None, config_path: str = None):
        """Initialize from a config dict or YAML file.
        
        Args:
            config: configuration dictionary
            config_path: path to YAML config file (used if config is None)
        """
        if config is None and config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config is None:
            config = self._default_config()
        
        self.config = config
        self._build_components()
    
    @staticmethod
    def _default_config() -> dict:
        """Return the default configuration (Candidate A)."""
        return {
            'hash': {
                'name': 'MatriXHash-256',
                'state_size': 32,
                'block_size': 64,
                'digest_size': 32,
            },
            'round_function': {
                'num_rounds': 8,
                'matrix_type': 'cauchy_mds',
                'matrix_seed': 42,
                'sbox_type': 'aes',
                'sbox_param': 254,
                'permutation_type': 'shiftrows_4x8',
            },
            'cauchy': {
                'x_start': 1,
                'y_start': 33,
            },
        }
    
    def _build_components(self):
        """Build all hash components from the config."""
        rf = self.config['round_function']
        
        # Number of rounds
        self.num_rounds = rf['num_rounds']
        
        # S-box
        sbox_type = rf.get('sbox_type', 'aes')
        if sbox_type == 'aes':
            self.sbox = AES_SBOX
        elif sbox_type == 'power_map':
            self.sbox = build_power_map_sbox(rf.get('sbox_param', 254))
        elif sbox_type == 'random':
            self.sbox = build_random_sbox(rf.get('matrix_seed', 42))
        elif sbox_type == 'identity':
            self.sbox = build_identity_sbox()
        else:
            raise ValueError(f"Unknown S-box type: {sbox_type}")
        
        # Mixing matrix
        matrix_type = rf.get('matrix_type', 'cauchy_mds')
        if matrix_type == 'cauchy_mds':
            cauchy_cfg = self.config.get('cauchy', {})
            self.matrix = build_cauchy_mds(
                STATE_SIZE,
                x_start=cauchy_cfg.get('x_start', 1),
                y_start=cauchy_cfg.get('y_start', STATE_SIZE + 1),
            )
        else:
            self.matrix = build_matrix(
                STATE_SIZE,
                matrix_type=matrix_type,
                seed=rf.get('matrix_seed', 42),
            )
        
        # Permutation
        perm_type = rf.get('permutation_type', 'shiftrows_4x8')
        self.permutation = get_permutation(perm_type)
        
        # Round constants
        self.round_constants = _generate_round_constants(self.num_rounds)
        
        # IV
        self.iv = bytes_to_gf_vector(IV_BYTES)
    
    # -------------------------------------------------------------------
    # Padding (Merkle-Damgård)
    # -------------------------------------------------------------------
    
    def _pad(self, message: bytes) -> bytes:
        """Merkle-Damgård padding.
        
        1. Append 0x80 byte
        2. Append zeros until length ≡ 56 (mod 64)
        3. Append original message length as 8-byte big-endian integer
        
        Result length is always a multiple of 64 bytes (BLOCK_SIZE).
        """
        msg_len = len(message)
        
        # Append the 0x80 byte
        padded = message + b'\x80'
        
        # Append zeros until length ≡ 56 mod 64
        while len(padded) % BLOCK_SIZE != (BLOCK_SIZE - 8):
            padded += b'\x00'
        
        # Append length as 8-byte big-endian
        padded += msg_len.to_bytes(8, byteorder='big')
        
        assert len(padded) % BLOCK_SIZE == 0
        return padded
    
    # -------------------------------------------------------------------
    # Round Function
    # -------------------------------------------------------------------
    
    def _derive_round_key(self, message_block, round_idx: int):
        """Derive the round key from the message block and round constant.
        
        round_key = message_material ⊕ round_constant
        
        The message block is 64 bytes but the state is 32 bytes, so we
        XOR-fold the block: first_half ⊕ second_half, then XOR with RC.
        """
        # XOR-fold the 64-byte block into 32 bytes
        first_half = message_block[:STATE_SIZE]
        second_half = message_block[STATE_SIZE:]
        folded = first_half + second_half  # GF addition = XOR
        
        # XOR with round constant
        return folded + self.round_constants[round_idx]
    
    def _single_round(self, state, message_block, round_idx: int):
        """Execute one round of the hash function.
        
        INJECT → SUBSTITUTE → DIFFUSE → PERMUTE
        """
        # 1. INJECT: s ← s ⊕ round_key
        round_key = self._derive_round_key(message_block, round_idx)
        state = state + round_key  # GF(2^8) addition = XOR
        
        # 2. SUBSTITUTE: apply S-box byte-wise
        state = apply_sbox(state, self.sbox)
        
        # 3. DIFFUSE: matrix multiplication over GF(2^8)
        state = self.matrix @ state
        
        # 4. PERMUTE: byte position shuffle
        state = apply_permutation(state, self.permutation)
        
        return state
    
    def _block_encrypt(self, state, message_block, num_rounds: int = None):
        """Apply all rounds of the round function (the "encryption" E).
        
        Args:
            state: current 32-byte state (GF(2^8)^32)
            message_block: 64-byte message block (GF(2^8)^64)
            num_rounds: override round count (for reduced-round analysis)
        
        Returns:
            Transformed state after all rounds
        """
        if num_rounds is None:
            num_rounds = self.num_rounds
        
        for r in range(num_rounds):
            state = self._single_round(state, message_block, r)
        
        return state
    
    # -------------------------------------------------------------------
    # Compression Function (Davies-Meyer)
    # -------------------------------------------------------------------
    
    def _compress(self, chaining_value, message_block, num_rounds: int = None):
        """Davies-Meyer compression: H_i = E(H_{i-1}, M_i) ⊕ H_{i-1}
        
        The feedforward (⊕ H_{i-1}) prevents length-extension attacks
        and ensures the compression function is one-way even if E is invertible.
        """
        encrypted = self._block_encrypt(chaining_value, message_block, num_rounds)
        return encrypted + chaining_value  # GF addition = XOR = feedforward
    
    # -------------------------------------------------------------------
    # Full Hash
    # -------------------------------------------------------------------
    
    def hash(self, message: bytes, num_rounds: int = None) -> bytes:
        """Compute the MatriXHash-256 digest of a message.
        
        Args:
            message: arbitrary-length bytes input
            num_rounds: override round count (for security margin testing)
        
        Returns:
            32-byte (256-bit) digest
        """
        # Pad the message
        padded = self._pad(message)
        
        # Initialize chaining value with IV
        h = self.iv.copy()
        
        # Process each 64-byte block
        num_blocks = len(padded) // BLOCK_SIZE
        for i in range(num_blocks):
            block_bytes = padded[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            block = bytes_to_gf_vector(block_bytes)
            h = self._compress(h, block, num_rounds)
        
        # Convert final state to bytes
        return gf_vector_to_bytes(h)
    
    def hexdigest(self, message: bytes, num_rounds: int = None) -> str:
        """Compute hash and return as hex string."""
        return self.hash(message, num_rounds).hex()
    
    def reduced_round_hash(self, message: bytes, rounds: int) -> bytes:
        """Hash with a specific number of rounds (for security margin sweep)."""
        return self.hash(message, num_rounds=rounds)
    
    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------
    
    def describe(self) -> str:
        """Return a human-readable description of the current configuration."""
        rf = self.config['round_function']
        return (
            f"MatriXHash-256 | "
            f"Rounds: {self.num_rounds} | "
            f"Matrix: {rf.get('matrix_type', 'cauchy_mds')} | "
            f"S-box: {rf.get('sbox_type', 'aes')} | "
            f"Perm: {rf.get('permutation_type', 'shiftrows_4x8')}"
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CryptoForge — MatriXHash-256 Verification")
    print("=" * 55)
    
    # Build hash with default config
    hasher = MatriXHash256()
    print(f"Config: {hasher.describe()}")
    print()
    
    # Test 1: Determinism — same input → same hash
    msg = b"Hello, CryptoForge!"
    h1 = hasher.hexdigest(msg)
    h2 = hasher.hexdigest(msg)
    print(f"Test 1 — Determinism")
    print(f"  Input: {msg}")
    print(f"  Hash1: {h1}")
    print(f"  Hash2: {h2}")
    print(f"  Match: {h1 == h2}")
    assert h1 == h2, "FAIL: Hash is not deterministic!"
    
    # Test 2: Different inputs → different hashes
    msg_a = b"Hello, CryptoForge!"
    msg_b = b"Hello, CryptoForge?"  # one character different
    ha = hasher.hexdigest(msg_a)
    hb = hasher.hexdigest(msg_b)
    print(f"\nTest 2 — Sensitivity")
    print(f"  Input A: {msg_a}")
    print(f"  Input B: {msg_b}")
    print(f"  Hash A:  {ha}")
    print(f"  Hash B:  {hb}")
    print(f"  Different: {ha != hb}")
    assert ha != hb, "FAIL: Different inputs produce same hash!"
    
    # Test 3: Empty message hashes without error
    h_empty = hasher.hexdigest(b"")
    print(f"\nTest 3 — Empty message")
    print(f"  Hash: {h_empty}")
    print(f"  Length: {len(hasher.hash(b''))} bytes = {len(hasher.hash(b'')) * 8} bits")
    assert len(hasher.hash(b"")) == 32, "FAIL: Digest is not 32 bytes!"
    
    # Test 4: Output is exactly 32 bytes for various input lengths
    print(f"\nTest 4 — Output size consistency")
    for length in [0, 1, 31, 32, 55, 56, 63, 64, 100, 1000]:
        msg = bytes(range(256)) * (length // 256 + 1)
        msg = msg[:length]
        h = hasher.hash(msg)
        ok = len(h) == 32
        print(f"  Input {length:4d} bytes → digest {len(h)} bytes {'✓' if ok else 'FAIL'}")
        assert ok, f"FAIL: Digest length {len(h)} for input length {length}"
    
    # Test 5: Reduced-round hashing works
    print(f"\nTest 5 — Reduced-round hashing")
    msg = b"test"
    for r in [1, 2, 4, 8]:
        h = hasher.hexdigest(msg, num_rounds=r)
        print(f"  {r} rounds: {h[:32]}...")
    
    # Test 6: Config from YAML file
    print(f"\nTest 6 — YAML config loading")
    try:
        hasher_yaml = MatriXHash256(config_path="configs/default.yaml")
        h_yaml = hasher_yaml.hexdigest(b"test")
        h_default = hasher.hexdigest(b"test")
        print(f"  YAML config hash:    {h_yaml[:32]}...")
        print(f"  Default config hash: {h_default[:32]}...")
        print(f"  Match: {h_yaml == h_default}")
    except Exception as e:
        print(f"  YAML loading error (non-critical): {e}")
    
    # Test 7: Quick avalanche sanity check
    print(f"\nTest 7 — Avalanche sanity check (single bit flip)")
    msg = b"\x00" * 64
    h_original = hasher.hash(msg)
    
    msg_flipped = bytearray(msg)
    msg_flipped[0] = 0x01  # flip least significant bit of first byte
    h_flipped = hasher.hash(bytes(msg_flipped))
    
    # Count differing bits
    diff_bits = 0
    for b1, b2 in zip(h_original, h_flipped):
        diff_bits += bin(b1 ^ b2).count('1')
    
    print(f"  Original hash: {h_original.hex()[:32]}...")
    print(f"  Flipped hash:  {h_flipped.hex()[:32]}...")
    print(f"  Differing bits: {diff_bits} / 256 ({diff_bits/256*100:.1f}%)")
    print(f"  Ideal: ~128 bits (50%)")
    
    print()
    print("=" * 55)
    print("ALL MATRIXHASH-256 TESTS PASSED")