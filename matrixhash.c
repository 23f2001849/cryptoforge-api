/*
 * MatriXHash-256 — C Implementation
 * 
 * Identical algorithm to core/matrixhash.py:
 *   - GF(2^8) with AES polynomial x^8 + x^4 + x^3 + x + 1
 *   - 32-byte state, 64-byte blocks, 256-bit digest
 *   - Merkle-Damgård + Davies-Meyer compression
 *   - Round: INJECT → SUBSTITUTE → DIFFUSE → PERMUTE
 *   - Default: 8 rounds, AES S-box, 32×32 Cauchy MDS, ShiftRows 4×8
 *
 * Compile:
 *   Windows: cl /O2 /LD matrixhash.c /Fe:matrixhash.dll
 *   GCC:    gcc -O2 -shared -fPIC -o matrixhash.so matrixhash.c
 *   MinGW:  gcc -O2 -shared -o matrixhash.dll matrixhash.c
 *
 * The Python wrapper (matrixhash_c.py) calls this via ctypes.
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define STATE_SIZE 32
#define BLOCK_SIZE 64
#define DIGEST_SIZE 32
#define NUM_ROUNDS 8
#define GF_POLY 0x11B  /* x^8 + x^4 + x^3 + x + 1 */

/* ═══════════════════════════════════════════════════════════════
   GF(2^8) ARITHMETIC
   ═══════════════════════════════════════════════════════════════ */

/* Multiplication lookup table: gf_mul_table[a][b] = a * b in GF(2^8) */
static uint8_t gf_mul_table[256][256];

/* Inverse table: gf_inv[a] = a^{-1} in GF(2^8), gf_inv[0] = 0 */
static uint8_t gf_inv[256];

static uint8_t gf_mul_slow(uint8_t a, uint8_t b) {
    uint16_t result = 0;
    uint16_t aa = a;
    for (int i = 0; i < 8; i++) {
        if (b & (1 << i))
            result ^= (aa << i);
    }
    /* Reduce modulo the polynomial */
    for (int i = 15; i >= 8; i--) {
        if (result & (1 << i))
            result ^= (GF_POLY << (i - 8));
    }
    return (uint8_t)(result & 0xFF);
}

static void init_gf_tables(void) {
    /* Build full multiplication table */
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++)
            gf_mul_table[a][b] = gf_mul_slow((uint8_t)a, (uint8_t)b);

    /* Build inverse table by brute force */
    gf_inv[0] = 0;
    for (int a = 1; a < 256; a++) {
        for (int b = 1; b < 256; b++) {
            if (gf_mul_table[a][b] == 1) {
                gf_inv[a] = (uint8_t)b;
                break;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   AES S-BOX
   ═══════════════════════════════════════════════════════════════ */

static uint8_t aes_sbox[256];

static uint8_t bit(uint8_t x, int i) { return (x >> i) & 1; }

static void init_aes_sbox(void) {
    /* AES S-box: multiplicative inverse in GF(2^8) + affine transform */
    for (int i = 0; i < 256; i++) {
        uint8_t inv = gf_inv[i]; /* 0 maps to 0 */
        /* Affine transform over GF(2): b_i = inv_i ^ inv_{(i+4)%8} ^ inv_{(i+5)%8}
           ^ inv_{(i+6)%8} ^ inv_{(i+7)%8} ^ c_i, where c = 0x63 */
        uint8_t result = 0;
        for (int j = 0; j < 8; j++) {
            uint8_t b = bit(inv, j) ^ bit(inv, (j+4)%8) ^ bit(inv, (j+5)%8)
                      ^ bit(inv, (j+6)%8) ^ bit(inv, (j+7)%8) ^ bit(0x63, j);
            result |= (b << j);
        }
        aes_sbox[i] = result;
    }
}

/* ═══════════════════════════════════════════════════════════════
   32×32 CAUCHY MDS MATRIX
   M[i][j] = 1 / (x_i + y_j) in GF(2^8)
   x_i = i + 1  (i = 0..31), y_j = j + 33 (j = 0..31)
   Addition in GF(2^8) = XOR
   ═══════════════════════════════════════════════════════════════ */

static uint8_t mds_matrix[STATE_SIZE][STATE_SIZE];

static void init_mds_matrix(void) {
    for (int i = 0; i < STATE_SIZE; i++) {
        uint8_t x_i = (uint8_t)(i + 1);
        for (int j = 0; j < STATE_SIZE; j++) {
            uint8_t y_j = (uint8_t)(j + 33);
            uint8_t denom = x_i ^ y_j;  /* GF(2^8) addition = XOR */
            mds_matrix[i][j] = gf_inv[denom];  /* 1/denom */
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   SHIFTROWS 4×8 PERMUTATION
   32 bytes as 4 rows × 8 cols, row i shifted left by i
   ═══════════════════════════════════════════════════════════════ */

static uint8_t shiftrows_perm[STATE_SIZE];

static void init_permutation(void) {
    int nrows = 4, ncols = 8;
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            int old_pos = row * ncols + col;
            int new_col = (col - row + ncols) % ncols;
            int new_pos = row * ncols + new_col;
            shiftrows_perm[new_pos] = (uint8_t)old_pos;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
   IV AND ROUND CONSTANTS
   IV = first 32 bytes of fractional part of π
   RC[r] = SHA-256("MatriXHash-256-RC-{r}")[:32]
   ═══════════════════════════════════════════════════════════════ */

static const uint8_t IV[STATE_SIZE] = {
    0x24, 0x3F, 0x6A, 0x88, 0x85, 0xA3, 0x08, 0xD3,
    0x13, 0x19, 0x8A, 0x2E, 0x03, 0x70, 0x73, 0x44,
    0xA4, 0x09, 0x38, 0x22, 0x29, 0x9F, 0x31, 0xD0,
    0x08, 0x2E, 0xFA, 0x98, 0xEC, 0x4E, 0x6C, 0x89
};

/*
 * Pre-computed round constants.
 * Each is SHA-256("MatriXHash-256-RC-{r}") truncated to 32 bytes.
 * Computed with Python:
 *   import hashlib
 *   for r in range(8):
 *     h = hashlib.sha256(f"MatriXHash-256-RC-{r}".encode()).hexdigest()
 *     print(f"  // RC[{r}]")
 *     print("  {" + ", ".join(f"0x{h[i:i+2]}" for i in range(0,64,2)) + "},")
 */
static uint8_t round_constants[NUM_ROUNDS][STATE_SIZE];
static int rc_initialized = 0;

/* Minimal SHA-256 for round constant generation only */
static const uint32_t sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define RR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^((~(x))&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (RR(x,2)^RR(x,13)^RR(x,22))
#define EP1(x) (RR(x,6)^RR(x,11)^RR(x,25))
#define SIG0(x) (RR(x,7)^RR(x,18)^((x)>>3))
#define SIG1(x) (RR(x,17)^RR(x,19)^((x)>>10))

static void sha256_hash(const uint8_t *msg, size_t len, uint8_t out[32]) {
    uint32_t h0=0x6a09e667, h1=0xbb67ae85, h2=0x3c6ef372, h3=0xa54ff53a;
    uint32_t h4=0x510e527f, h5=0x9b05688c, h6=0x1f83d9ab, h7=0x5be0cd19;

    /* Padding */
    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    uint8_t *padded = (uint8_t*)calloc(padded_len, 1);
    if (!padded) return;
    memcpy(padded, msg, len);
    padded[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++)
        padded[padded_len - 1 - i] = (uint8_t)(bit_len >> (i * 8));

    for (size_t offset = 0; offset < padded_len; offset += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)padded[offset+i*4]<<24)|((uint32_t)padded[offset+i*4+1]<<16)
                  |((uint32_t)padded[offset+i*4+2]<<8)|padded[offset+i*4+3];
        for (int i = 16; i < 64; i++)
            w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];

        uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + EP1(e) + CH(e,f,g) + sha256_k[i] + w[i];
            uint32_t t2 = EP0(a) + MAJ(a,b,c);
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h0+=a; h1+=b; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=hh;
    }
    free(padded);

    uint32_t hs[8] = {h0,h1,h2,h3,h4,h5,h6,h7};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++)
            out[i*4+j] = (uint8_t)(hs[i] >> (24 - j*8));
}

static void init_round_constants(void) {
    if (rc_initialized) return;
    for (int r = 0; r < NUM_ROUNDS; r++) {
        char tag[64];
        int tag_len = sprintf(tag, "MatriXHash-256-RC-%d", r);
        sha256_hash((const uint8_t*)tag, (size_t)tag_len, round_constants[r]);
    }
    rc_initialized = 1;
}

/* ═══════════════════════════════════════════════════════════════
   INITIALIZATION — call once before hashing
   ═══════════════════════════════════════════════════════════════ */

static int initialized = 0;

static void ensure_init(void) {
    if (initialized) return;
    init_gf_tables();
    init_aes_sbox();
    init_mds_matrix();
    init_permutation();
    init_round_constants();
    initialized = 1;
}

/* ═══════════════════════════════════════════════════════════════
   MATRIX-VECTOR MULTIPLY OVER GF(2^8)
   out[i] = Σ_j M[i][j] * v[j]
   ═══════════════════════════════════════════════════════════════ */

static void gf_matrix_vec_mul(const uint8_t M[STATE_SIZE][STATE_SIZE],
                               const uint8_t *v, uint8_t *out) {
    for (int i = 0; i < STATE_SIZE; i++) {
        uint8_t acc = 0;
        for (int j = 0; j < STATE_SIZE; j++) {
            acc ^= gf_mul_table[M[i][j]][v[j]];
        }
        out[i] = acc;
    }
}

/* ═══════════════════════════════════════════════════════════════
   SINGLE ROUND: INJECT → SUBSTITUTE → DIFFUSE → PERMUTE
   ═══════════════════════════════════════════════════════════════ */

static void single_round(uint8_t state[STATE_SIZE],
                          const uint8_t block[BLOCK_SIZE],
                          int round_idx) {
    uint8_t temp[STATE_SIZE];

    /* 1. INJECT: state ^= round_key
       round_key = (block[0:32] XOR block[32:64]) XOR round_constant[r] */
    for (int i = 0; i < STATE_SIZE; i++) {
        uint8_t folded = block[i] ^ block[i + STATE_SIZE];
        state[i] ^= folded ^ round_constants[round_idx][i];
    }

    /* 2. SUBSTITUTE: byte-wise S-box */
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = aes_sbox[state[i]];
    }

    /* 3. DIFFUSE: 32×32 matrix multiply over GF(2^8) */
    gf_matrix_vec_mul(mds_matrix, state, temp);
    memcpy(state, temp, STATE_SIZE);

    /* 4. PERMUTE: ShiftRows 4×8 */
    memcpy(temp, state, STATE_SIZE);
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = temp[shiftrows_perm[i]];
    }
}

/* ═══════════════════════════════════════════════════════════════
   COMPRESSION: Davies-Meyer
   H_i = E(H_{i-1}, M_i) XOR H_{i-1}
   ═══════════════════════════════════════════════════════════════ */

static void compress(uint8_t h[STATE_SIZE], const uint8_t block[BLOCK_SIZE]) {
    uint8_t save[STATE_SIZE];
    memcpy(save, h, STATE_SIZE);

    /* E(H, M): apply all rounds */
    for (int r = 0; r < NUM_ROUNDS; r++) {
        single_round(h, block, r);
    }

    /* Davies-Meyer feedforward: H_i = E(H_{i-1}, M) XOR H_{i-1} */
    for (int i = 0; i < STATE_SIZE; i++) {
        h[i] ^= save[i];
    }
}

/* ═══════════════════════════════════════════════════════════════
   PADDING: Merkle-Damgård
   1. Append 0x80
   2. Zeros until length ≡ 56 (mod 64)
   3. 8-byte big-endian original length
   ═══════════════════════════════════════════════════════════════ */

static uint8_t* pad_message(const uint8_t *msg, size_t msg_len, size_t *out_len) {
    /* Calculate padded length */
    size_t padded = msg_len + 1; /* +1 for 0x80 */
    while (padded % BLOCK_SIZE != (BLOCK_SIZE - 8))
        padded++;
    padded += 8; /* length field */

    uint8_t *result = (uint8_t*)calloc(padded, 1);
    if (!result) return NULL;

    memcpy(result, msg, msg_len);
    result[msg_len] = 0x80;

    /* Big-endian length at the end */
    uint64_t bit_count = (uint64_t)msg_len; /* store byte count, matching Python */
    for (int i = 0; i < 8; i++) {
        result[padded - 1 - i] = (uint8_t)(bit_count >> (i * 8));
    }

    *out_len = padded;
    return result;
}

/* ═══════════════════════════════════════════════════════════════
   FULL HASH
   ═══════════════════════════════════════════════════════════════ */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void matrixhash256(const uint8_t *msg, size_t msg_len, uint8_t digest[DIGEST_SIZE]) {
    ensure_init();

    /* Pad */
    size_t padded_len;
    uint8_t *padded = pad_message(msg, msg_len, &padded_len);
    if (!padded) {
        memset(digest, 0, DIGEST_SIZE);
        return;
    }

    /* Initialize state with IV */
    uint8_t h[STATE_SIZE];
    memcpy(h, IV, STATE_SIZE);

    /* Process each 64-byte block */
    size_t num_blocks = padded_len / BLOCK_SIZE;
    for (size_t i = 0; i < num_blocks; i++) {
        compress(h, padded + i * BLOCK_SIZE);
    }

    free(padded);
    memcpy(digest, h, DIGEST_SIZE);
}

/* Convenience: hash and return as hex string */
EXPORT void matrixhash256_hex(const uint8_t *msg, size_t msg_len, char hex_out[65]) {
    uint8_t digest[DIGEST_SIZE];
    matrixhash256(msg, msg_len, digest);
    for (int i = 0; i < DIGEST_SIZE; i++) {
        sprintf(hex_out + i * 2, "%02x", digest[i]);
    }
    hex_out[64] = '\0';
}

/* Get the AES S-box table (for verification) */
EXPORT void get_aes_sbox(uint8_t out[256]) {
    ensure_init();
    memcpy(out, aes_sbox, 256);
}

/* Get MDS matrix (for verification) */
EXPORT void get_mds_matrix(uint8_t out[STATE_SIZE * STATE_SIZE]) {
    ensure_init();
    memcpy(out, mds_matrix, STATE_SIZE * STATE_SIZE);
}