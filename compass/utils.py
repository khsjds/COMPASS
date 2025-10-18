import numpy as np, math, hashlib
from dataclasses import dataclass
from typing import List, Tuple

# ------------------ Utilities ------------------
def mod_q(x, q: int) -> np.ndarray:
    return (np.asarray(x, dtype=np.int64) % q).astype(np.int64)

def zero_center(vec: np.ndarray, q: int) -> np.ndarray:
    return ((vec + q//2) % q) - q//2

def negacyclic_conv(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    N = a.shape[0]
    full = np.convolve(a.astype(np.int64), b.astype(np.int64))
    res = np.zeros(N, dtype=np.int64)
    res[:N] = full[:N]
    tail = full[N:]
    for j, val in enumerate(tail):
        res[j % N] -= val
    return mod_q(res, q)

def shake256(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(data).digest(outlen)

def bytes_per_q(q: int) -> int:
    return ((q - 1).bit_length() + 7) // 8

def vec_modq_to_bytes(vec: np.ndarray, q: int) -> bytes:
    b = bytes_per_q(q)
    return b"".join(int(int(x) % q).to_bytes(b, "big") for x in vec)

def vec_modq_from_bytes(data: bytes, n: int, q: int) -> Tuple[np.ndarray, int]:
    b = bytes_per_q(q)
    need = n * b
    if len(data) < need:
        raise ValueError(f"need {need} bytes, got {len(data)}")
    chunk = data[:need]
    arr = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, b)
    vals = np.array([int.from_bytes(row.tobytes(), "big") for row in arr], dtype=np.int64)
    return vals, need

# ------------------ Compact PASS-style c encoding ------------------
# Format: 8 bytes of sign bits (bit i = 1 means -1, 0 means +1), then κ indices (uint16 BE)
# Assumptions: κ <= 64, N <= 65535
def pack_c(c: np.ndarray, kappa: int) -> bytes:
    nz = np.flatnonzero(c)
    assert len(nz) == kappa, "c must have exactly κ nonzeros"
    signs = [(0 if c[i] == 1 else 1) for i in nz]  # 0=>+1, 1=>-1
    sign_bits = 0
    for i, bit in enumerate(signs[:64]):
        sign_bits |= (bit & 1) << i
    idx_bytes = b"".join(int(i).to_bytes(2, "big") for i in nz)
    return sign_bits.to_bytes(8, "big") + idx_bytes

def unpack_c(data: bytes, N: int, kappa: int) -> Tuple[np.ndarray, int]:
    sign_bits = int.from_bytes(data[:8], "big")
    idx_bytes = data[8:8 + 2*kappa]
    nz_idx = [int.from_bytes(idx_bytes[i:i+2], "big") for i in range(0, 2*kappa, 2)]
    c = np.zeros(N, dtype=np.int64)
    for i, idx in enumerate(nz_idx):
        sign = -1 if ((sign_bits >> i) & 1) else 1
        c[idx] = sign
    return c, (8 + 2*kappa)

# ------------------ Partial Fourier (FΩ) ------------------
def find_primitive_root(q: int) -> int:
    for g in range(2, q-1):
        if pow(g, (q-1)//2, q) != 1:
            return g
    return 3

def find_g_2N(q: int, N: int) -> int:
    r = find_primitive_root(q)
    g = pow(r, (q-1)//(2*N), q)
    if pow(g, N, q) != q-1:
        for k in range(2, 2*N):
            cand = pow(r, ((q-1)//(2*N))*k, q)
            if pow(cand, N, q) == q-1:
                return cand
    return g

class PartialFourier:
    """
    Partial Fourier Transform with overflow protection for large q.

    Evaluates polynomials at subset Ω of roots using vectorized operations
    with uint64 arithmetic and periodic modular reduction.
    """

    def __init__(self, q: int, N: int, t: int, Omega=None, g=None):
        """
        Initialize Partial Fourier transform.

        Args:
            q: Modulus (must satisfy q ≡ 1 mod 2N)
            N: Ring dimension (power of 2)
            t: Number of evaluation points (|Ω|)
            Omega: Optional preset exponents (odd powers)
            g: Optional preset primitive 2N-th root
        """
        self.q, self.N, self.t = q, N, t
        self.g = g if g is not None else find_g_2N(q, N)

        # Choose Omega (odd exponents for roots of x^N + 1)
        if Omega is None:
            odd = np.arange(1, 2*N, 2, dtype=np.int64)
            rng = np.random.default_rng()
            self.Omega = np.sort(rng.choice(odd, size=t, replace=False))
        else:
            self.Omega = np.array(Omega, dtype=np.int64)
            assert np.all(Omega % 2 == 1), "Omega must contain odd exponents"
            self.Omega = np.sort(Omega)

        # Precompute Vandermonde matrix as uint64 for overflow safety
        self.P = np.empty((t, N), dtype=np.uint64)
        for j, r in enumerate(self.Omega):
            root = pow(self.g, int(r), self.q)
            power = 1
            for i in range(N):
                self.P[j, i] = power
                power = (power * root) % self.q

        # Determine evaluation strategy based on q
        if self.q > (1 << 29):  # ~500M threshold
            # For very large q (close to 2^32), be extra conservative
            if self.q > (1 << 31):  # q > 2^31
                # Can only accumulate 1 term at a time!
                self.reduce_every = 1
                print(f"  Using ultra-safe mode: q={q} requires reduce_every=1")
            else:
                # For large q, compute safe reduction frequency
                # uint64 can hold up to 2^64
                # Each term is at most q * q ≈ 2^60 for q ≈ 2^30
                # Safely accumulate multiple terms before reduction
                max_terms = (1 << 63) // (self.q * self.q)  # Conservative estimate
                self.reduce_every = max(1, max_terms // 2)
            self.use_safe_mode = True
            print(f"  Precomputed Vandermonde matrix: {self.P.shape} ({self.P.nbytes / 1024:.1f} KB)")
            print(f"  Using overflow-safe mode: reduce every {self.reduce_every} terms")
        else:
            self.reduce_every = self.N
            self.use_safe_mode = False
            print(f"  Precomputed Vandermonde matrix: {self.P.shape} ({self.P.nbytes / 1024:.1f} KB)")

    def F(self, f: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial f at roots in Ω.

        Computes f̂|Ω = P·f (mod q) with overflow protection for large q.

        Args:
            f: Coefficient vector (length N)

        Returns:
            Evaluation vector (length t): [f(ω^r1), ..., f(ω^rt)]

        Time: O(t·N) with vectorized NumPy operations
        """
        # Convert to uint64 and reduce modulo q
        f_mod = (f % self.q).astype(np.uint64)

        if not self.use_safe_mode:
            # Fast path for small q: single matmul
            result = (self.P @ f_mod) % self.q
            return result.astype(np.int64)

        # Safe path for large q: chunked matmul with periodic reduction
        result = np.zeros(self.t, dtype=np.uint64)

        for start in range(0, self.N, self.reduce_every):
            end = min(start + self.reduce_every, self.N)

            # Compute partial dot product (won't overflow due to small chunk)
            chunk_result = self.P[:, start:end] @ f_mod[start:end]

            # Accumulate with modular reduction
            result = (result + chunk_result) % self.q

        return result.astype(np.int64)

# ------------------ Witness serialization (compact c + mod-q vectors) ------------------
@dataclass
class WitnessPacked:
    """Packed witness: (Z_i, z_i, c1, c2) with compact encoding"""
    data: bytes

def witness_serialize(Zi: np.ndarray, zi: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                      N: int, q: int, kappa: int) -> WitnessPacked:
    """
    Serialize witness components into compact format.

    Args:
        Zi, zi: mod-q vectors (length N)
        c1, c2: ternary challenge vectors with exactly κ nonzeros
        N: ring dimension
        q: modulus
        kappa: challenge sparsity
    """
    out  = vec_modq_to_bytes(Zi, q)
    out += vec_modq_to_bytes(zi, q)
    out += pack_c(c1, kappa)
    out += pack_c(c2, kappa)
    return WitnessPacked(out)

def witness_deserialize(buf: bytes, N: int, q: int, kappa: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deserialize witness from compact format.

    Returns:
        (Zi, zi, c1, c2) tuple
    """
    off = 0
    Zi, used = vec_modq_from_bytes(buf[off:], N, q); off += used
    zi, used = vec_modq_from_bytes(buf[off:], N, q); off += used
    c1, used = unpack_c(buf[off:], N, kappa); off += used
    c2, used = unpack_c(buf[off:], N, kappa); off += used
    return (Zi % q, zi % q, c1, c2)

def Hash_C(data: bytes, outlen: int = 64) -> bytes:
    """Challenge hash: arbitrary data -> τ bits for FormatC"""
    return shake256(data, outlen)

def FormatC(seed: bytes, N: int, kappa: int) -> np.ndarray:
    """Deterministic sparse ternary: exactly κ nonzeros from {-1,+1}"""
    rng = np.random.default_rng(int.from_bytes(seed, 'big'))
    idx = rng.choice(N, size=kappa, replace=False)
    sgn = rng.choice([-1, 1], size=kappa)
    c = np.zeros(N, dtype=np.int64)
    c[idx] = sgn
    return c

def Hash_beta(c1: np.ndarray, c2: np.ndarray, fhat: np.ndarray, q: int) -> int:
    """Aggregation weight: ±1 based on challenge/element hash"""
    data = vec_modq_to_bytes(c1, q) + vec_modq_to_bytes(c2, q) + vec_modq_to_bytes(fhat % q, q)
    return 1 if (shake256(data, 1)[0] & 1) else -1

def Hash_Acc(Z_hat: np.ndarray, q: int, m: int) -> np.ndarray:
    """Accumulator digest: Z_hat -> m elements in Z_q"""
    need = 2 * m
    raw = shake256(vec_modq_to_bytes(Z_hat % q, q), need)
    vals = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    limbs16 = (vals[0::2] << 8) + vals[1::2]
    return (limbs16 % q)[:m]

def HashToLat(cert: bytes, N: int) -> np.ndarray:
    """
    Map certificate to short ring vector in B_∞(1).

    Implementation: HashToLat = Φ ∘ HashC where:
    - HashC: certificate -> raw bytes via SHAKE256
    - Φ: deterministic rounding/centering to {-1, 0, +1}^N

    This is preprocessing done BEFORE Accumulate.

    Args:
        cert: Certificate bytes (email, ID, etc.)
        N: Ring dimension

    Returns:
        f ∈ B_∞(1): short vector with coefficients in {-1, 0, +1}
    """
    raw = shake256(cert, N)

    # Deterministic map Φ: bytes -> {-1, 0, 1}^N
    # Use mod 3 to get uniform distribution over {0, 1, 2}, then shift
    f = np.zeros(N, dtype=np.int64)
    for i in range(N):
        val = raw[i] % 3  # {0, 1, 2}
        f[i] = val - 1     # {-1, 0, +1}

    return f