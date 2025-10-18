import numpy as np, time
from typing import List, Tuple
from .utils import (
    PartialFourier, negacyclic_conv, zero_center, shake256,
    vec_modq_to_bytes, vec_modq_from_bytes,
    witness_serialize, witness_deserialize, WitnessPacked,
    Hash_C, FormatC, Hash_beta, Hash_Acc,
    HashToLat,
)

class COMPASS:
    """
    COMPASS: Compact PASS-lineage Accumulator with Succinct Proofs
    """
    def __init__(self, K=20, param_set="set1", lambda_sec=128):
        """
        Initialize COMPASS parameters (but not keys).

        This sets up the mathematical structure but doesn't generate secrets yet.
        Setup() must be called separately to generate keys.
        """
        # Choose parameter set
        if param_set == "set1":
            self.N = 512
            self.q = 205207553
            self.kappa = 44
            self.sigma = 11336
            self.t = 256

        elif param_set == "set2":
            self.N = 1024
            self.q = 4294957057
            self.kappa = 36
            self.sigma = 167771
            self.t = 512

        self.m = 128    # HashAcc output dimension
        self.tau = 512  # HashC output bits
        self.lambda_sec = lambda_sec

        # Derived parameters
        self.sy = self._compute_gaussian_threshold()
        self.Boundz = self._compute_Boundz()
        self.M = self._compute_M()
        self.BoundZ = None
        self.K = K
        self.ctx = None
        # Compute BoundZ
        self.BoundZ = self._compute_BoundZ_for_K(K)
        print(f"  BoundZ for K={K}: {self.BoundZ:.2f}")

        # Initialize PartialFourier (generates Ω)
        self.PF = PartialFourier(self.q, self.N, self.t)
        assert pow(self.PF.g, self.N, self.q) == self.q - 1, "g^N must be -1 mod q"

        # Keys and oracles set to None (will be filled by Setup)
        self.sk = None
        self.pk = None
        self.Hc1 = None  # Domain-separated challenge hash
        self.Hc2 = None  # Domain-separated challenge hash
        self.Hbeta = None  # Aggregation weight hash
        self.HAcc = None  # Accumulator hash

        print(f"COMPASS initialized with {param_set}: N={self.N}, q={self.q}, κ={self.kappa}, t={self.t}")

    def Setup(self, ctx=None, epoch="2025-01"):
        """
        Setup(1^λ) → (pp, sk)

        Generate manager's keypair and define domain-separated hash oracles.

        Args:
            ctx: Context string for domain separation (scheme ID, epoch, etc.)
            epoch: Epoch/version string for domain separation

        Returns:
            pp: Public parameters (dict)
            sk: Secret key (stored internally as self.sk)
        """
        print(f"Running Setup with security parameter λ={self.lambda_sec}...")

        # 1. Sample secret key sk ← B_∞(1)
        self.sk = np.random.choice([-1, 0, 1], size=self.N).astype(np.int64)

        # 2. Compute public key pk = F_Ω(sk)
        self.pk = self.PF.F(self.sk)

        # 3. Define domain-separated hash oracles
        if ctx is None:
            # Hash of parameters for uniqueness
            params_str = f"N={self.N}||q={self.q}||t={self.t}||kappa={self.kappa}||sigma={self.sigma}"
            params_hash = shake256(params_str.encode(), 16).hex()
            ctx = f"COMPASS-v1||{params_hash}||{epoch}"

        self.ctx = ctx.encode()
        print(f"  Context: {ctx[:60]}...")

        # Store as methods with fixed domain separation
        # These capture self and ctx in closure
        def _Hc1(y_hat_omega, f_hat_omega):
            """Challenge hash c1: binds commitment to member vector"""
            data = self.ctx + b"||C1||" + vec_modq_to_bytes(y_hat_omega, self.q) + vec_modq_to_bytes(f_hat_omega, self.q)
            seed = Hash_C(data, 64)
            return FormatC(seed, self.N, self.kappa)

        def _Hc2(y_hat_omega, pk):
            """Challenge hash c2: binds commitment to public key"""
            data = self.ctx + b"||C2||" + vec_modq_to_bytes(y_hat_omega, self.q) + vec_modq_to_bytes(pk, self.q)
            seed = Hash_C(data, 64)
            return FormatC(seed, self.N, self.kappa)

        def _Hbeta(c1, c2, f_hat_omega):
            """Aggregation weight: ±1"""
            data = self.ctx + b"||BETA||"
            return Hash_beta(c1, c2, f_hat_omega, self.q)

        def _HAcc(Z_hat_omega):
            """Accumulator digest"""
            return Hash_Acc(Z_hat_omega, self.q, self.m)

        # Assign to instance
        self.Hc1 = _Hc1
        self.Hc2 = _Hc2
        self.Hbeta = _Hbeta
        self.HAcc = _HAcc

        # Build public parameters
        pp = {
            'N': self.N,
            'q': self.q,
            'g': self.PF.g,
            'sigma': self.sigma,
            'Omega': self.PF.Omega,
            't': self.t,
            'kappa': self.kappa,
            'm': self.m,
            'tau': self.tau,
            'pk': self.pk,
            'ctx': self.ctx,
            'Boundz': self.Boundz
        }

        print(f"✓ Setup complete. pk has dimension {len(self.pk)}")
        return pp, self.sk

    def Accumulate(self, F: List[np.ndarray]):
        """
        Accumulate(pp, sk, K, {f_i}) → (Acc, Z, L)

        Main accumulation algorithm with rejection sampling.
        This is the most computation-intensive part.

        Args:
            F: List of member vectors f_i ∈ B_∞(1) (NOT certificates!)
               Use HashToLat(cert, N) externally to convert certificates to f_i

        Returns:
            Acc: Accumulator value (m-dimensional vector in Z_q)
            Z: Aggregate state (polynomial in Rq)
            L: List of tuples (β_i, c_i1, c_i2, z_i) for witness extraction
        """
        if self.sk is None:
            raise RuntimeError("Must call Setup() before Accumulate()")

        K = len(F)
        #self.K = K
        #print(f"\nAccumulating {K} members...")
        if K > self.K:
            print(f"Entered member number exceeded set K, this will be unsafe.")

        F_hat = [self.PF.F(f) for f in F]  # Precompute evaluations

        # Compute BoundZ
        #self.BoundZ = self._compute_BoundZ_for_K(K)
        #print(f"  BoundZ for K={K}: {self.BoundZ:.2f}")

        outer_attempts = 0
        max_outer_attempts = 10
        while outer_attempts < max_outer_attempts:
            sum_attempts = 0
            outer_attempts += 1
            Z = np.zeros(self.N, dtype=np.int64)
            L = []
            for i in range(K):
                # Inner loop: rejection sampling for single member
                inner_attempts = 0
                max_inner_attempts = 30000
                while inner_attempts < max_inner_attempts:
                    inner_attempts += 1

                    # 1. Sample commitment y ← D^N_σ
                    y = np.random.normal(0.0, self.sigma, size=self.N).round().astype(np.int64)
                    if np.max(np.abs(y)) > self.sy:
                        continue  # Resample if clipping threshold exceeded
                    y_hat_omega = self.PF.F(y)

                    # 2. Compute dual challenges
                    c1 = self.Hc1(y_hat_omega, F_hat[i])
                    c2 = self.Hc2(y_hat_omega, self.pk)

                    # 3. Compute response z = sk * c1 + f * c2 + y
                    u = (negacyclic_conv(self.sk, c1, self.q) + negacyclic_conv(F[i], c2, self.q)) % self.q
                    z = (u + y) % self.q
                    z = zero_center(z, self.q)  # Center for norm computation
                    u = zero_center(u, self.q)  # Center for rejection sampling

                    # 4. Rejection sampling checks
                    z_norm = np.linalg.norm(z)
                    if z_norm > self.Boundz:
                        continue  # Reject: z too large

                    # 5. Weighted rejection sampling
                    if not self._rejection_sampling(u, z):
                        continue

                    # Accepted!
                    sum_attempts += inner_attempts
                    if inner_attempts > 1:
                        print(f"  Member {i+1}: accepted after {inner_attempts} attempts")
                    break

                if inner_attempts >= max_inner_attempts:
                    raise RuntimeError(f"Failed to find valid z for member {i} after {max_inner_attempts} attempts")


                # 6. Compute aggregation weight β_i and update Z
                beta_i = self.Hbeta(c1, c2, F_hat[i])
                Z = (Z + beta_i * z) % self.q
                L.append((beta_i, c1, c2, z))

            print(f"Empirical p ≈ {K/sum_attempts:.3e} "
                  f"(avg attempts/member ≈ {sum_attempts/K:.1f})")
            # Check aggregate bound
            Z_centered = zero_center(Z, self.q)
            Z_norm = np.linalg.norm(Z_centered)
            if Z_norm <= self.BoundZ:
                print(f"✓ Accumulation successful after {outer_attempts} outer attempt(s)")
                break
            else:
                print(f"  Outer attempt {outer_attempts}: ||Z|| = {Z_norm:.1f} > {self.BoundZ:.1f}, restarting...")

        if outer_attempts >= max_outer_attempts:
            raise RuntimeError(f"Failed to satisfy BoundZ after {max_outer_attempts} attempts")

        # Compute accumulator value
        Z_hat_omega = self.PF.F(Z)
        Acc = self.HAcc(Z_hat_omega)

        print(f"  Accumulator size: {len(Acc)} elements")
        print(f"  ||Z||_2 = {Z_norm:.2f}")

        return Acc, Z, L

    def Witness(self, Z: np.ndarray, L: List, i: int) -> WitnessPacked:
        """
        Witness(pp, Z, L, i) → w_i

        Extract witness for member i. Lightweight operation.

        Args:
            Z: Aggregate state
            L: Auxiliary list from Accumulate
            i: Member index (0-indexed)

        Returns:
            w_i: Witness tuple (Z_i, z_i, c_i1, c_i2)
        """
        if i < 0 or i >= len(L):
            raise ValueError(f"Invalid member index {i}")

        beta_i, ci1, ci2, zi = L[i]

        # Compute Z_i = Z - β_i * z_i (mod q)
        Zi = (Z - beta_i * zi) % self.q

        return self.serialize_witness((Zi, zi, ci1, ci2))

    def Verify(self, Acc: np.ndarray, f: np.ndarray, witness_packed: WitnessPacked) -> bool:
        """
        Verify(pp, Acc, f, w) → {0, 1}

        Verify membership witness.

        Args:
            Acc: Accumulator value
            f: Member vector
            witness: Tuple (Z_i, z_i, c_i1, c_i2)

        Returns:
            True if valid, False otherwise
        """
        if self.BoundZ is None:
            raise RuntimeError("BoundZ not set - must Accumulate before Verify")
        Zi, zi, ci1, ci2 = self.deserialize_witness(witness_packed.data)

        # 1. Norm check on z_i
        zi_centered = zero_center(zi, self.q)
        if np.linalg.norm(zi_centered) > self.Boundz:
            return False

        # 2. Compute evaluations (all return uint64 internally, cast to int64)
        zi_hat = self.PF.F(zi).astype(np.uint64)
        ci1_hat = self.PF.F(ci1).astype(np.uint64)
        ci2_hat = self.PF.F(ci2).astype(np.uint64)
        f_hat = self.PF.F(f).astype(np.uint64)
        pk_uint = self.pk.astype(np.uint64)

        # 3. Reconstruct y'_hat = z_hat - pk ⊙ c1_hat - f_hat ⊙ c2_hat
        #y_prime_hat = (zi_hat - self.pk * ci1_hat - f_hat * ci2_hat) % self.q
        # Compute each multiplication separately with immediate modulo
        term1 = (pk_uint * ci1_hat) % self.q
        term2 = (f_hat * ci2_hat) % self.q
        # Subtraction in modular arithmetic (handle underflow)
        y_prime_hat = (zi_hat + self.q - term1 + self.q - term2) % self.q
        y_prime_hat = y_prime_hat.astype(np.int64)

        # 4. Recompute challenges
        ci1_prime = self.Hc1(y_prime_hat, f_hat)
        ci2_prime = self.Hc2(y_prime_hat, self.pk)

        # 5. Challenge consistency check
        if not (np.array_equal(ci1_prime, ci1) and np.array_equal(ci2_prime, ci2)):
            return False

        # 6. Reconstruct aggregate Z' = Z_i + β_i * z_i
        beta_i = self.Hbeta(ci1, ci2, f_hat)
        Z_prime = (Zi + beta_i * zi) % self.q

        # 7. Aggregate norm check
        Z_prime_centered = zero_center(Z_prime, self.q)
        if np.linalg.norm(Z_prime_centered) > self.BoundZ:
            return False

        # 8. Recompute accumulator
        Z_prime_hat = self.PF.F(Z_prime)
        Acc_prime = self.HAcc(Z_prime_hat)

        # 9. Final check
        return np.array_equal(Acc_prime, Acc)

    def serialize_witness(self, wit):
        """Convenience wrapper for witness serialization"""
        return witness_serialize(*wit, self.N, self.q, self.kappa)

    def deserialize_witness(self, buf):
        """Convenience wrapper for witness deserialization"""
        return witness_deserialize(buf, self.N, self.q, self.kappa)

    def _compute_M(self):
        """
        Compute rejection sampling constant M.
        """
        U = 2 * self.kappa * np.sqrt(self.N)
        exponent = (U**2 + 2 * U * self.Boundz) / (2 * self.sigma**2)
        return np.exp(exponent)

    def _compute_BoundZ_for_K(self, K: int) -> float:
        """
        Compute aggregate bound BoundZ for K members.
        """
        return 2 * self.sigma * np.sqrt(K * self.N)

    def _compute_gaussian_threshold(self):
        """Compute clipping threshold s_y for Gaussian sampling"""
        return self.sigma * np.sqrt(2 * (self.lambda_sec * np.log(2) + np.log(2 * self.N)))

    def _compute_Boundz(self):
        """Compute L2 bound on z for verification"""
        # From paper: ||z||_2 ≤ 2σ√N
        return 2 * self.sigma * np.sqrt(self.N)

    def _rejection_sampling(self, u: np.ndarray, z: np.ndarray) -> bool:
        """
        Weighted rejection sampling (Lyubashevsky-style).

        Accept z with probability:
        P_accept = min(1, (exp(-2⟨z,u⟩ + ||u||²) / (2σ²))/M))

        From Equation (1) in paper.
        """
        # Compute exponent: (-2⟨z,u⟩ + ||u||²₂) / (2σ²)
        inner_product = np.dot(z, u)
        u_norm_sq = np.dot(u, u)

        exponent = (-2 * inner_product + u_norm_sq) / (2 * self.sigma**2)

        # P_accept = min(1, exp(exponent) / M)
        prob = min(1.0, np.exp(exponent) / self.M)

        # Sample uniform and accept if below probability
        return np.random.random() < prob