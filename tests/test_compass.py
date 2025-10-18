import numpy as np, time
from compass.compass import COMPASS
from compass.utils import HashToLat, bytes_per_q

def test_compass():
    """Test COMPASS with proper HashToLat usage and performance tracking"""
    print("="*60)
    print("TESTING CORRECTED COMPASS")
    print("="*60)

    # 1. Initialize
    compass = COMPASS(K=100000, param_set="set1", lambda_sec=128)

    # 2. Setup
    pp, sk = compass.Setup(ctx=None, epoch="2025-01")
    print(f"\n✓ Setup complete")

    # 3. Create certificates and convert to lattice vectors
    K = 200
    certificates = [f"user_{i}@example.com".encode() for i in range(K)]

    # IMPORTANT: Use HashToLat BEFORE calling Accumulate
    F = [HashToLat(cert, compass.N) for cert in certificates]
    print(f"\n✓ Converted {K} certificates to lattice vectors")

    # 4. Accumulate with timing
    print(f"\n{'='*60}")
    print(f"ACCUMULATE (K={K} members)")
    print(f"{'='*60}")
    start = time.time()
    Acc, Z, L = compass.Accumulate(F)
    time_accumulate = time.time() - start

    print(f"\n⏱️  Total Accumulate time: {time_accumulate:.3f}s ({time_accumulate*1000:.1f}ms)")
    print(f"   Average per member: {time_accumulate/K:.3f}s ({time_accumulate*1000/K:.1f}ms)")

    # 5. Generate witnesses (just show size for first few)
    print(f"\n{'='*60}")
    print(f"WITNESS GENERATION")
    print(f"{'='*60}")
    for i in range(min(5, K)):
        w_packed = compass.Witness(Z, L, i)
        print(f"  Witness {i}: {len(w_packed.data)} bytes")
    if K > 5:
        print(f"  ... ({K-5} more witnesses)")

    # 6. Verify with timing
    print(f"\n{'='*60}")
    print(f"VERIFICATION (K={K} members)")
    print(f"{'='*60}")

    verification_times = []
    all_valid = True

    for i in range(K):
        w_packed = compass.Witness(Z, L, i)

        start = time.time()
        valid = compass.Verify(Acc, F[i], w_packed)
        verify_time = time.time() - start
        verification_times.append(verify_time)

        status = "✓ VALID" if valid else "✗ INVALID"
        if i < 5 or not valid:  # Show first 5 and any failures
            print(f"  Member {i}: {status} ({verify_time*1000:.2f}ms)")

        if not valid:
            print(f"    ERROR: Verification failed!")
            all_valid = False

    if K > 5 and all_valid:
        print(f"  ... ({K-5} more members verified)")

    # Verification statistics
    avg_verify = np.mean(verification_times)
    min_verify = np.min(verification_times)
    max_verify = np.max(verification_times)

    print(f"\n⏱️  Verification times:")
    print(f"   Average: {avg_verify*1000:.2f}ms")
    print(f"   Min: {min_verify*1000:.2f}ms")
    print(f"   Max: {max_verify*1000:.2f}ms")
    print(f"   Total: {sum(verification_times):.3f}s")

    if not all_valid:
        return False

    # 7. Negative test
    print(f"\n{'='*60}")
    print(f"NEGATIVE TEST (Forgery Attempt)")
    print(f"{'='*60}")
    fake_cert = b"attacker@evil.com"
    fake_f = HashToLat(fake_cert, compass.N)
    w0_packed = compass.Witness(Z, L, 0)

    start = time.time()
    forged = compass.Verify(Acc, fake_f, w0_packed)
    forgery_time = time.time() - start

    if forged:
        print(f"  ✗ ERROR: Forgery succeeded!")
        return False
    else:
        print(f"  ✓ Forgery rejected ({forgery_time*1000:.2f}ms)")

    # Final summary
    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Parameter set: {compass.N=}, {compass.q=}, {compass.kappa=}, {compass.t=}")
    print(f"Members (K): {K}")
    print(f"")
    print(f"Accumulate:")
    print(f"  Total time: {time_accumulate:.3f}s")
    print(f"  Per member: {time_accumulate*1000/K:.1f}ms")
    print(f"")
    print(f"Witness size: {len(compass.Witness(Z, L, 0).data)} bytes")
    print(f"")
    print(f"Verify:")
    print(f"  Average time: {avg_verify*1000:.2f}ms")
    print(f"  Throughput: {1/avg_verify:.1f} verifications/sec")
    print(f"")
    print(f"Accumulator size: {len(Acc)} elements × {bytes_per_q(compass.q)} bytes = {len(Acc) * bytes_per_q(compass.q)} bytes")

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED! ✓")
    print(f"{'='*60}")
    return True

if __name__ == "__main__":
    ok = test_compass()
    raise SystemExit(0 if ok else 1)