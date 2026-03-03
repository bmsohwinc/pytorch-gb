#!/usr/bin/env python3
"""
Microbenchmark: mul_out(dst, src, scalar) vs src.mul_(scalar)
=============================================================
Simulates the two code paths in mark_variable_ready_dense:
  gb=0: at::mul_out(bucket_view, grad, 1/N) — read from grad, write to bucket_view
  gb=1: bucket_view.mul_(1/N)              — in-place on bucket_view (grad is aliased)

Also tests the "strided view" variant: where bucket_view is a .narrow() view 
into a larger contiguous allocation (like the real DDP bucket).
"""
import torch
import time
import argparse


def bench_op(label, op_fn, warmup=50, iters=200):
    """Run op_fn with CUDA sync timing."""
    for _ in range(warmup):
        op_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        op_fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us
    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    return avg, std


def run_test(num_elements, num_params=1, device="cuda:0"):
    """
    Simulate DDP with num_params parameters, each of size num_elements/num_params.
    Compare the total time across all params for:
      1. mul_out(view, separate_grad, scalar) — the gb=0 path
      2. view.mul_(scalar) — the gb=1 alias path
      3. Just mul_out but src=dst (reference for in-place via mul_out)
    """
    scalar = 1.0 / 8  # typical div_factor
    wrapped = torch.tensor(scalar, device=device)
    
    # Elements per param
    per_param = num_elements // num_params
    
    # === Setup for gb=0 path ===
    # Bucket is one contiguous allocation; grads are separate
    bucket_gb0 = torch.randn(num_elements, device=device)
    grads_gb0 = [torch.randn(per_param, device=device) for _ in range(num_params)]
    views_gb0 = [bucket_gb0.narrow(0, i * per_param, per_param) for i in range(num_params)]
    
    # === Setup for gb=1 path ===
    # Bucket is one contiguous allocation; grads are views INTO the bucket
    bucket_gb1 = torch.randn(num_elements, device=device)
    views_gb1 = [bucket_gb1.narrow(0, i * per_param, per_param) for i in range(num_params)]
    # In gb=1, grad IS the view, so we just mul_ the view
    
    # === Setup for standalone tensors (control) ===  
    standalone = torch.randn(num_elements, device=device)
    standalone2 = torch.randn(num_elements, device=device)

    def gb0_path():
        """Simulate gb=0: mul_out from separate grads into bucket views"""
        for g, v in zip(grads_gb0, views_gb0):
            torch.mul(g, wrapped, out=v)

    def gb1_path():
        """Simulate gb=1: in-place mul_ on bucket views"""
        for v in views_gb1:
            v.mul_(scalar)

    def whole_mul_out():
        """Control: mul_out on whole contiguous tensor"""
        torch.mul(standalone, wrapped, out=standalone2)

    def whole_mul_inplace():
        """Control: in-place mul_ on whole contiguous tensor"""
        standalone.mul_(scalar)

    print(f"\n{'='*60}")
    print(f"  num_elements={num_elements:,}  num_params={num_params}")
    print(f"  per_param={per_param:,}")
    print(f"{'='*60}")

    avg, std = bench_op("gb=0 (mul_out per param)", gb0_path)
    print(f"  gb=0  mul_out(view, grad, s)    : {avg:8.1f} ± {std:6.1f} μs")

    avg1, std1 = bench_op("gb=1 (mul_ per param)", gb1_path)
    print(f"  gb=1  view.mul_(s)              : {avg1:8.1f} ± {std1:6.1f} μs")

    pct = (avg - avg1) / avg * 100 if avg else 0
    print(f"  improvement (gb1 vs gb0)        : {pct:+.1f}%")

    avg2, std2 = bench_op("whole mul_out", whole_mul_out)
    print(f"  whole mul_out(dst, src, s)      : {avg2:8.1f} ± {std2:6.1f} μs")

    avg3, std3 = bench_op("whole mul_ inplace", whole_mul_inplace)
    print(f"  whole mul_(s)                   : {avg3:8.1f} ± {std3:6.1f} μs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("Microbenchmark: mul_out vs mul_ (DDP forward copy simulation)")
    print(f"Device: {args.device}")
    print(f"GPU: {torch.cuda.get_device_name(args.device)}")

    # Test with various sizes matching your sweep
    for expo in range(20, 29):
        n = 2 ** expo
        # model_params in your MLP: use realistic num_params count
        # For a big MLP with width=512, each layer has ~262K params
        # So for 2^27, you'd have ~512 parameters
        num_params = max(1, n // (512 * 512))
        run_test(n, num_params=num_params, device=args.device)

    # Also test with num_params=1 to isolate kernel-level difference
    print("\n\n=== SINGLE PARAM (isolate kernel perf) ===")
    for expo in [24, 26, 28]:
        n = 2 ** expo
        run_test(n, num_params=1, device=args.device)


if __name__ == "__main__":
    main()
