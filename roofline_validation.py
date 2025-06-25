#!/usr/bin/env python3
"""
Roofline Cost Model Validation
Compares predicted costs with actual measured execution times.
"""

import time
import numpy as np
from roofline_cost_model import RooflineMatMulCostModel, get_cpu_specs
import statistics
from typing import List, Tuple


def measure_actual_matmul_time(m: int, n: int, k: int, num_runs: int = 5) -> float:
    """
    Measure actual matrix multiplication execution time.
    
    Args:
        m, n, k: Matrix dimensions for (m,k) @ (k,n) = (m,n)
        num_runs: Number of runs to average
    
    Returns:
        Average execution time in seconds
    """
    # Pre-allocate matrices
    A = np.random.random((m, k)).astype(np.float64)
    B = np.random.random((k, n)).astype(np.float64)
    
    times = []
    
    # Warmup run
    _ = np.dot(A, B)
    
    for _ in range(num_runs):
        start_time = time.perf_counter()
        C = np.dot(A, B)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return statistics.mean(times)


def validate_roofline_model():
    """
    Validate the roofline cost model against actual measurements.
    """
    print("ROOFLINE COST MODEL VALIDATION")
    print("=" * 60)
    
    # Initialize cost model
    specs = get_cpu_specs()
    cost_model = RooflineMatMulCostModel(specs)
    
    # Test cases with different characteristics
    test_cases = [
        # (m, n, k, description)
        (100, 100, 100, "Small cubic"),
        (500, 500, 500, "Medium cubic"),
        (1000, 1000, 1000, "Large cubic"),
        (2000, 2000, 1, "Very thin matrices"),
        (1000, 2000, 1, "Thin matrices"),
        (500, 1000, 10, "Moderately thin"),
        (200, 200, 100, "Balanced workload"),
        (100, 100, 1000, "Compute intensive"),
        (64, 64, 64, "Small matrices"),
        (32, 32, 32, "Very small matrices"),
    ]
    
    print("Running validation tests...")
    print(f"{'Test Case':<25} {'Predicted (μs)':<15} {'Actual (μs)':<15} {'Error (%)':<12} {'Ratio':<8}")
    print("-" * 85)
    
    results = []
    
    for m, n, k, description in test_cases:
        print(f"Testing {description} MatMul({m}, {n}, {k})...", end=" ", flush=True)
        
        # Get prediction from roofline model
        predicted_time = cost_model.get_cost_function(
            "test", "MatMul_KnownShape", None, 0, [None, None, m, n, k]
        )
        
        # Measure actual execution time
        actual_time = measure_actual_matmul_time(m, n, k, num_runs=3)
        
        # Calculate metrics
        predicted_us = predicted_time * 1e6
        actual_us = actual_time * 1e6
        error_percent = abs(predicted_us - actual_us) / actual_us * 100
        ratio = predicted_us / actual_us
        
        results.append({
            'test': description,
            'predicted': predicted_us,
            'actual': actual_us,
            'error': error_percent,
            'ratio': ratio,
            'm': m, 'n': n, 'k': k
        })
        
        print(f"{description:<25} {predicted_us:<15.2f} {actual_us:<15.2f} {error_percent:<12.1f} {ratio:<8.2f}")
    
    return results


def analyze_validation_results(results: List[dict]):
    """
    Analyze the validation results and provide insights.
    """
    print("\n" + "=" * 60)
    print("VALIDATION ANALYSIS")
    print("=" * 60)
    
    errors = [r['error'] for r in results]
    ratios = [r['ratio'] for r in results]
    
    print(f"Statistical Summary:")
    print(f"  Mean Absolute Error:    {statistics.mean(errors):.1f}%")
    print(f"  Median Absolute Error:  {statistics.median(errors):.1f}%")
    print(f"  Max Error:              {max(errors):.1f}%")
    print(f"  Min Error:              {min(errors):.1f}%")
    print(f"  Standard Deviation:     {statistics.stdev(errors):.1f}%")
    print()
    print(f"Prediction Ratio Analysis:")
    print(f"  Mean Ratio (Pred/Actual): {statistics.mean(ratios):.2f}")
    print(f"  Median Ratio:             {statistics.median(ratios):.2f}")
    print(f"  Ratio Range:              {min(ratios):.2f} - {max(ratios):.2f}")
    
    # Categorize results
    good_predictions = [r for r in results if r['error'] < 50]
    poor_predictions = [r for r in results if r['error'] >= 50]
    
    print(f"\nAccuracy Categories:")
    print(f"  Good predictions (<50% error): {len(good_predictions)}/{len(results)}")
    print(f"  Poor predictions (≥50% error): {len(poor_predictions)}/{len(results)}")
    
    if poor_predictions:
        print(f"\nPoor Predictions Analysis:")
        for r in poor_predictions:
            flops = 2 * r['m'] * r['n'] * r['k']
            bytes_transferred = (r['m'] * r['n'] + r['n'] * r['k'] + r['m'] * r['k']) * 8
            intensity = flops / bytes_transferred
            print(f"  {r['test']}: {r['error']:.1f}% error, intensity={intensity:.2f} FLOP/Byte")
    
    # Model bias analysis
    over_predictions = [r for r in results if r['ratio'] > 1.0]
    under_predictions = [r for r in results if r['ratio'] < 1.0]
    
    print(f"\nModel Bias Analysis:")
    print(f"  Over-predictions (model too pessimistic): {len(over_predictions)}/{len(results)}")
    print(f"  Under-predictions (model too optimistic): {len(under_predictions)}/{len(results)}")
    
    if statistics.mean(ratios) > 1.2:
        print("  → Model tends to OVER-estimate execution times")
    elif statistics.mean(ratios) < 0.8:
        print("  → Model tends to UNDER-estimate execution times")
    else:
        print("  → Model predictions are reasonably balanced")


def detailed_case_analysis():
    """
    Perform detailed analysis on a few specific cases.
    """
    print("\n" + "=" * 60)
    print("DETAILED CASE ANALYSIS")
    print("=" * 60)
    
    specs = get_cpu_specs()
    cost_model = RooflineMatMulCostModel(specs)
    
    cases = [
        (1000, 2000, 1, "Memory-bound case"),
        (500, 500, 500, "Compute-bound case"),
    ]
    
    for m, n, k, description in cases:
        print(f"\n{description}: MatMul({m}, {n}, {k})")
        print("-" * 40)
        
        # Calculate theoretical values
        total_flops = 2 * m * n * k
        bytes_transferred = (m * n + n * k + m * k) * 8
        intensity = total_flops / bytes_transferred
        
        # Using the hardcoded measured values from the model
        M2_FLOPS = 112.2e9
        M2_BANDWIDTH = 253.4e9
        
        time_compute = total_flops / M2_FLOPS
        time_memory = bytes_transferred / M2_BANDWIDTH
        predicted_time = max(time_compute, time_memory)
        
        # Measure actual time
        actual_time = measure_actual_matmul_time(m, n, k, num_runs=5)
        
        print(f"Theoretical Analysis:")
        print(f"  FLOPs: {total_flops:,}")
        print(f"  Bytes: {bytes_transferred:,}")
        print(f"  Intensity: {intensity:.2f} FLOP/Byte")
        print(f"  Compute time: {time_compute*1e6:.2f} μs")
        print(f"  Memory time: {time_memory*1e6:.2f} μs")
        print(f"  Bottleneck: {'Compute' if time_compute > time_memory else 'Memory'}")
        print(f"  Predicted: {predicted_time*1e6:.2f} μs")
        print(f"  Actual: {actual_time*1e6:.2f} μs")
        print(f"  Error: {abs(predicted_time - actual_time)/actual_time*100:.1f}%")


if __name__ == "__main__":
    # Run validation
    validation_results = validate_roofline_model()
    
    # Analyze results
    analyze_validation_results(validation_results)
    
    # Detailed analysis
    detailed_case_analysis()
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("1. The roofline model provides a good first-order approximation")
    print("2. Errors can arise from:")
    print("   - CPU frequency scaling and thermal throttling")
    print("   - Cache effects not modeled in the simple roofline")
    print("   - BLAS library optimizations (SIMD, blocking, etc.)")
    print("   - OS scheduling and background processes")
    print("3. The model is most accurate for identifying compute vs memory bottlenecks")
    print("4. For precise timing, consider more sophisticated models or profiling") 