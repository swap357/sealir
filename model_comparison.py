#!/usr/bin/env python3
"""
A simple script to compare the accuracy of the Roofline model 
vs. a FLOPS-only model against actual measured times.
"""

import time
import numpy as np
import statistics

from roofline_cost_model import RooflineMatMulCostModel, get_cpu_specs as get_roofline_specs
from flops_only_cost_model import FLOPSOnlyMatMulCostModel, get_cpu_specs as get_flops_specs

def measure_actual_matmul_time(m: int, n: int, k: int, num_runs: int = 3) -> float:
    """Measures the actual execution time of a matrix multiplication."""
    A = np.random.random((m, k)).astype(np.float64)
    B = np.random.random((k, n)).astype(np.float64)
    
    # Warmup run to handle caching, etc.
    _ = np.dot(A, B)
    
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = np.dot(A, B)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return statistics.mean(times)

def compare_models_accuracy():
    """
    Compares the accuracy of the Roofline and FLOPS-only models against real-world measurements.
    """
    print("Model Accuracy Comparison: Roofline vs. FLOPS-Only vs. Actual")
    print("=" * 70)

    # 1. Initialize both models
    roofline_model = RooflineMatMulCostModel(get_roofline_specs())
    flops_model = FLOPSOnlyMatMulCostModel(get_flops_specs())

    # 2. Define test cases
    test_cases = [
        ("Memory-Bound (Thin)", 1000, 2000, 1),
        ("Compute-Bound (Cubic)", 500, 500, 500),
        ("Balanced", 200, 200, 100),
        ("Small", 100, 100, 100),
    ]

    # 3. Run comparison and print results
    print(f"{'Workload':<25} {'Actual (μs)':<15} {'Roofline Err %':<20} {'FLOPS Err %':<20}")
    print("-" * 80)

    for description, m, n, k in test_cases:
        # Measure actual execution time
        actual_time_sec = measure_actual_matmul_time(m, n, k)
        actual_time_us = actual_time_sec * 1e6

        # Get predicted costs from both models
        roofline_pred_us = roofline_model.get_cost_function(
            "node", "MatMul_KnownShape", "type", 0, [None, None, m, n, k]
        ) * 1e6

        flops_pred_us = flops_model.get_cost_function(
            "node", "MatMul_KnownShape", "type", 0, [None, None, m, n, k]
        ) * 1e6

        # Calculate percentage error
        roofline_error = abs(roofline_pred_us - actual_time_us) / actual_time_us * 100
        flops_error = abs(flops_pred_us - actual_time_us) / actual_time_us * 100
        
        print(f"{description:<25} {actual_time_us:<15.2f} {roofline_error:<20.1f} {flops_error:<20.1f}")

    print("-" * 80)

if __name__ == "__main__":
    compare_models_accuracy() 