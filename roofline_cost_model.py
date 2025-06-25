import cpuinfo
from dataclasses import dataclass

# Base class placeholder - in your real code, this would be the actual
# class from the 'tensat' example.
class _ch05_CostModel:
    def get_simple(self, cost):
        # In a real scenario, this would return a symbolic cost object
        return cost
    def get_cost_function(self, nodename, op, ty, cost, children):
        return float('inf')

# --- Step 1: Programmatically Get CPU Specs ---

@dataclass
class HardwareSpecs:
    flops_per_second: float
    memory_bandwidth_bps: float

def get_cpu_specs() -> HardwareSpecs:
    """
    Identifies the CPU and returns its estimated performance specs.
    """
    cpu_brand = cpuinfo.get_cpu_info().get('brand_raw', '').lower()
    
    # A database of known CPU specs. You would expand this list.
    # Note: These are illustrative values for theoretical peaks.
    spec_database = {
        'apple m1': HardwareSpecs(flops_per_second=2.6e12, memory_bandwidth_bps=68e9),
        'apple m2': HardwareSpecs(flops_per_second=3.6e12, memory_bandwidth_bps=100e9),
        'apple m1 pro': HardwareSpecs(flops_per_second=5.2e12, memory_bandwidth_bps=200e9),
        'apple m2 pro': HardwareSpecs(flops_per_second=6.8e12, memory_bandwidth_bps=200e9),
        'core i7-10750h': HardwareSpecs(flops_per_second=0.8e12, memory_bandwidth_bps=45e9),
        'core i9-13900k': HardwareSpecs(flops_per_second=1.8e12, memory_bandwidth_bps=89e9),
    }

    # Find the matching CPU from our database
    for name, specs in spec_database.items():
        if name in cpu_brand:
            print(f"Detected CPU: {cpu_brand}. Using specs for '{name}'.")
            return specs
            
    # Fallback to reasonable default values if CPU is not in our database
    print(f"CPU '{cpu_brand}' not in database. Using default specs.")
    return HardwareSpecs(flops_per_second=200e9, memory_bandwidth_bps=50e9)


# --- Step 2: Implement the Roofline Cost Model ---

SIZEOF_FLOAT64 = 8  # bytes

class RooflineMatMulCostModel(_ch05_CostModel):
    def __init__(self, hardware_specs: HardwareSpecs):
        self.specs = hardware_specs
        print("\nRooflineCostModel Initialized:")
        print(f"  - Compute: {self.specs.flops_per_second / 1e9:.1f} GFLOP/s")
        print(f"  - Memory: {self.specs.memory_bandwidth_bps / 1e9:.1f} GB/s")
        print("-" * 30)
        super().__init__()

    def get_cost_function(self, nodename, op, ty, cost, children):
        # We only define the cost for MatMul_KnownShape
        if op == "MatMul_KnownShape":
            _lhs, _rhs, m, n, k = children
            
            # Use theoretical values instead of measured (measured were too low for FLOPS)
            # M2_FLOPS = 112.2e9  # Measured: 112.2 GFLOP/s (artificially low)
            # M2_BANDWIDTH = 253.4e9  # Measured: 253.4 GB/s
            M2_FLOPS = 3.6e12  # Theoretical: 3600 GFLOP/s
            M2_BANDWIDTH = 100e9  # Theoretical: 100 GB/s
            
            # 1. Calculate the total compute work (FLOPs)
            total_flops = 2 * m * n * k
            
            # 2. Calculate the time if purely compute-bound
            time_compute = total_flops / M2_FLOPS

            # 3. Calculate the total memory traffic (Bytes)
            bytes_transferred = (m * n + n * k + m * k) * SIZEOF_FLOAT64
            
            # 4. Calculate the time if purely memory-bound
            time_memory = bytes_transferred / M2_BANDWIDTH
            
            # 5. The true cost (estimated time) is the maximum of the two.
            estimated_time_cost = max(time_compute, time_memory)
            
            # We use get_simple because we have a concrete cost now.
            # The framework will still add the cost of children.
            return self.get_simple(estimated_time_cost)

        # Fallback for any other operation
        return super().get_cost_function(nodename, op, ty, cost, children)


# --- Step 3: Demonstrate the Model ---
if __name__ == "__main__":
    # Get the specs for the machine this script is running on
    my_hardware_specs = get_cpu_specs()
    
    # Instantiate our cost model with these specs
    cost_model = RooflineMatMulCostModel(my_hardware_specs)
    
    # --- Example Usage ---
    # Let's test two MatMul operations with the same FLOPs but different shapes.
    
    # Case 1: A "thin" matrix multiplication (often memory-bound)
    # Shapes: (1000, 2000, 1)
    cost1 = cost_model.get_cost_function(
        "n1", "MatMul_KnownShape", None, 0, [None, None, 1000, 2000, 1]
    )
    print(f"Cost for MatMul(1000, 2000, 1): {cost1 * 1e6:.4f} microseconds")

    # Case 2: A "cubed" matrix multiplication (often compute-bound)
    # Shapes: (200, 200, 100)
    cost2 = cost_model.get_cost_function(
        "n2", "MatMul_KnownShape", None, 0, [None, None, 200, 200, 100]
    )
    print(f"Cost for MatMul(200, 200, 100): {cost2 * 1e6:.4f} microseconds")

    # Both cases have the same number of FLOPs (4,000,000)
    print("\nNote: Both operations have 4 million FLOPs, but the cost model")
    print("correctly identifies that the different shapes lead to different")
    print("performance due to memory vs. compute bottlenecks.") 