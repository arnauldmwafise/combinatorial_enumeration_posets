import time
import json
import numpy as np
from poset_operad import PosetOperad

def generate_random_lower_triangular(dim: int) -> np.ndarray:
    """Generates a valid random lower triangular poset matrix with an active diagonal."""
    mat = np.random.randint(0, 2, size=(dim, dim)).astype('int64')
    lower_mat = np.tril(mat, k=-1)
    np.fill_diagonal(lower_mat, 1)
    return lower_mat

def execute_scaling_benchmark():
    # Structural scale points to measure (Matrix Dimensions p x p)
    test_dimensions = [4, 8, 16, 32, 64]
    # Fixed size for the incoming substitution poset (q x q)
    q_dim = 8 
    iterations = 50
    
    variants = ["general", "boundary", "total", "fully_isolated"]
    log_file = "benchmark_results.txt"
    
    with open(log_file, "w") as f:
        header = f"{'Dimension (p)':<15}{'Variant':<18}{'Total Time (s)':<18}{'Avg Time/Comp (ms)':<20}\n"
        f.write("========================================================================\n")
        f.write("      ALGEBRAIC POSET OPERAD LOWER-TRIANGULAR PIPELINE BENCHMARK        \n")
        f.write("========================================================================\n")
        f.write(header)
        f.write("------------------------------------------------------------------------\n")
        print("Starting performance benchmarking metrics scaling sequence...")

        for p_dim in test_dimensions:
            # Prepare randomized test beds
            l_mat = generate_random_lower_triangular(p_dim)
            r_mat = generate_random_lower_triangular(q_dim)
            
            l_sig = PosetOperad.matrix_to_bits(l_mat)
            r_sig = PosetOperad.matrix_to_bits(r_mat)
            
            for var in variants:
                start_time = time.perf_counter()
                
                # Execute over steady-state iterations to average noise
                for _ in range(iterations):
                    _ = PosetOperad.generate_compositions_bitpacked(
                        l_sig, r_sig, p_dim, q_dim, comp_type=var
                    )
                    
                end_time = time.perf_counter()
                total_duration = end_time - start_time
                
                # Calculate metric scales (Total compositions calculated = iterations * p_dim)
                total_compositions = iterations * p_dim
                avg_time_ms = (total_duration / total_compositions) * 1000
                
                log_line = f"{p_dim:<15}{var:<18}{total_duration:<18.5f}{avg_time_ms:<20.5f}\n"
                f.write(log_line)
                print(f"Completed Dimension {p_dim} [{var}] - Throughput Avg: {avg_time_ms:.4f} ms")
                
        f.write("========================================================================\n")
        print(f"\nBenchmark completed successfully! Logs saved to -> {log_file}")

if __name__ == "__main__":
    execute_scaling_benchmark()
