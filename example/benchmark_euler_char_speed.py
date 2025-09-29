import numpy as np
import time
import os
import gc

from tools21cm import topology

def create_test_data(box_dim=128):
    """Creates a reasonably complex 3D binary array for testing."""
    print(f"Generating a {box_dim}x{box_dim}x{box_dim} test data cube...")
    arr = np.zeros((box_dim, box_dim, box_dim), dtype=np.int32)
    
    # A solid core
    size = box_dim // 4
    start = box_dim // 2 - size // 2
    end = start + size
    arr[start:end, start:end, start:end] = 1
    
    # Random noise to make it less trivial
    noise = np.random.randint(0, 200, size=arr.shape)
    arr[noise > 198] = 1
    print(f"Test data generated with {arr.sum()} active cells.\n")
    return arr

if __name__ == "__main__":
    # --- Setup ---
    BOX_DIM = 32 #128
    test_data = create_test_data(BOX_DIM)
    timings = {}

    # Define the backends to test
    backends_to_test = ['python', 'numba', 'cython', 'torch']
    chi_results = {}
    for backend in backends_to_test:
        print(f"--- Benchmarking '{backend}' Backend ---")
        
        # Check if the backend is available
        available = False
        if backend == 'python': available = True
        elif backend == 'numba' and topology.VB.numba_available: available = True
        elif backend == 'cython' and topology.VB.cython_available: available = True
        elif backend == 'torch' and topology.VB.torch_available: available = True

        if not available:
            print(f"   Backend not available. Skipping.\n")
            continue

        # For parallel Cython, set thread count to max
        if backend == 'cython':
            n_threads = os.cpu_count()
            os.environ['OMP_NUM_THREADS'] = str(n_threads)
            print(f"   (Using {n_threads} threads for Cython/OpenMP)")

        # Perform a warm-up run for JIT or GPU backends
        if backend in ['numba', 'torch']:
            print("   (Warm-up run...)")
            topology.EulerCharacteristic(test_data, speed_up=backend, verbose=False)

        # Run the actual benchmark
        print("   (Benchmarking run...)")
        t_start = time.time()
        chi_value = topology.EulerCharacteristic(test_data, speed_up=backend, verbose=True)
        t_end = time.time()
        
        duration = t_end - t_start
        timings[backend] = duration
        
        print(f"   Result Chi = {chi_value}, Time = {duration:.4f} seconds\n")
        chi_results[backend] = chi_value
        gc.collect()

    # Clean up environment variable
    if 'OMP_NUM_THREADS' in os.environ:
        del os.environ['OMP_NUM_THREADS']
        
    # --- Final Summary ---
    print("="*65)
    print("                    Backend Benchmark Summary")
    print("="*65)
    if timings:
        sorted_results = sorted(timings.items(), key=lambda item: item[1])
        baseline_time = timings.get('python', 1e-9)
        
        # Add the 'Chi Value' column to the header
        print(f"{'Implementation':<20} | {'Chi Value':<12} | {'Time (s)':<15} | {'Speedup'}")
        print("-"*65)
        
        for name, t in sorted_results:
            chi_val = chi_results.get(name, 'N/A') 
            speedup = baseline_time / t
            print(f"{name:<20} | {chi_val:<12.0f} | {t:<15.4f} | {speedup:.2f}x")
    else:
        print("No backends were benchmarked.")
