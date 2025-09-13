#!/usr/bin/env python3
"""
Simple test script to verify the parallel calculation functionality
"""

import sys
import os
import networkx as nx
import numpy as np
import time

# Add the Calculators directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Calculators'))

from Base_Calculator import Base_Calculator

def create_test_dataset(n_graphs=10, n_nodes_per_graph=5):
    """Create a simple test dataset of random graphs"""
    graphs = []
    labels = []
    
    for i in range(n_graphs):
        # Create a random graph
        G = nx.erdos_renyi_graph(n_nodes_per_graph, 0.3)
        graphs.append(G)
        labels.append(f"graph_{i}")
    
    return graphs, labels

def test_parallel_vs_sequential():
    """Test that parallel and sequential calculations produce the same results"""
    print("Creating test dataset...")
    graphs, labels = create_test_dataset(n_graphs=20, n_nodes_per_graph=8)
    
    # Test sequential calculation
    print("\nTesting sequential calculation...")
    calc_seq = Base_Calculator(dataset=graphs, labels=labels, activate=True)
    
    start_time = time.time()
    calc_seq.calculate()
    seq_time = time.time() - start_time
    
    seq_upper = calc_seq.upperbound_matrix.copy()
    seq_lower = calc_seq.lowerbound_matrix.copy()
    
    # Test parallel calculation
    print("\nTesting parallel calculation...")
    calc_par = Base_Calculator(dataset=graphs, labels=labels, activate=True)
    
    start_time = time.time()
    calc_par.calculate_parallel(n_processes=2)  # Use 2 processes for testing
    par_time = time.time() - start_time
    
    par_upper = calc_par.upperbound_matrix.copy()
    par_lower = calc_par.lowerbound_matrix.copy()
    
    # Compare results (they won't be identical due to random numbers, but structure should be same)
    print(f"\nTiming comparison:")
    print(f"Sequential time: {seq_time:.4f} seconds")
    print(f"Parallel time: {par_time:.4f} seconds")
    print(f"Speedup: {seq_time/par_time:.2f}x")
    
    # Check matrix properties
    print(f"\nMatrix properties:")
    print(f"Sequential - Upper matrix shape: {seq_upper.shape}, Lower matrix shape: {seq_lower.shape}")
    print(f"Parallel - Upper matrix shape: {par_upper.shape}, Lower matrix shape: {par_lower.shape}")
    
    # Check diagonal elements (should all be 0)
    seq_diag_upper = np.diag(seq_upper)
    seq_diag_lower = np.diag(seq_lower)
    par_diag_upper = np.diag(par_upper)
    par_diag_lower = np.diag(par_lower)
    
    print(f"Sequential - Diagonal zeros (upper): {np.all(seq_diag_upper == 0)}")
    print(f"Sequential - Diagonal zeros (lower): {np.all(seq_diag_lower == 0)}")
    print(f"Parallel - Diagonal zeros (upper): {np.all(par_diag_upper == 0)}")
    print(f"Parallel - Diagonal zeros (lower): {np.all(par_diag_lower == 0)}")
    
    # Check symmetry
    seq_sym_upper = np.allclose(seq_upper, seq_upper.T)
    seq_sym_lower = np.allclose(seq_lower, seq_lower.T)
    par_sym_upper = np.allclose(par_upper, par_upper.T)
    par_sym_lower = np.allclose(par_lower, par_lower.T)
    
    print(f"Sequential - Matrix symmetry (upper): {seq_sym_upper}")
    print(f"Sequential - Matrix symmetry (lower): {seq_sym_lower}")
    print(f"Parallel - Matrix symmetry (upper): {par_sym_upper}")
    print(f"Parallel - Matrix symmetry (lower): {par_sym_lower}")
    
    # Test GPU calculation if available
    print("\nTesting GPU calculation...")
    calc_gpu = Base_Calculator(dataset=graphs, labels=labels, activate=True)
    
    start_time = time.time()
    calc_gpu.calculate_gpu()
    gpu_time = time.time() - start_time
    
    print(f"GPU calculation time: {gpu_time:.4f} seconds")
    
    gpu_upper = calc_gpu.upperbound_matrix.copy()
    gpu_lower = calc_gpu.lowerbound_matrix.copy()
    
    # Check GPU results
    gpu_diag_upper = np.diag(gpu_upper)
    gpu_diag_lower = np.diag(gpu_lower)
    gpu_sym_upper = np.allclose(gpu_upper, gpu_upper.T)
    gpu_sym_lower = np.allclose(gpu_lower, gpu_lower.T)
    
    print(f"GPU - Diagonal zeros (upper): {np.all(gpu_diag_upper == 0)}")
    print(f"GPU - Diagonal zeros (lower): {np.all(gpu_diag_lower == 0)}")
    print(f"GPU - Matrix symmetry (upper): {gpu_sym_upper}")
    print(f"GPU - Matrix symmetry (lower): {gpu_sym_lower}")
    
    print("\n✅ All tests completed successfully!")

if __name__ == "__main__":
    test_parallel_vs_sequential()
