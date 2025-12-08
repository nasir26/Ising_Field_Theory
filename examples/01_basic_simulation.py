#!/usr/bin/env python3
"""
Example 1: Basic Ising Field Theory Simulation

This example demonstrates:
1. Setting up a quantum Ising chain simulation
2. Computing the ground state and energy spectrum
3. Time evolution of initial states
4. Measuring observables
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from ising_field_theory import (
    IsingHamiltonian,
    IsingFieldTheorySimulation,
    E8MassRatios
)
from ising_field_theory.simulation import SimulationParameters


def main():
    print("=" * 70)
    print("Ising Field Theory - Basic Simulation Example")
    print("=" * 70)
    
    # ==========================================================================
    # Part 1: Hamiltonian Construction and Diagonalization
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 1: Hamiltonian Construction")
    print("-" * 70)
    
    # Create Ising Hamiltonian at the critical point
    # H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ
    ham = IsingHamiltonian(
        n_sites=8,
        J=1.0,
        h_transverse=1.0,  # g = hₓ/J = 1 (critical point)
        h_longitudinal=0.0,
        boundary='open'
    )
    
    print(f"Number of sites: {ham.n_sites}")
    print(f"Hilbert space dimension: {ham.dim}")
    print(f"Critical point (g=1): {ham.is_critical}")
    
    # Diagonalize to find eigenspectrum
    eigenvalues, eigenvectors = ham.diagonalize(n_states=10)
    
    print(f"\nEnergy eigenvalues (first 10):")
    for i, E in enumerate(eigenvalues):
        print(f"  E_{i} = {E:.6f}")
    
    # Compute mass gaps
    gaps = ham.compute_mass_gaps()
    print(f"\nMass gaps (Δₙ = Eₙ - E₀):")
    for i, gap in enumerate(gaps[:5], 1):
        print(f"  Δ_{i} = {gap:.6f}")
    
    # ==========================================================================
    # Part 2: Full Simulation Setup
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 2: Full Simulation")
    print("-" * 70)
    
    # Set up simulation parameters
    params = SimulationParameters(
        n_sites=10,
        J=1.0,
        h_transverse=1.0,
        h_longitudinal=0.0,
        boundary='open',
        total_time=5.0,
        n_trotter_steps=50,
        trotter_order=2
    )
    
    sim = IsingFieldTheorySimulation(params)
    print(sim.summary())
    
    # ==========================================================================
    # Part 3: Ground State Analysis
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 3: Ground State Analysis")
    print("-" * 70)
    
    # Prepare ground state
    ground_state = sim.prepare_ground_state(method='exact')
    
    # Compute observables
    E_ground = sim.compute_energy(ground_state)
    M_ground = sim.compute_magnetization(ground_state)
    
    print(f"Ground state energy: {E_ground:.6f}")
    print(f"Ground state magnetization: {M_ground:.6f}")
    
    # Correlation function
    correlations = sim.compute_correlation_function(ground_state, operator='X')
    print(f"\nSpin-spin correlation C(r) = ⟨σ₀ˣσᵣˣ⟩:")
    for r in range(min(6, len(correlations))):
        print(f"  C({r}) = {correlations[r]:.6f}")
    
    # Correlation length
    xi = sim.compute_correlation_length(ground_state)
    print(f"\nCorrelation length ξ = {xi:.4f}")
    
    # Entanglement entropy
    S_ent = sim.compute_entanglement_entropy(ground_state)
    print(f"Entanglement entropy (half-chain): {S_ent:.4f}")
    
    # ==========================================================================
    # Part 4: Time Evolution
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 4: Time Evolution")
    print("-" * 70)
    
    # Prepare initial state: domain wall
    initial_state = sim.prepare_domain_wall_state(position=5)
    print(f"Initial state: domain wall at position 5")
    
    # Time evolution
    results = sim.time_evolve(
        initial_state,
        total_time=3.0,
        n_time_points=30,
        method='exact',
        compute_observables=True
    )
    
    print(f"\nTime evolution results:")
    print(f"  Time points: {len(results.times)}")
    print(f"  Energy conservation: ΔE/E = {np.std(results.energies)/np.abs(np.mean(results.energies)):.2e}")
    print(f"  Magnetization range: [{results.magnetization.min():.4f}, {results.magnetization.max():.4f}]")
    
    # ==========================================================================
    # Part 5: Quantum Circuit Statistics
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 5: Quantum Circuit for Time Evolution")
    print("-" * 70)
    
    stats = sim.get_circuit_stats(total_time=1.0, n_steps=20)
    print(f"Circuit statistics for t=1.0, n_steps=20:")
    print(f"  Number of qubits: {stats['n_qubits']}")
    print(f"  Circuit depth: {stats['depth']}")
    print(f"  Total gates: {stats['total_gates']}")
    print(f"  Gate breakdown: {stats['gate_counts']}")
    print(f"  Estimated Trotter error: {stats['estimated_error']:.4e}")
    
    # ==========================================================================
    # Part 6: Comparison of Exact vs Trotter Evolution
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 6: Trotter Error Analysis")
    print("-" * 70)
    
    test_state = sim.prepare_single_particle_state(position=5)
    
    for n_steps in [5, 10, 20, 40]:
        error = sim.trotter_error(test_state, time=1.0)
        params.n_trotter_steps = n_steps
        sim_temp = IsingFieldTheorySimulation(params)
        error = sim_temp.trotter_error(test_state, time=1.0)
        print(f"  n_steps = {n_steps:3d}: ||exact - trotter|| = {error:.6e}")
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
