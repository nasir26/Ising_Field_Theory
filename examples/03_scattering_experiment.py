#!/usr/bin/env python3
"""
Example 3: Scattering Experiment on Quantum Computer

This example demonstrates:
1. Setting up a two-particle scattering experiment
2. Preparing wavepacket initial states
3. Time evolution through collision
4. Analyzing particle production and S-matrix elements
5. Verifying integrability at the E₈ point

"The Ising field theory is probably the simplest interesting situation 
where one can study scattering on a quantum computer and ask questions 
regarding particle production."
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from ising_field_theory import (
    IsingFieldTheorySimulation,
    ScatteringAnalysis
)
from ising_field_theory.simulation import SimulationParameters
from ising_field_theory.scattering import ScatteringSetup, create_scattering_experiment


def main():
    print("=" * 70)
    print("Ising Field Theory - Scattering Experiment")
    print("=" * 70)
    
    # ==========================================================================
    # Part 1: Setting Up the Scattering Experiment
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 1: Scattering Setup")
    print("-" * 70)
    
    # Create simulation at E₈ point (integrable)
    n_sites = 16
    
    params = SimulationParameters(
        n_sites=n_sites,
        J=1.0,
        h_transverse=1.0,       # Critical point
        h_longitudinal=0.1,     # E₈ perturbation
        boundary='open',
        n_trotter_steps=50,
        trotter_order=2
    )
    
    sim = IsingFieldTheorySimulation(params)
    analysis = ScatteringAnalysis(sim)
    
    # Configure scattering
    setup = ScatteringSetup(
        n_sites=n_sites,
        initial_separation=8,           # Particles start 8 sites apart
        momenta=(0.5, -0.5),            # Counter-propagating
        wavepacket_width=2.0            # Gaussian width
    )
    
    print(f"System size: {n_sites} sites")
    print(f"Initial separation: {setup.initial_separation} sites")
    print(f"Momenta: k₁ = {setup.momenta[0]}, k₂ = {setup.momenta[1]}")
    print(f"Wavepacket width: σ = {setup.wavepacket_width}")
    print(f"Estimated collision time: t_c ≈ {setup.collision_time:.2f}")
    print(f"Total simulation time: t = {setup.total_time:.2f}")
    
    # ==========================================================================
    # Part 2: Running the Scattering Simulation
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 2: Running Scattering Simulation")
    print("-" * 70)
    
    print("Evolving two-particle state through collision...")
    results = analysis.run_scattering(setup, n_time_points=50, method='exact')
    
    print(f"\nSimulation completed:")
    print(f"  Time points: {len(results.times)}")
    print(f"  Initial particle number: {results.particle_numbers[0]:.4f}")
    print(f"  Final particle number: {results.particle_numbers[-1]:.4f}")
    
    # ==========================================================================
    # Part 3: Particle Production Analysis
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 3: Particle Production Analysis")
    print("-" * 70)
    
    production = analysis.analyze_particle_production(results)
    
    print(f"\nParticle production results:")
    print(f"  Initial particles: {production['initial_particles']:.4f}")
    print(f"  Final particles: {production['final_particles']:.4f}")
    print(f"  Change in particle number: {production['particle_change']:.4f}")
    print(f"  Is elastic (no production): {production['is_elastic']}")
    print(f"  Particle number variance: {production['particle_number_variance']:.6f}")
    
    # ==========================================================================
    # Part 4: Integrability Verification
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 4: Integrability Verification")
    print("-" * 70)
    
    integrability = analysis.verify_integrability(results)
    
    print(f"\nIntegrability check:")
    print(f"  Is integrable: {integrability['is_integrable']}")
    print(f"  Particle number mean: {integrability['particle_number_mean']:.4f}")
    print(f"  Particle number std: {integrability['particle_number_std']:.6f}")
    print(f"  Particle drift: {integrability['particle_number_drift']:.6f}")
    print(f"  Conservation quality: {integrability['conservation_quality']*100:.2f}%")
    
    # ==========================================================================
    # Part 5: S-matrix Extraction
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 5: S-matrix Element Extraction")
    print("-" * 70)
    
    # Extract S-matrix element from asymptotic states
    S_element = analysis.extract_s_matrix_element(
        results.states[0],   # Initial state (t = -∞)
        results.states[-1]   # Final state (t = +∞)
    )
    
    print(f"\nNumerical S-matrix element:")
    print(f"  S = {S_element}")
    print(f"  |S| = {abs(S_element):.6f}")
    print(f"  arg(S) = {np.angle(S_element):.6f} rad")
    
    # Compare with E₈ analytical result
    rapidity = 2 * abs(setup.momenta[0])  # Approximate rapidity
    S_analytical = analysis.compute_e8_s_matrix((1, 1), rapidity)
    
    print(f"\nE₈ analytical S₁₁(θ={rapidity:.2f}):")
    print(f"  S = {S_analytical}")
    print(f"  |S| = {abs(S_analytical):.6f}")
    
    # ==========================================================================
    # Part 6: Momentum Distribution Evolution
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 6: Momentum Distribution")
    print("-" * 70)
    
    print("\nMomentum distribution n(k) at different times:")
    print("-" * 50)
    
    time_indices = [0, len(results.times)//4, len(results.times)//2, -1]
    labels = ['t=0 (initial)', 't=T/4', 't=T/2 (collision)', 't=T (final)']
    
    for idx, label in zip(time_indices, labels):
        n_k = results.momentum_distribution[idx]
        k_peak = np.argmax(n_k)
        print(f"  {label}:")
        print(f"    Peak at k_index = {k_peak}, n(k_peak) = {n_k[k_peak]:.4f}")
    
    # ==========================================================================
    # Part 7: Comparison: Integrable vs Non-integrable
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 7: Integrable vs Non-integrable Comparison")
    print("-" * 70)
    
    print("\nComparing scattering at different points in phase space:\n")
    
    test_cases = [
        ("E₈ point (g=1, hᵤ=0.1)", 1.0, 0.1),
        ("Free fermion (g=1, hᵤ=0)", 1.0, 0.0),
        ("Away from criticality (g=0.8, hᵤ=0)", 0.8, 0.0),
        ("Non-integrable (g=0.9, hᵤ=0.1)", 0.9, 0.1),
    ]
    
    for name, g, h_z in test_cases:
        params = SimulationParameters(
            n_sites=12,
            J=1.0,
            h_transverse=g,
            h_longitudinal=h_z,
            boundary='open'
        )
        
        sim = IsingFieldTheorySimulation(params)
        analysis = ScatteringAnalysis(sim)
        
        setup = ScatteringSetup(
            n_sites=12,
            initial_separation=6,
            momenta=(0.4, -0.4),
            wavepacket_width=1.5
        )
        
        results = analysis.run_scattering(setup, n_time_points=30, method='exact')
        integ = analysis.verify_integrability(results)
        
        print(f"  {name}:")
        print(f"    Particle conservation: {integ['conservation_quality']*100:.1f}%")
        print(f"    Integrable: {integ['is_integrable']}\n")
    
    print("=" * 70)
    print("Scattering experiment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
