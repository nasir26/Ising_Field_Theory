#!/usr/bin/env python3
"""
Example 2: E₈ Integrable Field Theory Spectrum

This example demonstrates:
1. The E₈ integrable field theory at the critical point with longitudinal field
2. Computing mass ratios and comparing with exact E₈ predictions
3. Verification of integrability through spectral properties

The E₈ theory emerges when:
- Transverse field is at critical value: hₓ = J (g = 1)
- Small longitudinal field is applied: hᵤ ≠ 0

The spectrum contains 8 stable particles with masses related to the 
E₈ Lie algebra structure.
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
    print("E₈ Integrable Field Theory - Mass Spectrum Analysis")
    print("=" * 70)
    
    # ==========================================================================
    # Part 1: Exact E₈ Mass Ratios
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 1: Exact E₈ Mass Ratios (Analytical)")
    print("-" * 70)
    
    e8_ratios = E8MassRatios.compute_ratios()
    golden = E8MassRatios.golden_ratio_relations()
    
    print(f"\nGolden ratio φ = (1+√5)/2 = {golden['phi']:.6f}")
    print(f"\nE₈ mass ratios mₙ/m₁:")
    print("-" * 40)
    
    mass_formulas = [
        "m₁ = m (fundamental)",
        "m₂ = 2m cos(π/5) = φ·m",
        "m₃ = 2m cos(π/30)",
        "m₄ = 2m₂ cos(7π/30)",
        "m₅ = 2m₂ cos(2π/15)",
        "m₆ = 2m₂ cos(π/30)",
        "m₇ = 4m₂ cos(π/5)cos(7π/30)",
        "m₈ = 4m₂ cos(π/5)cos(2π/15)"
    ]
    
    for i, (ratio, formula) in enumerate(zip(e8_ratios, mass_formulas), 1):
        print(f"  m{i}/m₁ = {ratio:.6f}  ({formula})")
    
    # Verify golden ratio relationships
    print(f"\nGolden ratio checks:")
    print(f"  m₂/m₁ = {e8_ratios[1]:.6f} ≈ φ = {golden['phi']:.6f}")
    print(f"  m₃/m₁ = {e8_ratios[2]:.6f} ≈ φ² = {golden['phi']**2:.6f}")
    
    # ==========================================================================
    # Part 2: Numerical Spectrum vs E₈ Predictions
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 2: Numerical Spectrum Comparison")
    print("-" * 70)
    
    # Study convergence with system size
    print("\nConvergence study with system size:")
    print("-" * 60)
    print(f"{'N':>4} | {'m₂/m₁':>8} | {'m₃/m₁':>8} | {'m₄/m₁':>8} | {'Error':>8}")
    print("-" * 60)
    
    for n_sites in [8, 10, 12, 14]:
        params = SimulationParameters(
            n_sites=n_sites,
            J=1.0,
            h_transverse=1.0,       # Critical
            h_longitudinal=0.05,    # E₈ perturbation
            boundary='open'
        )
        
        sim = IsingFieldTheorySimulation(params)
        comparison = sim.compare_with_e8(n_states=6)
        
        computed = comparison['computed']
        errors = comparison['relative_error']
        avg_error = np.mean(errors[:3]) if len(errors) >= 3 else np.nan
        
        if len(computed) >= 3:
            print(f"{n_sites:>4} | {computed[0]:>8.4f} | {computed[1]:>8.4f} | {computed[2]:>8.4f} | {avg_error*100:>7.2f}%")
    
    # ==========================================================================
    # Part 3: Detailed Spectrum Analysis at Fixed Size
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 3: Detailed Spectrum (N=12 sites)")
    print("-" * 70)
    
    params = SimulationParameters(
        n_sites=12,
        J=1.0,
        h_transverse=1.0,
        h_longitudinal=0.05,
        boundary='open'
    )
    
    sim = IsingFieldTheorySimulation(params)
    comparison = sim.compare_with_e8(n_states=8)
    
    print(f"\n{'Particle':>10} | {'Computed':>10} | {'E₈ Exact':>10} | {'Rel Error':>10}")
    print("-" * 50)
    
    for i, (comp, exact, err) in enumerate(zip(
        comparison['computed'],
        comparison['e8_exact'],
        comparison['relative_error']
    ), 2):
        print(f"{'m'+str(i)+'/m₁':>10} | {comp:>10.4f} | {exact:>10.4f} | {err*100:>9.2f}%")
    
    # ==========================================================================
    # Part 4: Effect of Longitudinal Field Strength
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 4: Dependence on Longitudinal Field hᵤ")
    print("-" * 70)
    
    print("\nMass gap Δ₁ vs longitudinal field hᵤ:")
    print("-" * 40)
    
    n_sites = 10
    for h_z in [0.01, 0.02, 0.05, 0.1, 0.2]:
        params = SimulationParameters(
            n_sites=n_sites,
            J=1.0,
            h_transverse=1.0,
            h_longitudinal=h_z,
            boundary='open'
        )
        
        sim = IsingFieldTheorySimulation(params)
        gaps = sim.compute_mass_spectrum(n_states=3)
        
        # Theoretical scaling: Δ₁ ∝ |hᵤ|^(8/15) for small hᵤ
        print(f"  hᵤ = {h_z:.3f}: Δ₁ = {gaps[0]:.6f}")
    
    # ==========================================================================
    # Part 5: Conformal Dimensions at Criticality
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 5: CFT at Critical Point (hᵤ = 0)")
    print("-" * 70)
    
    ham = IsingHamiltonian(
        n_sites=12,
        J=1.0,
        h_transverse=1.0,
        h_longitudinal=0.0,
        boundary='open'
    )
    
    scaling = ham.scaling_dimension()
    print(f"\nCritical Ising model: c = 1/2 minimal model M(3,4)")
    print(f"\nConformal dimensions of primary fields:")
    print(f"  Identity (I):        h = {scaling['identity']}")
    print(f"  Energy density (ε):  h = {scaling['energy_density']}")
    print(f"  Spin field (σ):      h = {scaling['spin_field']}")
    print(f"\nCentral charge: c = {scaling['central_charge']}")
    
    # Free fermion spectrum
    print(f"\nFree fermion single-particle spectrum:")
    eps_k = ham.free_fermion_spectrum()
    print(f"  Single-particle energies εₖ (first 5): {eps_k[:5]}")
    
    print("\n" + "=" * 70)
    print("E₈ spectrum analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
