#!/usr/bin/env python3
"""
Example 4: Quantum Circuit Construction

This example demonstrates:
1. Building Trotterized time evolution circuits
2. State preparation circuits
3. Measurement circuits for observables
4. Circuit depth and gate count analysis
5. Trotter error bounds

These circuits can be run on actual quantum hardware or simulators.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from ising_field_theory.circuits import (
    QuantumCircuit,
    TrotterCircuit,
    StatePreparation,
    MeasurementCircuit,
    Gate,
    GateType
)


def main():
    print("=" * 70)
    print("Ising Field Theory - Quantum Circuit Construction")
    print("=" * 70)
    
    n_qubits = 6
    
    # ==========================================================================
    # Part 1: Basic Quantum Circuit Operations
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 1: Basic Quantum Circuit")
    print("-" * 70)
    
    # Create a simple circuit
    circuit = QuantumCircuit(n_qubits)
    
    # Add some gates
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.rx(2, np.pi/4)
    circuit.rzz(0, 1, np.pi/3)
    
    print(f"Circuit with {n_qubits} qubits:")
    print(f"  Gates: {[str(g) for g in circuit.gates]}")
    print(f"  Depth: {circuit.depth()}")
    print(f"  Gate counts: {circuit.gate_count()}")
    
    # Convert to unitary
    U = circuit.to_unitary()
    print(f"  Unitary shape: {U.shape}")
    print(f"  Unitarity check: ||U†U - I|| = {np.linalg.norm(U.conj().T @ U - np.eye(2**n_qubits)):.2e}")
    
    # ==========================================================================
    # Part 2: Trotter Evolution Circuits
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 2: Trotter Time Evolution Circuits")
    print("-" * 70)
    
    # Create Trotter circuit builder
    trotter = TrotterCircuit(
        n_qubits=n_qubits,
        J=1.0,
        h_x=1.0,
        h_z=0.1,
        boundary='open'
    )
    
    print(f"\nHamiltonian: H = -J Σ ZᵢZᵢ₊₁ - hₓ Σ Xᵢ - hᵤ Σ Zᵢ")
    print(f"  J = {trotter.J}, hₓ = {trotter.h_x}, hᵤ = {trotter.h_z}")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Number of ZZ bonds: {trotter.n_bonds}")
    
    # Build first-order Trotter circuit
    print("\n--- First-Order Trotter ---")
    circuit_1st = trotter.build_evolution_circuit(
        total_time=1.0,
        n_steps=10,
        order=1
    )
    
    print(f"  Total time: t = 1.0")
    print(f"  Trotter steps: 10")
    print(f"  Circuit depth: {circuit_1st.depth()}")
    print(f"  Total gates: {len(circuit_1st.gates)}")
    print(f"  Gate breakdown: {circuit_1st.gate_count()}")
    error_1st = trotter.trotter_error_bound(1.0, 10, order=1)
    print(f"  Error bound: {error_1st:.4e}")
    
    # Build second-order Trotter circuit
    print("\n--- Second-Order Trotter ---")
    circuit_2nd = trotter.build_evolution_circuit(
        total_time=1.0,
        n_steps=10,
        order=2
    )
    
    print(f"  Total time: t = 1.0")
    print(f"  Trotter steps: 10")
    print(f"  Circuit depth: {circuit_2nd.depth()}")
    print(f"  Total gates: {len(circuit_2nd.gates)}")
    print(f"  Gate breakdown: {circuit_2nd.gate_count()}")
    error_2nd = trotter.trotter_error_bound(1.0, 10, order=2)
    print(f"  Error bound: {error_2nd:.4e}")
    
    # Trotter error scaling
    print("\n--- Trotter Error Scaling ---")
    print(f"{'Steps':>6} | {'1st Order':>12} | {'2nd Order':>12}")
    print("-" * 35)
    for n_steps in [5, 10, 20, 40, 80]:
        err_1 = trotter.trotter_error_bound(1.0, n_steps, order=1)
        err_2 = trotter.trotter_error_bound(1.0, n_steps, order=2)
        print(f"{n_steps:>6} | {err_1:>12.4e} | {err_2:>12.4e}")
    
    # ==========================================================================
    # Part 3: State Preparation Circuits
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 3: State Preparation")
    print("-" * 70)
    
    prep = StatePreparation(n_qubits)
    
    # All zeros state (ferromagnetic)
    state_00 = prep.all_zeros_state()
    print(f"\n|00...0⟩ state (ferromagnetic):")
    print(f"  Non-zero amplitude at index 0: {state_00[0]}")
    
    # Product state in X basis
    state_plus = prep.product_state_x()
    print(f"\n|+⟩^⊗n state (paramagnetic):")
    print(f"  All amplitudes equal: {np.allclose(np.abs(state_plus), 1/np.sqrt(2**n_qubits))}")
    
    # Domain wall state
    domain_wall = prep.domain_wall_state(position=3)
    print(f"\nDomain wall at position 3:")
    nonzero_idx = np.where(np.abs(domain_wall) > 0.5)[0]
    print(f"  Non-zero at index: {nonzero_idx}")
    binary = format(nonzero_idx[0], f'0{n_qubits}b')
    print(f"  Binary: |{binary}⟩")
    
    # Momentum eigenstate
    momentum_state = prep.momentum_eigenstate(momentum_index=2)
    print(f"\nMomentum eigenstate (k = 2π/(N+1)):")
    print(f"  Superposition of {np.sum(np.abs(momentum_state) > 1e-10)} basis states")
    
    # Gaussian wavepacket
    wavepacket = prep.wavepacket_state(center=3.0, width=1.0, momentum=0.5)
    print(f"\nGaussian wavepacket (center=3, σ=1, k=0.5):")
    print(f"  Normalized: {np.isclose(np.linalg.norm(wavepacket), 1.0)}")
    
    # Two-particle state
    two_particle = prep.two_particle_state(site1=1, site2=4)
    print(f"\nTwo-particle state (sites 1 and 4):")
    nonzero_idx = np.where(np.abs(two_particle) > 0.5)[0]
    binary = format(nonzero_idx[0], f'0{n_qubits}b')
    print(f"  Binary: |{binary}⟩")
    
    # Variational circuit
    print("\n--- Variational Ansatz Circuit ---")
    n_layers = 2
    theta = np.random.randn(n_qubits * n_layers) * 0.1
    var_circuit = prep.variational_ground_state_circuit(theta)
    print(f"  Layers: {n_layers}")
    print(f"  Parameters: {len(theta)}")
    print(f"  Circuit depth: {var_circuit.depth()}")
    print(f"  Gate counts: {var_circuit.gate_count()}")
    
    # ==========================================================================
    # Part 4: Measurement Circuits
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 4: Measurement Circuits")
    print("-" * 70)
    
    meas = MeasurementCircuit(n_qubits)
    
    # Z basis measurement
    z_meas = meas.z_basis_measurement()
    print(f"\nZ basis measurement:")
    print(f"  Gates: {z_meas.gate_count()}")
    
    # X basis measurement (requires H gates)
    x_meas = meas.x_basis_measurement()
    print(f"\nX basis measurement:")
    print(f"  Gates: {x_meas.gate_count()}")
    
    # Y basis measurement
    y_meas = meas.y_basis_measurement()
    print(f"\nY basis measurement:")
    print(f"  Gates: {y_meas.gate_count()}")
    
    # Energy measurement circuits
    print("\n--- Energy Measurement Circuits ---")
    energy_circuits = meas.energy_measurement_circuits(
        J=1.0, h_x=1.0, h_z=0.1, boundary='open'
    )
    
    zz_circuits = [c for c in energy_circuits if 'ZZ' in c[2]]
    x_circuits = [c for c in energy_circuits if c[2].startswith('X')]
    z_circuits = [c for c in energy_circuits if c[2].startswith('Z_')]
    
    print(f"  Total measurement circuits: {len(energy_circuits)}")
    print(f"  ZZ term circuits: {len(zz_circuits)}")
    print(f"  X term circuits: {len(x_circuits)}")
    print(f"  Z term circuits: {len(z_circuits)}")
    
    # ==========================================================================
    # Part 5: Full Evolution + Measurement Pipeline
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Part 5: Complete Quantum Simulation Pipeline")
    print("-" * 70)
    
    # 1. Prepare initial state
    initial_state = prep.domain_wall_state(position=n_qubits//2)
    
    # 2. Build evolution circuit
    evolution = trotter.build_evolution_circuit(total_time=2.0, n_steps=20, order=2)
    
    # 3. Apply evolution
    evolved_state = evolution.apply(initial_state)
    
    # 4. Measure in different bases
    # (In a real quantum computer, we'd run the circuit many times)
    
    print(f"\nPipeline summary:")
    print(f"  Initial state: domain wall at center")
    print(f"  Evolution time: t = 2.0")
    print(f"  Trotter steps: 20 (second-order)")
    print(f"  Evolution circuit depth: {evolution.depth()}")
    print(f"  Total gates in evolution: {len(evolution.gates)}")
    
    # Simulate measurement outcomes
    # Z-basis probabilities
    probs = np.abs(evolved_state)**2
    most_likely = np.argsort(probs)[-5:][::-1]
    
    print(f"\n  Most likely measurement outcomes (Z basis):")
    for idx in most_likely:
        binary = format(idx, f'0{n_qubits}b')
        print(f"    |{binary}⟩: probability = {probs[idx]:.4f}")
    
    print("\n" + "=" * 70)
    print("Quantum circuit construction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
