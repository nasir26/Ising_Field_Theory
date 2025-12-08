# Ising Field Theory Quantum Simulation

A mathematically rigorous simulation package for studying the Ising Field Theory (IFT) on a quantum computer, enabling scattering experiments and particle production analysis.

> *"The Ising field theory is probably the simplest interesting situation where one can study scattering on a quantum computer and ask questions regarding particle production."*

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Package Structure](#package-structure)
- [Usage Examples](#usage-examples)
- [Physics Background](#physics-background)
- [API Reference](#api-reference)

## Overview

This package provides a complete simulation framework for the Ising Field Theory, including:

1. **Hamiltonian Construction**: Exact representation of the quantum Ising chain
2. **Jordan-Wigner Transformation**: Mapping to fermionic (Majorana) representation
3. **Quantum Circuits**: Trotterized time evolution with rigorous error bounds
4. **State Preparation**: Vacuum, particle, wavepacket, and domain wall states
5. **Scattering Analysis**: S-matrix extraction and particle production detection
6. **E₈ Verification**: Comparison with exact E₈ integrable field theory results

## Mathematical Foundation

### The Quantum Ising Chain

The quantum Ising chain Hamiltonian in transverse and longitudinal fields:

```
H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ - hᵤ Σᵢ σᵢᶻ
```

where:
- `J`: Ising coupling strength
- `hₓ`: Transverse field (controls quantum fluctuations)
- `hᵤ`: Longitudinal field (breaks Z₂ symmetry)

### Phase Diagram

| Regime | Condition | Physics |
|--------|-----------|---------|
| Ferromagnetic | g = hₓ/J < 1 | Spontaneous magnetization |
| Critical | g = 1 | Conformal field theory (c = 1/2) |
| Paramagnetic | g > 1 | Disordered phase |
| E₈ Point | g = 1, hᵤ ≠ 0 | Integrable field theory with 8 stable particles |

### Continuum Limit

Near the critical point (g → 1), the low-energy physics is described by a massive Majorana fermion:

```
S = ∫ d²x [ψ̄(∂_μγ^μ + m)ψ]
```

with mass gap m ∝ |g - 1|.

### E₈ Mass Spectrum

At the E₈ integrable point (g = 1, hᵤ ≠ 0), the spectrum contains 8 stable particles with mass ratios:

| Particle | Mass Ratio mₙ/m₁ | Golden Ratio Form |
|----------|------------------|-------------------|
| m₁ | 1 | 1 |
| m₂ | 2cos(π/5) ≈ 1.618 | φ |
| m₃ | 2cos(π/30) ≈ 1.989 | φ² |
| m₄ | 2m₂cos(7π/30) ≈ 2.405 | - |
| m₅ | 2m₂cos(2π/15) ≈ 2.956 | - |
| m₆ | 2m₂cos(π/30) ≈ 3.218 | - |
| m₇ | 4m₂cos(π/5)cos(7π/30) ≈ 3.891 | - |
| m₈ | 4m₂cos(π/5)cos(2π/15) ≈ 4.783 | - |

where φ = (1+√5)/2 is the golden ratio.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ising_field_theory

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.4 (optional, for visualization)

## Quick Start

```python
import numpy as np
from ising_field_theory import (
    IsingFieldTheorySimulation,
    SimulationParameters,
    ScatteringAnalysis,
    E8MassRatios
)

# Set up simulation near E₈ point
params = SimulationParameters(
    n_sites=12,
    J=1.0,
    h_transverse=1.0,      # Critical transverse field
    h_longitudinal=0.1,    # Small longitudinal perturbation → E₈
    boundary='open'
)

sim = IsingFieldTheorySimulation(params)
print(sim.summary())

# Compute ground state and mass spectrum
ground_state = sim.prepare_ground_state()
mass_gaps = sim.compute_mass_spectrum(n_states=8)
print(f"Mass gaps: {mass_gaps}")

# Compare with E₈ predictions
comparison = sim.compare_with_e8()
print(f"E₈ comparison: {comparison['relative_error']}")
```

## Package Structure

```
ising_field_theory/
├── __init__.py          # Package exports
├── hamiltonian.py       # Hamiltonian construction and diagonalization
├── jordan_wigner.py     # Jordan-Wigner transformation
├── circuits.py          # Quantum circuit construction
├── simulation.py        # Main simulation orchestration
└── scattering.py        # Scattering analysis
```

### Module Descriptions

#### `hamiltonian.py`
- `IsingHamiltonian`: Full Hamiltonian construction using sparse matrices
- `E8MassRatios`: Analytical E₈ mass ratios
- Exact diagonalization and eigenspectrum computation
- Correlation length and scaling dimension analysis

#### `jordan_wigner.py`
- `JordanWignerTransform`: Spin-to-fermion mapping
- Majorana fermion operators
- Bogoliubov transformation for free fermion case
- Fermion parity computation

#### `circuits.py`
- `QuantumCircuit`: General quantum circuit representation
- `TrotterCircuit`: Trotterized time evolution with error bounds
- `StatePreparation`: Vacuum, particle, wavepacket states
- `MeasurementCircuit`: Observable measurement circuits

#### `simulation.py`
- `IsingFieldTheorySimulation`: Main simulation class
- Exact and Trotterized time evolution
- Observable computation (energy, magnetization, correlations)
- Spectral function and dynamic structure factor

#### `scattering.py`
- `ScatteringAnalysis`: Scattering experiment analysis
- Particle production detection
- S-matrix extraction
- E₈ integrability verification

## Usage Examples

### 1. Time Evolution and Observables

```python
from ising_field_theory import IsingFieldTheorySimulation, SimulationParameters

params = SimulationParameters(n_sites=10, J=1.0, h_transverse=0.8)
sim = IsingFieldTheorySimulation(params)

# Prepare initial state (domain wall)
initial_state = sim.prepare_domain_wall_state(position=5)

# Time evolution
results = sim.time_evolve(
    initial_state, 
    total_time=10.0, 
    n_time_points=100,
    method='exact'
)

# Access observables
print(f"Energies: {results.energies}")
print(f"Magnetization: {results.magnetization}")
```

### 2. Quantum Circuit Construction

```python
from ising_field_theory.circuits import TrotterCircuit, StatePreparation

# Build Trotter evolution circuit
trotter = TrotterCircuit(n_qubits=8, J=1.0, h_x=1.0, h_z=0.1)
circuit = trotter.build_evolution_circuit(
    total_time=5.0,
    n_steps=50,
    order=2  # Second-order Trotter
)

print(f"Circuit depth: {circuit.depth()}")
print(f"Gate counts: {circuit.gate_count()}")
print(f"Trotter error bound: {trotter.trotter_error_bound(5.0, 50, 2)}")

# State preparation
prep = StatePreparation(n_qubits=8)
wavepacket = prep.wavepacket_state(center=4.0, width=1.5, momentum=0.5)
```

### 3. Scattering Experiment

```python
from ising_field_theory.scattering import (
    ScatteringAnalysis, 
    ScatteringSetup,
    create_scattering_experiment
)

# Quick setup
sim, analysis, setup = create_scattering_experiment(
    n_sites=16,
    g=1.0,        # Critical point
    h_z=0.1,      # E₈ perturbation
    momenta=(0.5, -0.5)  # Counter-propagating particles
)

# Run scattering
results = analysis.run_scattering(setup, n_time_points=100)

# Analyze particle production
production = analysis.analyze_particle_production(results)
print(f"Elastic scattering: {production['is_elastic']}")
print(f"Particle number change: {production['particle_change']}")

# Verify integrability
integrability = analysis.verify_integrability(results)
print(f"Is integrable: {integrability['is_integrable']}")
```

### 4. E₈ Spectrum Verification

```python
from ising_field_theory import IsingFieldTheorySimulation, SimulationParameters, E8MassRatios

# Set up at E₈ point
params = SimulationParameters(
    n_sites=14,
    J=1.0,
    h_transverse=1.0,
    h_longitudinal=0.05
)
sim = IsingFieldTheorySimulation(params)

# Compare mass ratios
comparison = sim.compare_with_e8(n_states=8)

print("Mass Ratio Comparison:")
print("=" * 50)
print(f"{'n':>4} {'Computed':>12} {'E₈ Exact':>12} {'Error (%)':>12}")
print("-" * 50)
for i, (comp, exact, err) in enumerate(zip(
    comparison['computed'], 
    comparison['e8_exact'],
    comparison['relative_error']
), 1):
    print(f"{i:>4} {comp:>12.4f} {exact:>12.4f} {err*100:>12.2f}")
```

### 5. Jordan-Wigner Transformation

```python
from ising_field_theory import JordanWignerTransform

jw = JordanWignerTransform(n_sites=4)

# Verify anticommutation relations
print(f"Anticommutation verified: {jw.verify_anticommutation()}")

# Get Majorana operators
gamma_a = jw.majorana_operator(site=1, which='a')  # γⱼᵃ = cⱼ + cⱼ†
gamma_b = jw.majorana_operator(site=1, which='b')  # γⱼᵇ = i(cⱼ† - cⱼ)

# Bogoliubov transformation for free fermion case
energies, angles = jw.bogoliubov_transform(J=1.0, h_x=1.5)
print(f"Single-particle energies: {energies}")
```

## Physics Background

### Why Ising Field Theory?

The Ising Field Theory is the simplest quantum field theory that exhibits:

1. **Non-trivial scattering**: Particles interact and can scatter off each other
2. **Particle production**: Away from integrability, collisions can create new particles
3. **Exact solvability**: At special points (free fermion, E₈), exact results are known
4. **Quantum simulation relevance**: Naturally maps to qubit systems

### Scattering on Quantum Computers

The key questions addressable with quantum simulation:

1. **Elastic scattering**: Do particles scatter without particle production?
2. **S-matrix elements**: What is the scattering amplitude?
3. **Particle production thresholds**: At what energy do new particles appear?
4. **Integrability breaking**: How does particle production emerge as we move away from integrable points?

### Trotterization Error Analysis

The Trotter decomposition approximates:

```
exp(-iHt) ≈ [exp(-iH_ZZ δt) exp(-iH_X δt) exp(-iH_Z δt)]^n
```

**First-order error**: O(t²/n) per commutator
**Second-order error**: O(t³/n²) 

For the Ising model:
- `[H_ZZ, H_X]` ≠ 0 → non-trivial error
- `[H_ZZ, H_Z] = 0` → these terms can be combined freely

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `IsingHamiltonian` | Constructs and diagonalizes the Ising Hamiltonian |
| `JordanWignerTransform` | Spin-fermion mapping utilities |
| `QuantumCircuit` | General quantum circuit representation |
| `TrotterCircuit` | Trotterized time evolution circuits |
| `StatePreparation` | Initial state preparation methods |
| `MeasurementCircuit` | Observable measurement circuits |
| `IsingFieldTheorySimulation` | Main simulation orchestration |
| `ScatteringAnalysis` | Scattering experiment analysis |

### Data Classes

| Class | Description |
|-------|-------------|
| `SimulationParameters` | Simulation configuration |
| `SimulationResults` | Time evolution results |
| `ScatteringSetup` | Scattering experiment configuration |
| `ScatteringResults` | Scattering analysis results |
| `E8MassRatios` | E₈ mass spectrum data |

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ising_field_theory,
  title = {Ising Field Theory Quantum Simulation},
  author = {[Author]},
  year = {2024},
  description = {Quantum simulation of Ising Field Theory for scattering and particle production studies}
}
```

## License

MIT License - see LICENSE file for details.
