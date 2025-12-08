"""
Ising Field Theory Quantum Simulation Package

Mathematical Foundation:
------------------------
The Ising Field Theory (IFT) is the continuum limit of the quantum Ising chain:

    H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ - hᵤ Σᵢ σᵢᶻ

Near the critical point (hₓ = J), the low-energy physics is described by a 
massive Majorana fermion with action:

    S = ∫ d²x [ψ̄(∂_μγ^μ + m)ψ]

where m ∝ (hₓ - J) is the mass gap.

The longitudinal field hᵤ perturbation at criticality yields the E₈ integrable
field theory with 8 stable particles whose mass ratios are:

    m₁ : m₂ : m₃ : ... : m₈ = 1 : 2cos(π/5) : 2cos(π/30) : ...

This package implements:
    1. Exact Hamiltonian construction
    2. Jordan-Wigner transformation to qubit representation
    3. Trotterized time evolution circuits
    4. Initial state preparation (vacuum, particle states)
    5. Scattering amplitude measurement protocols
    6. Particle production analysis
"""

from .hamiltonian import IsingHamiltonian, E8MassRatios
from .jordan_wigner import JordanWignerTransform
from .circuits import (
    TrotterCircuit,
    StatePreparation,
    MeasurementCircuit
)
from .simulation import IsingFieldTheorySimulation
from .scattering import ScatteringAnalysis

__version__ = "1.0.0"
__all__ = [
    "IsingHamiltonian",
    "E8MassRatios", 
    "JordanWignerTransform",
    "TrotterCircuit",
    "StatePreparation",
    "MeasurementCircuit",
    "IsingFieldTheorySimulation",
    "ScatteringAnalysis"
]
