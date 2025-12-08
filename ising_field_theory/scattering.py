"""
Scattering Analysis for Ising Field Theory

This module implements scattering amplitude calculations and particle production
analysis for the Ising field theory on a quantum computer.

Physical Background:
--------------------
The Ising Field Theory supports various scattering processes:

1. Elastic scattering: 
   - Kink-kink → Kink-kink
   - Particle-particle at integrable points (no particle production)

2. Inelastic scattering (away from integrability):
   - Particle production: 1 + 1 → n particles
   - Energy threshold: E_cm > Σ mᵢ for n-particle final state

3. E₈ Integrable Scattering:
   At the E₈ point (hₓ = J, hᵤ ≠ 0), scattering is purely elastic.
   S-matrix elements are known exactly from bootstrap equations.

Scattering Setup on Quantum Computer:
-------------------------------------
1. Initial state: Two well-separated wavepackets with opposite momenta
2. Time evolution: Let particles collide
3. Final state: Analyze particle content via correlation functions

Key observables:
- S-matrix elements from asymptotic states
- Particle number after collision
- Energy distribution in final state
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from .hamiltonian import IsingHamiltonian, E8MassRatios
from .jordan_wigner import JordanWignerTransform
from .circuits import QuantumCircuit, TrotterCircuit, StatePreparation
from .simulation import IsingFieldTheorySimulation, SimulationParameters


@dataclass
class ScatteringSetup:
    """
    Configuration for a scattering experiment.
    
    Attributes:
        n_sites: Number of lattice sites
        initial_separation: Initial distance between particles
        momenta: Tuple of (p1, p2) initial momenta
        wavepacket_width: Gaussian width of wavepackets
        collision_time: Estimated time for collision
        total_time: Total simulation time
    """
    n_sites: int
    initial_separation: float
    momenta: Tuple[float, float]
    wavepacket_width: float
    collision_time: float = None
    total_time: float = None
    
    def __post_init__(self):
        if self.collision_time is None:
            # Estimate collision time from velocities
            # v ≈ ∂ε/∂k at small k for relativistic dispersion
            avg_momentum = 0.5 * (abs(self.momenta[0]) + abs(self.momenta[1]))
            if avg_momentum > 0:
                self.collision_time = self.initial_separation / (2 * avg_momentum)
            else:
                self.collision_time = self.n_sites / 2
        
        if self.total_time is None:
            self.total_time = 3 * self.collision_time


@dataclass 
class ScatteringResults:
    """
    Results from a scattering simulation.
    
    Attributes:
        times: Array of time points
        states: State vectors at each time
        particle_numbers: Expected particle number vs time
        momentum_distribution: k-space distribution at each time
        energy_spectrum: Energy distribution in final state
        s_matrix_element: Extracted S-matrix element (if applicable)
        elastic_probability: Probability of elastic scattering
        inelastic_channels: Particle production channels and probabilities
    """
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    states: List[np.ndarray] = field(default_factory=list)
    particle_numbers: np.ndarray = field(default_factory=lambda: np.array([]))
    momentum_distribution: List[np.ndarray] = field(default_factory=list)
    energy_spectrum: Optional[np.ndarray] = None
    s_matrix_element: Optional[complex] = None
    elastic_probability: Optional[float] = None
    inelastic_channels: Dict[int, float] = field(default_factory=dict)
    
    # Correlation functions for particle detection
    density_density: List[np.ndarray] = field(default_factory=list)
    spin_spin: List[np.ndarray] = field(default_factory=list)


class ScatteringAnalysis:
    """
    Analysis tools for scattering processes in Ising Field Theory.
    
    This class provides methods to:
    1. Set up two-particle initial states for scattering
    2. Evolve through collision
    3. Analyze final states for particle production
    4. Extract S-matrix elements
    
    Example usage:
    
        # Create simulation
        params = SimulationParameters(n_sites=20, J=1.0, h_transverse=1.0)
        sim = IsingFieldTheorySimulation(params)
        
        # Set up scattering
        analysis = ScatteringAnalysis(sim)
        setup = ScatteringSetup(
            n_sites=20,
            initial_separation=10,
            momenta=(0.5, -0.5),  # Counter-propagating
            wavepacket_width=2.0
        )
        
        # Run scattering experiment
        results = analysis.run_scattering(setup)
        
        # Analyze particle production
        analysis.compute_particle_production(results)
    """
    
    def __init__(self, simulation: IsingFieldTheorySimulation):
        """
        Initialize scattering analysis.
        
        Parameters:
            simulation: IsingFieldTheorySimulation instance
        """
        self.sim = simulation
        self.n_qubits = simulation.n_qubits
        self.params = simulation.params
        
        # Precompute operators for analysis
        self._build_operators()
    
    def _build_operators(self):
        """Build operators needed for scattering analysis."""
        dim = self.sim.dim
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Build local operators
        self.sigma_x_ops = []
        self.sigma_z_ops = []
        
        for j in range(self.n_qubits):
            op_x = np.array([[1.0]])
            op_z = np.array([[1.0]])
            
            for i in range(self.n_qubits):
                if i == j:
                    op_x = np.kron(op_x, sigma_x)
                    op_z = np.kron(op_z, sigma_z)
                else:
                    op_x = np.kron(op_x, np.eye(2))
                    op_z = np.kron(op_z, np.eye(2))
            
            self.sigma_x_ops.append(op_x)
            self.sigma_z_ops.append(op_z)
        
        # Particle number operator (domain wall counting)
        # N = (1/2) Σᵢ (1 - σᵢᶻσᵢ₊₁ᶻ)
        self.particle_number_op = np.zeros((dim, dim), dtype=complex)
        for i in range(self.n_qubits - 1):
            zz_op = self.sigma_z_ops[i] @ self.sigma_z_ops[i + 1]
            self.particle_number_op += 0.5 * (np.eye(dim) - zz_op)
    
    def prepare_scattering_state(
        self,
        setup: ScatteringSetup
    ) -> np.ndarray:
        """
        Prepare initial state for scattering.
        
        Creates two well-separated wavepackets with opposite momenta.
        
        For the Ising model, particles are domain walls (kinks).
        Two-particle state: |↑↑...↑↓↓...↓↑↑...↑⟩
        
        Parameters:
            setup: ScatteringSetup configuration
        
        Returns:
            Initial state vector
        """
        n = self.n_qubits
        sigma = setup.wavepacket_width
        k1, k2 = setup.momenta
        
        # Positions for two wavepackets
        x1 = (n - setup.initial_separation) / 2
        x2 = (n + setup.initial_separation) / 2
        
        # Build superposition of two-particle (domain wall) states
        state = np.zeros(self.sim.dim, dtype=complex)
        
        # Two domain walls at positions i < j
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Amplitude = product of Gaussian wavepackets with phase
                amp_i = np.exp(-(i - x1)**2 / (4 * sigma**2)) * np.exp(1j * k1 * i)
                amp_j = np.exp(-(j - x2)**2 / (4 * sigma**2)) * np.exp(1j * k2 * j)
                
                # Two-kink state: all up, then down from i to j, then up
                # Binary representation
                index = 0
                for site in range(n):
                    if i <= site < j:
                        index |= (1 << (n - 1 - site))
                
                state[index] += amp_i * amp_j
        
        # Normalize
        state = state / np.linalg.norm(state)
        
        return state
    
    def prepare_single_particle_wavepacket(
        self,
        center: float,
        momentum: float,
        width: float
    ) -> np.ndarray:
        """
        Prepare a single-particle (domain wall) wavepacket.
        
        |ψ⟩ = Σⱼ exp(-(j-x₀)²/4σ²) exp(ikj) |kink at j⟩
        
        where |kink at j⟩ = |↑↑...↑↓↓...⟩ with domain wall at position j.
        """
        n = self.n_qubits
        state = np.zeros(self.sim.dim, dtype=complex)
        
        for j in range(n):
            # Gaussian amplitude with momentum phase
            amplitude = np.exp(-(j - center)**2 / (4 * width**2))
            amplitude *= np.exp(1j * momentum * j)
            
            # Domain wall state at position j
            # All spins up to j are |0⟩, rest are |1⟩
            index = (1 << (n - j)) - 1 if j < n else (1 << n) - 1
            state[index] += amplitude
        
        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state = state / norm
        
        return state
    
    def run_scattering(
        self,
        setup: ScatteringSetup,
        n_time_points: int = 100,
        method: str = 'exact'
    ) -> ScatteringResults:
        """
        Run a full scattering simulation.
        
        1. Prepare two-particle initial state
        2. Evolve through collision
        3. Compute observables at each time step
        
        Parameters:
            setup: ScatteringSetup configuration
            n_time_points: Number of time points to record
            method: 'exact' or 'trotter'
        
        Returns:
            ScatteringResults with full analysis
        """
        results = ScatteringResults()
        
        # Prepare initial state
        initial_state = self.prepare_scattering_state(setup)
        
        # Time array
        times = np.linspace(0, setup.total_time, n_time_points)
        results.times = times
        
        # Evolve and compute observables
        if method == 'exact':
            results.states = self.sim.exact_time_evolution(initial_state, times)
        else:
            results.states = []
            for t in times:
                if t == 0:
                    results.states.append(initial_state.copy())
                else:
                    state, _ = self.sim.trotter_time_evolution(
                        initial_state, t,
                        n_steps=max(1, int(t * 10)),
                        order=2
                    )
                    results.states.append(state)
        
        # Compute particle numbers
        results.particle_numbers = np.array([
            self.compute_particle_number(state) 
            for state in results.states
        ])
        
        # Compute momentum distributions
        results.momentum_distribution = [
            self.compute_momentum_distribution(state)
            for state in results.states
        ]
        
        # Compute density-density correlations
        results.density_density = [
            self.compute_density_correlation(state)
            for state in results.states
        ]
        
        return results
    
    def compute_particle_number(self, state: np.ndarray) -> float:
        """
        Compute expected number of particles (domain walls).
        
        N = ⟨ψ| (1/2) Σᵢ (1 - σᵢᶻσᵢ₊₁ᶻ) |ψ⟩
        
        Each domain wall contributes 1 to this count.
        """
        return np.real(np.conj(state) @ self.particle_number_op @ state)
    
    def compute_momentum_distribution(self, state: np.ndarray) -> np.ndarray:
        """
        Compute momentum distribution n(k).
        
        n(k) = |⟨k|ψ⟩|² where |k⟩ is a momentum eigenstate.
        
        For the spin chain, we use the Fourier transform of magnetization.
        """
        n = self.n_qubits
        
        # Compute local magnetization profile
        m_x = np.zeros(n, dtype=complex)
        for j in range(n):
            m_x[j] = np.conj(state) @ self.sigma_x_ops[j] @ state
        
        # Fourier transform to momentum space
        n_k = np.abs(np.fft.fft(m_x)) ** 2 / n
        
        return n_k
    
    def compute_density_correlation(self, state: np.ndarray) -> np.ndarray:
        """
        Compute density-density correlation function.
        
        C(i,j) = ⟨nᵢnⱼ⟩ - ⟨nᵢ⟩⟨nⱼ⟩
        
        where nᵢ is the local particle density (domain wall indicator).
        """
        n = self.n_qubits
        dim = self.sim.dim
        
        # Local density: nᵢ = (1 - σᵢᶻσᵢ₊₁ᶻ)/2
        correlations = np.zeros((n-1, n-1))
        
        for i in range(n - 1):
            zz_i = self.sigma_z_ops[i] @ self.sigma_z_ops[i + 1]
            n_i = 0.5 * (np.eye(dim) - zz_i)
            expect_ni = np.real(np.conj(state) @ n_i @ state)
            
            for j in range(i, n - 1):
                zz_j = self.sigma_z_ops[j] @ self.sigma_z_ops[j + 1]
                n_j = 0.5 * (np.eye(dim) - zz_j)
                expect_nj = np.real(np.conj(state) @ n_j @ state)
                
                # ⟨nᵢnⱼ⟩
                expect_ninj = np.real(np.conj(state) @ n_i @ n_j @ state)
                
                # Connected correlator
                correlations[i, j] = expect_ninj - expect_ni * expect_nj
                correlations[j, i] = correlations[i, j]
        
        return correlations
    
    def analyze_particle_production(
        self, 
        results: ScatteringResults,
        threshold: float = 0.1
    ) -> Dict:
        """
        Analyze particle production from scattering results.
        
        Compare initial and final particle numbers to detect
        inelastic scattering and particle production.
        
        Parameters:
            results: ScatteringResults from run_scattering
            threshold: Threshold for particle number change detection
        
        Returns:
            Dictionary with production analysis
        """
        # Initial and final particle numbers
        N_initial = results.particle_numbers[0]
        N_final = results.particle_numbers[-1]
        
        # Detect production
        delta_N = N_final - N_initial
        
        # Analyze time evolution of particle number
        # At integrable points, N should be conserved
        N_variance = np.var(results.particle_numbers)
        N_max = np.max(results.particle_numbers)
        N_min = np.min(results.particle_numbers)
        
        # Determine if scattering is elastic or inelastic
        is_elastic = abs(delta_N) < threshold and N_variance < threshold**2
        
        analysis = {
            'initial_particles': N_initial,
            'final_particles': N_final,
            'particle_change': delta_N,
            'is_elastic': is_elastic,
            'particle_number_variance': N_variance,
            'max_particles': N_max,
            'min_particles': N_min,
        }
        
        if not is_elastic:
            # Estimate probability of n-particle final states
            # This is approximate based on particle number distribution
            probs = {}
            for n_part in range(int(N_min), int(N_max) + 2):
                # Probability of being near this particle number
                mask = np.abs(results.particle_numbers - n_part) < 0.5
                probs[n_part] = np.mean(mask)
            
            analysis['particle_probabilities'] = probs
            results.inelastic_channels = probs
        
        return analysis
    
    def extract_s_matrix_element(
        self,
        initial_state: np.ndarray,
        final_state: np.ndarray,
        energy: float = None
    ) -> complex:
        """
        Extract S-matrix element from in/out states.
        
        S = ⟨out|S|in⟩ = ⟨ψ(t→∞)|ψ(t→-∞)⟩ × phase
        
        For elastic scattering of identical particles, this gives
        the scattering phase shift δ via S = e^{2iδ}.
        
        Parameters:
            initial_state: Asymptotic in-state
            final_state: Asymptotic out-state
            energy: Center of mass energy (for phase factor)
        
        Returns:
            S-matrix element
        """
        # Overlap
        S = np.conj(final_state) @ initial_state
        
        # For proper S-matrix, need to account for free propagation
        # S_full = S × exp(-iEt) where E is the total energy
        # This is the scattering amplitude (momentum-space S-matrix)
        
        return S
    
    def compute_scattering_phase(
        self,
        results: ScatteringResults,
        momentum: float
    ) -> float:
        """
        Extract scattering phase shift from wavepacket dynamics.
        
        The phase shift δ(k) is extracted by comparing the position
        of the scattered wavepacket with free propagation.
        
        For elastic 2→2 scattering:
        Δx = 2 × ∂δ/∂k (Wigner time delay)
        
        Parameters:
            results: ScatteringResults with evolved states
            momentum: Momentum at which to extract phase
        
        Returns:
            Scattering phase shift δ (in radians)
        """
        # This is a simplified extraction based on peak tracking
        # Full analysis requires momentum-resolved correlation functions
        
        # Track peak position in momentum distribution
        n_time = len(results.times)
        peak_positions = []
        
        for t_idx in range(n_time):
            n_k = results.momentum_distribution[t_idx]
            peaks, _ = find_peaks(n_k)
            if len(peaks) > 0:
                # Find peak closest to expected momentum
                k_values = 2 * np.pi * np.arange(len(n_k)) / len(n_k)
                closest = peaks[np.argmin(np.abs(k_values[peaks] - momentum))]
                peak_positions.append(k_values[closest])
            else:
                peak_positions.append(momentum)
        
        peak_positions = np.array(peak_positions)
        
        # Phase shift from momentum shift (approximate)
        dk = np.mean(peak_positions) - momentum
        # δ ≈ L × dk where L is system size
        phase_shift = self.n_qubits * dk / 2
        
        return phase_shift
    
    def compute_e8_s_matrix(
        self,
        particle_types: Tuple[int, int],
        rapidity: float
    ) -> complex:
        """
        Compute exact E₈ S-matrix element.
        
        The E₈ S-matrix satisfies:
        - Unitarity: S(θ)S(-θ) = 1
        - Crossing: S_{ab}(θ) = S_{ab̄}(iπ - θ)
        - Bootstrap: Fusion rules from E₈ algebra
        
        S_{ab}(θ) = ∏ᵢ fᵢ(θ)^{nᵢ}
        
        where fᵢ(θ) are building blocks and nᵢ depend on particle types.
        
        Parameters:
            particle_types: (a, b) where a, b ∈ {1, ..., 8}
            rapidity: Rapidity difference θ = θ₁ - θ₂
        
        Returns:
            S-matrix element S_{ab}(θ)
        """
        a, b = particle_types
        theta = rapidity
        
        # E₈ S-matrix building blocks
        # f(θ) = sinh(θ + iπ/h) / sinh(θ - iπ/h)
        # where h = 30 (Coxeter number of E₈)
        
        h = 30  # E₈ Coxeter number
        
        def f_block(theta: complex, x: float) -> complex:
            """Building block f_x(θ) = sinh((θ + iπx/h)/2) / sinh((θ - iπx/h)/2)"""
            num = np.sinh((theta + 1j * np.pi * x / h) / 2)
            den = np.sinh((theta - 1j * np.pi * x / h) / 2)
            return num / den if abs(den) > 1e-15 else 1.0
        
        # Simplified: S₁₁ for lightest particle
        # S₁₁(θ) = f₁(θ) f₇(θ) f₁₁(θ) f₁₃(θ) f₁₇(θ) f₁₉(θ) f₂₃(θ) f₂₉(θ)
        if a == 1 and b == 1:
            exponents = [1, 7, 11, 13, 17, 19, 23, 29]
            S = 1.0
            for x in exponents:
                S *= f_block(theta, x)
            return S
        
        # For other particles, the S-matrix elements are more complex
        # Here we return 1 as placeholder (elastic scattering)
        return 1.0 + 0j
    
    def verify_integrability(
        self,
        results: ScatteringResults,
        tolerance: float = 0.05
    ) -> Dict:
        """
        Verify integrability by checking particle number conservation.
        
        At integrable points (E₈ or free fermion), the number of
        particles is conserved and there is no particle production.
        
        Parameters:
            results: ScatteringResults from scattering simulation
            tolerance: Tolerance for particle number conservation
        
        Returns:
            Dictionary with integrability analysis
        """
        N = results.particle_numbers
        
        # Check conservation
        N_mean = np.mean(N)
        N_std = np.std(N)
        N_drift = abs(N[-1] - N[0])
        
        is_integrable = N_std < tolerance and N_drift < tolerance
        
        # Additional checks: factorization of S-matrix
        # For integrable models, multi-particle S-matrix factorizes
        
        return {
            'is_integrable': is_integrable,
            'particle_number_mean': N_mean,
            'particle_number_std': N_std,
            'particle_number_drift': N_drift,
            'conservation_quality': 1.0 - N_std / N_mean if N_mean > 0 else 1.0
        }
    
    def run_e8_scattering_test(
        self,
        rapidity: float = 0.5,
        n_time_points: int = 100
    ) -> Dict:
        """
        Run a comprehensive E₈ scattering test.
        
        At the E₈ point (g=1, h_z≠0), verify:
        1. Particle number conservation
        2. S-matrix matches analytical E₈ result
        3. Mass ratios match E₈ spectrum
        
        Parameters:
            rapidity: Initial rapidity for wavepackets
            n_time_points: Number of time points
        
        Returns:
            Dictionary with E₈ test results
        """
        results = {}
        
        # Check we're at E₈ point
        results['is_e8_point'] = (
            self.params.is_critical and 
            self.params.h_longitudinal != 0
        )
        
        if not results['is_e8_point']:
            results['warning'] = "Not at E₈ point (need g=1, h_z≠0)"
        
        # Compute mass ratios
        e8_comparison = self.sim.compare_with_e8(n_states=8)
        results['mass_ratios'] = e8_comparison
        
        # Run scattering
        setup = ScatteringSetup(
            n_sites=self.n_qubits,
            initial_separation=self.n_qubits // 2,
            momenta=(rapidity, -rapidity),
            wavepacket_width=self.n_qubits / 8
        )
        
        scattering_results = self.run_scattering(setup, n_time_points)
        
        # Verify integrability
        integrability = self.verify_integrability(scattering_results)
        results['integrability'] = integrability
        
        # Extract S-matrix
        S_numerical = self.extract_s_matrix_element(
            scattering_results.states[0],
            scattering_results.states[-1]
        )
        results['S_numerical'] = S_numerical
        
        # Compare with analytical E₈ S-matrix
        S_analytical = self.compute_e8_s_matrix((1, 1), 2 * rapidity)
        results['S_analytical'] = S_analytical
        results['S_matrix_error'] = abs(abs(S_numerical) - abs(S_analytical))
        
        return results


def create_scattering_experiment(
    n_sites: int = 16,
    J: float = 1.0,
    g: float = 1.0,
    h_z: float = 0.1,
    initial_separation: int = None,
    momenta: Tuple[float, float] = (0.5, -0.5)
) -> Tuple[IsingFieldTheorySimulation, ScatteringAnalysis, ScatteringSetup]:
    """
    Convenience function to set up a scattering experiment.
    
    Parameters:
        n_sites: Number of lattice sites
        J: Ising coupling
        g: Dimensionless coupling hₓ/J
        h_z: Longitudinal field (set to 0 for free fermion, nonzero for E₈)
        initial_separation: Initial particle separation (default: n_sites/2)
        momenta: Initial momenta of particles
    
    Returns:
        Tuple of (simulation, analysis, setup)
    """
    if initial_separation is None:
        initial_separation = n_sites // 2
    
    params = SimulationParameters(
        n_sites=n_sites,
        J=J,
        h_transverse=g * J,
        h_longitudinal=h_z,
        boundary='open'
    )
    
    sim = IsingFieldTheorySimulation(params)
    analysis = ScatteringAnalysis(sim)
    
    setup = ScatteringSetup(
        n_sites=n_sites,
        initial_separation=initial_separation,
        momenta=momenta,
        wavepacket_width=n_sites / 8
    )
    
    return sim, analysis, setup
