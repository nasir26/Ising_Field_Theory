"""
Ising Field Theory Simulation - Main Orchestration Module

This module provides the main simulation class that integrates:
1. Hamiltonian construction and diagonalization
2. Quantum circuit construction for time evolution
3. State preparation and measurement
4. Observable computation and analysis

Physical Setup:
---------------
The Ising Field Theory (IFT) emerges from the quantum Ising chain near criticality:

    H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ - hᵤ Σᵢ σᵢᶻ

Critical point: g = hₓ/J = 1 (with hᵤ = 0)

In the scaling limit (a → 0, keeping correlation length finite):
    - Transverse field perturbation: massive Majorana fermion with m ∝ |g - 1|
    - Longitudinal field at criticality: E₈ integrable field theory

Key Observables:
----------------
1. Magnetization: ⟨M⟩ = ⟨Σᵢ σᵢˣ⟩/N (order parameter)
2. Energy density: ⟨ε⟩ = ⟨σᵢᶻσᵢ₊₁ᶻ⟩
3. Two-point correlator: G(r) = ⟨σᵢˣσᵢ₊ᵣˣ⟩
4. Dynamic structure factor: S(k,ω)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from scipy.linalg import expm

from .hamiltonian import IsingHamiltonian, E8MassRatios
from .jordan_wigner import JordanWignerTransform, compute_fermion_parity
from .circuits import (
    QuantumCircuit, 
    TrotterCircuit, 
    StatePreparation, 
    MeasurementCircuit
)


@dataclass
class SimulationParameters:
    """
    Parameters for Ising Field Theory simulation.
    
    Physical parameters:
        n_sites: Number of lattice sites (qubits)
        J: Ising coupling strength
        h_transverse: Transverse field (hₓ)
        h_longitudinal: Longitudinal field (hᵤ)
        boundary: 'open' or 'periodic'
    
    Simulation parameters:
        total_time: Total evolution time
        n_trotter_steps: Number of Trotter steps
        trotter_order: Order of Trotter decomposition (1 or 2)
        n_shots: Number of measurement shots (for sampling)
    """
    n_sites: int
    J: float = 1.0
    h_transverse: float = 1.0
    h_longitudinal: float = 0.0
    boundary: str = 'open'
    total_time: float = 1.0
    n_trotter_steps: int = 10
    trotter_order: int = 2
    n_shots: int = 1000
    
    @property
    def g(self) -> float:
        """Dimensionless coupling g = hₓ/J."""
        return self.h_transverse / self.J if self.J != 0 else np.inf
    
    @property
    def is_critical(self) -> bool:
        """Check if system is at critical point g = 1."""
        return np.isclose(self.g, 1.0, rtol=1e-10)
    
    @property
    def is_paramagnetic(self) -> bool:
        """Check if system is in paramagnetic phase g > 1."""
        return self.g > 1.0
    
    @property
    def is_ferromagnetic(self) -> bool:
        """Check if system is in ferromagnetic phase g < 1."""
        return self.g < 1.0


@dataclass
class SimulationResults:
    """
    Container for simulation results.
    
    Attributes:
        times: Array of time points
        states: List of state vectors at each time
        energies: Energy expectation values ⟨H⟩(t)
        magnetization: Magnetization ⟨M⟩(t) = ⟨Σσˣ⟩/N
        correlations: Two-point correlations
        eigenvalues: Hamiltonian eigenvalues (if computed)
        eigenvectors: Hamiltonian eigenvectors (if computed)
        mass_gaps: Mass gaps Δₙ = Eₙ - E₀
        trotter_errors: Estimated Trotter errors
    """
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    states: List[np.ndarray] = field(default_factory=list)
    energies: np.ndarray = field(default_factory=lambda: np.array([]))
    magnetization: np.ndarray = field(default_factory=lambda: np.array([]))
    correlations: Dict[str, np.ndarray] = field(default_factory=dict)
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    mass_gaps: Optional[np.ndarray] = None
    trotter_errors: Optional[np.ndarray] = None
    
    # Additional physics results
    fermion_parity: Optional[np.ndarray] = None
    entanglement_entropy: Optional[np.ndarray] = None
    fidelity: Optional[np.ndarray] = None


class IsingFieldTheorySimulation:
    """
    Main simulation class for Ising Field Theory on quantum computer.
    
    This class orchestrates:
    1. Hamiltonian construction and exact diagonalization
    2. Quantum circuit construction (Trotterized evolution)
    3. State preparation for various initial conditions
    4. Time evolution (exact and Trotterized)
    5. Observable measurements and analysis
    
    Example usage:
    
        # Set up simulation near E₈ point
        params = SimulationParameters(
            n_sites=8,
            J=1.0,
            h_transverse=1.0,  # Critical transverse field
            h_longitudinal=0.1  # Small longitudinal perturbation
        )
        
        sim = IsingFieldTheorySimulation(params)
        
        # Prepare initial state and evolve
        initial_state = sim.prepare_ground_state()
        results = sim.time_evolve(initial_state, total_time=10.0)
        
        # Analyze mass spectrum
        mass_ratios = sim.extract_mass_ratios()
    """
    
    def __init__(self, params: SimulationParameters):
        """Initialize simulation with given parameters."""
        self.params = params
        self.n_qubits = params.n_sites
        self.dim = 2 ** params.n_sites
        
        # Initialize core components
        self.hamiltonian = IsingHamiltonian(
            n_sites=params.n_sites,
            J=params.J,
            h_transverse=params.h_transverse,
            h_longitudinal=params.h_longitudinal,
            boundary=params.boundary
        )
        
        self.jw_transform = JordanWignerTransform(params.n_sites)
        
        self.trotter = TrotterCircuit(
            n_qubits=params.n_sites,
            J=params.J,
            h_x=params.h_transverse,
            h_z=params.h_longitudinal,
            boundary=params.boundary
        )
        
        self.state_prep = StatePreparation(params.n_sites)
        self.measurement = MeasurementCircuit(params.n_sites)
        
        # Cache Hamiltonian matrix
        self._H_matrix = None
        self._H_dense = None
    
    @property
    def H_matrix(self) -> np.ndarray:
        """Get (dense) Hamiltonian matrix."""
        if self._H_dense is None:
            self._H_matrix = self.hamiltonian.build_hamiltonian()
            self._H_dense = self._H_matrix.toarray()
        return self._H_dense
    
    # =========================================================================
    # State Preparation Methods
    # =========================================================================
    
    def prepare_ground_state(self, method: str = 'exact') -> np.ndarray:
        """
        Prepare the ground state of the Ising Hamiltonian.
        
        Parameters:
            method: 'exact' for diagonalization, 'variational' for VQE
        
        Returns:
            Ground state vector |Ω⟩
        """
        if method == 'exact':
            eigenvalues, eigenvectors = self.hamiltonian.diagonalize(n_states=1)
            return eigenvectors[:, 0]
        elif method == 'variational':
            # Simple variational optimization
            from scipy.optimize import minimize
            
            n_layers = 3
            n_params = self.n_qubits * n_layers
            
            def energy(theta):
                circuit = self.state_prep.variational_ground_state_circuit(theta)
                state = circuit.apply(self.state_prep.all_zeros_state())
                return np.real(np.conj(state) @ self.H_matrix @ state)
            
            result = minimize(energy, np.random.randn(n_params) * 0.1)
            circuit = self.state_prep.variational_ground_state_circuit(result.x)
            return circuit.apply(self.state_prep.all_zeros_state())
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def prepare_excited_state(self, n: int) -> np.ndarray:
        """
        Prepare nth excited state via exact diagonalization.
        
        Parameters:
            n: Excitation level (0 = ground state, 1 = first excited, etc.)
        
        Returns:
            nth eigenstate |n⟩
        """
        eigenvalues, eigenvectors = self.hamiltonian.diagonalize(n_states=n+1)
        return eigenvectors[:, n]
    
    def prepare_single_particle_state(
        self, 
        momentum: float = None,
        position: int = None,
        width: float = None
    ) -> np.ndarray:
        """
        Prepare a single-particle excitation state.
        
        Three modes:
        1. Momentum eigenstate: |k⟩ (specify momentum)
        2. Position eigenstate: localized spin flip (specify position)
        3. Wavepacket: Gaussian with momentum (specify all three)
        
        In the Ising model, single-particle states correspond to
        domain wall excitations (kinks/antikinks).
        """
        if width is not None and position is not None and momentum is not None:
            # Gaussian wavepacket
            return self.state_prep.wavepacket_state(
                center=position,
                width=width,
                momentum=momentum
            )
        elif momentum is not None:
            # Momentum eigenstate (approximate)
            k_index = max(1, min(self.n_qubits, 
                         int(momentum * (self.n_qubits + 1) / np.pi)))
            return self.state_prep.momentum_eigenstate(k_index)
        elif position is not None:
            # Localized state
            return self.state_prep.single_spin_flip_state(position)
        else:
            # Default: create at center
            return self.state_prep.single_spin_flip_state(self.n_qubits // 2)
    
    def prepare_two_particle_state(
        self,
        positions: Tuple[int, int] = None,
        momenta: Tuple[float, float] = None,
        width: float = None
    ) -> np.ndarray:
        """
        Prepare a two-particle state for scattering studies.
        
        For studying particle-particle scattering in IFT.
        
        Parameters:
            positions: Tuple of initial positions
            momenta: Tuple of initial momenta (for wavepackets)
            width: Wavepacket width (if using wavepackets)
        
        Returns:
            Two-particle state vector
        """
        if positions is None:
            # Default: particles at 1/4 and 3/4 of chain
            positions = (self.n_qubits // 4, 3 * self.n_qubits // 4)
        
        if width is not None and momenta is not None:
            # Two wavepackets
            wp1 = self.state_prep.wavepacket_state(
                center=positions[0], width=width, momentum=momenta[0]
            )
            wp2 = self.state_prep.wavepacket_state(
                center=positions[1], width=width, momentum=momenta[1]
            )
            # Approximate two-particle state (assuming weak overlap)
            # In fermionic language: c†_k1 c†_k2 |0⟩
            # This is more complex; we use direct construction
            return self.state_prep.two_particle_state(positions[0], positions[1])
        else:
            return self.state_prep.two_particle_state(positions[0], positions[1])
    
    def prepare_domain_wall_state(self, position: int = None) -> np.ndarray:
        """
        Prepare domain wall (kink) state.
        
        |↑↑...↑↓↓...↓⟩ - interface between two ferromagnetic domains.
        
        In the continuum limit, this corresponds to a massive kink
        excitation in the Ising field theory.
        """
        if position is None:
            position = self.n_qubits // 2
        return self.state_prep.domain_wall_state(position)
    
    # =========================================================================
    # Time Evolution Methods
    # =========================================================================
    
    def exact_time_evolution(
        self, 
        initial_state: np.ndarray,
        times: np.ndarray
    ) -> List[np.ndarray]:
        """
        Compute exact time evolution using matrix exponentiation.
        
        |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩
        
        Parameters:
            initial_state: Initial state vector |ψ(0)⟩
            times: Array of time points to evaluate
        
        Returns:
            List of state vectors at each time point
        """
        H = self.H_matrix
        states = []
        
        for t in times:
            U = expm(-1j * H * t)
            state_t = U @ initial_state
            states.append(state_t)
        
        return states
    
    def trotter_time_evolution(
        self,
        initial_state: np.ndarray,
        total_time: float,
        n_steps: int = None,
        order: int = None
    ) -> Tuple[np.ndarray, QuantumCircuit]:
        """
        Compute Trotterized time evolution.
        
        exp(-iHt) ≈ [exp(-iH_A δt) exp(-iH_B δt)]^n
        
        Parameters:
            initial_state: Initial state vector
            total_time: Total evolution time t
            n_steps: Number of Trotter steps (default from params)
            order: Trotter order (default from params)
        
        Returns:
            Tuple of (final_state, circuit)
        """
        if n_steps is None:
            n_steps = self.params.n_trotter_steps
        if order is None:
            order = self.params.trotter_order
        
        circuit = self.trotter.build_evolution_circuit(
            total_time=total_time,
            n_steps=n_steps,
            order=order
        )
        
        final_state = circuit.apply(initial_state)
        
        return final_state, circuit
    
    def time_evolve(
        self,
        initial_state: np.ndarray,
        total_time: float = None,
        n_time_points: int = 50,
        method: str = 'exact',
        compute_observables: bool = True
    ) -> SimulationResults:
        """
        Full time evolution with observable computation.
        
        Parameters:
            initial_state: Initial state vector
            total_time: Total simulation time (default from params)
            n_time_points: Number of time points to record
            method: 'exact' or 'trotter'
            compute_observables: Whether to compute observables
        
        Returns:
            SimulationResults containing states and observables
        """
        if total_time is None:
            total_time = self.params.total_time
        
        times = np.linspace(0, total_time, n_time_points)
        results = SimulationResults(times=times)
        
        if method == 'exact':
            results.states = self.exact_time_evolution(initial_state, times)
        else:
            # Trotter evolution at each time point
            results.states = []
            for t in times:
                if t == 0:
                    results.states.append(initial_state.copy())
                else:
                    n_steps = max(1, int(t * self.params.n_trotter_steps / total_time))
                    state, _ = self.trotter_time_evolution(
                        initial_state, t, n_steps, self.params.trotter_order
                    )
                    results.states.append(state)
        
        if compute_observables:
            results = self._compute_observables(results)
        
        return results
    
    # =========================================================================
    # Observable Computation
    # =========================================================================
    
    def _compute_observables(self, results: SimulationResults) -> SimulationResults:
        """Compute all observables from state evolution."""
        n_times = len(results.times)
        
        results.energies = np.zeros(n_times)
        results.magnetization = np.zeros(n_times)
        results.fermion_parity = np.zeros(n_times)
        
        for i, state in enumerate(results.states):
            results.energies[i] = self.compute_energy(state)
            results.magnetization[i] = self.compute_magnetization(state)
            results.fermion_parity[i] = compute_fermion_parity(state)
        
        return results
    
    def compute_energy(self, state: np.ndarray) -> float:
        """
        Compute energy expectation value ⟨H⟩.
        
        E = ⟨ψ|H|ψ⟩
        """
        return np.real(np.conj(state) @ self.H_matrix @ state)
    
    def compute_magnetization(self, state: np.ndarray) -> float:
        """
        Compute magnetization (order parameter).
        
        M = ⟨Σᵢ σᵢˣ⟩/N
        
        This is the order parameter that distinguishes phases:
        - M ≠ 0 in ferromagnetic phase (g < 1)
        - M = 0 in paramagnetic phase (g > 1)
        """
        total_mx = 0.0
        
        for site in range(self.n_qubits):
            # Build σˣ operator at site
            sigma_x_op = np.array([[1.0]])
            for i in range(self.n_qubits):
                if i == site:
                    sigma_x_op = np.kron(sigma_x_op, 
                                        np.array([[0, 1], [1, 0]], dtype=complex))
                else:
                    sigma_x_op = np.kron(sigma_x_op, np.eye(2))
            
            total_mx += np.real(np.conj(state) @ sigma_x_op @ state)
        
        return total_mx / self.n_qubits
    
    def compute_correlation_function(
        self, 
        state: np.ndarray,
        operator: str = 'Z'
    ) -> np.ndarray:
        """
        Compute two-point correlation function.
        
        C(r) = ⟨σ₀ᵅ σᵣᵅ⟩ - ⟨σ₀ᵅ⟩⟨σᵣᵅ⟩
        
        where α = X, Y, or Z.
        
        Parameters:
            state: State vector to measure
            operator: 'X', 'Y', or 'Z'
        
        Returns:
            Array of correlation values C(r) for r = 0, 1, ..., n-1
        """
        pauli = {'X': np.array([[0, 1], [1, 0]], dtype=complex),
                 'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
                 'Z': np.array([[1, 0], [0, -1]], dtype=complex)}[operator]
        
        correlations = np.zeros(self.n_qubits)
        
        # Compute single-site expectation at site 0
        op_0 = np.array([[1.0]])
        for i in range(self.n_qubits):
            if i == 0:
                op_0 = np.kron(op_0, pauli)
            else:
                op_0 = np.kron(op_0, np.eye(2))
        expect_0 = np.real(np.conj(state) @ op_0 @ state)
        
        for r in range(self.n_qubits):
            # Build two-site operator σ₀σᵣ
            if r == 0:
                correlations[r] = 1.0  # ⟨σ₀²⟩ = 1
                continue
            
            op_0r = np.array([[1.0]])
            for i in range(self.n_qubits):
                if i == 0 or i == r:
                    op_0r = np.kron(op_0r, pauli)
                else:
                    op_0r = np.kron(op_0r, np.eye(2))
            
            expect_0r = np.real(np.conj(state) @ op_0r @ state)
            
            # Build single-site expectation at site r
            op_r = np.array([[1.0]])
            for i in range(self.n_qubits):
                if i == r:
                    op_r = np.kron(op_r, pauli)
                else:
                    op_r = np.kron(op_r, np.eye(2))
            expect_r = np.real(np.conj(state) @ op_r @ state)
            
            # Connected correlator
            correlations[r] = expect_0r - expect_0 * expect_r
        
        return correlations
    
    def compute_correlation_length(self, state: np.ndarray = None) -> float:
        """
        Compute correlation length from exponential decay of correlations.
        
        C(r) ~ exp(-r/ξ) for large r
        
        Near criticality: ξ ~ |g - 1|^(-ν) with ν = 1 for 2D Ising.
        """
        if state is None:
            state = self.prepare_ground_state()
        
        corr = self.compute_correlation_function(state, operator='X')
        
        # Fit exponential decay (avoiding r=0)
        r_vals = np.arange(1, self.n_qubits)
        log_corr = np.log(np.abs(corr[1:]) + 1e-15)
        
        # Linear fit: log(C) = -r/ξ + const
        if len(r_vals) > 2:
            coeffs = np.polyfit(r_vals, log_corr, 1)
            xi = -1.0 / coeffs[0] if coeffs[0] < 0 else np.inf
            return max(0, xi)
        
        return np.inf
    
    def compute_entanglement_entropy(
        self, 
        state: np.ndarray,
        subsystem_size: int = None
    ) -> float:
        """
        Compute entanglement entropy of bipartition.
        
        S = -Tr(ρ_A log ρ_A)
        
        where ρ_A = Tr_B(|ψ⟩⟨ψ|) is the reduced density matrix.
        
        Parameters:
            state: State vector
            subsystem_size: Size of subsystem A (default: half)
        
        Returns:
            von Neumann entanglement entropy
        """
        if subsystem_size is None:
            subsystem_size = self.n_qubits // 2
        
        # Reshape state to bipartite form
        dim_A = 2 ** subsystem_size
        dim_B = 2 ** (self.n_qubits - subsystem_size)
        
        psi_matrix = state.reshape(dim_A, dim_B)
        
        # Compute reduced density matrix via SVD
        # ρ_A = Σᵢ λᵢ² |uᵢ⟩⟨uᵢ|
        _, s, _ = np.linalg.svd(psi_matrix)
        
        # Schmidt coefficients squared (eigenvalues of ρ_A)
        schmidt_sq = s ** 2
        schmidt_sq = schmidt_sq[schmidt_sq > 1e-15]  # Remove zeros
        
        # von Neumann entropy
        entropy = -np.sum(schmidt_sq * np.log(schmidt_sq))
        
        return entropy
    
    # =========================================================================
    # Mass Spectrum Analysis
    # =========================================================================
    
    def compute_mass_spectrum(self, n_states: int = 10) -> np.ndarray:
        """
        Compute mass spectrum (energy gaps above ground state).
        
        Δₙ = Eₙ - E₀
        
        Near E₈ point, ratios Δₙ/Δ₁ should approach E₈ mass ratios.
        """
        eigenvalues, _ = self.hamiltonian.diagonalize(n_states=n_states)
        E0 = eigenvalues[0]
        return eigenvalues[1:] - E0
    
    def extract_mass_ratios(self, n_states: int = 10) -> np.ndarray:
        """
        Extract mass ratios mₙ/m₁ from spectrum.
        
        Compare with E₈ analytical predictions for validation.
        """
        gaps = self.compute_mass_spectrum(n_states)
        if len(gaps) < 2 or gaps[0] < 1e-10:
            return np.array([])
        return gaps / gaps[0]
    
    def compare_with_e8(self, n_states: int = 8) -> Dict[str, np.ndarray]:
        """
        Compare computed mass ratios with E₈ predictions.
        
        Returns dictionary with:
        - 'computed': Numerically computed mass ratios
        - 'e8_exact': Exact E₈ predictions
        - 'relative_error': (computed - exact) / exact
        """
        computed = self.extract_mass_ratios(n_states + 1)
        e8_exact = E8MassRatios.compute_ratios()[:len(computed)]
        
        rel_error = np.abs(computed - e8_exact) / e8_exact
        
        return {
            'computed': computed,
            'e8_exact': e8_exact,
            'relative_error': rel_error
        }
    
    # =========================================================================
    # Spectral Analysis
    # =========================================================================
    
    def compute_spectral_function(
        self,
        initial_state: np.ndarray,
        operator: np.ndarray,
        omega_range: Tuple[float, float],
        n_omega: int = 100,
        broadening: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral function A(ω) using Lehmann representation.
        
        A(ω) = Σₙ |⟨n|O|ψ⟩|² δ(ω - (Eₙ - E₀))
        
        With Lorentzian broadening:
        A(ω) = Σₙ |⟨n|O|ψ⟩|² η / [(ω - Eₙ + E₀)² + η²]
        
        Parameters:
            initial_state: Initial state |ψ⟩
            operator: Operator O to probe
            omega_range: (ωmin, ωmax) frequency range
            n_omega: Number of frequency points
            broadening: Lorentzian broadening η
        
        Returns:
            (omega_array, spectral_function)
        """
        # Get all eigenstates
        eigenvalues, eigenvectors = self.hamiltonian.diagonalize(
            n_states=min(50, self.dim - 2)
        )
        E0 = eigenvalues[0]
        
        # Compute matrix elements ⟨n|O|ψ⟩
        O_psi = operator @ initial_state
        matrix_elements = np.abs(eigenvectors.T.conj() @ O_psi) ** 2
        
        # Energy differences
        excitation_energies = eigenvalues - E0
        
        # Compute spectral function
        omega = np.linspace(omega_range[0], omega_range[1], n_omega)
        A_omega = np.zeros(n_omega)
        
        for n, (En, mel) in enumerate(zip(excitation_energies, matrix_elements)):
            # Lorentzian
            A_omega += mel * broadening / ((omega - En)**2 + broadening**2)
        
        return omega, A_omega / np.pi
    
    def compute_dynamic_structure_factor(
        self,
        k_values: np.ndarray = None,
        omega_range: Tuple[float, float] = (0, 5),
        n_omega: int = 100,
        broadening: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute dynamic structure factor S(k, ω).
        
        S(k, ω) = Σₙ |⟨n|σₖˣ|0⟩|² δ(ω - Eₙ + E₀)
        
        where σₖˣ = Σⱼ exp(ikxⱼ) σⱼˣ / √N
        
        Returns:
            (k_array, omega_array, S_k_omega)
        """
        if k_values is None:
            # Allowed momenta for open boundaries
            k_values = np.pi * np.arange(1, self.n_qubits + 1) / (self.n_qubits + 1)
        
        omega = np.linspace(omega_range[0], omega_range[1], n_omega)
        S = np.zeros((len(k_values), n_omega))
        
        # Get ground state and eigenstates
        eigenvalues, eigenvectors = self.hamiltonian.diagonalize(
            n_states=min(50, self.dim - 2)
        )
        ground_state = eigenvectors[:, 0]
        E0 = eigenvalues[0]
        
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        
        for ki, k in enumerate(k_values):
            # Build momentum-space operator σₖˣ
            sigma_k = np.zeros((self.dim, self.dim), dtype=complex)
            
            for j in range(self.n_qubits):
                x_j = j + 0.5  # Site position
                phase = np.exp(1j * k * x_j) / np.sqrt(self.n_qubits)
                
                # σⱼˣ
                op_j = np.array([[1.0]])
                for i in range(self.n_qubits):
                    if i == j:
                        op_j = np.kron(op_j, pauli_x)
                    else:
                        op_j = np.kron(op_j, np.eye(2))
                
                sigma_k += phase * op_j
            
            # Compute spectral function for this k
            _, S[ki, :] = self.compute_spectral_function(
                ground_state, sigma_k, omega_range, n_omega, broadening
            )
        
        return k_values, omega, S
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute fidelity between two states.
        
        F = |⟨ψ₁|ψ₂⟩|²
        """
        return np.abs(np.conj(state1) @ state2) ** 2
    
    def trotter_error(self, state: np.ndarray, time: float) -> float:
        """
        Compute Trotter error by comparing with exact evolution.
        
        Error = ||exp(-iHt)|ψ⟩ - U_Trotter|ψ⟩||
        """
        # Exact evolution
        exact_state = expm(-1j * self.H_matrix * time) @ state
        
        # Trotter evolution
        trotter_state, _ = self.trotter_time_evolution(
            state, time, 
            self.params.n_trotter_steps,
            self.params.trotter_order
        )
        
        return np.linalg.norm(exact_state - trotter_state)
    
    def get_circuit_stats(
        self, 
        total_time: float = None, 
        n_steps: int = None
    ) -> Dict:
        """Get statistics about the quantum circuit."""
        if total_time is None:
            total_time = self.params.total_time
        if n_steps is None:
            n_steps = self.params.n_trotter_steps
        
        circuit = self.trotter.build_evolution_circuit(
            total_time, n_steps, self.params.trotter_order
        )
        
        return {
            'n_qubits': self.n_qubits,
            'depth': circuit.depth(),
            'gate_counts': circuit.gate_count(),
            'total_gates': len(circuit.gates),
            'trotter_steps': n_steps,
            'trotter_order': self.params.trotter_order,
            'estimated_error': self.trotter.trotter_error_bound(
                total_time, n_steps, self.params.trotter_order
            )
        }
    
    def summary(self) -> str:
        """Return summary of simulation setup."""
        lines = [
            "=" * 60,
            "Ising Field Theory Simulation",
            "=" * 60,
            f"System size: {self.n_qubits} sites",
            f"Coupling J: {self.params.J}",
            f"Transverse field hₓ: {self.params.h_transverse}",
            f"Longitudinal field hᵤ: {self.params.h_longitudinal}",
            f"Dimensionless coupling g = hₓ/J: {self.params.g:.4f}",
            f"Phase: {'critical' if self.params.is_critical else 'paramagnetic' if self.params.is_paramagnetic else 'ferromagnetic'}",
            f"Boundary conditions: {self.params.boundary}",
            "-" * 60,
            f"Hilbert space dimension: {self.dim}",
            f"Trotter steps: {self.params.n_trotter_steps}",
            f"Trotter order: {self.params.trotter_order}",
            "=" * 60,
        ]
        return "\n".join(lines)
