"""
Ising Field Theory Hamiltonian Construction

The quantum Ising chain Hamiltonian in transverse and longitudinal fields:

    H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ - hᵤ Σᵢ σᵢᶻ

Critical point: hₓ = J (with hᵤ = 0)

Scaling limit near criticality:
    - Lattice spacing a → 0
    - Correlation length ξ = a/|1 - hₓ/J| → finite
    - Continuum Majorana field theory emerges

E₈ mass spectrum (hₓ = J, hᵤ ≠ 0):
    mₙ/m₁ given by roots of characteristic polynomial of E₈ Cartan matrix
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass


# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_I = np.array([[1, 0], [0, 1]], dtype=complex)


@dataclass
class E8MassRatios:
    """
    E₈ integrable field theory mass ratios.
    
    At the critical transverse field with longitudinal perturbation,
    the spectrum contains 8 stable particles with masses:
    
        m₁ = m (fundamental mass scale)
        m₂ = 2m cos(π/5)
        m₃ = 2m cos(π/30)
        m₄ = 2m₂ cos(7π/30)
        m₅ = 2m₂ cos(2π/15)
        m₆ = 2m₂ cos(π/30)
        m₇ = 4m₂ cos(π/5) cos(7π/30)
        m₈ = 4m₂ cos(π/5) cos(2π/15)
    
    These arise from the E₈ Lie algebra structure.
    """
    
    @staticmethod
    def compute_ratios() -> np.ndarray:
        """Compute exact E₈ mass ratios m_n/m_1."""
        m1 = 1.0
        m2 = 2 * np.cos(np.pi / 5)
        m3 = 2 * np.cos(np.pi / 30)
        m4 = 2 * m2 * np.cos(7 * np.pi / 30)
        m5 = 2 * m2 * np.cos(2 * np.pi / 15)
        m6 = 2 * m2 * np.cos(np.pi / 30)
        m7 = 4 * m2 * np.cos(np.pi / 5) * np.cos(7 * np.pi / 30)
        m8 = 4 * m2 * np.cos(np.pi / 5) * np.cos(2 * np.pi / 15)
        
        return np.array([m1, m2, m3, m4, m5, m6, m7, m8])
    
    @staticmethod
    def golden_ratio_relations() -> dict:
        """
        Express mass ratios in terms of golden ratio φ = (1+√5)/2.
        
        m₂/m₁ = φ (golden ratio)
        m₃/m₁ = φ + 1 = φ²
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        return {
            "phi": phi,
            "m2/m1": phi,
            "m3/m1": phi**2,
            "m4/m1": phi**2 + phi,
            "m5/m1": phi**3,
        }


class IsingHamiltonian:
    """
    Construct and analyze the quantum Ising chain Hamiltonian.
    
    H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ - hᵤ Σᵢ σᵢᶻ
    
    Parameters:
        n_sites: Number of lattice sites
        J: Ising coupling strength (σᶻσᶻ term)
        h_transverse: Transverse field strength (σˣ term)
        h_longitudinal: Longitudinal field strength (σᶻ term)
        boundary: 'open' or 'periodic'
    """
    
    def __init__(
        self,
        n_sites: int,
        J: float = 1.0,
        h_transverse: float = 1.0,
        h_longitudinal: float = 0.0,
        boundary: str = 'open'
    ):
        self.n_sites = n_sites
        self.J = J
        self.h_x = h_transverse
        self.h_z = h_longitudinal
        self.boundary = boundary
        self.dim = 2 ** n_sites
        
        # Compute criticality parameter
        self.g = h_transverse / J if J != 0 else np.inf
        self.is_critical = np.isclose(self.g, 1.0, rtol=1e-10)
        
        # Cache for Hamiltonian matrix
        self._H_matrix = None
        self._eigenvalues = None
        self._eigenvectors = None
    
    def _single_site_operator(self, op: np.ndarray, site: int) -> csr_matrix:
        r"""
        Construct operator acting on single site embedded in full Hilbert space.
        
        O_site = I ⊗ ... ⊗ I ⊗ op ⊗ I ⊗ ... ⊗ I
                 \_________/       \___________/
                   site-1            n-site-1
        """
        if site < 0 or site >= self.n_sites:
            raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
        
        result = csr_matrix(np.array([[1.0]]))
        for i in range(self.n_sites):
            if i == site:
                result = kron(result, csr_matrix(op))
            else:
                result = kron(result, eye(2, format='csr'))
        return result
    
    def _two_site_operator(
        self, 
        op1: np.ndarray, 
        site1: int, 
        op2: np.ndarray, 
        site2: int
    ) -> csr_matrix:
        """
        Construct two-site operator: op1_site1 ⊗ op2_site2
        """
        result = csr_matrix(np.array([[1.0]]))
        for i in range(self.n_sites):
            if i == site1:
                result = kron(result, csr_matrix(op1))
            elif i == site2:
                result = kron(result, csr_matrix(op2))
            else:
                result = kron(result, eye(2, format='csr'))
        return result
    
    def build_hamiltonian(self) -> csr_matrix:
        """
        Construct the full Hamiltonian matrix.
        
        H = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ - hᵤ Σᵢ σᵢᶻ
        """
        if self._H_matrix is not None:
            return self._H_matrix
        
        H = csr_matrix((self.dim, self.dim), dtype=complex)
        
        # Ising coupling: -J Σᵢ σᵢᶻσᵢ₊₁ᶻ
        n_bonds = self.n_sites - 1 if self.boundary == 'open' else self.n_sites
        for i in range(n_bonds):
            j = (i + 1) % self.n_sites
            H -= self.J * self._two_site_operator(SIGMA_Z, i, SIGMA_Z, j)
        
        # Transverse field: -hₓ Σᵢ σᵢˣ
        for i in range(self.n_sites):
            H -= self.h_x * self._single_site_operator(SIGMA_X, i)
        
        # Longitudinal field: -hᵤ Σᵢ σᵢᶻ
        if self.h_z != 0:
            for i in range(self.n_sites):
                H -= self.h_z * self._single_site_operator(SIGMA_Z, i)
        
        self._H_matrix = H
        return H
    
    def get_terms(self) -> Tuple[List[csr_matrix], List[csr_matrix], List[csr_matrix]]:
        """
        Return Hamiltonian decomposed into terms for Trotter decomposition.
        
        Returns:
            zz_terms: List of σᵢᶻσᵢ₊₁ᶻ operators with coefficient -J
            x_terms: List of σᵢˣ operators with coefficient -hₓ
            z_terms: List of σᵢᶻ operators with coefficient -hᵤ
        """
        zz_terms = []
        x_terms = []
        z_terms = []
        
        # ZZ coupling terms
        n_bonds = self.n_sites - 1 if self.boundary == 'open' else self.n_sites
        for i in range(n_bonds):
            j = (i + 1) % self.n_sites
            zz_terms.append((-self.J, i, j))
        
        # X field terms
        for i in range(self.n_sites):
            x_terms.append((-self.h_x, i))
        
        # Z field terms
        if self.h_z != 0:
            for i in range(self.n_sites):
                z_terms.append((-self.h_z, i))
        
        return zz_terms, x_terms, z_terms
    
    def diagonalize(self, n_states: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute lowest n_states eigenvalues and eigenvectors.
        
        For critical Ising chain, the spectrum should show:
        - Linear dispersion (Majorana fermion)
        - E₈ mass ratios when h_z ≠ 0
        """
        H = self.build_hamiltonian()
        
        if self.dim <= 64:
            # Full diagonalization for small systems
            H_dense = H.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            idx = np.argsort(eigenvalues)[:n_states]
            self._eigenvalues = eigenvalues[idx]
            self._eigenvectors = eigenvectors[:, idx]
        else:
            # Lanczos for large systems
            n_states = min(n_states, self.dim - 2)
            eigenvalues, eigenvectors = eigsh(H, k=n_states, which='SA')
            idx = np.argsort(eigenvalues)
            self._eigenvalues = eigenvalues[idx]
            self._eigenvectors = eigenvectors[:, idx]
        
        return self._eigenvalues, self._eigenvectors
    
    def compute_mass_gaps(self) -> np.ndarray:
        """
        Compute mass gaps Δₙ = Eₙ - E₀.
        
        Near E₈ point, ratios Δₙ/Δ₁ should approach E₈ mass ratios.
        """
        if self._eigenvalues is None:
            self.diagonalize()
        
        E0 = self._eigenvalues[0]
        gaps = self._eigenvalues[1:] - E0
        return gaps
    
    def correlation_length(self) -> float:
        """
        Compute correlation length ξ from mass gap.
        
        ξ = 1/m where m = E₁ - E₀ (in lattice units)
        
        At criticality, ξ → ∞.
        """
        gaps = self.compute_mass_gaps()
        if len(gaps) == 0 or gaps[0] < 1e-10:
            return np.inf
        return 1.0 / gaps[0]
    
    def free_fermion_spectrum(self) -> np.ndarray:
        """
        Exact free fermion spectrum for h_z = 0.
        
        At h_z = 0, the model is exactly solvable via Jordan-Wigner transformation.
        Single-particle energies:
        
            ε_k = 2√[(J - hₓcos(k))² + (hₓsin(k))²]
        
        where k = πn/(N+1) for open boundary conditions.
        """
        if self.h_z != 0:
            raise ValueError("Free fermion spectrum only valid for h_z = 0")
        
        if self.boundary == 'open':
            k_values = np.pi * np.arange(1, self.n_sites + 1) / (self.n_sites + 1)
        else:
            k_values = 2 * np.pi * np.arange(self.n_sites) / self.n_sites
        
        # Single-particle dispersion relation
        epsilon_k = 2 * np.sqrt(
            (self.J - self.h_x * np.cos(k_values))**2 + 
            (self.h_x * np.sin(k_values))**2
        )
        
        return np.sort(epsilon_k)
    
    def scaling_dimension(self) -> dict:
        """
        Return conformal scaling dimensions at criticality.
        
        The critical Ising model is described by the c=1/2 minimal model M(3,4).
        
        Primary fields:
            - Identity (I): h = 0
            - Energy density (ε): h = 1/2  
            - Spin field (σ): h = 1/16
        """
        return {
            "central_charge": 0.5,
            "identity": 0.0,
            "energy_density": 0.5,
            "spin_field": 1/16,
        }


def build_ising_hamiltonian_matrix(
    n_qubits: int,
    J: float,
    h_x: float,
    h_z: float = 0.0,
    periodic: bool = False
) -> np.ndarray:
    """
    Utility function to build dense Hamiltonian matrix directly.
    
    H = -J Σᵢ ZᵢZᵢ₊₁ - hₓ Σᵢ Xᵢ - hᵤ Σᵢ Zᵢ
    """
    ham = IsingHamiltonian(
        n_sites=n_qubits,
        J=J,
        h_transverse=h_x,
        h_longitudinal=h_z,
        boundary='periodic' if periodic else 'open'
    )
    return ham.build_hamiltonian().toarray()
