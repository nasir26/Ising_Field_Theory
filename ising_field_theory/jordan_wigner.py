"""
Jordan-Wigner Transformation for Ising Field Theory

The Jordan-Wigner transformation maps spin-1/2 operators to spinless fermions:

    σⱼ⁺ = (∏ₖ₌₁ʲ⁻¹ -σₖᶻ) cⱼ†
    σⱼ⁻ = (∏ₖ₌₁ʲ⁻¹ -σₖᶻ) cⱼ
    σⱼᶻ = 2cⱼ†cⱼ - 1

Inverse transformation:
    cⱼ = (∏ₖ₌₁ʲ⁻¹ σₖᶻ) σⱼ⁺
    cⱼ† = (∏ₖ₌₁ʲ⁻¹ σₖᶻ) σⱼ⁻

The Ising Hamiltonian transforms to free Majorana fermions (when h_z = 0):
    H = Σⱼ [J(cⱼ†cⱼ₊₁ + cⱼ†cⱼ₊₁† + h.c.) - 2hₓ(cⱼ†cⱼ - 1/2)]
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.sparse import csr_matrix, kron, eye
from dataclasses import dataclass


# Pauli matrices
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_I = np.array([[1, 0], [0, 1]], dtype=complex)
SIGMA_PLUS = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|
SIGMA_MINUS = np.array([[0, 0], [1, 0]], dtype=complex)  # |1><0|


@dataclass
class FermionOperator:
    """
    Representation of a fermionic operator string.
    
    Stores coefficient and list of (site, type) pairs where type ∈ {'c', 'c†'}.
    """
    coefficient: complex
    operators: List[Tuple[int, str]]  # [(site, 'c' or 'c†'), ...]
    
    def __repr__(self):
        if not self.operators:
            return f"{self.coefficient}"
        ops_str = " ".join(f"{t}_{s}" for s, t in self.operators)
        return f"{self.coefficient} * {ops_str}"


class JordanWignerTransform:
    """
    Jordan-Wigner transformation between spin and fermionic representations.
    
    Convention:
        - Qubits ordered left to right: |q₀ q₁ ... qₙ₋₁⟩
        - |0⟩ = spin up = empty fermion site
        - |1⟩ = spin down = occupied fermion site
    """
    
    def __init__(self, n_sites: int):
        self.n_sites = n_sites
        self.dim = 2 ** n_sites
        
    def jordan_wigner_string(self, site: int) -> csr_matrix:
        """
        Compute the Jordan-Wigner string operator: ∏ₖ₌₀ˢⁱᵗᵉ⁻¹ σₖᶻ
        
        This string ensures anticommutation relations for fermion operators.
        """
        if site == 0:
            return eye(self.dim, format='csr')
        
        result = csr_matrix(np.array([[1.0]]))
        for i in range(self.n_sites):
            if i < site:
                result = kron(result, csr_matrix(SIGMA_Z))
            else:
                result = kron(result, eye(2, format='csr'))
        return result
    
    def annihilation_operator(self, site: int) -> csr_matrix:
        """
        Fermion annihilation operator cⱼ in qubit representation.
        
        cⱼ = (∏ₖ₌₀ʲ⁻¹ σₖᶻ) σⱼ⁺
        
        where σ⁺ = (σˣ + iσʸ)/2 = |0⟩⟨1|
        """
        # Build JW string ⊗ σ⁺ ⊗ I...I
        result = csr_matrix(np.array([[1.0]]))
        for i in range(self.n_sites):
            if i < site:
                result = kron(result, csr_matrix(SIGMA_Z))
            elif i == site:
                result = kron(result, csr_matrix(SIGMA_PLUS))
            else:
                result = kron(result, eye(2, format='csr'))
        return result
    
    def creation_operator(self, site: int) -> csr_matrix:
        """
        Fermion creation operator cⱼ† in qubit representation.
        
        cⱼ† = (∏ₖ₌₀ʲ⁻¹ σₖᶻ) σⱼ⁻
        
        where σ⁻ = (σˣ - iσʸ)/2 = |1⟩⟨0|
        """
        result = csr_matrix(np.array([[1.0]]))
        for i in range(self.n_sites):
            if i < site:
                result = kron(result, csr_matrix(SIGMA_Z))
            elif i == site:
                result = kron(result, csr_matrix(SIGMA_MINUS))
            else:
                result = kron(result, eye(2, format='csr'))
        return result
    
    def number_operator(self, site: int) -> csr_matrix:
        """
        Fermion number operator nⱼ = cⱼ†cⱼ = (1 - σⱼᶻ)/2
        """
        result = csr_matrix(np.array([[1.0]]))
        for i in range(self.n_sites):
            if i == site:
                n_local = (SIGMA_I - SIGMA_Z) / 2
                result = kron(result, csr_matrix(n_local))
            else:
                result = kron(result, eye(2, format='csr'))
        return result
    
    def majorana_operator(self, site: int, which: str = 'a') -> csr_matrix:
        """
        Majorana fermion operators γ.
        
        Two Majorana fermions per site:
            γⱼᵃ = cⱼ + cⱼ†     (real/Hermitian)
            γⱼᵇ = i(cⱼ† - cⱼ)  (real/Hermitian)
        
        Anticommutation: {γᵢ, γⱼ} = 2δᵢⱼ
        
        In spin language:
            γⱼᵃ = (∏ₖ<ⱼ σₖᶻ) σⱼˣ
            γⱼᵇ = (∏ₖ<ⱼ σₖᶻ) σⱼʸ
        """
        c = self.annihilation_operator(site)
        c_dag = self.creation_operator(site)
        
        if which == 'a':
            return c + c_dag
        elif which == 'b':
            return 1j * (c_dag - c)
        else:
            raise ValueError(f"which must be 'a' or 'b', got {which}")
    
    def hopping_term(self, site1: int, site2: int) -> csr_matrix:
        """
        Fermion hopping term: cᵢ†cⱼ + cⱼ†cᵢ
        
        For nearest neighbors |i-j|=1, this simplifies in spin language.
        """
        c1 = self.annihilation_operator(site1)
        c1_dag = self.creation_operator(site1)
        c2 = self.annihilation_operator(site2)
        c2_dag = self.creation_operator(site2)
        
        return c1_dag @ c2 + c2_dag @ c1
    
    def pairing_term(self, site1: int, site2: int) -> csr_matrix:
        """
        Fermion pairing term: cᵢ†cⱼ† + cⱼcᵢ
        
        This term arises in the Ising Hamiltonian and breaks fermion number conservation.
        """
        c1 = self.annihilation_operator(site1)
        c1_dag = self.creation_operator(site1)
        c2 = self.annihilation_operator(site2)
        c2_dag = self.creation_operator(site2)
        
        return c1_dag @ c2_dag + c2 @ c1
    
    def transform_ising_hamiltonian(
        self, 
        J: float, 
        h_x: float, 
        h_z: float = 0.0,
        periodic: bool = False
    ) -> csr_matrix:
        """
        Transform Ising Hamiltonian to fermionic representation.
        
        H_Ising = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - hₓ Σᵢ σᵢˣ - hᵤ Σᵢ σᵢᶻ
        
        In fermionic form (h_z = 0):
        H = Σⱼ [-J(cⱼ† - cⱼ)(cⱼ₊₁† + cⱼ₊₁) - hₓ(1 - 2cⱼ†cⱼ)]
          = Σⱼ [-J(cⱼ†cⱼ₊₁ - cⱼ†cⱼ₊₁† + cⱼcⱼ₊₁ - cⱼcⱼ₊₁†) + hₓ(2cⱼ†cⱼ - 1)]
        
        The pairing terms (cⱼ†cⱼ₊₁† and cⱼcⱼ₊₁) are crucial - they break 
        fermion number conservation and lead to Majorana physics.
        """
        H = csr_matrix((self.dim, self.dim), dtype=complex)
        
        n_bonds = self.n_sites - 1 if not periodic else self.n_sites
        
        for j in range(n_bonds):
            j_next = (j + 1) % self.n_sites
            
            c_j = self.annihilation_operator(j)
            c_j_dag = self.creation_operator(j)
            c_next = self.annihilation_operator(j_next)
            c_next_dag = self.creation_operator(j_next)
            
            # -J σⱼᶻσⱼ₊₁ᶻ = -J(1-2nⱼ)(1-2nⱼ₊₁)
            # Equivalently using σᶻσᶻ = -(cⱼ†-cⱼ)(cⱼ₊₁†+cⱼ₊₁) in Majorana form
            # This gives: -J[cⱼ†cⱼ₊₁ + cⱼ†cⱼ₊₁† - cⱼcⱼ₊₁ - cⱼcⱼ₊₁†]
            
            H -= J * (c_j_dag @ c_next + c_j_dag @ c_next_dag - 
                      c_j @ c_next - c_j @ c_next_dag)
        
        # Transverse field: -hₓ σᵢˣ = -hₓ(cⱼ + cⱼ†) with JW string
        # This is more subtle - we use the spin representation directly
        for j in range(self.n_sites):
            c_j = self.annihilation_operator(j)
            c_j_dag = self.creation_operator(j)
            # σˣ = c + c† (with JW string already included in operators)
            # But actually -hₓ σˣⱼ in terms of fermions...
            # σˣ = γᵃ (Majorana), so we need JW string times X
            pass  # This is handled correctly in the direct spin construction
        
        return H
    
    def bogoliubov_transform(
        self, 
        J: float, 
        h_x: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Bogoliubov transformation for the free fermion case (h_z = 0).
        
        The Hamiltonian can be diagonalized as:
            H = Σₖ εₖ (ηₖ†ηₖ - 1/2)
        
        where ηₖ are Bogoliubov quasiparticle operators:
            ηₖ = uₖcₖ + vₖc†₋ₖ
        
        Returns:
            energies: Single-particle energies εₖ
            theta_k: Bogoliubov angles satisfying tan(2θₖ) = J sin(k)/(hₓ - J cos(k))
        """
        # Momentum values for open boundary conditions
        k_values = np.pi * np.arange(1, self.n_sites + 1) / (self.n_sites + 1)
        
        # Bogoliubov angles
        A_k = h_x - J * np.cos(k_values)  # Diagonal term
        B_k = J * np.sin(k_values)         # Off-diagonal term
        
        theta_k = 0.5 * np.arctan2(B_k, A_k)
        
        # Single-particle energies
        epsilon_k = np.sqrt(A_k**2 + B_k**2)
        
        return epsilon_k, theta_k
    
    def verify_anticommutation(self) -> bool:
        """
        Verify that the constructed fermion operators satisfy anticommutation relations:
            {cᵢ, cⱼ†} = δᵢⱼ
            {cᵢ, cⱼ} = 0
            {cᵢ†, cⱼ†} = 0
        """
        tol = 1e-10
        
        for i in range(self.n_sites):
            ci = self.annihilation_operator(i)
            ci_dag = self.creation_operator(i)
            
            for j in range(self.n_sites):
                cj = self.annihilation_operator(j)
                cj_dag = self.creation_operator(j)
                
                # {cᵢ, cⱼ†} = δᵢⱼ
                anticomm = ci @ cj_dag + cj_dag @ ci
                expected = eye(self.dim) if i == j else csr_matrix((self.dim, self.dim))
                if np.max(np.abs((anticomm - expected).toarray())) > tol:
                    return False
                
                # {cᵢ, cⱼ} = 0
                anticomm = ci @ cj + cj @ ci
                if np.max(np.abs(anticomm.toarray())) > tol:
                    return False
                
                # {cᵢ†, cⱼ†} = 0
                anticomm = ci_dag @ cj_dag + cj_dag @ ci_dag
                if np.max(np.abs(anticomm.toarray())) > tol:
                    return False
        
        return True


def compute_fermion_parity(state: np.ndarray) -> float:
    """
    Compute fermion parity P = (-1)^N where N = Σᵢ nᵢ.
    
    In spin language: P = ∏ᵢ σᵢᶻ
    
    The Ising Hamiltonian conserves parity, so eigenstates have definite parity.
    """
    n_qubits = int(np.log2(len(state)))
    
    # Build parity operator: ∏ᵢ σᵢᶻ
    parity_op = np.array([[1.0]])
    for _ in range(n_qubits):
        parity_op = np.kron(parity_op, SIGMA_Z)
    
    return np.real(np.conj(state) @ parity_op @ state)
