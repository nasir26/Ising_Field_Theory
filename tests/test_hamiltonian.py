"""
Unit tests for Hamiltonian module.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from ising_field_theory.hamiltonian import (
    IsingHamiltonian,
    E8MassRatios,
    SIGMA_X, SIGMA_Y, SIGMA_Z, SIGMA_I
)


class TestPauliMatrices:
    """Test Pauli matrix definitions."""
    
    def test_pauli_hermitian(self):
        """Pauli matrices should be Hermitian."""
        for sigma in [SIGMA_X, SIGMA_Y, SIGMA_Z]:
            assert np.allclose(sigma, sigma.conj().T)
    
    def test_pauli_eigenvalues(self):
        """Pauli matrices should have eigenvalues ±1."""
        for sigma in [SIGMA_X, SIGMA_Y, SIGMA_Z]:
            eigenvalues = np.linalg.eigvalsh(sigma)
            assert np.allclose(sorted(eigenvalues), [-1, 1])
    
    def test_pauli_anticommutation(self):
        """Pauli matrices should anticommute."""
        paulis = [SIGMA_X, SIGMA_Y, SIGMA_Z]
        for i, sigma_i in enumerate(paulis):
            for j, sigma_j in enumerate(paulis):
                anticomm = sigma_i @ sigma_j + sigma_j @ sigma_i
                if i == j:
                    assert np.allclose(anticomm, 2 * SIGMA_I)
                else:
                    assert np.allclose(anticomm, np.zeros((2, 2)))


class TestE8MassRatios:
    """Test E8 mass ratio calculations."""
    
    def test_fundamental_mass(self):
        """First mass ratio should be 1."""
        ratios = E8MassRatios.compute_ratios()
        assert np.isclose(ratios[0], 1.0)
    
    def test_golden_ratio(self):
        """m2/m1 should equal the golden ratio."""
        ratios = E8MassRatios.compute_ratios()
        phi = (1 + np.sqrt(5)) / 2
        assert np.isclose(ratios[1], phi)
    
    def test_mass_ordering(self):
        """Mass ratios should be increasing."""
        ratios = E8MassRatios.compute_ratios()
        for i in range(len(ratios) - 1):
            assert ratios[i] < ratios[i+1]
    
    def test_eight_particles(self):
        """Should have exactly 8 mass ratios."""
        ratios = E8MassRatios.compute_ratios()
        assert len(ratios) == 8


class TestIsingHamiltonian:
    """Test Ising Hamiltonian construction."""
    
    def test_dimension(self):
        """Hilbert space dimension should be 2^n."""
        for n in [2, 3, 4, 5]:
            ham = IsingHamiltonian(n_sites=n)
            assert ham.dim == 2**n
    
    def test_hermitian(self):
        """Hamiltonian should be Hermitian."""
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=0.5)
        H = ham.build_hamiltonian().toarray()
        assert np.allclose(H, H.conj().T)
    
    def test_critical_point(self):
        """Critical point detection at g=1."""
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=1.0)
        assert ham.is_critical
        
        ham2 = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=0.5)
        assert not ham2.is_critical
    
    def test_ground_state_energy_bound(self):
        """Ground state energy should be bounded below."""
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=1.0)
        eigenvalues, _ = ham.diagonalize(n_states=1)
        # Energy per site should be reasonable
        E_per_site = eigenvalues[0] / 4
        assert E_per_site > -5  # Rough bound
        assert E_per_site < 0   # Should be negative
    
    def test_eigenvalue_ordering(self):
        """Eigenvalues should be in ascending order."""
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=1.0)
        eigenvalues, _ = ham.diagonalize(n_states=10)
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] <= eigenvalues[i+1]
    
    def test_mass_gap_positive(self):
        """Mass gaps should be positive away from criticality."""
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=0.5)
        gaps = ham.compute_mass_gaps()
        assert all(gap > 0 for gap in gaps)
    
    def test_periodic_vs_open(self):
        """Periodic and open BCs should give different results."""
        ham_open = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=1.0, boundary='open')
        ham_periodic = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=1.0, boundary='periodic')
        
        H_open = ham_open.build_hamiltonian().toarray()
        H_periodic = ham_periodic.build_hamiltonian().toarray()
        
        assert not np.allclose(H_open, H_periodic)
    
    def test_free_fermion_spectrum(self):
        """Free fermion spectrum should be real and positive."""
        ham = IsingHamiltonian(n_sites=6, J=1.0, h_transverse=1.5, h_longitudinal=0.0)
        spectrum = ham.free_fermion_spectrum()
        
        assert all(np.isreal(spectrum))
        assert all(spectrum >= 0)
    
    def test_scaling_dimensions(self):
        """Scaling dimensions should match CFT predictions."""
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=1.0)
        dims = ham.scaling_dimension()
        
        assert dims['central_charge'] == 0.5
        assert dims['identity'] == 0.0
        assert dims['energy_density'] == 0.5
        assert dims['spin_field'] == 1/16


class TestHamiltonianSymmetries:
    """Test symmetries of the Ising Hamiltonian."""
    
    def test_z2_symmetry_hz_zero(self):
        """Z2 symmetry should be present when h_z = 0."""
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=1.0, h_longitudinal=0.0)
        H = ham.build_hamiltonian().toarray()
        
        # Build Z2 symmetry operator: product of all X operators
        Z2 = np.array([[1.0]])
        for _ in range(4):
            Z2 = np.kron(Z2, SIGMA_X)
        
        # [H, Z2] = 0
        commutator = H @ Z2 - Z2 @ H
        assert np.allclose(commutator, 0)
    
    def test_parity_conservation(self):
        """Fermion parity P = ∏σᶻ should be conserved (commutes with H)."""
        # Note: Parity ∏σᶻ commutes with ZZ and Z terms, but NOT with X terms
        # So we test with h_x = 0 for strict parity conservation
        ham = IsingHamiltonian(n_sites=4, J=1.0, h_transverse=0.0, h_longitudinal=0.5)
        H = ham.build_hamiltonian().toarray()
        
        # Parity operator: product of all Z operators
        parity = np.array([[1.0]])
        for _ in range(4):
            parity = np.kron(parity, SIGMA_Z)
        
        # [H, P] = 0 when h_x = 0
        commutator = H @ parity - parity @ H
        assert np.allclose(commutator, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
