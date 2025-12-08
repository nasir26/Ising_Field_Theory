"""
Unit tests for Jordan-Wigner transformation module.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from ising_field_theory.jordan_wigner import (
    JordanWignerTransform,
    compute_fermion_parity,
    SIGMA_X, SIGMA_Y, SIGMA_Z
)


class TestJordanWignerTransform:
    """Test Jordan-Wigner transformation."""
    
    def test_dimension(self):
        """JW transform should preserve Hilbert space dimension."""
        jw = JordanWignerTransform(n_sites=4)
        assert jw.dim == 16
    
    def test_anticommutation_small(self):
        """Verify anticommutation relations for small system."""
        jw = JordanWignerTransform(n_sites=3)
        assert jw.verify_anticommutation()
    
    def test_anticommutation_medium(self):
        """Verify anticommutation relations for medium system."""
        jw = JordanWignerTransform(n_sites=4)
        assert jw.verify_anticommutation()
    
    def test_creation_annihilation_adjoint(self):
        """c† should be adjoint of c."""
        jw = JordanWignerTransform(n_sites=3)
        
        for site in range(3):
            c = jw.annihilation_operator(site).toarray()
            c_dag = jw.creation_operator(site).toarray()
            assert np.allclose(c_dag, c.conj().T)
    
    def test_number_operator_eigenvalues(self):
        """Number operator eigenvalues should be 0 or 1."""
        jw = JordanWignerTransform(n_sites=3)
        
        for site in range(3):
            n_op = jw.number_operator(site).toarray()
            eigenvalues = np.linalg.eigvalsh(n_op)
            assert all(np.isclose(e, 0) or np.isclose(e, 1) for e in eigenvalues)
    
    def test_majorana_hermitian(self):
        """Majorana operators should be Hermitian."""
        jw = JordanWignerTransform(n_sites=3)
        
        for site in range(3):
            gamma_a = jw.majorana_operator(site, 'a').toarray()
            gamma_b = jw.majorana_operator(site, 'b').toarray()
            
            assert np.allclose(gamma_a, gamma_a.conj().T)
            assert np.allclose(gamma_b, gamma_b.conj().T)
    
    def test_majorana_anticommutation(self):
        """Majorana operators should satisfy {γ_i, γ_j} = 2δ_ij."""
        jw = JordanWignerTransform(n_sites=2)
        
        # Same site, same type
        gamma_0a = jw.majorana_operator(0, 'a').toarray()
        anticomm = gamma_0a @ gamma_0a + gamma_0a @ gamma_0a
        assert np.allclose(anticomm, 2 * np.eye(4))
        
        # Same site, different type
        gamma_0b = jw.majorana_operator(0, 'b').toarray()
        anticomm = gamma_0a @ gamma_0b + gamma_0b @ gamma_0a
        assert np.allclose(anticomm, 0)
    
    def test_jw_string_identity_at_zero(self):
        """JW string at site 0 should be identity."""
        jw = JordanWignerTransform(n_sites=3)
        jw_string = jw.jordan_wigner_string(0).toarray()
        assert np.allclose(jw_string, np.eye(8))
    
    def test_hopping_hermitian(self):
        """Hopping term should be Hermitian."""
        jw = JordanWignerTransform(n_sites=4)
        hop = jw.hopping_term(0, 1).toarray()
        assert np.allclose(hop, hop.conj().T)
    
    def test_pairing_hermitian(self):
        """Pairing term should be Hermitian."""
        jw = JordanWignerTransform(n_sites=4)
        pair = jw.pairing_term(0, 1).toarray()
        assert np.allclose(pair, pair.conj().T)
    
    def test_bogoliubov_energies_positive(self):
        """Bogoliubov single-particle energies should be non-negative."""
        jw = JordanWignerTransform(n_sites=6)
        energies, _ = jw.bogoliubov_transform(J=1.0, h_x=1.5)
        assert all(e >= 0 for e in energies)


class TestFermionParity:
    """Test fermion parity computation."""
    
    def test_vacuum_parity(self):
        """Vacuum state should have parity +1."""
        n_qubits = 4
        vacuum = np.zeros(2**n_qubits)
        vacuum[0] = 1.0  # |0000⟩
        
        parity = compute_fermion_parity(vacuum)
        assert np.isclose(parity, 1.0)
    
    def test_single_excitation_parity(self):
        """Single excitation should have parity -1."""
        n_qubits = 4
        state = np.zeros(2**n_qubits)
        state[1] = 1.0  # |0001⟩ (one fermion)
        
        parity = compute_fermion_parity(state)
        assert np.isclose(parity, -1.0)
    
    def test_even_excitations_parity(self):
        """Even number of excitations should have parity +1."""
        n_qubits = 4
        state = np.zeros(2**n_qubits)
        state[3] = 1.0  # |0011⟩ (two fermions)
        
        parity = compute_fermion_parity(state)
        assert np.isclose(parity, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
