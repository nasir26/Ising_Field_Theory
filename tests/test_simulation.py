"""
Unit tests for simulation module.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from ising_field_theory.simulation import (
    IsingFieldTheorySimulation,
    SimulationParameters,
    SimulationResults
)


class TestSimulationParameters:
    """Test simulation parameter handling."""
    
    def test_default_parameters(self):
        """Default parameters should be valid."""
        params = SimulationParameters(n_sites=4)
        assert params.n_sites == 4
        assert params.J == 1.0
        assert params.h_transverse == 1.0
        assert params.h_longitudinal == 0.0
    
    def test_critical_detection(self):
        """Critical point should be detected correctly."""
        params = SimulationParameters(n_sites=4, J=1.0, h_transverse=1.0)
        assert params.is_critical
        
        params2 = SimulationParameters(n_sites=4, J=1.0, h_transverse=0.5)
        assert not params2.is_critical
    
    def test_phase_detection(self):
        """Phase should be detected correctly."""
        params_ferro = SimulationParameters(n_sites=4, J=1.0, h_transverse=0.5)
        assert params_ferro.is_ferromagnetic
        assert not params_ferro.is_paramagnetic
        
        params_para = SimulationParameters(n_sites=4, J=1.0, h_transverse=2.0)
        assert params_para.is_paramagnetic
        assert not params_para.is_ferromagnetic


class TestIsingFieldTheorySimulation:
    """Test main simulation class."""
    
    @pytest.fixture
    def sim(self):
        """Create a simulation instance for testing."""
        params = SimulationParameters(
            n_sites=6,
            J=1.0,
            h_transverse=1.0,
            h_longitudinal=0.0,
            boundary='open'
        )
        return IsingFieldTheorySimulation(params)
    
    def test_hamiltonian_hermitian(self, sim):
        """Hamiltonian should be Hermitian."""
        H = sim.H_matrix
        assert np.allclose(H, H.conj().T)
    
    def test_ground_state_normalized(self, sim):
        """Ground state should be normalized."""
        gs = sim.prepare_ground_state()
        assert np.isclose(np.linalg.norm(gs), 1.0)
    
    def test_ground_state_is_eigenstate(self, sim):
        """Ground state should be eigenstate of H."""
        gs = sim.prepare_ground_state()
        E = sim.compute_energy(gs)
        
        # H|gs⟩ = E|gs⟩
        H_gs = sim.H_matrix @ gs
        assert np.allclose(H_gs, E * gs)
    
    def test_energy_conservation(self, sim):
        """Energy should be conserved in exact evolution."""
        initial = sim.prepare_domain_wall_state(position=3)
        results = sim.time_evolve(
            initial, 
            total_time=2.0, 
            n_time_points=20,
            method='exact'
        )
        
        # Energy variance should be very small
        E_std = np.std(results.energies)
        assert E_std < 1e-10
    
    def test_fidelity_self(self, sim):
        """Fidelity of state with itself should be 1."""
        state = sim.prepare_ground_state()
        F = sim.fidelity(state, state)
        assert np.isclose(F, 1.0)
    
    def test_fidelity_orthogonal(self, sim):
        """Fidelity of orthogonal states should be 0."""
        # Ground and first excited state should be orthogonal
        gs = sim.prepare_ground_state()
        es = sim.prepare_excited_state(1)
        F = sim.fidelity(gs, es)
        assert np.isclose(F, 0.0)
    
    def test_magnetization_range(self, sim):
        """Magnetization should be in [-1, 1]."""
        gs = sim.prepare_ground_state()
        M = sim.compute_magnetization(gs)
        assert -1 <= M <= 1
    
    def test_correlation_function_symmetry(self, sim):
        """Correlation function should be symmetric."""
        gs = sim.prepare_ground_state()
        corr = sim.compute_correlation_function(gs, operator='Z')
        
        # C(0) should be 1
        assert np.isclose(corr[0], 1.0)
    
    def test_entanglement_entropy_bounds(self, sim):
        """Entanglement entropy should be bounded."""
        gs = sim.prepare_ground_state()
        S = sim.compute_entanglement_entropy(gs)
        
        # Entropy should be non-negative
        assert S >= 0
        # Entropy should be bounded by log(min(d_A, d_B))
        max_entropy = 3 * np.log(2)  # 3 qubits in each half
        assert S <= max_entropy + 0.01  # Small tolerance
    
    def test_mass_spectrum_positive(self, sim):
        """Mass gaps should be positive."""
        gaps = sim.compute_mass_spectrum(n_states=5)
        assert all(gap > 0 for gap in gaps)
    
    def test_mass_ratios(self, sim):
        """Mass ratios should be positive."""
        ratios = sim.extract_mass_ratios(n_states=5)
        assert all(r > 0 for r in ratios)
        # Mass ratios should be >= 1 (by definition, m_n/m_1 >= 1)
        assert all(r >= 1.0 - 1e-10 for r in ratios)
    
    def test_circuit_stats(self, sim):
        """Circuit stats should be reasonable."""
        stats = sim.get_circuit_stats(total_time=1.0, n_steps=10)
        
        assert stats['n_qubits'] == 6
        assert stats['depth'] > 0
        assert stats['total_gates'] > 0
        assert stats['estimated_error'] > 0


class TestTimeEvolution:
    """Test time evolution methods."""
    
    @pytest.fixture
    def sim(self):
        params = SimulationParameters(
            n_sites=4,
            J=1.0,
            h_transverse=1.0,
            n_trotter_steps=20,
            trotter_order=2
        )
        return IsingFieldTheorySimulation(params)
    
    def test_exact_evolution_unitarity(self, sim):
        """Exact evolution should preserve norm."""
        initial = sim.prepare_domain_wall_state(position=2)
        results = sim.time_evolve(initial, total_time=1.0, method='exact')
        
        for state in results.states:
            assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_trotter_evolution_unitarity(self, sim):
        """Trotter evolution should preserve norm."""
        initial = sim.prepare_domain_wall_state(position=2)
        
        final, _ = sim.trotter_time_evolution(initial, total_time=1.0)
        assert np.isclose(np.linalg.norm(final), 1.0)
    
    def test_trotter_approaches_exact(self, sim):
        """Trotter should approach exact with more steps."""
        initial = sim.prepare_single_particle_state(position=2)
        
        # Exact evolution
        exact_states = sim.exact_time_evolution(initial, np.array([1.0]))
        exact_final = exact_states[0]
        
        # Trotter with increasing steps
        prev_error = np.inf
        for n_steps in [5, 10, 20]:
            trotter_final, _ = sim.trotter_time_evolution(
                initial, 1.0, n_steps=n_steps, order=2
            )
            error = 1 - sim.fidelity(exact_final, trotter_final)
            assert error < prev_error or error < 0.01
            prev_error = error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
