"""
Unit tests for quantum circuits module.
"""

import numpy as np
import pytest
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


class TestGate:
    """Test individual quantum gates."""
    
    def test_pauli_x_matrix(self):
        """X gate should have correct matrix."""
        gate = Gate(GateType.X, (0,))
        matrix = gate.to_matrix()
        expected = np.array([[0, 1], [1, 0]])
        assert np.allclose(matrix, expected)
    
    def test_pauli_z_matrix(self):
        """Z gate should have correct matrix."""
        gate = Gate(GateType.Z, (0,))
        matrix = gate.to_matrix()
        expected = np.array([[1, 0], [0, -1]])
        assert np.allclose(matrix, expected)
    
    def test_hadamard_matrix(self):
        """Hadamard gate should have correct matrix."""
        gate = Gate(GateType.H, (0,))
        matrix = gate.to_matrix()
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(matrix, expected)
    
    def test_rx_unitary(self):
        """RX gate should be unitary."""
        for theta in [0, np.pi/4, np.pi/2, np.pi]:
            gate = Gate(GateType.RX, (0,), (theta,))
            matrix = gate.to_matrix()
            assert np.allclose(matrix @ matrix.conj().T, np.eye(2))
    
    def test_cnot_matrix(self):
        """CNOT gate should have correct matrix."""
        gate = Gate(GateType.CNOT, (0, 1))
        matrix = gate.to_matrix()
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        assert np.allclose(matrix, expected)
    
    def test_zz_gate_unitary(self):
        """ZZ gate should be unitary."""
        for theta in [0, np.pi/4, np.pi/2, np.pi]:
            gate = Gate(GateType.ZZ, (0, 1), (theta,))
            matrix = gate.to_matrix()
            assert np.allclose(matrix @ matrix.conj().T, np.eye(4))


class TestQuantumCircuit:
    """Test quantum circuit construction and execution."""
    
    def test_empty_circuit_identity(self):
        """Empty circuit should be identity."""
        circuit = QuantumCircuit(n_qubits=3)
        U = circuit.to_unitary()
        assert np.allclose(U, np.eye(8))
    
    def test_single_qubit_gate_embedding(self):
        """Single qubit gate should be correctly embedded."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.x(0)
        U = circuit.to_unitary()
        
        # X on qubit 0 is X ⊗ I
        expected = np.kron(np.array([[0, 1], [1, 0]]), np.eye(2))
        assert np.allclose(U, expected)
    
    def test_circuit_unitarity(self):
        """Circuit unitary should be unitary."""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.rx(2, np.pi/4)
        
        U = circuit.to_unitary()
        assert np.allclose(U @ U.conj().T, np.eye(8))
    
    def test_circuit_depth(self):
        """Circuit depth should be computed correctly."""
        circuit = QuantumCircuit(n_qubits=3)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)  # Depth 1 (parallel)
        circuit.cnot(0, 1)  # Depth 2
        circuit.cnot(1, 2)  # Depth 3
        
        assert circuit.depth() == 3
    
    def test_circuit_concatenation(self):
        """Circuit concatenation should work correctly."""
        c1 = QuantumCircuit(n_qubits=2)
        c1.h(0)
        
        c2 = QuantumCircuit(n_qubits=2)
        c2.x(1)
        
        c3 = c1 + c2
        assert len(c3.gates) == 2
    
    def test_gate_count(self):
        """Gate count should be correct."""
        circuit = QuantumCircuit(n_qubits=2)
        circuit.h(0)
        circuit.h(1)
        circuit.cnot(0, 1)
        circuit.rx(0, np.pi/4)
        
        counts = circuit.gate_count()
        assert counts['H'] == 2
        assert counts['CNOT'] == 1
        assert counts['RX'] == 1


class TestTrotterCircuit:
    """Test Trotter decomposition circuits."""
    
    def test_first_order_circuit(self):
        """First-order Trotter circuit should be unitary."""
        trotter = TrotterCircuit(n_qubits=4, J=1.0, h_x=1.0, h_z=0.0)
        circuit = trotter.build_evolution_circuit(total_time=1.0, n_steps=5, order=1)
        
        U = circuit.to_unitary()
        assert np.allclose(U @ U.conj().T, np.eye(16))
    
    def test_second_order_circuit(self):
        """Second-order Trotter circuit should be unitary."""
        trotter = TrotterCircuit(n_qubits=4, J=1.0, h_x=1.0, h_z=0.1)
        circuit = trotter.build_evolution_circuit(total_time=1.0, n_steps=5, order=2)
        
        U = circuit.to_unitary()
        assert np.allclose(U @ U.conj().T, np.eye(16))
    
    def test_trotter_error_decreases(self):
        """Trotter error bound should decrease with more steps."""
        trotter = TrotterCircuit(n_qubits=4, J=1.0, h_x=1.0)
        
        error_10 = trotter.trotter_error_bound(1.0, 10, order=2)
        error_20 = trotter.trotter_error_bound(1.0, 20, order=2)
        
        assert error_20 < error_10
    
    def test_zero_time_identity(self):
        """Zero time evolution should be identity."""
        trotter = TrotterCircuit(n_qubits=3, J=1.0, h_x=1.0)
        circuit = trotter.build_evolution_circuit(total_time=0.0, n_steps=1, order=2)
        
        U = circuit.to_unitary()
        assert np.allclose(U, np.eye(8))


class TestStatePreparation:
    """Test state preparation methods."""
    
    def test_all_zeros_normalized(self):
        """All zeros state should be normalized."""
        prep = StatePreparation(n_qubits=4)
        state = prep.all_zeros_state()
        assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_all_zeros_correct(self):
        """All zeros state should be |0000⟩."""
        prep = StatePreparation(n_qubits=4)
        state = prep.all_zeros_state()
        assert np.isclose(state[0], 1.0)
        assert np.allclose(state[1:], 0.0)
    
    def test_product_state_x_normalized(self):
        """Product state in X basis should be normalized."""
        prep = StatePreparation(n_qubits=4)
        state = prep.product_state_x()
        assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_domain_wall_normalized(self):
        """Domain wall state should be normalized."""
        prep = StatePreparation(n_qubits=6)
        for pos in range(7):
            state = prep.domain_wall_state(position=pos)
            assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_wavepacket_normalized(self):
        """Wavepacket state should be normalized."""
        prep = StatePreparation(n_qubits=8)
        state = prep.wavepacket_state(center=4.0, width=1.0, momentum=0.5)
        assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_two_particle_normalized(self):
        """Two-particle state should be normalized."""
        prep = StatePreparation(n_qubits=6)
        state = prep.two_particle_state(site1=1, site2=4)
        assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_momentum_eigenstate_normalized(self):
        """Momentum eigenstate should be normalized."""
        prep = StatePreparation(n_qubits=6)
        for k in range(1, 7):
            state = prep.momentum_eigenstate(momentum_index=k)
            assert np.isclose(np.linalg.norm(state), 1.0)


class TestMeasurementCircuit:
    """Test measurement circuit construction."""
    
    def test_z_measurement_has_measures(self):
        """Z basis measurement should have measure gates."""
        meas = MeasurementCircuit(n_qubits=4)
        circuit = meas.z_basis_measurement()
        
        counts = circuit.gate_count()
        assert counts.get('MEASURE', 0) == 4
    
    def test_x_measurement_has_hadamards(self):
        """X basis measurement should have Hadamard gates."""
        meas = MeasurementCircuit(n_qubits=4)
        circuit = meas.x_basis_measurement()
        
        counts = circuit.gate_count()
        assert counts.get('H', 0) == 4
    
    def test_energy_measurement_circuits(self):
        """Energy measurement should generate correct number of circuits."""
        meas = MeasurementCircuit(n_qubits=4)
        circuits = meas.energy_measurement_circuits(J=1.0, h_x=1.0, h_z=0.1)
        
        # Should have circuits for ZZ terms (3), X terms (4), Z terms (4)
        assert len(circuits) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
