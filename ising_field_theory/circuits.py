"""
Quantum Circuit Construction for Ising Field Theory Simulation

This module implements:
1. Trotterized time evolution circuits
2. State preparation circuits (vacuum, particle states)
3. Measurement circuits for observables

Circuit decomposition follows from:
    exp(-iHt) ≈ [exp(-iH_ZZ δt) exp(-iH_X δt) exp(-iH_Z δt)]^n

where δt = t/n is the Trotter step.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum


class GateType(Enum):
    """Supported quantum gate types."""
    X = "X"
    Y = "Y"
    Z = "Z"
    H = "H"
    S = "S"
    T = "T"
    RX = "RX"
    RY = "RY"
    RZ = "RZ"
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"
    XX = "XX"
    YY = "YY"
    ZZ = "ZZ"
    MEASURE = "MEASURE"


@dataclass
class Gate:
    """
    Representation of a quantum gate.
    
    Attributes:
        gate_type: Type of gate
        qubits: Tuple of qubit indices the gate acts on
        params: Optional parameters (angles for rotation gates)
    """
    gate_type: GateType
    qubits: Tuple[int, ...]
    params: Optional[Tuple[float, ...]] = None
    
    def __repr__(self):
        if self.params:
            return f"{self.gate_type.value}({self.params})@{self.qubits}"
        return f"{self.gate_type.value}@{self.qubits}"
    
    def to_matrix(self) -> np.ndarray:
        """Return the matrix representation of the gate."""
        if self.gate_type == GateType.X:
            return np.array([[0, 1], [1, 0]], dtype=complex)
        elif self.gate_type == GateType.Y:
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        elif self.gate_type == GateType.Z:
            return np.array([[1, 0], [0, -1]], dtype=complex)
        elif self.gate_type == GateType.H:
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif self.gate_type == GateType.S:
            return np.array([[1, 0], [0, 1j]], dtype=complex)
        elif self.gate_type == GateType.T:
            return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        elif self.gate_type == GateType.RX:
            theta = self.params[0]
            return np.array([
                [np.cos(theta/2), -1j * np.sin(theta/2)],
                [-1j * np.sin(theta/2), np.cos(theta/2)]
            ], dtype=complex)
        elif self.gate_type == GateType.RY:
            theta = self.params[0]
            return np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2), np.cos(theta/2)]
            ], dtype=complex)
        elif self.gate_type == GateType.RZ:
            theta = self.params[0]
            return np.array([
                [np.exp(-1j * theta/2), 0],
                [0, np.exp(1j * theta/2)]
            ], dtype=complex)
        elif self.gate_type == GateType.CNOT:
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=complex)
        elif self.gate_type == GateType.CZ:
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=complex)
        elif self.gate_type == GateType.SWAP:
            return np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=complex)
        elif self.gate_type == GateType.XX:
            # exp(-i θ/2 XX)
            theta = self.params[0]
            c, s = np.cos(theta/2), np.sin(theta/2)
            return np.array([
                [c, 0, 0, -1j*s],
                [0, c, -1j*s, 0],
                [0, -1j*s, c, 0],
                [-1j*s, 0, 0, c]
            ], dtype=complex)
        elif self.gate_type == GateType.YY:
            # exp(-i θ/2 YY)
            theta = self.params[0]
            c, s = np.cos(theta/2), np.sin(theta/2)
            return np.array([
                [c, 0, 0, 1j*s],
                [0, c, -1j*s, 0],
                [0, -1j*s, c, 0],
                [1j*s, 0, 0, c]
            ], dtype=complex)
        elif self.gate_type == GateType.ZZ:
            # exp(-i θ/2 ZZ)
            theta = self.params[0]
            return np.array([
                [np.exp(-1j*theta/2), 0, 0, 0],
                [0, np.exp(1j*theta/2), 0, 0],
                [0, 0, np.exp(1j*theta/2), 0],
                [0, 0, 0, np.exp(-1j*theta/2)]
            ], dtype=complex)
        else:
            raise ValueError(f"Unknown gate type: {self.gate_type}")


class QuantumCircuit:
    """
    Quantum circuit representation.
    
    Stores gates as a list for both execution and analysis.
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.gates: List[Gate] = []
        self.dim = 2 ** n_qubits
    
    def add_gate(self, gate: Gate):
        """Add a gate to the circuit."""
        for q in gate.qubits:
            if q < 0 or q >= self.n_qubits:
                raise ValueError(f"Qubit index {q} out of range [0, {self.n_qubits})")
        self.gates.append(gate)
    
    def x(self, qubit: int):
        self.add_gate(Gate(GateType.X, (qubit,)))
    
    def y(self, qubit: int):
        self.add_gate(Gate(GateType.Y, (qubit,)))
    
    def z(self, qubit: int):
        self.add_gate(Gate(GateType.Z, (qubit,)))
    
    def h(self, qubit: int):
        self.add_gate(Gate(GateType.H, (qubit,)))
    
    def s(self, qubit: int):
        self.add_gate(Gate(GateType.S, (qubit,)))
    
    def t(self, qubit: int):
        self.add_gate(Gate(GateType.T, (qubit,)))
    
    def rx(self, qubit: int, theta: float):
        self.add_gate(Gate(GateType.RX, (qubit,), (theta,)))
    
    def ry(self, qubit: int, theta: float):
        self.add_gate(Gate(GateType.RY, (qubit,), (theta,)))
    
    def rz(self, qubit: int, theta: float):
        self.add_gate(Gate(GateType.RZ, (qubit,), (theta,)))
    
    def cnot(self, control: int, target: int):
        self.add_gate(Gate(GateType.CNOT, (control, target)))
    
    def cz(self, qubit1: int, qubit2: int):
        self.add_gate(Gate(GateType.CZ, (qubit1, qubit2)))
    
    def swap(self, qubit1: int, qubit2: int):
        self.add_gate(Gate(GateType.SWAP, (qubit1, qubit2)))
    
    def rxx(self, qubit1: int, qubit2: int, theta: float):
        """exp(-i θ/2 X⊗X)"""
        self.add_gate(Gate(GateType.XX, (qubit1, qubit2), (theta,)))
    
    def ryy(self, qubit1: int, qubit2: int, theta: float):
        """exp(-i θ/2 Y⊗Y)"""
        self.add_gate(Gate(GateType.YY, (qubit1, qubit2), (theta,)))
    
    def rzz(self, qubit1: int, qubit2: int, theta: float):
        """exp(-i θ/2 Z⊗Z)"""
        self.add_gate(Gate(GateType.ZZ, (qubit1, qubit2), (theta,)))
    
    def measure(self, qubit: int):
        self.add_gate(Gate(GateType.MEASURE, (qubit,)))
    
    def measure_all(self):
        for q in range(self.n_qubits):
            self.measure(q)
    
    def _embed_single_qubit_gate(self, matrix: np.ndarray, qubit: int) -> np.ndarray:
        """Embed single-qubit gate into full Hilbert space."""
        result = np.array([[1.0]], dtype=complex)
        for i in range(self.n_qubits):
            if i == qubit:
                result = np.kron(result, matrix)
            else:
                result = np.kron(result, np.eye(2))
        return result
    
    def _embed_two_qubit_gate(
        self, 
        matrix: np.ndarray, 
        qubit1: int, 
        qubit2: int
    ) -> np.ndarray:
        """
        Embed two-qubit gate into full Hilbert space.
        
        The matrix is given in the computational basis ordered as:
        |00⟩, |01⟩, |10⟩, |11⟩ where first index is qubit1.
        """
        if qubit1 > qubit2:
            # Swap and adjust matrix
            qubit1, qubit2 = qubit2, qubit1
            # Permute matrix to account for qubit order swap
            perm = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
            matrix = perm @ matrix @ perm.T
        
        # Build: I^{⊗q1} ⊗ (I^{⊗(q2-q1-1)} contracted with matrix) ⊗ I^{⊗(n-q2-1)}
        # More direct approach: construct full operator
        
        result = np.eye(self.dim, dtype=complex)
        
        for i in range(self.dim):
            for j in range(self.dim):
                # Extract bits for qubit1 and qubit2 from indices i, j
                b1_i = (i >> (self.n_qubits - 1 - qubit1)) & 1
                b2_i = (i >> (self.n_qubits - 1 - qubit2)) & 1
                b1_j = (j >> (self.n_qubits - 1 - qubit1)) & 1
                b2_j = (j >> (self.n_qubits - 1 - qubit2)) & 1
                
                # Check if all other bits match
                mask = ~((1 << (self.n_qubits - 1 - qubit1)) | (1 << (self.n_qubits - 1 - qubit2)))
                if (i & mask) != (j & mask):
                    result[i, j] = 0
                else:
                    # Index into 4x4 matrix
                    idx_i = b1_i * 2 + b2_i
                    idx_j = b1_j * 2 + b2_j
                    result[i, j] = matrix[idx_i, idx_j]
        
        return result
    
    def to_unitary(self) -> np.ndarray:
        """
        Convert circuit to unitary matrix.
        
        U = Gₙ Gₙ₋₁ ... G₁
        """
        U = np.eye(self.dim, dtype=complex)
        
        for gate in self.gates:
            if gate.gate_type == GateType.MEASURE:
                continue  # Skip measurement for unitary construction
            
            matrix = gate.to_matrix()
            
            if len(gate.qubits) == 1:
                G = self._embed_single_qubit_gate(matrix, gate.qubits[0])
            elif len(gate.qubits) == 2:
                G = self._embed_two_qubit_gate(matrix, gate.qubits[0], gate.qubits[1])
            else:
                raise ValueError(f"Gates on more than 2 qubits not supported")
            
            U = G @ U
        
        return U
    
    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply circuit to a state vector."""
        return self.to_unitary() @ state
    
    def depth(self) -> int:
        """Compute circuit depth (excluding measurements)."""
        if not self.gates:
            return 0
        
        qubit_depths = [0] * self.n_qubits
        for gate in self.gates:
            if gate.gate_type == GateType.MEASURE:
                continue
            max_depth = max(qubit_depths[q] for q in gate.qubits)
            for q in gate.qubits:
                qubit_depths[q] = max_depth + 1
        
        return max(qubit_depths)
    
    def gate_count(self) -> Dict[str, int]:
        """Count gates by type."""
        counts = {}
        for gate in self.gates:
            name = gate.gate_type.value
            counts[name] = counts.get(name, 0) + 1
        return counts
    
    def copy(self) -> 'QuantumCircuit':
        """Create a copy of the circuit."""
        new_circuit = QuantumCircuit(self.n_qubits)
        new_circuit.gates = self.gates.copy()
        return new_circuit
    
    def __add__(self, other: 'QuantumCircuit') -> 'QuantumCircuit':
        """Concatenate circuits."""
        if self.n_qubits != other.n_qubits:
            raise ValueError("Circuits must have same number of qubits")
        result = self.copy()
        result.gates.extend(other.gates)
        return result


class TrotterCircuit:
    """
    Trotter decomposition circuits for Ising Hamiltonian time evolution.
    
    H = -J Σᵢ ZᵢZᵢ₊₁ - hₓ Σᵢ Xᵢ - hᵤ Σᵢ Zᵢ
    
    First-order Trotter:
        exp(-iHt) ≈ [exp(-iH_ZZ δt) exp(-iH_X δt) exp(-iH_Z δt)]^n
    
    Second-order Trotter (symmetric):
        exp(-iHt) ≈ [exp(-iH_Z δt/2) exp(-iH_X δt/2) exp(-iH_ZZ δt) 
                     exp(-iH_X δt/2) exp(-iH_Z δt/2)]^n
    
    Gate decompositions:
        exp(i θ ZᵢZⱼ) = CNOT(i,j) · RZ(2θ)_j · CNOT(i,j)
        exp(i θ Xᵢ) = RX(2θ)_i
        exp(i θ Zᵢ) = RZ(2θ)_i
    """
    
    def __init__(
        self, 
        n_qubits: int,
        J: float = 1.0,
        h_x: float = 1.0,
        h_z: float = 0.0,
        boundary: str = 'open'
    ):
        self.n_qubits = n_qubits
        self.J = J
        self.h_x = h_x
        self.h_z = h_z
        self.boundary = boundary
        self.n_bonds = n_qubits - 1 if boundary == 'open' else n_qubits
    
    def zz_layer(self, circuit: QuantumCircuit, dt: float):
        """
        Apply exp(i J dt Σᵢ ZᵢZᵢ₊₁) layer.
        
        exp(i θ ZZ) = CNOT · RZ(2θ) · CNOT
        
        With θ = J·dt for each bond.
        """
        theta = 2 * self.J * dt  # Factor of 2 from RZZ convention
        
        # Even bonds first (can be parallelized)
        for i in range(0, self.n_bonds, 2):
            j = (i + 1) % self.n_qubits
            circuit.rzz(i, j, theta)
        
        # Odd bonds (can be parallelized)
        for i in range(1, self.n_bonds, 2):
            j = (i + 1) % self.n_qubits
            circuit.rzz(i, j, theta)
    
    def x_layer(self, circuit: QuantumCircuit, dt: float):
        """
        Apply exp(i hₓ dt Σᵢ Xᵢ) layer.
        
        exp(i θ X) = RX(2θ)
        """
        theta = 2 * self.h_x * dt
        for i in range(self.n_qubits):
            circuit.rx(i, theta)
    
    def z_layer(self, circuit: QuantumCircuit, dt: float):
        """
        Apply exp(i hᵤ dt Σᵢ Zᵢ) layer.
        
        exp(i θ Z) = RZ(2θ)
        """
        if self.h_z == 0:
            return
        
        theta = 2 * self.h_z * dt
        for i in range(self.n_qubits):
            circuit.rz(i, theta)
    
    def first_order_step(self, circuit: QuantumCircuit, dt: float):
        """
        Single first-order Trotter step.
        
        exp(-iHδt) ≈ exp(-iH_ZZ δt) · exp(-iH_X δt) · exp(-iH_Z δt)
        
        Error: O(δt²)
        """
        self.zz_layer(circuit, dt)
        self.x_layer(circuit, dt)
        self.z_layer(circuit, dt)
    
    def second_order_step(self, circuit: QuantumCircuit, dt: float):
        """
        Single second-order (symmetric) Trotter step.
        
        exp(-iHδt) ≈ exp(-iH_Z δt/2) · exp(-iH_X δt/2) · exp(-iH_ZZ δt) 
                     · exp(-iH_X δt/2) · exp(-iH_Z δt/2)
        
        Error: O(δt³)
        """
        self.z_layer(circuit, dt/2)
        self.x_layer(circuit, dt/2)
        self.zz_layer(circuit, dt)
        self.x_layer(circuit, dt/2)
        self.z_layer(circuit, dt/2)
    
    def build_evolution_circuit(
        self, 
        total_time: float, 
        n_steps: int,
        order: int = 2
    ) -> QuantumCircuit:
        """
        Build complete time evolution circuit.
        
        Parameters:
            total_time: Total evolution time t
            n_steps: Number of Trotter steps
            order: Trotter order (1 or 2)
        
        Returns:
            Quantum circuit implementing exp(-iHt)
        """
        circuit = QuantumCircuit(self.n_qubits)
        dt = total_time / n_steps
        
        step_func = self.first_order_step if order == 1 else self.second_order_step
        
        for _ in range(n_steps):
            step_func(circuit, dt)
        
        return circuit
    
    def trotter_error_bound(self, total_time: float, n_steps: int, order: int = 2) -> float:
        """
        Upper bound on Trotter error.
        
        First-order: ||exp(-iHt) - U_T|| ≤ t²||[H_A, H_B]||/n
        Second-order: ||exp(-iHt) - U_T|| ≤ t³||[[H,H_A],H_B]||/n²
        
        Returns conservative estimate based on operator norms.
        """
        dt = total_time / n_steps
        
        # Estimate commutator norms (rough upper bounds)
        # ||H_ZZ|| ~ J * n_bonds
        # ||H_X|| ~ h_x * n_qubits
        # ||[H_ZZ, H_X]|| ~ 4 * J * h_x * n_qubits (each X can fail to commute with 2 ZZ)
        
        norm_zz = abs(self.J) * self.n_bonds
        norm_x = abs(self.h_x) * self.n_qubits
        norm_z = abs(self.h_z) * self.n_qubits
        
        commutator_norm = 4 * abs(self.J) * abs(self.h_x) * self.n_qubits
        
        if order == 1:
            return total_time**2 * commutator_norm / n_steps
        else:
            # Second-order has smaller error
            nested_commutator = commutator_norm * (norm_zz + norm_x + norm_z)
            return total_time**3 * nested_commutator / (n_steps**2)


class StatePreparation:
    """
    Prepare initial states for Ising field theory simulation.
    
    Key states:
    1. Vacuum state (ground state of H)
    2. Single-particle states (localized excitations)
    3. Multi-particle states (for scattering)
    4. Domain wall states
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
    
    def all_zeros_state(self) -> np.ndarray:
        """
        Prepare |00...0⟩ state.
        
        This is the ferromagnetic state in the Z basis (all spins up).
        Ground state of H with hₓ = 0 and J > 0.
        """
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0
        return state
    
    def all_ones_state(self) -> np.ndarray:
        """
        Prepare |11...1⟩ state.
        
        The other ferromagnetic state (all spins down).
        """
        state = np.zeros(self.dim, dtype=complex)
        state[-1] = 1.0
        return state
    
    def product_state_x(self) -> np.ndarray:
        """
        Prepare |+⟩^⊗n = H^⊗n|0⟩^⊗n state.
        
        This is the paramagnetic state, ground state of H with J = 0.
        """
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        state = plus.copy()
        for _ in range(self.n_qubits - 1):
            state = np.kron(state, plus)
        return state
    
    def domain_wall_state(self, position: int) -> np.ndarray:
        """
        Prepare domain wall state: |00...011...1⟩
        
        All zeros up to position, all ones after.
        This represents a kink excitation in the Ising model.
        
        Parameters:
            position: Location of domain wall (0 to n_qubits)
        """
        # Binary number with position trailing ones
        index = (1 << (self.n_qubits - position)) - 1
        state = np.zeros(self.dim, dtype=complex)
        state[index] = 1.0
        return state
    
    def single_spin_flip_state(self, site: int) -> np.ndarray:
        """
        Prepare state with single spin flipped: |00...010...0⟩
        
        Represents a localized magnon excitation.
        """
        index = 1 << (self.n_qubits - 1 - site)
        state = np.zeros(self.dim, dtype=complex)
        state[index] = 1.0
        return state
    
    def superposition_spin_flip(self, amplitude_func: Callable[[int], complex] = None) -> np.ndarray:
        """
        Prepare superposition of single spin-flip states:
        
        |ψ⟩ = Σᵢ f(i) |0...1ᵢ...0⟩
        
        For plane wave: f(i) = exp(ikxᵢ)/√N
        
        Parameters:
            amplitude_func: Function f(site) -> amplitude. Default is uniform.
        """
        if amplitude_func is None:
            amplitude_func = lambda i: 1.0 / np.sqrt(self.n_qubits)
        
        state = np.zeros(self.dim, dtype=complex)
        for site in range(self.n_qubits):
            index = 1 << (self.n_qubits - 1 - site)
            state[index] = amplitude_func(site)
        
        # Normalize
        state = state / np.linalg.norm(state)
        return state
    
    def momentum_eigenstate(self, momentum_index: int) -> np.ndarray:
        """
        Prepare single-particle momentum eigenstate.
        
        |k⟩ = Σⱼ exp(ikxⱼ)/√N σⱼ⁺|0⟩
        
        where k = πn/(N+1) for open boundaries.
        
        Parameters:
            momentum_index: n in k = πn/(N+1), range 1 to N
        """
        if momentum_index < 1 or momentum_index > self.n_qubits:
            raise ValueError(f"momentum_index must be in [1, {self.n_qubits}]")
        
        k = np.pi * momentum_index / (self.n_qubits + 1)
        
        def amplitude(site):
            x = site + 1  # Position (1-indexed)
            return np.exp(1j * k * x) / np.sqrt(self.n_qubits)
        
        return self.superposition_spin_flip(amplitude)
    
    def wavepacket_state(
        self, 
        center: float, 
        width: float, 
        momentum: float
    ) -> np.ndarray:
        """
        Prepare Gaussian wavepacket state.
        
        |ψ⟩ = Σⱼ exp(-(xⱼ-x₀)²/4σ²) exp(ikxⱼ) σⱼ⁺|0⟩
        
        Parameters:
            center: Wavepacket center x₀ (in lattice units)
            width: Wavepacket width σ
            momentum: Wavepacket momentum k
        """
        def amplitude(site):
            x = site + 0.5  # Site position
            gaussian = np.exp(-(x - center)**2 / (4 * width**2))
            phase = np.exp(1j * momentum * x)
            return gaussian * phase
        
        return self.superposition_spin_flip(amplitude)
    
    def two_particle_state(
        self, 
        site1: int, 
        site2: int
    ) -> np.ndarray:
        """
        Prepare two-particle (two spin-flip) state.
        
        |ψ⟩ = |0...1ᵢ...1ⱼ...0⟩
        
        For studying two-particle scattering.
        """
        if site1 == site2:
            raise ValueError("Sites must be different")
        
        index = (1 << (self.n_qubits - 1 - site1)) | (1 << (self.n_qubits - 1 - site2))
        state = np.zeros(self.dim, dtype=complex)
        state[index] = 1.0
        return state
    
    def variational_ground_state_circuit(
        self, 
        theta: np.ndarray
    ) -> QuantumCircuit:
        """
        Variational circuit for ground state preparation.
        
        Uses hardware-efficient ansatz with RY rotations and CZ entanglers.
        
        Parameters:
            theta: Variational parameters (n_qubits * n_layers,)
        """
        n_layers = len(theta) // self.n_qubits
        circuit = QuantumCircuit(self.n_qubits)
        
        idx = 0
        for layer in range(n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                circuit.ry(q, theta[idx])
                idx += 1
            
            # Entangling layer
            for q in range(self.n_qubits - 1):
                circuit.cz(q, q + 1)
        
        return circuit


class MeasurementCircuit:
    """
    Construct measurement circuits for Ising field theory observables.
    """
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
    
    def z_basis_measurement(self) -> QuantumCircuit:
        """
        Measure all qubits in Z basis.
        
        Directly measures ⟨σᵢᶻ⟩ and correlators ⟨σᵢᶻσⱼᶻ⟩.
        """
        circuit = QuantumCircuit(self.n_qubits)
        circuit.measure_all()
        return circuit
    
    def x_basis_measurement(self) -> QuantumCircuit:
        """
        Measure all qubits in X basis.
        
        Rotate to X eigenbasis via H, then measure in Z.
        Gives ⟨σᵢˣ⟩ (order parameter).
        """
        circuit = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            circuit.h(q)
        circuit.measure_all()
        return circuit
    
    def y_basis_measurement(self) -> QuantumCircuit:
        """
        Measure all qubits in Y basis.
        
        Rotate via S†H, then measure in Z.
        """
        circuit = QuantumCircuit(self.n_qubits)
        for q in range(self.n_qubits):
            circuit.rz(q, -np.pi/2)  # S†
            circuit.h(q)
        circuit.measure_all()
        return circuit
    
    def correlation_measurement_zz(self, site1: int, site2: int) -> QuantumCircuit:
        """
        Measure ⟨σᵢᶻσⱼᶻ⟩ correlation.
        
        Measure both qubits in Z basis and compute product.
        """
        circuit = QuantumCircuit(self.n_qubits)
        circuit.measure(site1)
        circuit.measure(site2)
        return circuit
    
    def energy_measurement_circuits(
        self, 
        J: float, 
        h_x: float, 
        h_z: float = 0.0,
        boundary: str = 'open'
    ) -> List[Tuple[QuantumCircuit, float, str]]:
        """
        Generate circuits needed to measure ⟨H⟩.
        
        H = -J Σ ZᵢZᵢ₊₁ - hₓ Σ Xᵢ - hᵤ Σ Zᵢ
        
        Returns list of (circuit, coefficient, term_type) for each term.
        """
        circuits = []
        
        n_bonds = self.n_qubits - 1 if boundary == 'open' else self.n_qubits
        
        # ZZ terms - measure in Z basis
        for i in range(n_bonds):
            j = (i + 1) % self.n_qubits
            circuit = self.z_basis_measurement()
            circuits.append((circuit, -J, f"ZZ_{i}_{j}"))
        
        # X terms - measure in X basis
        circuit = self.x_basis_measurement()
        for i in range(self.n_qubits):
            circuits.append((circuit, -h_x, f"X_{i}"))
        
        # Z terms - measure in Z basis
        if h_z != 0:
            circuit = self.z_basis_measurement()
            for i in range(self.n_qubits):
                circuits.append((circuit, -h_z, f"Z_{i}"))
        
        return circuits
