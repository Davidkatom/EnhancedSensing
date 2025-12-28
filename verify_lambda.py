import numpy as np
import qutip as qt
import sys
import os

# Add local path to find the module
sys.path.append(os.path.join(os.getcwd(), 'BruteForceAproach'))

try:
    from LambdaSystemDescriptor import LambdaSystem
except ImportError as e:
    print(f"Error importing LambdaSystem: {e}")
    sys.exit(1)

def test_initialization():
    print("Testing Initialization...")
    sys = LambdaSystem(J1=1.0, J2=0.5)
    assert sys.J1 == 1.0
    assert sys.J2 == 0.5
    assert sys.state == qt.tensor(qt.basis(2,0), qt.basis(2,0), qt.basis(2,0))
    print("Initialization OK.")

def test_hamiltonian():
    print("Testing Hamiltonian...")
    sys = LambdaSystem(J1=4.0, J2=4.0) # J/4 becomes 1.0
    # H = (1-Z1)(1-Ze) + (1-Z2)(1-Ze)
    # If state = |101> (indices 1, 0, 1), 
    # Z1|1> = -|1>, (1-Z1)|1> = 2|1>
    # Z2|0> = |0>, (1-Z2)|0> = 0
    # Ze|1> = -|1>, (1-Ze)|1> = 2|1>
    # Term 1 should provide 2*2 = 4 on this state.
    # Term 2 should provide 0.
    # Total H on |101> should be 4 * |101>
    
    state_101 = qt.tensor(qt.basis(2,1), qt.basis(2,0), qt.basis(2,1))
    H = sys.hamiltonian
    
    energy = H.matrix_element(state_101.dag(), state_101).real
    print(f"Energy of |101> with J=4: {energy}")
    assert np.isclose(energy, 4.0)
    
    # State |000> -> (1-Z)|0>=0 -> Energy 0
    state_000 = qt.tensor(qt.basis(2,0), qt.basis(2,0), qt.basis(2,0))
    energy_000 = H.matrix_element(state_000.dag(), state_000).real
    assert np.isclose(energy_000, 0.0)
    print("Hamiltonian Check OK.")

def test_unitary():
    print("Testing Unitary Application...")
    sys = LambdaSystem(J1=1.0, J2=1.0)
    # Apply X on qubit 1 (index 0)
    # |000> -> |100>
    sys.apply_unitary(qt.sigmax(), targets=[0])
    expected = qt.tensor(qt.basis(2,1), qt.basis(2,0), qt.basis(2,0))
    overlap = sys.state.overlap(expected)
    print(f"Overlap after X1: {abs(overlap)}")
    assert np.isclose(abs(overlap), 1.0)
    
    # Apply CNOT on 1 and e (0 and 2)
    # Control 0, Target 2. 
    # Current state |100>. Control is 1, so target flips -> |101>
    # cnot = qt.cnot() # Standard 2-qubit CNOT might be moved
    # Manual CNOT
    cnot = qt.Qobj([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]], dims=[[2, 2], [2, 2]])
    # But wait, qutip cnot() is 4x4. We need to apply it to targets [0, 2].
    sys.apply_unitary(cnot, targets=[0, 2])
    
    expected_final = qt.tensor(qt.basis(2,1), qt.basis(2,0), qt.basis(2,1))
    overlap_final = sys.state.overlap(expected_final)
    print(f"Overlap after CNOT(0,2): {abs(overlap_final)}")
    assert np.isclose(abs(overlap_final), 1.0)
    print("Unitary Application OK.")

def test_evolution():
    print("Testing Time Evolution...")
    # Initialize in superposition |+00> = (|000> + |100>)/sqrt(2)
    # H with J1=4.
    # |000> energy 0.
    # |100> (spin 1 down? No, basis(2,1) is down in qutip usually z=-1?
    # Wait, Z = [[1, 0], [0, -1]]. basis(0) -> ev 1. basis(1) -> ev -1.
    # (1-Z)|0> -> (1-1)|0> = 0.
    # (1-Z)|1> -> (1-(-1))|1> = 2|1>.
    # So basis(0) is 'up' (0), basis(1) is 'down' (1).
    # H = (1-Z1)(1-Ze)...
    # If e is 0 (up), (1-Ze)|0> = 0. So H|psi> = 0 for any state where e=0.
    # So |+00> should correspond to energy 0 (stationary state).
    
    sys = LambdaSystem(J1=np.pi, J2=0) 
    # To get dynamics, we need e in state 1.
    # Start in |+01> -> qubit 1 is +, qubit 2 is 0, qubit e is 1.
    # |001>: 1=0, e=1. (1-Z1)=0. H=0.
    # |101>: 1=1, e=1. (1-Z1)=2. (1-Ze)=2. H = J/4 * 4 = J.
    # So |101> picks up phase exp(-i J t).
    # |001> picks up phase 1.
    # Result at t=1 (J=pi): |0> + exp(-i pi)|1> = |0> - |1> = |->.
    
    initial_s = qt.tensor((qt.basis(2,0)+qt.basis(2,1)).unit(), qt.basis(2,0), qt.basis(2,1))
    sys.state = initial_s
    
    t_list = [0, 1.0]
    sys.time_evolve(t_list)
    
    # Check final state
    expected_state = qt.tensor((qt.basis(2,0)-qt.basis(2,1)).unit(), qt.basis(2,0), qt.basis(2,1))
    
    overlap = sys.state.overlap(expected_state)
    print(f"Overlap after evolution (expect 1.0): {abs(overlap)}")
    assert np.isclose(abs(overlap), 1.0)
    print("Time Evolution OK.")

if __name__ == "__main__":
    test_initialization()
    test_hamiltonian()
    test_unitary()
    test_evolution()
