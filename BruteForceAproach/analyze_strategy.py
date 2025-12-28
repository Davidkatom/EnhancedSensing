import numpy as np
import qutip as qt

def parameterized_state(a, b):
    # |psi> = cos(a)|0> + e^(ib)sin(a)|1>
    q0 = qt.basis(2, 0)
    q1 = qt.basis(2, 1)
    return np.cos(a) * q0 + np.exp(1j * b) * np.sin(a) * q1

def get_bloch_vector(state):
    # Expectation of x, y, z
    ax = qt.expect(qt.sigmax(), state)
    ay = qt.expect(qt.sigmay(), state)
    az = qt.expect(qt.sigmaz(), state)
    return np.array([ax, ay, az])

def analyze():
    # Parameters from previous run
    a = [0.5532402,  5.39062026, 1.47187824]
    b = [1.05784828, 1.69389217, 1.34972795]
    t = 7.316168450635615
    theta = [1.60402809, 3.95884429, 4.5593682]
    phi = [4.41519156, 5.36255327, 4.86245582]
    
    print(f"Optimal Time: {t:.4f} (units of 1/J)")
    
    print("\n--- Initial State Preparation ---")
    qubit_labels = ['1', '2', 'e']
    for i in range(3):
        psi = parameterized_state(a[i], b[i])
        bloch = get_bloch_vector(psi)
        print(f"Qubit {qubit_labels[i]}:")
        print(f"  Angles: a={a[i]:.3f}, b={b[i]:.3f}")
        print(f"  Bloch Vector (x,y,z): [{bloch[0]:.3f}, {bloch[1]:.3f}, {bloch[2]:.3f}]")
        
        # Check closeness to poles or equator
        if np.abs(bloch[2]) > 0.95:
            print("  -> Approximately |0> or |1> (Z-axis)")
        elif np.abs(bloch[2]) < 0.1:
            print("  -> Approximately on Equator (Superposition)")
            
    print("\n--- Measurement Basis ---")
    # Measurement basis is defined by rotation W. 
    # Measuring in Z after W is equivalent to measuring in basis defined by W^dag |0> / |1>.
    # Basis vectors |u0> = W^dag |0>, |u1> = W^dag |1>
    
    for i in range(3):
        # Reconstruct W_i
        # Rz(phi) = exp(-i phi/2 Z)
        rz = (-1j * phi[i] / 2.0 * qt.sigmaz()).expm()
        # Ry(theta) = exp(-i theta/2 Y)
        ry = (-1j * theta[i] / 2.0 * qt.sigmay()).expm()
        W = ry.dag() * rz.dag() # Based on previous script logic
        
        # Basis states in standard coords
        u0 = W.dag() * qt.basis(2, 0)
        bloch = get_bloch_vector(u0)
        
        print(f"Qubit {qubit_labels[i]} Basis (+1 outcome axis):")
        print(f"  Rotation Angles: theta={theta[i]:.3f}, phi={phi[i]:.3f}")
        print(f"  Measurement Axis (Bloch): [{bloch[0]:.3f}, {bloch[1]:.3f}, {bloch[2]:.3f}]")

if __name__ == "__main__":
    analyze()
