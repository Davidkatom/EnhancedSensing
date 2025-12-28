import numpy as np
import qutip as qt
from scipy.optimize import minimize
from LambdaSystemDescriptor import LambdaSystem

import numpy as np
import qutip as qt
from scipy.optimize import minimize
from LambdaSystemDescriptor import LambdaSystem

# Global constants for current run (updated by optimize_fisher)
J1_GLOBAL = 1.0
J2_GLOBAL = 1.0
GAMMAS_GLOBAL = [0.0, 0.0, 0.0]

def get_basis_states():
    # 3 qubits -> 8 states
    return [qt.tensor(qt.basis(2, i), qt.basis(2, j), qt.basis(2, k)) 
            for i in range(2) for j in range(2) for k in range(2)]

BASIS_STATES = get_basis_states()

def parameterized_state(a, b):
    # |psi> = cos(a)|0> + e^(ib)sin(a)|1>
    q0 = qt.basis(2, 0)
    q1 = qt.basis(2, 1)
    return np.cos(a) * q0 + np.exp(1j * b) * np.sin(a) * q1

def single_qubit_unitary(theta, phi):
    # Manual Rotation implementations
    # Rz(phi) = exp(-i phi/2 Z)
    rz = (-1j * phi / 2.0 * qt.sigmaz()).expm()
    
    # Ry(theta) = exp(-i theta/2 Y)
    ry = (-1j * theta / 2.0 * qt.sigmay()).expm()
    return ry.dag() * rz.dag()

def get_liouvillian_extended(J1, J2, gammas):
    """
    Constructs the extended Liouvillian for evolving [rho, d_rho/dJ].
    L_ext = [[L, 0], [L_grad, L]]
    """
    sys = LambdaSystem(J1=J1, J2=J2)
    H = sys.hamiltonian
    
    # Collapse operators: sqrt(gamma) * sigma_z
    c_ops = []
    # Qubit 1
    if gammas[0] > 0:
        c_ops.append(np.sqrt(gammas[0]) * qt.tensor(qt.sigmaz(), qt.qeye(2), qt.qeye(2)))
    # Qubit 2
    if gammas[1] > 0:
        c_ops.append(np.sqrt(gammas[1]) * qt.tensor(qt.qeye(2), qt.sigmaz(), qt.qeye(2)))
    # Qubit e
    if gammas[2] > 0:
        c_ops.append(np.sqrt(gammas[2]) * qt.tensor(qt.qeye(2), qt.qeye(2), qt.sigmaz()))
        
    # Calculate Liouvillians
    L = qt.liouvillian(H, c_ops)
    L_grad = qt.liouvillian(H, [])
    
    # Convert to pure matrix forms (strip superoperator dims/type)
    # L is 64x64. Treat as standard operator on the 'vector' space.
    L_mat = qt.Qobj(L.full())
    L_grad_mat = qt.Qobj(L_grad.full())
    
    # Define Aux operators (force dense)
    id2 = qt.qeye(2).to('dense')
    lower_op = qt.Qobj([[0, 0], [1, 0]]).to('dense')
    
    L_ext = qt.tensor(id2, L_mat) + qt.tensor(lower_op, L_grad_mat)
    
    return L_ext


def calculate_fisher_info(params):
    J1 = J1_GLOBAL
    J2 = J2_GLOBAL
    gammas = GAMMAS_GLOBAL
    
    a = params[0:3]
    b = params[3:6]
    t = params[6]
    th = params[7:10]
    ph = params[10:13]
    
    if t < 0: return 0
    
    # 1. Prepare Initial State
    psi0_list = [parameterized_state(a[i], b[i]) for i in range(3)]
    psi0 = qt.tensor(psi0_list)
    rho0 = qt.ket2dm(psi0)
    
    # Vectorize and ensure dense simple Qobj
    rho0_vec = qt.operator_to_vector(rho0)
    rho0_vec_simple = qt.Qobj(rho0_vec.full())
    
    # Extended state: rho0_vec in top block (Aux |0>)
    # Force basis to dense
    rho_ext = qt.tensor(qt.basis(2, 0).to('dense'), rho0_vec_simple)
    
    # 2. Time Evolution
    # Construct L_ext
    # Note: recomputing L_ext every time is expensive if J1,J2 are constant. 
    # But params don't change J1/J2, only optimize_fisher outer loop does. Called once per optimize.
    # Actually calculate_fisher_info is called repeatedly by optimizer.
    # We should cache L_ext if J/gamma are constant.
    # For now, reconstruct.
    
    # OPTIMIZATION: Cache L_ext global?
    global L_EXT_CACHE
    if 'L_EXT_CACHE' not in globals() or L_EXT_CACHE['id'] != (J1, J2, tuple(gammas)):
        L_ext_obj = get_liouvillian_extended(J1, J2, gammas)
        L_EXT_CACHE = {'id': (J1, J2, tuple(gammas)), 'obj': L_ext_obj}
    
    L_ext = L_EXT_CACHE['obj']
    
    # Exp evolution
    # L_ext is 8192x8192 sparse. .expm() might be dense and slow.
    # Use propagator? Or mesolve?
    # Simple expm on dense might crash or be slow.
    # But 8192 is small enough for sparse expm? No, High memory.
    # 8000^2 * 16 bytes = 1GB. It's fine for dense in 64-bit systems.
    # But faster to use scipy.sparse.linalg.expm or similar.
    # Qutip .expm() uses dense usually.
    # Let's try direct evolution?
    
    # Try iterative calculation if t is large?
    # t is optimizing parameter.
    
    U_super = (L_ext * t).expm()
    
    rho_ext_t = U_super * rho_ext
    
    # Extract rho(t) and dot_rho(t)
    # Split the vector
    # Extract rho(t) and dot_rho(t)
    # Convert to numpy array first to avoid Qobj slicing ambiguity
    rho_ext_arr = rho_ext_t.full()
    
    half_len = rho_ext_arr.shape[0] // 2
    # Slice the numpy array
    rho_vec_data = rho_ext_arr[0:half_len, 0:1] # Keep 2D shape (Nx1)
    d_rho_vec_data = rho_ext_arr[half_len:2*half_len, 0:1]
    
    # Target dims: [[[2,2,2], [2,2,2]], [1]]
    target_dims = [[[2, 2, 2], [2, 2, 2]], [1]]
    
    # Create new Qobjs with correct dims from the sliced data
    rho_vec_t = qt.Qobj(rho_vec_data, dims=target_dims)
    d_rho_vec_t = qt.Qobj(d_rho_vec_data, dims=target_dims)
    
    rho_t = qt.vector_to_operator(rho_vec_t)
    d_rho_t = qt.vector_to_operator(d_rho_vec_t)
    
    # 3. Apply Local Rotations
    w_list = [single_qubit_unitary(th[i], ph[i]) for i in range(3)]
    W = qt.tensor(w_list)
    
    # Rotate density matrix: rho_final = W * rho * W_dag
    # d_rho_final = W * d_rho * W_dag
    rho_final = W * rho_t * W.dag()
    d_rho_final = W * d_rho_t * W.dag()
    
    # 4. Fisher Info
    fisher = 0.0
    
    for k_state in BASIS_STATES:
        # P_k = <k|rho|k>
        # dP_k = <k|d_rho|k>
        # Check trace logic. <k|A|k> = tr(|k><k| A).
        # We can just compute scalar elements.
        
        P_k = k_state.dag() * rho_final * k_state
        dP_k = k_state.dag() * d_rho_final * k_state
        
        # Extract real scalar
        # If P_k is Qobj 1x1
        if isinstance(P_k, qt.Qobj): 
             val_P = np.abs(P_k[0,0]) # Density matrix diag should be real positive
             val_dP = np.real(dP_k[0,0])
        else:
             val_P = np.abs(P_k)
             val_dP = np.real(dP_k)
             
        if val_P > 1e-12:
            fisher += (val_dP**2) / val_P
            
    return fisher

def objective(params):
    return -calculate_fisher_info(params)

def optimize_fisher(J1, J2, gammas):
    global J1_GLOBAL, J2_GLOBAL, GAMMAS_GLOBAL
    J1_GLOBAL = J1
    J2_GLOBAL = J2
    GAMMAS_GLOBAL = gammas
    
    print(f"Starting Optimization with J1={J1}, J2={J2}, gammas={gammas}...")
    
    # Initial Guess
    np.random.seed(42)
    x0 = np.random.uniform(0, 2*np.pi, 13)
    x0[6] = 1.0 # Time
    
    bounds = [(0, 2*np.pi)]*6 + [(0, 10)] + [(0, 2*np.pi)]*6
    
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    
    max_fisher = -res.fun
    print("\nOptimization Complete!")
    print(f"Max Fisher Information: {max_fisher}")
    print(f"Optimal Time t: {res.x[6]}")
    print("Optimal Parameters:")
    print(f" a: {res.x[0:3]}")
    print(f" b: {res.x[3:6]}")
    print(f" theta: {res.x[7:10]}")
    print(f" phi: {res.x[10:13]}")
    
    return res

if __name__ == "__main__":
    # Default behavior or testing
    # User might call this function from another script or modify main
    optimize_fisher(J1=1.0, J2=1.0, gammas=[0.1, 0.1, 0.1])

