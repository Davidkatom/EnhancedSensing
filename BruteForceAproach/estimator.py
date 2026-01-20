import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm

N = 12 
shots = 100
J = 2.0
Omega = 1.0
omega = 1.0

# Time parameters
t_max = 2
steps = 60
times = np.linspace(0, t_max, steps)





exp_value_x = []

def estimateJ_naive():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    # Construct S_x, S_z
    S_x = np.kron(sx, np.eye(2**N, dtype=complex))
    S_z = np.kron(sz, np.eye(2**N, dtype=complex))

    # 2. Bath Operators
    # I_alpha = sum_k (1/2) sigma_alpha^k
    # k runs from 1 to N. System is index 0.

    I_x = np.zeros((2**N, 2**N), dtype=complex)
    I_y = np.zeros((2**N, 2**N), dtype=complex)
    I_z = np.zeros((2**N, 2**N), dtype=complex)

    for k in range(N):
        # Construct sigma_alpha on k-th spin
        vx = [id2] * N
        vy = [id2] * N
        vz = [id2] * N
        
        vx[k] = sx
        vy[k] = sy
        vz[k] = sz
        
        # Compute tensor products
        term_x = vx[0]
        term_y = vy[0]
        term_z = vz[0]
        
        for i in range(1, N):
            term_x = np.kron(term_x, vx[i])
            term_y = np.kron(term_y, vy[i])
            term_z = np.kron(term_z, vz[i])
            
        I_x += 0.5 * term_x
        I_y += 0.5 * term_y
        I_z += 0.5 * term_z




    H = J * np.kron(sz, I_z) 
    print(H.shape)
    plus = np.array([1.0, 1.0]) / np.sqrt(2.0)
    psi0 = np.array([1.0, 0.0])
    for _ in range(N):
        psi0 = np.kron(psi0, plus)
    
    for t in times:
        exp_value_xt = []
        for s in range(shots):
            U = expm(-1j * H * t)
            psit = U @ psi0
            H_total = np.kron(id2, scipy.linalg.hadamard(2**N) / np.sqrt(2**N))
            psit_x = H_total @ psit
            probs = np.abs(psit_x)**2
            sample_idx = np.random.choice(2**(N+1), p=probs/np.sum(probs))
            bitstring = format(sample_idx, f'0{N+1}b')
            # mean_x = np.mean([int(bitstring[i+1]) for i in range(N)])
            mean_x = np.mean([1 - 2 * int(bitstring[i+1]) for i in range(N)])

            exp_value_xt.append(mean_x)
        exp_value_x.append(np.mean(exp_value_xt))
        analytical = np.cos(J * t)

    from scipy.optimize import curve_fit
    popt, _ = curve_fit(lambda t, j_est: np.cos(j_est * t), times, exp_value_x, p0=[0.5])
    print(f"Estimated J: {popt[0]}")



Sy2_t = []
S_opt_t = []
def estimateJ_squeezed():
    # optimal_time = 0.22
    for t in times:        
        S_t = (N*(N-1))/8 * np.cos(2*t*J*J/Omega)**(N-2)
        Sy2 = (N*(N+1))/8 - S_t
        Sy2_t.append(4*Sy2/N)
        Sz2 = N/4
        # S_opt = (Sy2 + Sz2)/2 - 0.5 * (np.sqrt((Sy2 - Sz2)**2))
        # S_opt_t.append(S_opt)

    




# estimateJ_naive()

# plt.plot(times, exp_value_x)
# plt.plot(times, np.cos(J * times))


estimateJ_squeezed()
plt.plot(times, Sy2_t, label='Sy2')
# plt.plot(times, S_opt_t, label='S_opt')
plt.legend()
plt.show()
    
    

    

    