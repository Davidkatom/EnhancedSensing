import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm

def get_system_operators():
    """Returns system Pauli matrices and identity."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id_S = np.eye(2, dtype=complex)
    return sx, sy, sz, id_S

def get_bath_operators(N):
    """Returns bath spin operators in Dickie basis."""
    dim_B = N + 1
    J_bath = N / 2.0
    m_values = np.arange(-N//2, N//2 + 1)
    
    # J_z is diagonal
    Jz_B = np.diag(m_values).astype(complex)
    
    # J_+ ladder operator
    Jp_B = np.zeros((dim_B, dim_B), dtype=complex)
    for i, m in enumerate(m_values):
        if i < dim_B - 1:
            coeff = np.sqrt(J_bath*(J_bath+1) - m*(m+1))
            Jp_B[i+1, i] = coeff   
            
    Jm_B = Jp_B.conj().T
    
    Jx_B = 0.5 * (Jp_B + Jm_B)
    Jy_B = 0.5 * (Jp_B - Jm_B) / 1j
    id_B = np.eye(dim_B, dtype=complex)
    
    return Jx_B, Jy_B, Jz_B, Jp_B, id_B, m_values

def get_total_operators(N):
    """Constructs total system x bath operators."""
    sx, sy, sz, id_S = get_system_operators()
    Jx_B, Jy_B, Jz_B, Jp_B, id_B, m_values = get_bath_operators(N)
    
    # System operators in full space
    S_x = np.kron(sx, id_B)
    S_z = np.kron(sz, id_B)
    
    # Bath operators in full space
    I_x = np.kron(id_S, Jx_B)
    I_y = np.kron(id_S, Jy_B)
    I_z = np.kron(id_S, Jz_B)
    I_i = np.kron(id_S, id_B)
    
    I_p = np.kron(id_S, Jp_B)
    
    return {
        'S_x': S_x, 'S_z': S_z,
        'I_x': I_x, 'I_y': I_y, 'I_z': I_z,
        'I_p': I_p, 'I_i': I_i,
        'm_values': m_values
    }

def get_psi0(N, m_values):
    """Constructs the initial state."""
    def cm(m):
        c = math.comb(N, N//2 + m)
        return (1/(2**(N/2))) * math.sqrt(c)
    
    dim_B = N + 1
    psiB0 = np.zeros(dim_B, dtype=complex)
    for i, m in enumerate(m_values):
        psiB0[i] = cm(m)
    
    psiS0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    psi0 = np.kron(psiS0, psiB0)
    return psi0

def get_hamiltonian(ops, Omega, omega, g_int):
    """Constructs the Hamiltonian."""
    S_x = ops['S_x']
    I_x = ops['I_x']
    I_z = ops['I_z']
    I_i = ops['I_i']
    
    I_z2 = I_z @ I_z
    
    # H = (Omega/2) S_x + omega I_x + g_int * S_x * I_z^2
    H = (Omega / 2.0) * S_x + omega * I_x + g_int * (I_i @ I_z2)
    return H

def get_psit(H, psi0, t):
    """Calculates the state at time t."""
    U = expm(-1j * H * t)
    return U @ psi0

def run_simulation(N, times, H, psi0, ops):
    """Runs the time evolution and calculates observables."""
    I_x = ops['I_x']
    I_y = ops['I_y']
    I_z = ops['I_z']
    I_p = ops['I_p']
    J_bath = N / 2.0
    
    I_p2 = I_p @ I_p
    Iyz_sym = 0.5 * (I_y @ I_z + I_z @ I_y)
    
    results = {
        'mean_Ix': [], 'mean_Iy': [], 'mean_Iz': [],
        'mean_Ix2': [], 'mean_Iy2': [], 'mean_Iz2': [],
        'mean_Ip2': [], 'mean_Iyz': [],
        'min_var': [], 'xiS2': [], 'xiR2': [], 'theta_opt': []
    }
    
    print("Starting time evolution...")
    for t in times:
        psi = get_psit(H, psi0, t)
        
        def expect(Op):
            return np.real(np.vdot(psi, Op @ psi))
        
        # Calculate moments
        mean_Ix = expect(I_x)
        mean_Iy = expect(I_y)
        mean_Iz = expect(I_z)
        
        mean_Ix2 = expect(I_x @ I_x)
        mean_Iy2 = expect(I_y @ I_y)
        mean_Iz2 = expect(I_z @ I_z) 
        mean_Ip2 = expect(I_p2) 
        
        mean_Iyz_sym = expect(Iyz_sym)
        
        # ---- variances and covariance (physical units) ----
        dIy2 = mean_Iy2 - mean_Iy**2
        dIz2 = mean_Iz2 - mean_Iz**2
        cov  = mean_Iyz_sym - mean_Iy*mean_Iz
        
        # ---- minimal variance in the transverse (y-z) plane ----
        tr  = dIy2 + dIz2
        det_term = (dIy2 - dIz2)**2 + 4*cov**2
        min_var = 0.5 * (tr - np.sqrt(det_term))
        results['min_var'].append(min_var)
        
        # ---- squeezing parameters ----
        xiS2 = (2.0 * min_var) / J_bath
        results['xiS2'].append(xiS2)
        
        if abs(mean_Ix) > 1e-12:
            xiR2 = (N * min_var) / (mean_Ix**2)
        else:
            xiR2 = np.nan
        results['xiR2'].append(xiR2)
        
        results['mean_Ix'].append(4*mean_Ix/N)
        results['mean_Iy'].append(4*mean_Iy/N)
        results['mean_Iz'].append(4*mean_Iz/N)
        results['mean_Ix2'].append(4*mean_Ix2/N)
        results['mean_Iy2'].append(4*mean_Iy2/N)
        results['mean_Iz2'].append(4*mean_Iz2/N)
        results['mean_Ip2'].append(mean_Ip2)
        results['mean_Iyz'].append(4*mean_Iyz_sym/N)
        
        theta_opt = np.tan(2.0 * cov/ (dIy2 - dIz2))
        results['theta_opt'].append(theta_opt)
        
    return results

def plot_results(times, results, N, g_int):
    """Plots the simulation results."""
    mean_Ix_list = results['mean_Ix']
    mean_Iy_list = results['mean_Iy']
    mean_Iz_list = results['mean_Iz']
    mean_Ix2_list = results['mean_Ix2']
    mean_Iy2_list = results['mean_Iy2']
    mean_Iz2_list = results['mean_Iz2']
    mean_Ip2_list = results['mean_Ip2']
    xiS2_list = results['xiS2']
    theta_opt_list = results['theta_opt']
    
    print("Evolution complete. Plotting...")

    theta2 = np.unwrap(np.array(theta_opt_list))
    theta_opt_list = 0.5 * theta2

    # Plotting
    fig, axes = plt.subplots(5, 2, figsize=(12, 25), sharex=True)

    # --- Row 1: Means ---
    axes[0, 0].plot(times, mean_Ix_list, label='<Ix>')
    axes[0, 0].set_ylabel('<Ix>')
    axes[0, 0].set_title('Mean Ix')
    axes[0, 0].grid(True)

    axes[1, 0].plot(times, mean_Iy_list, label='<Iy>')
    axes[1, 0].set_ylabel('<Iy>')
    axes[1, 0].set_title('Mean Iy')
    axes[1, 0].grid(True)

    axes[2, 0].plot(times, mean_Iz_list, label='<Iz>')
    axes[2, 0].set_ylabel('<Iz>')
    axes[2, 0].set_title('Mean Iz')
    axes[2, 0].grid(True)

    # --- Column 2: Second moments ---
    axes[0, 1].plot(times, mean_Ix2_list, label='<Ix^2>', color='orange')
    axes[0, 1].set_ylabel('<Ix^2>')
    axes[0, 1].set_title('Mean Ix^2')
    axes[0, 1].grid(True)

    axes[1, 1].plot(times, mean_Iy2_list, label='<Iy^2>', color='orange')
    axes[1, 1].set_ylabel('<Iy^2>')
    axes[1, 1].set_title('Mean Iy^2')
    axes[1, 1].grid(True)

    axes[2, 1].plot(times, mean_Iz2_list, label='<Iz^2>', color='orange')
    axes[2, 1].set_ylabel('<Iz^2>')
    axes[2, 1].set_title('Mean Iz^2')
    axes[2, 1].grid(True)
    axes[2, 1].ticklabel_format(style='plain', useOffset=False)

    # --- Row 4: Optimal angle & Squeezing ---
    axes[3, 0].plot(times, theta_opt_list, label=r'$\theta_{\mathrm{opt}}$')
    axes[3, 0].set_ylabel(r'$\theta_{\mathrm{opt}}$ (rad)')
    axes[3, 0].set_xlabel('Time')
    axes[3, 0].set_title('Optimal squeezing angle in y-z plane')
    axes[3, 0].grid(True)

    axes[3, 1].plot(times, xiS2_list, label=r'$\xi_S^2$')
    axes[3, 1].axhline(1.0, linestyle='--', linewidth=1)
    axes[3, 1].set_ylabel(r'$\xi_S^2$')
    axes[3, 1].set_xlabel('Time')
    axes[3, 1].set_title('Spin squeezing (Kitagawa–Ueda)')
    axes[3, 1].grid(True)

    # --- Row 5: <S+^2> ---
    axes[4, 0].plot(times, np.real(mean_Ip2_list), label='Re[<S+^2>]')
    axes[4, 0].plot(times, np.imag(mean_Ip2_list), label='Im[<S+^2>]', linestyle='--')
    
    # Analytical
    a_param = g_int
    p_t = np.exp(-2 * N * (a_param**2) * (times**2)) * (N**2/4.0 - N/4.0)
    axes[4, 0].plot(times, p_t, label='Analytical', linestyle=':', color='black')
    
    axes[4, 0].set_ylabel('<S+^2>')
    axes[4, 0].set_xlabel('Time')
    axes[4, 0].set_title('Expectation of S_+^2')
    axes[4, 0].legend()
    axes[4, 0].grid(True)
    
    axes[4, 1].axis('off')

    plt.tight_layout()
    plt.savefig('spin_squeezing_dickie.png')
    print("Saved plot to 'spin_squeezing_dickie.png'")


def main():
    # Parameters
    N = 10
    J_bath = N / 2.0
    
    # Physical Parameters
    J_coupling = 2.0
    Omega = 0.0
    omega = 0.0
    g_int = (J_coupling**2) / 1.0 # J^2/Omega assumed 1.0 denominator here? User code had 1.0.
    
    # Time parameters
    t_max = 3.0
    steps = 200
    times = np.linspace(0, t_max, steps)

    print(f"Generating operators for N={N} (Bath Spin J={J_bath})...")
    
    ops = get_total_operators(N)
    psi0 = get_psi0(N, ops['m_values'])
    H = get_hamiltonian(ops, Omega, omega, g_int)
    
    results = run_simulation(N, times, H, psi0, ops)
    plot_results(times, results, N, g_int)
    
    # Old debug print
    # print(results['mean_Iz2'][-1]) # not exactly trivial to match the exact end print

if __name__ == "__main__":
    main()
