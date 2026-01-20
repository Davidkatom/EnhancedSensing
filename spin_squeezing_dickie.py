
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm

def main():
    # Parameters
    N = 10  # Number of bath spins
    J_bath = N / 2.0 # Total spin of the bath
    
    # Physical Parameters
    # Using same values as previous script where applicable
    J_coupling = 1.0   # J in Hamiltonian
    Omega = 0.0 # System Rabi frequency
    omega = 0.0  # Bath splitting
    
    # Hamiltonian term prefactor for interaction: J^2 / Omega
    # H_int = (J^2 / Omega) * S_x * I_z^2
    g_int = (J_coupling**2) / 1.0 #switch to Omega

    # Time parameters
    t_max = 3
    steps = 200
    times = np.linspace(0, t_max, steps)

    print(f"Generating operators for N={N} (Bath Spin J={J_bath})...")
    
    # Dimensions
    dim_S = 2
    dim_B = N +1
    dim_total = dim_S * dim_B
    print(f"Total dimension: {dim_total}")

    # 1. System Operators (Pauli matrices)
    # Basis: |0> (up), |1> (down) => sigma_z = diag(1, -1)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id_S = np.eye(dim_S, dtype=complex)

    # 2. Bath Operators (Spin-J matrices)
    # Basis: |J, m> ordered from m = J to m = -J
    # m values: J, J-1, ..., -J
    m_values = np.arange(-N//2, N//2 + 1)
    print("m_values: ", m_values)
    # J_z is diagonal
    Jz_B = np.diag(m_values).astype(complex)
    
    # J_+ ladder operator
    # J_+ |m> = sqrt(J(J+1) - m(m+1)) |m+1>
    # Matrix element (m', m): <m'|J+|m> is non-zero if m' = m+1
    # Indices i correspond to m_values[i]. m_values is descending.
    # m_values[i] = m. m_values[i-1] = m+1.
    # So J_+ connects i to i-1. Element (i-1, i).
    
    Jp_B = np.zeros((dim_B, dim_B), dtype=complex)
    for i, m in enumerate(m_values):
        if i < dim_B - 1:                    # <-- prevents wraparound
            coeff = np.sqrt(J_bath*(J_bath+1) - m*(m+1))
            Jp_B[i+1, i] = coeff   
            
    Jm_B = Jp_B.conj().T
    
    Jx_B = 0.5 * (Jp_B + Jm_B)
    Jy_B = 0.5 * (Jp_B - Jm_B) / 1j
    id_B = np.eye(dim_B, dtype=complex)

    # 3. Full Operators (System x Bath)
    # System is first subsystem
    
    S_x = np.kron(sx, id_B)
    S_z = np.kron(sz, id_B)
    
    I_x = np.kron(id_S, Jx_B)
    I_y = np.kron(id_S, Jy_B)
    I_z = np.kron(id_S, Jz_B)
    
    I_z2 = I_z @ I_z
    
    # 4. Hamiltonian
    # H = (Omega/2) S_x + omega I_x + (J^2 / Omega) S_x I_z^2
    H = (Omega / 2.0) * S_x + omega * I_x + g_int * (S_x @ I_z2)
    
    def cm(m):
        c = math.comb(N, N//2 + m)
        return (1/(2**(N/2))) * np.sqrt(c)
    
    psiB0 = np.zeros(dim_B, dtype=complex)
    for i, m in enumerate(m_values):
        psiB0[i] = cm(m)
    
    psiS0 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    
    psi0 = np.kron(psiS0, psiB0)
    

    mean_Ix_list = []
    mean_Iy_list = []
    mean_Iz_list = []
    mean_Ix2_list = []
    mean_Iy2_list = []
    mean_Iz2_list = []
    mean_Iyz_list = []
    min_var_list  = []
    xiS2_list     = []
    xiR2_list     = []
    theta_opt_list = []   # optimal squeezing angle in the y-z plane

    Iyz_sym = 0.5 * (I_y @ I_z + I_z @ I_y)


    def psi_t(t):
        U = expm(-1j * H * t)
        return U @ psi0
        # return np.exp(-1j * H * t) @ psi0
        # psiBt = np.zeros(dim_B, dtype=complex)
        # for i, m in enumerate(m_values):
        #     psiBt[i] = cm(m)*np.exp(-1j * g_int * m * m * t)
        # return np.kron(psiS0, psiBt)
    
    print("Starting time evolution...")
    for t in times:
      
        def expect(Op):
            return np.real(np.vdot(psi_t(t), Op @ psi_t(t)))
        
        # Calculate moments
        # <Sx> (Bath)
        mean_Ix = expect(I_x)
        mean_Iy = expect(I_y)
        mean_Iz = expect(I_z)
        
        mean_Ix2 = expect(I_x @ I_x)
        mean_Iy2 = expect(I_y @ I_y)
        mean_Iz2 = expect(I_z @ I_z) 
        
        mean_Iyz_sym = expect(Iyz_sym)
    # ---- variances and covariance (physical units) ----
        dIy2 = mean_Iy2 - mean_Iy**2
        dIz2 = mean_Iz2 - mean_Iz**2
        cov  = mean_Iyz_sym - mean_Iy*mean_Iz   # since mean_Iyz_sym = 1/2< IyIz+IzIy >

        # ---- minimal variance in the transverse (y-z) plane ----
        # eigenvalues of 2x2 covariance matrix:
        # [dIy2  cov]
        # [cov   dIz2]
        tr  = dIy2 + dIz2
        det_term = (dIy2 - dIz2)**2 + 4*cov**2
        min_var = 0.5 * (tr - np.sqrt(det_term))
        min_var_list.append(min_var)

        # ---- squeezing parameters ----
        # Kitagawa–Ueda: baseline for coherent state is J/2
        xiS2 = (2.0 * min_var) / J_bath
        xiS2_list.append(xiS2)

        # Wineland-like (avoid divide-by-zero)
        if abs(mean_Ix) > 1e-12:
            xiR2 = (N * min_var) / (mean_Ix**2)
        else:
            xiR2 = np.nan
        xiR2_list.append(xiR2)

        mean_Ix_list.append(4*mean_Ix/N)
        mean_Iy_list.append(4*mean_Iy/N)
        mean_Iz_list.append(4*mean_Iz/N)
        mean_Ix2_list.append(4*mean_Ix2/N)
        mean_Iy2_list.append(4*mean_Iy2/N)
        mean_Iz2_list.append(4*mean_Iz2/N)
        mean_Iyz_list.append(4*mean_Iyz_sym/N)
        

        theta_opt = np.tan(2.0 * cov/ (dIy2 - dIz2))
        theta_opt_list.append(theta_opt)

    print("Evolution complete. Plotting...")

    theta2 = np.unwrap(np.array(theta_opt_list))   # store 2*theta during loop
    theta_opt_list = 0.5 * theta2
    # theta_opt_list = np.unwrap(np.array(theta_opt_list))

    # Plotting (now 4x2 to include squeezing diagnostics)
    fig, axes = plt.subplots(4, 2, figsize=(12, 20), sharex=True)

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

    # --- Row 5: Optimal angle ---
    axes[3, 0].plot(times, theta_opt_list, label=r'$\theta_{\mathrm{opt}}$')
    axes[3, 0].set_ylabel(r'$\theta_{\mathrm{opt}}$ (rad)')
    axes[3, 0].set_xlabel('Time')
    axes[3, 0].set_title('Optimal squeezing angle in y-z plane')
    axes[3, 0].grid(True)

    # Right: squeezing parameter xi_S^2  (no manual colors per your other plots is fine; but keeping default is ok)
    axes[3, 1].plot(times, xiS2_list, label=r'$\xi_S^2$')
    axes[3, 1].axhline(1.0, linestyle='--', linewidth=1)
    axes[3, 1].set_ylabel(r'$\xi_S^2$')
    axes[3, 1].set_xlabel('Time')
    axes[3, 1].set_title('Spin squeezing (Kitagawa–Ueda)')
    axes[3, 1].grid(True)

    # print(2*(3/2)**1/6 * 1 / (N**(2/3)*0.25))
    print(mean_Iz2)


    plt.tight_layout()
    plt.savefig('spin_squeezing_dickie.png')
    print("Saved plot to 'spin_squeezing_dickie.png'")

if __name__ == "__main__":
    main()
