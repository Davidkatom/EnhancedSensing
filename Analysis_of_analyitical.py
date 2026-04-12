import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

def get_Sx2_stats(Omega, J, tlist, N=10, gamma=1.0, a_param=1.0):
    """
    Computes probabilities of Sx^2 outcomes and <Sx^2>, var(Sx^2) 
    using Analytical Solution 2, with robust numerical limits.
    """
    S_spin = N / 2.0
    dim_bath = int(2 * S_spin + 1)
    
    # Operators
    Jx = qt.jmat(S_spin, 'x') * 2.0
    Sx2 = Jx**2
    Sx2_mat = Sx2.full()
    Sx4_mat = (Sx2**2).full()
    
    # Eigenstates
    evals_x, evecs_x = Jx.eigenstates()
    Sx2_evals = np.round(evals_x**2, 5)
    unique_evals = np.unique(Sx2_evals)
    
    projector_mats = []
    for val in unique_evals:
        proj = qt.Qobj(np.zeros((dim_bath, dim_bath)))
        for i, ev in enumerate(Sx2_evals):
            if np.abs(ev - val) < 1e-4:
                proj += evecs_x[i] * evecs_x[i].dag()
        projector_mats.append(proj.full())
        
    plus_state_bath = qt.spin_coherent(S_spin, np.pi/2, 0)
    rho0_bath = plus_state_bath * plus_state_bath.dag()
    rho0_mat = rho0_bath.full()
    
    m_vals = np.diag(qt.jmat(S_spin, 'z').full())
    n_mat, m_mat = np.meshgrid(m_vals, m_vals, indexing='ij')
    
    r0 = 1.0
    x0 = 1.0
    
    delta_mat = (J**2 / (2 * Omega)) * (n_mat**2 - m_mat**2) if Omega != 0 else np.zeros_like(n_mat)
    k_mat = a_param * (J**2 / (2 * Omega**2)) * gamma * (n_mat - m_mat)**2 if Omega != 0 else np.zeros_like(n_mat)
    mu_mat = 0.5 * np.sqrt((k_mat - 2 * gamma + 0j)**2 - 64 * delta_mat**2 + 0j)
    
    probs = np.zeros((len(unique_evals), len(tlist)))
    sx2_expect = np.zeros(len(tlist))
    sx2_var = np.zeros(len(tlist))
    
    for idx, t in enumerate(tlist):
        E_plus = -(k_mat + 2 * gamma) * t / 2.0 + mu_mat * t
        E_minus = -(k_mat + 2 * gamma) * t / 2.0 - mu_mat * t
        
        r_nm_t = np.zeros_like(mu_mat, dtype=np.complex128)
        
        # Where mu is effectively zero:
        zero_mask = np.abs(mu_mat) < 1e-12
        if np.any(zero_mask):
            A = (2 * gamma - k_mat[zero_mask]) * r0 + 8j * delta_mat[zero_mask] * x0
            r_nm_t[zero_mask] = np.exp(-(k_mat[zero_mask] + 2 * gamma) * t / 2.0) * (r0 + A * t / 2.0)
            
        # Where mu is non-zero:
        nonzero_mask = ~zero_mask
        if np.any(nonzero_mask):
            term_C = ((2 * gamma - k_mat[nonzero_mask]) * r0 + 8j * delta_mat[nonzero_mask] * x0) / (2 * mu_mat[nonzero_mask])
            r_nm_t[nonzero_mask] = 0.5 * np.exp(E_plus[nonzero_mask]) * (r0 + term_C) + \
                                   0.5 * np.exp(E_minus[nonzero_mask]) * (r0 - term_C)
                                   
        rho_t_mat = rho0_mat * r_nm_t
        
        # Calculate probabilities of Sx^2 outcomes
        for k, proj_mat in enumerate(projector_mats):
            probs[k, idx] = np.real(np.sum(proj_mat.T * rho_t_mat))
            
        # Expectation value & variance
        ex_sx2 = np.real(np.sum(Sx2_mat.T * rho_t_mat))
        ex_sx4 = np.real(np.sum(Sx4_mat.T * rho_t_mat))
        
        sx2_expect[idx] = ex_sx2
        sx2_var[idx] = ex_sx4 - ex_sx2**2
        
    return probs, sx2_expect, sx2_var

def main():
    # Parameters matches SWcomparison.py
    N = 50
    gamma = 1.0
    J_nominal = 1.0
    dJ = 1e-4
    a_param = 1.0
    
    t_max = 5.0
    n_steps = 150
    tlist = np.linspace(0.01, t_max, n_steps)
    
    # Omega scan range
    Omega_list = np.linspace(0.5, 10.0, 60)
    
    fi_matrix = np.zeros((len(Omega_list), len(tlist)))
    
    print(f"Calculating Cramér-Rao Bound (CRB) map for {len(Omega_list)} values of Omega...")
    
    for i, Omega in enumerate(Omega_list):
        # Calculate statistics at J + dJ and J - dJ
        probs_plus, ex_plus, var_plus = get_Sx2_stats(Omega, J_nominal + dJ, tlist, N=N, gamma=gamma, a_param=a_param)
        probs_minus, ex_minus, var_minus = get_Sx2_stats(Omega, J_nominal - dJ, tlist, N=N, gamma=gamma, a_param=a_param)
        
        # CFI Calculation: sum_k (dP_k / dJ)^2 / P_k
        dP_dJ = (probs_plus - probs_minus) / (2 * dJ)
        probs_nominal = (probs_plus + probs_minus) / 2.0  # Approx probability at J_nominal
        
        # Avoid division by zero by adding epsilon
        eps = 1e-12
        cfi_t = np.sum((dP_dJ)**2 / (probs_nominal + eps), axis=0)
        
        fi_matrix[i, :] = cfi_t
        
    # Cramér-Rao Bound is 1 / FI.
    # Add a small epsilon to avoid division by zero.
    crb_matrix = 1.0 / (fi_matrix + 1e-15)
    
    # Min CRB over time for each Omega
    min_crb_per_omega = np.min(crb_matrix, axis=1)
    
    optimal_omega_idx = np.argmin(min_crb_per_omega)
    optimal_omega = Omega_list[optimal_omega_idx]
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D Heatmap
    # We clip CRB for plotting to avoid inf values at t=0 overtaking the colormap
    vmax_crb = np.percentile(min_crb_per_omega, 95) * 5.0 # dynamic cap based on the valid bounds
    im = axes[0].pcolormesh(tlist, Omega_list, crb_matrix, shading='auto', cmap='viridis', vmax=vmax_crb)
    axes[0].set_xlabel("Time (t)")
    axes[0].set_ylabel(r"Transverse Field ($\Omega$)")
    axes[0].set_title(r"Cramér-Rao Bound (CRB)")
    fig.colorbar(im, ax=axes[0])
    
    # 1D plot of min CRB vs Omega
    axes[1].plot(Omega_list, min_crb_per_omega, marker='o', linestyle='-', color='teal', label='Minimum CRB')
    axes[1].axvline(optimal_omega, color='red', linestyle='--', label=f'Optimal $\\Omega$ = {optimal_omega:.2f}')
    axes[1].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[1].set_ylabel(r"Min (over $t$) CRB")
    axes[1].set_title(r"Minimum CRB vs $\Omega$")
    
    # Optional: use log scale if CRB varies over orders of magnitude
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', which='both')
    
    plt.tight_layout()
    # Save the figure to file
    plt.savefig('cramer_rao_bound_analysis.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
