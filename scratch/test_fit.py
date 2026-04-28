import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Let's mock a few values of Omega and T to see the trend
# To do this accurately, I'll extract a portion of the original code just up to the optimal times.
# For speed I'll reduce the grid sizes.

from Analysis_of_analyitical_dicky import SimulationConfig, get_bath_density_matrices, compute_bath_qfi_trajectory

def test():
    cfg = SimulationConfig(n_steps=20, n_omegas=10, omega_max=10.0, omega_min=1.0)
    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    omega_list = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_omegas)

    qfi_matrix = np.zeros((len(omega_list), len(tlist)))
    
    for i, Omega in enumerate(omega_list):
        bath_rhos_plus = get_bath_density_matrices(Omega, cfg.J_nominal + cfg.dJ, tlist, cfg.N, cfg.gamma, cfg.beta, cfg.num_model)
        bath_rhos_minus = get_bath_density_matrices(Omega, cfg.J_nominal - cfg.dJ, tlist, cfg.N, cfg.gamma, cfg.beta, cfg.num_model)
        
        qfi_t, _ = compute_bath_qfi_trajectory(bath_rhos_plus, bath_rhos_minus, cfg.dJ, cfg.qfi_tol)
        qfi_matrix[i, :] = qfi_t

    qcrb_matrix = 1.0 / (qfi_matrix + cfg.qcrb_eps)
    optimal_time_idx = np.argmin(qcrb_matrix, axis=1)
    optimal_times = tlist[optimal_time_idx]
    
    print("Omega:", omega_list)
    print("Optimal Times:", optimal_times)

test()
