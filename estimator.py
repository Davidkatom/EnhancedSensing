import numpy as np
import scipy.linalg
from scipy.linalg import expm
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
import spin_squeezing_dickie as ssd

# --- Configuration ---
# N = 20
J_TRUE = 2.0
T_MAX = 1
STEPS = 50
# TIMES = np.linspace(0, T_MAX, STEPS)
SHOTS_MAX = 10

def get_naive_probs(N, times, J):
    """
    Simulates Method 1: Central Spin driving Bath.
    H = J * Sz * Iz.
    P(+X) = cos^2(Jt/2).
    """
    p0_list = []
    for t in times:
        p0 = np.cos(J * t / 2.0)**2
        p0_list.append(p0)
    return np.array(p0_list)

def get_squeezed_probs(N, times, J):
    ops = ssd.get_total_operators(N)

    Omega = 0.0
    omega = 0.0
    g_int = J**2

    psi0 = ssd.get_psi0(N, ops['m_values'])
    H = ssd.get_hamiltonian(ops, Omega, omega, g_int)

    probs_list = []
    m_values = ops['m_values']
    numerical = []
    numerical_y2 = []
    numerical_cov = []
    Iy = ops['I_y']
    Iz = ops['I_z']

    for t in times:
        psi = ssd.get_psit(H, psi0, t)

        # --- compute moments ---
        mean_Iy  = np.real(np.vdot(psi, Iy @ psi))
        mean_Iz  = np.real(np.vdot(psi, Iz @ psi))

        mean_Iy2 = np.real(np.vdot(psi, Iy @ Iy @ psi))
        mean_Iz2 = np.real(np.vdot(psi, Iz @ Iz @ psi))
        mean_Iyz = np.real(np.vdot(psi, (Iy @ Iz + Iz @ Iy) @ psi) / 2)

        dIy2 = mean_Iy2 - mean_Iy**2
        dIz2 = mean_Iz2 - mean_Iz**2
        cov  = mean_Iyz - mean_Iy * mean_Iz

        # --- optimal squeezing angle ---
        theta = 0.5 * np.arctan2(2 * cov, dIy2 - dIz2)

        # --- rotate so I_theta -> Iz ---
        R = expm(1j * theta * ops['I_x'])
        psi_rot = R @ psi

        # --- sample Iz ---
        psi_sq = np.abs(psi_rot)**2
        dim_B = N + 1
        probs_m = psi_sq[:dim_B] + psi_sq[dim_B:]

        probs_list.append(probs_m)
        numerical.append(0.5 * (mean_Iy2 + mean_Iz2 - np.sqrt((mean_Iy2 - mean_Iz2)**2 + 4*(cov)**2)))
        numerical_y2.append(mean_Iy2)
        numerical_cov.append(cov)

    return probs_list, m_values, numerical, numerical_y2, numerical_cov


def fit_naive(times, y_data):
    # Fit cos(J * t)
    def model(t, J_est):
        return np.cos(J_est * t)
    
    try:
        popt, _ = curve_fit(model, times, y_data, p0=np.random.uniform(1, 3, 1), bounds=(0, 10))
        return popt[0]
    except:
        return np.nan

def fit_squeezed_exact(times, y_data, N):

    def model(t, J):
        Sz2 = N / 4

        Sy2 = (N*(N+1))/8 - (N*(N-1))/8 * np.cos(2*(J)*t)**(N-2)

        cov = (N*(N-1))/4 * np.sin(2*(J)*t) * np.cos(2*(J)*t)**(N-3)

        return 0.5 * (
            Sy2 + Sz2 - np.sqrt((Sy2 - Sz2)**2 + 2*cov**2)
        )

    try:
        popt, _ = curve_fit(model, times, y_data, p0=np.random.uniform(1, 3, 1), bounds=(0,5))
        return popt[0]
    except:
        return np.nan

def fit_squeezed(times, y_data, N):

    def model(t, J):

        Sy2 = (N_val*(N_val+1))/8 - (N_val*(N_val-1))/8 * np.cos((2*J**2)*t)**(N_val-2)
        Sz2 = N_val / 4
        cov = (N_val*(N_val-1))/4 * np.sin((J**2)*t) * np.cos((J**2)*t)**(N_val-3)

        val_exact = 0.5 * (Sy2 + Sz2 - np.sqrt((Sy2 - Sz2)**2 + 4*(cov)**2))
        return val_exact

    try:
        popt, _ = curve_fit(model, times, y_data, p0=np.random.uniform(1, 3, 1), bounds=(0,5))
        return popt[0]
    except:
        return np.nan



def run_comparison():
    print(f"Running comparison for Fixed Shots={SHOTS_MAX}, True J={J_TRUE}")

    
    # N sweep configuration
    N_SWEEP = [4, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]


    avg_err_naive = []
    avg_err_squeezed = []
    avg_err_squeezed_exact = []
    
    NUM_AVERAGES = 100
    FIXED_SHOTS = SHOTS_MAX

    debug_squeezing_data = {} 
    
    print("Starting N sweep...")

    for N_val in N_SWEEP:
        print(f" Simulating N={N_val}...")
        
        # Checking logic: `p0 = np.cos(J * t / 2.0)**2`. Yes independent.
        times_naive = np.linspace(0, T_MAX, STEPS)
        opt_time = 2 / (np.sqrt(2 * (N_val - 2)) * J_TRUE**2)
        times_squeezed = np.linspace(0, opt_time, STEPS)
        probs_naive = get_naive_probs(N_val, times_naive, J_TRUE) 
        
        # Squeezed: Depends heavily on N
        probs_squeezed, m_vals, numerical, numerical_y2, numerical_cov = get_squeezed_probs(N_val, times_squeezed, J_TRUE) 
        
        temp_err_naive = []
        temp_err_squeezed = []
        temp_err_squeezed_exact = []

        is_last_N = (N_val == N_SWEEP[-1])

        for run_idx in range(NUM_AVERAGES):
            is_debug_run = is_last_N and (run_idx == 0)

            # --- Method 1: Naive ---
            mx_means = []
            for p0 in probs_naive:
                n_total_bits = N_val * FIXED_SHOTS
                count_ones = np.random.binomial(n_total_bits, p0)
                mx_sample = (2 * count_ones - n_total_bits) / n_total_bits
                mx_means.append(mx_sample)
                
            j_naive = fit_naive(times_naive, mx_means)
            temp_err_naive.append(abs(j_naive - J_TRUE))
            
            # --- Method 2 & 3: Squeezed ---
            iy2_means = []
            
            if is_debug_run:
                debug_squeezing_data['times'] = times_squeezed
                debug_squeezing_data['sampled'] = []
                debug_squeezing_data['analytical_approx'] = []
                debug_squeezing_data['analytical_exact'] = []
                debug_squeezing_data['numerical'] = []
                debug_squeezing_data['numerical_y2'] = []
                debug_squeezing_data['analytical_y2'] = []
                debug_squeezing_data['numerical_cov'] = []
                debug_squeezing_data['analytical_cov'] = []
            
            term1 = (N_val*(N_val+1))/8.0
            term2 = (N_val*N_val - N_val)/4.0
            for idx, probs_m in enumerate(probs_squeezed):
                samples_m = np.random.choice(m_vals, size=FIXED_SHOTS, p=probs_m / np.sum(probs_m))
                mean_m  = np.mean(samples_m)
                mean_m2 = np.mean(samples_m**2)
                var_m   = mean_m2 - mean_m**2
                iy2_means.append(var_m)

                if is_debug_run:
                    t = times_squeezed[idx]
                    # exact squeezed variance
                    Sy2 = (N_val*(N_val+1))/8 - (N_val*(N_val-1))/8 * np.cos((2*J_TRUE**2)*t)**(N_val-2)
                    Sz2 = N_val / 4
                    cov = (N_val*(N_val-1))/4 * np.sin((J_TRUE**2)*t) * np.cos((J_TRUE**2)*t)**(N_val-3)

                    val_exact = 0.5 * (Sy2 + Sz2 - np.sqrt((Sy2 - Sz2)**2 + 4*(cov)**2))

                    pref = (N_val/4) * ((N_val-1)/2)**(1/3)
                    val_approx = pref * np.exp(-2*(N_val-2)*(J_TRUE**2)*t**2)

                    debug_squeezing_data['sampled'].append(var_m)
                    debug_squeezing_data['analytical_exact'].append(val_exact)
                    debug_squeezing_data['analytical_y2'].append(Sy2)
                    debug_squeezing_data['analytical_approx'].append(val_approx)
                    debug_squeezing_data['numerical'].append(numerical[idx])
                    debug_squeezing_data['numerical_y2'].append(numerical_y2[idx])
                    debug_squeezing_data['numerical_cov'].append(numerical_cov[idx])
                    debug_squeezing_data['analytical_cov'].append(cov)

            j_squeezed = fit_squeezed(times_squeezed, var_m, N_val)

            temp_err_squeezed.append(abs(j_squeezed - J_TRUE))
            
            j_squeezed_ex = fit_squeezed_exact(times_squeezed, iy2_means, N_val)
            temp_err_squeezed_exact.append(abs(j_squeezed_ex - J_TRUE))
        
        # Average errors
        avg_err_naive.append(np.mean(temp_err_naive))
        avg_err_squeezed.append(np.mean(temp_err_squeezed))
        avg_err_squeezed_exact.append(np.mean(temp_err_squeezed_exact))
        
        print(f"N={N_val}: Naive={avg_err_naive[-1]:.4f}, SqApprox={avg_err_squeezed[-1]:.4f}, SqExact={avg_err_squeezed_exact[-1]:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(N_SWEEP, avg_err_naive, 'o-', label=f'Naive (M={FIXED_SHOTS})')
    # plt.plot(N_SWEEP, avg_err_squeezed, 's-', label=f'Squeezed Approx (M={FIXED_SHOTS})')
    plt.plot(N_SWEEP, avg_err_squeezed_exact, '^-', label=f'Squeezed Exact (M={FIXED_SHOTS})')
    
    # Scaling lines
    # SQL: 1/sqrt(N)
    # HL: 1/N
    # We normalized them to match the last point roughly or just plotting slope?
    # Let's plot raw functions and user can see slope.
    
    n_arr = np.array(N_SWEEP)
    plt.plot(n_arr, 1 /( STEPS*SHOTS_MAX * n_arr)**(1/2), '--', color='gray', label=r'$1/\sqrt{N}$ (SQL)')
    plt.plot(n_arr, 1 /((np.sqrt(STEPS*SHOTS_MAX)) * n_arr**(2/3)), ':', color='black', label=r'$1/N$ (Heisenberg)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Spins N')
    plt.ylabel('Estimation Error |J_est - J_true|')
    plt.title(f'J Estimation Error vs N (Shots={FIXED_SHOTS})')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('error_comparison_N_sweep.png')
    print("Saved 'error_comparison_N_sweep.png'")
    
    # --- Plotting Debug ---
    if 'times' in debug_squeezing_data:
        plt.figure(figsize=(10, 6))
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_approx'], label='Analytical Approx', color='black', linestyle='--')
        plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_exact'], label='Analytical Exact', color='green', linestyle='-')
        plt.plot(debug_squeezing_data['times'], debug_squeezing_data['sampled'], 'o', label=f'Sampled (N={N_SWEEP[-1]})', alpha=0.3)
        plt.plot(debug_squeezing_data['times'], debug_squeezing_data['numerical'], 'x', label=f'Numerical (N={N_SWEEP[-1]})', alpha=0.3)
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['numerical_y2'], 's', label=f'Numerical Y^2 (N={N_SWEEP[-1]})', alpha=0.3)
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_y2'], label=f'Analytical Y^2 (N={N_SWEEP[-1]})', alpha=0.3)

        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['numerical_cov'], 's', label=f'Numerical Cov (N={N_SWEEP[-1]})', alpha=0.3)
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_cov'], label=f'Analytical Cov (N={N_SWEEP[-1]})', alpha=0.3)
        
        plt.xlabel('Time')
        plt.ylabel('<V-^2>')
        plt.title(f'Debug: Squeezed Component (N={N_SWEEP[-1]})')
        plt.legend()
        plt.grid(True)
        plt.savefig('debug_squeezing.png')
        print("Saved 'debug_squeezing.png'")


if __name__ == "__main__":
    try:
        run_comparison()
    except Exception:
        import traceback
        traceback.print_exc()