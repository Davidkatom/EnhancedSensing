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
STEPS = 20
# TIMES = np.linspace(0, T_MAX, STEPS)
SHOTS_MAX = 1

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
        theta = np.pi/2 #measure sy^2

        # --- rotate so I_theta -> Iz ---
        R = expm(1j * theta * ops['I_x'])
        psi_rot = R @ psi

        # --- sample Iz ---
        psi_sq = np.abs(psi_rot)**2
        dim_B = N + 1
        probs_m = psi_sq[:dim_B] + psi_sq[dim_B:]

        probs_list.append(probs_m)
        numerical.append(0.5 * (dIy2 + dIz2 - np.sqrt((dIy2 - dIz2)**2 + 4*(cov)**2)))
        numerical_y2.append(dIy2)
        numerical_cov.append(cov)

    return probs_list, m_values, numerical, numerical_y2, numerical_cov


def fit_naive(times, y_data):
    # Fit cos(J * t)
    def model(t, J_est):
        return np.cos(J_est * t)
    
    best_popt = None
    best_cost = np.inf
    bounds = (0, 10)
    guesses = np.linspace(max(0.1, bounds[0]), bounds[1], 20)
    
    for g in guesses:
        try:
            popt, _ = curve_fit(model, times, y_data, p0=[g], bounds=bounds)
            cost = np.sum((model(times, popt[0]) - y_data)**2)
            if cost < best_cost:
                best_cost = cost
                best_popt = popt
        except:
            pass
            
    if best_popt is not None:
        return best_popt[0]
    return np.nan

def fit_squeezed_exact(times, y_data, N):

    def model(t, J):
        Sz2 = N / 4

        Sy2 = (N*(N+1))/8 - (N*(N-1))/8 * np.cos(2*(J)**2*t)**(N-2)

        cov = (N*(N-1))/4 * np.sin((J)**2*t) * np.cos((J)**2*t)**(N-3)

        return Sy2 #debug measure only sy^2
        return 0.5 * (
            Sy2 + Sz2 - np.sqrt((Sy2 - Sz2)**2 + 4*cov**2)
        )

    best_popt = None
    best_cost = np.inf
    bounds = (0, 5)
    guesses = np.linspace(max(0.1, bounds[0]), bounds[1], 20)
    
    for g in guesses:
        try:
            popt, _ = curve_fit(model, times, y_data, p0=[g], bounds=bounds)
            cost = np.sum((model(times, popt[0]) - y_data)**2)
            if cost < best_cost:
                best_cost = cost
                best_popt = popt
        except:
            pass
            
    if best_popt is not None:
        return best_popt[0]
    return np.nan

def fit_squeezed(times, y_data, N_val):

    def model(t, J):
        Sy2 = (N_val*(N_val+1))/8 - (N_val*(N_val-1))/8 * np.cos((2*J**2)*t)**(N_val-2)
        Sz2 = N_val / 4
        cov = (N_val*(N_val-1))/4 * np.sin((J**2)*t) * np.cos((J**2)*t)**(N_val-3)

        val_exact = 0.5 * (Sy2 + Sz2 - np.sqrt((Sy2 - Sz2)**2 + 4*(cov)**2))
        return val_exact

    best_popt = None
    best_cost = np.inf
    bounds = (0, 5)
    guesses = np.linspace(max(0.1, bounds[0]), bounds[1], 20)
    
    for g in guesses:
        try:
            popt, _ = curve_fit(model, times, y_data, p0=[g], bounds=bounds)
            cost = np.sum((model(times, popt[0]) - y_data)**2)
            if cost < best_cost:
                best_cost = cost
                best_popt = popt
        except:
            pass
            
    if best_popt is not None:
        return best_popt[0]
    return np.nan



def run_comparison():
    print(f"Running comparison for Fixed Shots={SHOTS_MAX}, True J={J_TRUE}")

    
    # T_total sweep configuration
    T_TOTAL_SWEEP = np.array([10 , 15, 20, 25, 30])
    N_val = 10  # Fixed N for the sweep

    avg_err_naive = []
    avg_err_squeezed = []
    avg_err_squeezed_exact = []
    
    NUM_AVERAGES = 50
    
    T_ramsey = 2.0
    T_squeezed = 0.05 * T_ramsey

    debug_squeezing_data = {} 
    
    print(f"Starting T_total sweep... (N={N_val}, T_ramsey={T_ramsey}, T_squeezed={T_squeezed})")

    for T_total in T_TOTAL_SWEEP:
        shots_ramsey = max(1, int(T_total / T_ramsey)) * N_val
        shots_squeezed = max(1, int(T_total / T_squeezed))
        print(f" Simulating T_total={T_total}... (shots_ramsey={shots_ramsey}, shots_squeezed={shots_squeezed})")
        
        # Checking logic: `p0 = np.cos(J * t / 2.0)**2`. Yes independent.
        T_MAX = T_ramsey 
        times_naive = np.linspace(0, T_MAX, STEPS)
        # opt_time = 2 / (np.sqrt(2 * (N_val - 2)) * J_TRUE**2)
        times_squeezed = np.linspace(0, T_squeezed, STEPS)
        # times_squeezed = np.array([0.06]) #debug
        probs_naive = get_naive_probs(N_val, times_naive, J_TRUE) 
        
        # Squeezed: Depends heavily on N
        probs_squeezed, m_vals, numerical, numerical_y2, numerical_cov = get_squeezed_probs(N_val, times_squeezed, J_TRUE) 
        temp_err_naive = []
        temp_err_squeezed = []
        temp_err_squeezed_exact = []

        is_last_Total = (T_total == T_TOTAL_SWEEP[-1])

        for run_idx in range(NUM_AVERAGES):
            is_debug_run = is_last_Total and (run_idx == 0)

            # --- Method 1: Naive ---
            mx_means = []
            for p0 in probs_naive:
                n_total_bits = shots_ramsey//STEPS
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
                samples_m = np.random.choice(m_vals, size=shots_squeezed, p=probs_m / np.sum(probs_m)) #after debug devide by STEPS
                mean_m  = np.mean(samples_m)
                mean_m2 = np.mean(samples_m**2)
                var_m   = mean_m2 - mean_m**2
                iy2_means.append(var_m)

                if is_debug_run:
                    t = times_squeezed[idx]
                    N_debug = N_val 
                    # exact squeezed variance
                    Sy2 = (N_debug*(N_debug+1))/8 - (N_debug*(N_debug-1))/8 * np.cos((2*J_TRUE**2)*t)**(N_debug-2)
                    Sz2 = N_debug / 4
                    cov = (N_debug*(N_debug-1))/4 * np.sin((J_TRUE**2)*t) * np.cos((J_TRUE**2)*t)**(N_debug-2)

                    val_exact = 0.5 * (Sy2 + Sz2 - np.sqrt((Sy2 - Sz2)**2 + 4*(cov)**2))
                    val_exact = Sy2 #debug measure only sy2

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


            j_squeezed = fit_squeezed(times_squeezed, iy2_means, N_val)

            temp_err_squeezed.append(abs(j_squeezed - J_TRUE))

            j_squeezed_ex = fit_squeezed_exact(times_squeezed, iy2_means, N_val)
            temp_err_squeezed_exact.append(abs(j_squeezed_ex - J_TRUE))
        
        # Average errors
        avg_err_naive.append(np.mean(temp_err_naive))
        avg_err_squeezed.append(np.mean(temp_err_squeezed))
        avg_err_squeezed_exact.append(np.mean(temp_err_squeezed_exact))
        
        print(f"T_total={T_total}: Naive={avg_err_naive[-1]:.4f}, SqApprox={avg_err_squeezed[-1]:.4f}, SqExact={avg_err_squeezed_exact[-1]:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(T_TOTAL_SWEEP, avg_err_naive, 'o-', label=f'Naive (N={N_val})')
    # plt.plot(T_TOTAL_SWEEP, avg_err_squeezed, 's-', label=f'Squeezed Approx (N={N_val})')
    plt.plot(T_TOTAL_SWEEP, avg_err_squeezed_exact, '^-', label=f'Squeezed Exact (N={N_val})')
    
    # Scaling lines
    t_arr = np.array(T_TOTAL_SWEEP)
    plt.plot(t_arr, 1 /( (t_arr//T_ramsey) * N_val)**(1/2), '--', color='gray', label=r'$1/\sqrt{T_{total}}$ (SQL)')
    plt.plot(t_arr, 1 /((np.sqrt((t_arr/T_squeezed))) * N_val**(2/3)), ':', color='black', label='Heisenberg like')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Total Time Budget (T_total)')
    plt.ylabel('Estimation Error |J_est - J_true|')
    plt.title(f'J Estimation Error vs T_total (N={N_val})')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig('error_comparison_T_sweep.png')
    print("Saved 'error_comparison_T_sweep.png'")
    
    # --- Plotting Debug ---
    if 'times' in debug_squeezing_data:
        plt.figure(figsize=(10, 6))
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_approx'], label='Analytical Approx', color='black', linestyle='--')
        plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_exact'], label='Analytical Exact', color='green', linestyle='-')
        plt.plot(debug_squeezing_data['times'], debug_squeezing_data['sampled'], 'o', label=f'Sampled (T={T_TOTAL_SWEEP[-1]})', alpha=0.3)
        plt.plot(debug_squeezing_data['times'], debug_squeezing_data['numerical'], 'x', label=f'Numerical (T={T_TOTAL_SWEEP[-1]})', alpha=0.3)
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['numerical_y2'], 's', label=f'Numerical Y^2 (T={T_TOTAL_SWEEP[-1]})', alpha=0.3)
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_y2'], label=f'Analytical Y^2 (T={T_TOTAL_SWEEP[-1]})', alpha=0.3)

        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['numerical_cov'], 's', label=f'Numerical Cov (T={T_TOTAL_SWEEP[-1]})', alpha=0.3)
        # plt.plot(debug_squeezing_data['times'], debug_squeezing_data['analytical_cov'], label=f'Analytical Cov (T={T_TOTAL_SWEEP[-1]})', alpha=0.3)
        
        plt.xlabel('Time')
        plt.ylabel('<V-^2>')
        plt.title(f'Debug: Squeezed Component (T={T_TOTAL_SWEEP[-1]})')
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