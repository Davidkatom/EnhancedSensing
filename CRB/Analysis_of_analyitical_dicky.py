import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
import qutip as qt
from dataclasses import dataclass
# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SimulationConfig:
    N: int = 6
    gamma: float = 0.01
    beta: float = 1.0
    
    J_nominal: float = 1
    dJ: float = 1e-3

    t_min: float = 0.01
    t_max: float = 100.0
    n_steps: int = 500

    omega_min: float = 0.0
    omega_max: float = 30.0
    n_omegas: int = 60

    num_model: int = 1  # 1 = exact H1, 2 = effective H2
    qfi_tol: float = 1e-12
    qcrb_eps: float = 1e-15

    output_figure: str = "bath_only_qfi_analysis.png"


# ============================================================
# Operator / state construction
# ============================================================

def build_spin_operators(N: int):
    """
    Build all operators needed for the full central-spin + bath model.
    """
    S_spin = N / 2.0
    dim_bath = int(2 * S_spin + 1)

    Jz = qt.jmat(S_spin, "z") * 2.0
    I_bath = qt.qeye(dim_bath)

    sx = qt.sigmax()
    sz = qt.sigmaz()
    si = qt.qeye(2)

    sx_s = qt.tensor(sx, I_bath)
    sz_s = qt.tensor(sz, I_bath)
    Sz_op = qt.tensor(si, Jz)

    return {
        "S_spin": S_spin,
        "dim_bath": dim_bath,
        "Jz": Jz,
        "I_bath": I_bath,
        "sx": sx,
        "sz": sz,
        "si": si,
        "sx_s": sx_s,
        "sz_s": sz_s,
        "Sz_op": Sz_op,
    }


def build_initial_state(S_spin: float) -> qt.Qobj:
    """
    Initial state:
      central spin in |+>
      bath in spin-coherent state along +x
    """
    plus_state_central = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    plus_state_bath = qt.spin_coherent(S_spin, np.pi / 2, 0)
    return qt.tensor(plus_state_central, plus_state_bath)


def build_hamiltonian(Omega: float, J: float, N: int, num_model: int) -> qt.Qobj:
    """
    Build the numerical Hamiltonian.

    num_model = 1:
        H1 = Omega * sigma_x + J * sigma_z * S_z

    num_model = 2:
        H2 = Omega * sigma_x + (J^2 / (2 Omega)) * sigma_x * S_z^2
    """
    ops = build_spin_operators(N)
    sx_s = ops["sx_s"]
    sz_s = ops["sz_s"]
    Sz_op = ops["Sz_op"]

    if num_model == 1:
        return Omega * sx_s + J * sz_s * Sz_op

    if num_model == 2:
        if abs(Omega) < 1e-15:
            return Omega * sx_s
        return Omega * sx_s + (J**2 / (2.0 * Omega)) * sx_s * (Sz_op**2)

    raise ValueError(f"Unsupported num_model={num_model}. Use 1 or 2.")


# ============================================================
# Time evolution and reduced bath states
# ============================================================

def get_bath_density_matrices(
    Omega: float,
    J: float,
    tlist: np.ndarray,
    N: int = 10,
    gamma: float = 1.0,
    beta: float = 1.0,
    num_model: int = 1,
):
    """
    Evolve the full state numerically and return the reduced bath
    density matrices rho_B(t) = Tr_central[rho_full(t)].
    """
    ops = build_spin_operators(N)

    H = build_hamiltonian(Omega, J, N, num_model)
    psi0 = build_initial_state(ops["S_spin"])

    collapse_ops = [np.sqrt(beta) * ops["sx_s"], np.sqrt(gamma) * ops["sz_s"]]

    result = qt.mesolve(H, psi0, tlist, c_ops=collapse_ops, e_ops=[])

    bath_rhos = []
    for state in result.states:
        rho_full = state * state.dag() if state.isket else state
        rho_bath = rho_full.ptrace(1)
        bath_rhos.append(rho_bath.full())

    return bath_rhos


# ============================================================
# QFI
# ============================================================

def qfi_from_rho_and_drho(rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12) -> tuple[float, np.ndarray]:
    r"""
    Compute mixed-state QFI from rho and d rho / dJ using

        F_Q = 2 \sum_{m,n : \lambda_m + \lambda_n > 0}
              | <m| drho |n> |^2 / (\lambda_m + \lambda_n)

    where rho = sum_n lambda_n |n><n|.
    """
    # Hermitize numerically
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)

    evals, evecs = np.linalg.eigh(rho)

    # Clean tiny negative eigenvalues from numerical noise
    evals = np.real(evals)
    evals[np.abs(evals) < tol] = 0.0

    qfi = 0.0
    dim = len(evals)
    L = np.zeros_like(rho, dtype=np.complex128)

    for m in range(dim):
        vm = evecs[:, m]
        for n in range(dim):
            denom = evals[m] + evals[n]
            if denom > tol:
                vn = evecs[:, n]
                elem = np.vdot(vm, drho @ vn)
                qfi += 2.0 * (np.abs(elem) ** 2) / denom
                L += (2.0 * elem / denom) * np.outer(vm, vn.conj())

    return float(np.real(qfi)), L


def get_measurement_projectors(O: np.ndarray, tol: float = 1e-10) -> list[np.ndarray]:
    """
    Get the projectors onto the eigenspaces of an observable O.
    """
    evals, evecs = np.linalg.eigh(O)
    unique_evals = []
    projectors = []
    for i, val in enumerate(evals):
        found = False
        for j, uval in enumerate(unique_evals):
            if abs(val - uval) < tol:
                projectors[j] += np.outer(evecs[:, i], evecs[:, i].conj())
                found = True
                break
        if not found:
            unique_evals.append(val)
            projectors.append(np.outer(evecs[:, i], evecs[:, i].conj()))
    return projectors


def compute_cfi_from_projectors(rho: np.ndarray, drho: np.ndarray, projectors: list[np.ndarray]) -> float:
    """
    Compute the Classical Fisher Information of a measurement defined by its projectors.
    """
    cfi = 0.0
    for P in projectors:
        p = np.real(np.trace(P @ rho))
        dp = np.real(np.trace(P @ drho))
        if p > 1e-15:
            cfi += (dp ** 2) / p
    return cfi


def build_general_observable(
    params: np.ndarray, Jx: np.ndarray, Jy: np.ndarray, Jz: np.ndarray
) -> np.ndarray:
    r"""
    Build the full general quadratic observable in all three spin directions:
        O = a_x * J_x + a_y * J_y + a_z * J_z
              + b_xx * J_x^2 + b_yy * J_y^2 + b_zz * J_z^2
              + b_xy * {J_x, J_y} + b_xz * {J_x, J_z} + b_yz * {J_y, J_z}

    params = [a_x, a_y, a_z, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz, c_xz2, c_yz2, c_z3, c_xyz]
    """
    a_x, a_y, a_z, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz, c_xz2, c_yz2, c_z3, c_xyz = params

    Jz2 = Jz @ Jz
    Jyz = Jy @ Jz + Jz @ Jy

    return (
        a_x * Jx + a_y * Jy + a_z * Jz
        + b_xx * (Jx @ Jx)
        + b_yy * (Jy @ Jy)
        + b_zz * Jz2
        + b_xy * (Jx @ Jy + Jy @ Jx)
        + b_xz * (Jx @ Jz + Jz @ Jx)
        + b_yz * Jyz
        + c_xz2 * (Jx @ Jz2 + Jz2 @ Jx)
        + c_yz2 * (Jy @ Jz2 + Jz2 @ Jy)
        + c_z3 * (Jz2 @ Jz)
        + c_xyz * (Jx @ Jyz + Jyz @ Jx)
    )


def cfi_of_general_observable(
    params: np.ndarray,
    rho: np.ndarray,
    drho: np.ndarray,
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
) -> float:
    """
    Return NEGATIVE CFI (for minimization) of the general observable at given rho/drho.
    The observable is normalized to unit Frobenius norm to make the scan scale-invariant.
    """
    O = build_general_observable(params, Jx, Jy, Jz)
    norm = np.linalg.norm(O, ord="fro")
    if norm < 1e-15:
        return 0.0
    O = O / norm
    projectors = get_measurement_projectors(O)
    return -compute_cfi_from_projectors(rho, drho, projectors)


def fisher_metric_project_sld(rho, drho, basis_ops, reg=1e-10):
    """
    Project the SLD onto span{basis_ops} using the Fisher/SLD metric.

    Finds O = sum_i alpha_i O_i such that
        sum_j G_ij alpha_j = b_i

    where
        G_ij = 1/2 Tr[rho(O_i O_j + O_j O_i)]
        b_i  = Tr[O_i drho]
    """
    K = len(basis_ops)

    G = np.zeros((K, K), dtype=np.complex128)
    b = np.zeros(K, dtype=np.complex128)

    for i, Oi in enumerate(basis_ops):
        Oi = 0.5 * (Oi + Oi.conj().T)

        b[i] = np.trace(Oi @ drho)

        for j, Oj in enumerate(basis_ops):
            Oj = 0.5 * (Oj + Oj.conj().T)
            G[i, j] = 0.5 * np.trace(rho @ (Oi @ Oj + Oj @ Oi))

    # numerical cleanup
    G = 0.5 * (G + G.conj().T)
    b = np.real_if_close(b)

    # regularized solve, because G can be ill-conditioned
    alpha = np.linalg.solve(G + reg * np.eye(K), b)

    O_proj = np.zeros_like(rho, dtype=np.complex128)
    for ai, Oi in zip(alpha, basis_ops):
        O_proj += ai * Oi

    O_proj = 0.5 * (O_proj + O_proj.conj().T)

    return O_proj, alpha


def find_optimal_observable_params(
    rho: np.ndarray,
    drho: np.ndarray,
    Jx: np.ndarray,
    Jy: np.ndarray,
    Jz: np.ndarray,
    param_bounds: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Use differential evolution to scan over all 13 parameters
    [a_x, a_y, a_z, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz, c_xz2, c_yz2, c_z3, c_xyz]
    and find the observable that maximises the Classical Fisher Information
    at the given (rho, drho).

    Returns:
        best_params  - the 13 optimal coefficients
        best_cfi     - the achieved CFI value
    """
    bounds = [(-param_bounds, param_bounds)] * 13

    result = differential_evolution(
        cfi_of_general_observable,
        bounds,
        args=(rho, drho, Jx, Jy, Jz),
        strategy="best1bin",
        maxiter=500,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.9,
        seed=seed,
        polish=True,
        workers=1,
    )
    best_params = result.x
    best_cfi = -result.fun
    return best_params, best_cfi


def compute_bath_qfi_trajectory(
    bath_rhos_plus,
    bath_rhos_minus,
    dJ: float,
    tol: float = 1e-12,
):
    """
    Given bath density matrices at J+dJ and J-dJ, compute the bath-only
    QFI as a function of time.
    """
    n_times = len(bath_rhos_plus)
    qfi_t = np.zeros(n_times)
    L_t = []
    rho_t = []
    drho_t = []

    for k in range(n_times):
        rho_plus = bath_rhos_plus[k]
        rho_minus = bath_rhos_minus[k]

        rho = 0.5 * (rho_plus + rho_minus)
        drho = (rho_plus - rho_minus) / (2.0 * dJ)

        qfi, L = qfi_from_rho_and_drho(rho, drho, tol=tol)
        qfi_t[k] = qfi
        L_t.append(L)
        rho_t.append(rho)
        drho_t.append(drho)

    return qfi_t, L_t, rho_t, drho_t


# ============================================================
# Plotting
# ============================================================

def plot_qfi_results(
    tlist: np.ndarray,
    omega_list: np.ndarray,
    min_qcrb_per_omega: np.ndarray,
    optimal_times: np.ndarray,
    opt_quadrature_angles: np.ndarray,
    opt_cfi_results: dict,
    opt_general_cfi: np.ndarray,
    opt_sld_cfi: np.ndarray,
    opt_sld_proj_cfi: np.ndarray,
    output_figure: str,
):
    """
    Plot:
      1) minimum bath-only QCRB vs Omega
      2) optimal time t*(Omega) vs Omega
      3) optimal measurement quadrature vs Omega
      4) CFI of given measurements + optimal general observable vs Omega
    """
    optimal_idx = np.argmin(min_qcrb_per_omega)
    optimal_omega = omega_list[optimal_idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # --- First panel: min QCRB ---
    axes[0].plot(
        omega_list,
        min_qcrb_per_omega,
        marker="o",
        linestyle="-",
        label="Min bath-only QCRB",
    )
    axes[0].axvline(
        optimal_omega,
        linestyle="--",
        color="red",
        label=fr"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[0].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[0].set_ylabel(r"Min over $t$ of QCRB")
    axes[0].set_title("Minimum bath-only QCRB vs Omega")
    # axes[0].set_yscale("log")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()

    # --- Second panel: optimal time t*(Omega) ---
    axes[1].plot(
        omega_list,
        optimal_times,
        marker="s",
        linestyle="",
        color="darkorange",
        label=r"Data $t^*(\Omega)$",
    )
    
    # Fit for T(Omega) = b + c * Omega
    # Using curve_fit to find parameters b and c
    def fit_func(omega, b, c):
        return b + c * omega

    try:
        # Initial guess: b is roughly the intercept, c is small
        p0_guess = [optimal_times[0], 0.0]
        popt, pcov = curve_fit(fit_func, omega_list, optimal_times, p0=p0_guess, maxfev=10000)
        b_fit, c_fit = popt
        
        omega_dense = np.linspace(min(omega_list), max(omega_list), 200)
        axes[1].plot(
            omega_dense, 
            fit_func(omega_dense, b_fit, c_fit), 
            linestyle="-", 
            color="black",
            label=fr"Fit: $T=b+c\Omega$" + "\n" + f"b={b_fit:.3f}, c={c_fit:.3e}"
        )
    except Exception as e:
        print(f"Curve fitting failed: {e}")

    axes[1].axvline(
        optimal_omega,
        color="red",
        linestyle="--",
        label=fr"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[1].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[1].set_ylabel(r"Optimal time $t^*$")
    axes[1].set_title(r"Optimal measurement time $t^*$ vs $\Omega$")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    # --- Third panel: optimal quadrature ---
    axes[2].plot(
        omega_list,
        opt_quadrature_angles,
        marker="^",
        linestyle="-",
        color="green",
        label="Optimal Quadrature",
    )
    axes[2].axvline(
        optimal_omega,
        color="red",
        linestyle="--",
        label=fr"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[2].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[2].set_ylabel(r"Angle (radians)")
    axes[2].set_title(r"Optimal Quadrature Angle vs $\Omega$")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    # --- Fourth panel: CFI of measurements ---
    for name, cfi_vals in opt_cfi_results.items():
        axes[3].plot(
            omega_list,
            cfi_vals,
            marker="o",
            linestyle="-",
            markersize=4,
            label=f"{name}",
        )

    # Optimal general observable CFI
    if opt_general_cfi is not None:
        axes[3].plot(
            omega_list,
            opt_general_cfi,
            marker="D",
            linestyle="-",
            linewidth=2.5,
            color="purple",
            markersize=5,
            label="Opt. General Observable",
        )

    # SLD Basis CFI
    if opt_sld_cfi is not None:
        axes[3].plot(
            omega_list,
            opt_sld_cfi,
            marker="*",
            linestyle="-",
            linewidth=2.5,
            color="cyan",
            markersize=6,
            label="Opt. SLD Basis",
        )

    # SLD Projected CFI
    if opt_sld_proj_cfi is not None:
        axes[3].plot(
            omega_list,
            opt_sld_proj_cfi,
            marker="X",
            linestyle=":",
            linewidth=2.0,
            color="magenta",
            markersize=6,
            label="Opt. SLD Projected",
        )

    # Also plot the QFI at the optimal time for comparison
    qfi_opt = 1.0 / min_qcrb_per_omega
    axes[3].plot(
        omega_list,
        qfi_opt,
        marker="s",
        linestyle="--",
        color="black",
        linewidth=2,
        label="QFI (Optimal)",
    )

    axes[3].axvline(
        optimal_omega,
        color="red",
        linestyle="--",
        label=fr"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[3].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[3].set_ylabel(r"Fisher Information")
    axes[3].set_title(r"CFI of Measurements at Optimal Time vs $\Omega$")
    axes[3].grid(True, linestyle=":")
    axes[3].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_figure, bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    cfg = SimulationConfig()

    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    omega_list = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_omegas)

    qfi_matrix = np.zeros((len(omega_list), len(tlist)))
    L_all = []
    rho_all_list = []   # rho_all_list[i][k]  = rho   at omega i, time k
    drho_all_list = []  # drho_all_list[i][k] = drho  at omega i, time k

    # --- Define measurements for CFI ---
    # Make it easy to add or change observables here.
    ops = build_spin_operators(cfg.N)
    Jx = qt.jmat(ops["S_spin"], "x").full() * 2.0
    Jy = qt.jmat(ops["S_spin"], "y").full() * 2.0
    Jz = qt.jmat(ops["S_spin"], "z").full() * 2.0

    measurements = {
        "X": Jx,
        "Y": Jy,
        "Z": Jz,
        "X^2": Jx @ Jx,
        "Y^2": Jy @ Jy,
        "Z^2": Jz @ Jz,
    }
    
    measurement_projectors = {
        name: get_measurement_projectors(O) for name, O in measurements.items()
    }
    
    cfi_matrices = {name: np.zeros((len(omega_list), len(tlist))) for name in measurements}
    # ------------------------------------

    print(f"Computing bath-only QFI for {len(omega_list)} values of Omega...")

    for i, Omega in enumerate(omega_list):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Processed {i + 1}/{len(omega_list)} Omega values")

        bath_rhos_plus = get_bath_density_matrices(
            Omega=Omega,
            J=cfg.J_nominal + cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            gamma=cfg.gamma,
            beta=cfg.beta,
            num_model=cfg.num_model,
        )

        bath_rhos_minus = get_bath_density_matrices(
            Omega=Omega,
            J=cfg.J_nominal - cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            gamma=cfg.gamma,
            beta=cfg.beta,
            num_model=cfg.num_model,
        )

        qfi_t, L_t, rho_t, drho_t = compute_bath_qfi_trajectory(
            bath_rhos_plus=bath_rhos_plus,
            bath_rhos_minus=bath_rhos_minus,
            dJ=cfg.dJ,
            tol=cfg.qfi_tol,
        )

        qfi_matrix[i, :] = qfi_t
        L_all.append(L_t)
        rho_all_list.append(rho_t)
        drho_all_list.append(drho_t)
        
        for k in range(len(tlist)):
            for name, projs in measurement_projectors.items():
                cfi_matrices[name][i, k] = compute_cfi_from_projectors(rho_t[k], drho_t[k], projs)

    qcrb_matrix = 1.0 / (qfi_matrix + cfg.qcrb_eps)
    min_qcrb_per_omega = np.min(qcrb_matrix, axis=1)

    # Optimal time t*(Omega): the time at which the QCRB is minimised for each Omega
    optimal_time_idx = np.argmin(qcrb_matrix, axis=1)
    optimal_times = tlist[optimal_time_idx]

    optimal_idx = np.argmin(min_qcrb_per_omega)
    optimal_omega = omega_list[optimal_idx]

    # Calculate optimal YZ plane squeezing quadrature angle for each Omega at optimal time
    # (Jy, Jz already computed above as numpy matrices)
    Jyz = Jy @ Jz + Jz @ Jy
    Jy2_minus_Jz2 = Jy @ Jy - Jz @ Jz
    
    opt_quadrature_angles = np.zeros(len(omega_list))
    for i in range(len(omega_list)):
        t_idx = optimal_time_idx[i]
        L_opt = L_all[i][t_idx]
        cyz = np.real(np.trace(L_opt @ Jyz))
        cy2z2 = np.real(np.trace(L_opt @ Jy2_minus_Jz2))
        opt_quadrature_angles[i] = 0.5 * np.arctan2(cyz, cy2z2)

    # Extract CFI at each observable's own individually optimal time
    opt_cfi_results = {name: np.zeros(len(omega_list)) for name in measurements}
    for i in range(len(omega_list)):
        for name in measurements:
            # argmax over all times for this specific measurement
            best_t_idx = np.argmax(cfi_matrices[name][i, :])
            opt_cfi_results[name][i] = cfi_matrices[name][i, best_t_idx]

    print("\nDone.")
    print(f"Optimal Omega (bath-only QFI criterion): {optimal_omega:.6f}")
    print(f"Minimum bath-only QCRB: {min_qcrb_per_omega[optimal_idx]:.6e}")
    print(f"Optimal time at best Omega: {optimal_times[optimal_idx]:.6f}")

    # ----------------------------------------------------------------
    # Optimize the general observable at global (Omega*, t_quantum*),
    # then evaluate its CFI at each Omega using each observable's own
    # individually optimal time (max over all t).
    # ----------------------------------------------------------------
    rho_all = rho_all_list
    drho_all = drho_all_list

    print("\nOptimizing general observable parameters at optimal Omega...")
    rho_opt  = rho_all[optimal_idx][optimal_time_idx[optimal_idx]]
    drho_opt = drho_all[optimal_idx][optimal_time_idx[optimal_idx]]

    best_params, best_cfi_val = find_optimal_observable_params(
        rho=rho_opt, drho=drho_opt, Jx=Jx, Jy=Jy, Jz=Jz
    )
    a_x, a_y, a_z, b_xx, b_yy, b_zz, b_xy, b_xz, b_yz, c_xz2, c_yz2, c_z3, c_xyz = best_params
    print(f"  Optimal CFI (general observable): {best_cfi_val:.6e}")
    print(f"  QFI at that point              : {1.0/min_qcrb_per_omega[optimal_idx]:.6e}")
    print(f"  Parameters:")
    print(f"    a_x={a_x:.4f}, a_y={a_y:.4f}, a_z={a_z:.4f}")
    print(f"    b_xx={b_xx:.4f}, b_yy={b_yy:.4f}, b_zz={b_zz:.4f}")
    print(f"    b_xy={b_xy:.4f}, b_xz={b_xz:.4f}, b_yz={b_yz:.4f}")
    print(f"    c_xz2={c_xz2:.4f}, c_yz2={c_yz2:.4f}, c_z3={c_z3:.4f}, c_xyz={c_xyz:.4f}")

    # Build the fixed optimal observable (normalized)
    O_best = build_general_observable(best_params, Jx, Jy, Jz)
    norm_best = np.linalg.norm(O_best, ord="fro")
    if norm_best > 1e-15:
        O_best = O_best / norm_best
    projectors_best = get_measurement_projectors(O_best)

    # Scan all Omegas: for each Omega take the best time for this observable
    print("Computing CFI of optimal general observable for all Omegas (individual optimal time)...")
    opt_general_cfi = np.zeros(len(omega_list))
    for i in range(len(omega_list)):
        best_cfi_i = 0.0
        for k in range(len(tlist)):
            cfi_k = compute_cfi_from_projectors(
                rho_all[i][k], drho_all[i][k], projectors_best
            )
            if cfi_k > best_cfi_i:
                best_cfi_i = cfi_k
        opt_general_cfi[i] = best_cfi_i

    print("Computing CFI of SLD basis for all Omegas (individual optimal time)...")
    L_opt_global = L_all[optimal_idx][optimal_time_idx[optimal_idx]]

    print("\nProjection of SLD onto raw operator basis (Fisher metric):")
    Jz2 = Jz @ Jz
    Jyz = Jy @ Jz + Jz @ Jy
    basis_operators = {
        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,
        "Jx2": Jx @ Jx,
        "Jy2": Jy @ Jy,
        "Jz2": Jz2,
        "Jxy": Jx @ Jy + Jy @ Jx,
        "Jxz": Jx @ Jz + Jz @ Jx,
        "Jyz": Jyz,
        "Jxz2": Jx @ Jz2 + Jz2 @ Jx,
        "Jyz2": Jy @ Jz2 + Jz2 @ Jy,
        "Jz3": Jz2 @ Jz,
        "Jxyz": Jx @ Jyz + Jyz @ Jx,
    }
    
    basis_names = list(basis_operators.keys())
    basis_ops = list(basis_operators.values())
    
    O_sld_proj, alpha = fisher_metric_project_sld(rho_opt, drho_opt, basis_ops)
    
    for name, coef in zip(basis_names, alpha):
        print(f"    {name}: {np.real(coef):.6f}")

    norm_sld_proj = np.linalg.norm(O_sld_proj, ord="fro")
    if norm_sld_proj > 1e-15:
        O_sld_proj = O_sld_proj / norm_sld_proj

    projectors_SLD = get_measurement_projectors(L_opt_global)
    projectors_sld_proj = get_measurement_projectors(O_sld_proj)
    
    opt_sld_cfi = np.zeros(len(omega_list))
    opt_sld_proj_cfi = np.zeros(len(omega_list))
    
    for i in range(len(omega_list)):
        best_cfi_i = 0.0
        best_proj_cfi_i = 0.0
        for k in range(len(tlist)):
            cfi_k = compute_cfi_from_projectors(
                rho_all[i][k], drho_all[i][k], projectors_SLD
            )
            if cfi_k > best_cfi_i:
                best_cfi_i = cfi_k
                
            proj_cfi_k = compute_cfi_from_projectors(
                rho_all[i][k], drho_all[i][k], projectors_sld_proj
            )
            if proj_cfi_k > best_proj_cfi_i:
                best_proj_cfi_i = proj_cfi_k
                
        opt_sld_cfi[i] = best_cfi_i
        opt_sld_proj_cfi[i] = best_proj_cfi_i

    plot_qfi_results(
        tlist=tlist,
        omega_list=omega_list,
        min_qcrb_per_omega=min_qcrb_per_omega,
        optimal_times=optimal_times,
        opt_quadrature_angles=opt_quadrature_angles,
        opt_cfi_results=opt_cfi_results,
        opt_general_cfi=opt_general_cfi,
        opt_sld_cfi=opt_sld_cfi,
        opt_sld_proj_cfi=opt_sld_proj_cfi,
        output_figure=cfg.output_figure,
    )


if __name__ == "__main__":
    main()