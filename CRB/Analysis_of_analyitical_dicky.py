import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SimulationConfig:
    N: int = 15
    gamma: float = 0.1
    beta: float = 1.0
    J_nominal: float = 1.0
    dJ: float = 1e-3

    t_min: float = 0.01
    t_max: float = 100.0
    n_steps: int = 400

    omega_min: float = 0.0
    omega_max: float = 50.0
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

def qfi_from_rho_and_drho(rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12) -> float:
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

    for m in range(dim):
        vm = evecs[:, m]
        for n in range(dim):
            denom = evals[m] + evals[n]
            if denom > tol:
                vn = evecs[:, n]
                elem = np.vdot(vm, drho @ vn)
                qfi += 2.0 * (np.abs(elem) ** 2) / denom

    return float(np.real(qfi))


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

    for k in range(n_times):
        rho_plus = bath_rhos_plus[k]
        rho_minus = bath_rhos_minus[k]

        rho = 0.5 * (rho_plus + rho_minus)
        drho = (rho_plus - rho_minus) / (2.0 * dJ)

        qfi_t[k] = qfi_from_rho_and_drho(rho, drho, tol=tol)

    return qfi_t


# ============================================================
# Plotting
# ============================================================

def plot_qfi_results(
    tlist: np.ndarray,
    omega_list: np.ndarray,
    qfi_matrix: np.ndarray,
    qcrb_matrix: np.ndarray,
    min_qcrb_per_omega: np.ndarray,
    optimal_times: np.ndarray,
    output_figure: str,
):
    """
    Plot:
      1) bath-only QCRB heatmap  (with t*(Omega) overlay)
      2) minimum bath-only QCRB vs Omega
      3) optimal time t*(Omega) vs Omega
    """
    optimal_idx = np.argmin(min_qcrb_per_omega)
    optimal_omega = omega_list[optimal_idx]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    vmax_qcrb = np.percentile(min_qcrb_per_omega, 95) * 5.0

    im = axes[0].pcolormesh(
        tlist,
        omega_list,
        qcrb_matrix,
        shading="auto",
        cmap="viridis",
        vmax=vmax_qcrb,
    )
    # Overlay the optimal time trajectory on the heatmap
    axes[0].plot(
        optimal_times,
        omega_list,
        color="red",
        linewidth=1.5,
        linestyle="--",
        label=r"$t^*(\Omega)$",
    )
    axes[0].set_xlabel("Time (t)")
    axes[0].set_ylabel(r"Transverse Field ($\Omega$)")
    axes[0].set_title("Bath-only Quantum Cramér-Rao Bound")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0])

    axes[1].plot(
        omega_list,
        min_qcrb_per_omega,
        marker="o",
        linestyle="-",
        label="Min bath-only QCRB",
    )
    axes[1].axvline(
        optimal_omega,
        linestyle="--",
        color="red",
        label=fr"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[1].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[1].set_ylabel(r"Min over $t$ of QCRB")
    axes[1].set_title("Minimum bath-only QCRB vs Omega")
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

    # --- Third panel: optimal time t*(Omega) ---
    axes[2].plot(
        omega_list,
        optimal_times,
        marker="s",
        linestyle="-",
        color="darkorange",
        label=r"$t^*(\Omega)$",
    )
    axes[2].axvline(
        optimal_omega,
        color="red",
        linestyle="--",
        label=fr"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[2].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[2].set_ylabel(r"Optimal time $t^*$")
    axes[2].set_title(r"Optimal measurement time $t^*$ vs $\Omega$")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

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

        qfi_t = compute_bath_qfi_trajectory(
            bath_rhos_plus=bath_rhos_plus,
            bath_rhos_minus=bath_rhos_minus,
            dJ=cfg.dJ,
            tol=cfg.qfi_tol,
        )

        qfi_matrix[i, :] = qfi_t

    qcrb_matrix = 1.0 / (qfi_matrix + cfg.qcrb_eps)
    min_qcrb_per_omega = np.min(qcrb_matrix, axis=1)

    # Optimal time t*(Omega): the time at which the QCRB is minimised for each Omega
    optimal_time_idx = np.argmin(qcrb_matrix, axis=1)
    optimal_times = tlist[optimal_time_idx]

    optimal_idx = np.argmin(min_qcrb_per_omega)
    optimal_omega = omega_list[optimal_idx]

    print("\nDone.")
    print(f"Optimal Omega (bath-only QFI criterion): {optimal_omega:.6f}")
    print(f"Minimum bath-only QCRB: {min_qcrb_per_omega[optimal_idx]:.6e}")
    print(f"Optimal time at best Omega: {optimal_times[optimal_idx]:.6f}")

    plot_qfi_results(
        tlist=tlist,
        omega_list=omega_list,
        qfi_matrix=qfi_matrix,
        qcrb_matrix=qcrb_matrix,
        min_qcrb_per_omega=min_qcrb_per_omega,
        optimal_times=optimal_times,
        output_figure=cfg.output_figure,
    )


if __name__ == "__main__":
    main()