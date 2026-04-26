import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SimulationConfig:
    N: int = 4                   # number of bath qubits
    gamma_bath: float = 10.0     # independent dephasing rate on each bath qubit
    J_nominal: float = 1.0
    dJ: float = 1e-4

    t_min: float = 0.01
    t_max: float = 5.0
    n_steps: int = 120

    omega_min: float = 0.0
    omega_max: float = 10.0
    n_omegas: int = 50

    qfi_tol: float = 1e-12
    qcrb_eps: float = 1e-15

    output_figure: str = "full_hilbert_bath_only_qfi.png"


# ============================================================
# Basic operator helpers
# ============================================================

def local_operator(op: qt.Qobj, target: int, total_sites: int) -> qt.Qobj:
    """
    Place a single-qubit operator 'op' on site 'target'
    in a tensor-product Hilbert space of 'total_sites' qubits.

    Convention:
      target = 0       -> central qubit
      target = 1..N    -> bath qubits
    """
    ops = [qt.qeye(2) for _ in range(total_sites)]
    ops[target] = op
    return qt.tensor(ops)


def build_full_hamiltonian(N: int, Omega: float, J: float) -> qt.Qobj:
    """
    Full-space Hamiltonian:
        H = Omega * sigma_x^(c) + J * sigma_z^(c) * sum_i sigma_z^(i)
    """
    total_sites = N + 1
    sx = qt.sigmax()
    sz = qt.sigmaz()

    sx_c = local_operator(sx, target=0, total_sites=total_sites)
    sz_c = local_operator(sz, target=0, total_sites=total_sites)

    H = Omega * sx_c
    for i in range(N):
        sz_i = local_operator(sz, target=i + 1, total_sites=total_sites)
        H += J * sz_c * sz_i

    return H


def build_initial_state(N: int) -> qt.Qobj:
    """
    Initial product state:
      central qubit in |+>
      all bath qubits in |+>
    """
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    return qt.tensor([plus for _ in range(N + 1)])


def build_bath_dephasing_c_ops(N: int, gamma_bath: float):
    """
    Independent dephasing on each bath qubit:
      c_i = sqrt(gamma_bath) * sigma_z^(i)
    """
    total_sites = N + 1
    sz = qt.sigmaz()

    c_ops = []
    for i in range(N):
        z_i = local_operator(sz, target=i + 1, total_sites=total_sites)
        c_ops.append(np.sqrt(gamma_bath) * z_i)

    return c_ops


# ============================================================
# Time evolution
# ============================================================

def evolve_and_get_bath_rhos(
    Omega: float,
    J: float,
    tlist: np.ndarray,
    N: int,
    gamma_bath: float,
):
    """
    Evolve the full state and return reduced bath density matrices
    at all times.
    """
    H = build_full_hamiltonian(N=N, Omega=Omega, J=J)
    psi0 = build_initial_state(N=N)
    c_ops = build_bath_dephasing_c_ops(N=N, gamma_bath=gamma_bath)

    result = qt.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[])

    bath_rhos = []
    for state in result.states:
        rho_full = state * state.dag() if state.isket else state
        rho_bath = rho_full.ptrace(list(range(1, N + 1)))  # trace out central qubit
        bath_rhos.append(rho_bath.full())

    return bath_rhos


# ============================================================
# QFI
# ============================================================

def qfi_from_rho_and_drho(rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12) -> float:
    """
    Mixed-state Quantum Fisher Information:
        F_Q = 2 sum_{m,n} |<m|drho|n>|^2 / (lambda_m + lambda_n)
    over terms with lambda_m + lambda_n > tol.
    """
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)

    evals, evecs = np.linalg.eigh(rho)
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
    Compute bath-only QFI(t) from finite differences in J.
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

def plot_results(
    tlist: np.ndarray,
    omega_list: np.ndarray,
    qcrb_matrix: np.ndarray,
    min_qcrb_per_omega: np.ndarray,
    optimal_times: np.ndarray,
    output_figure: str,
):
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
    axes[0].set_title("Bath-only QCRB (full Hilbert space)")
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
        color="red",
        linestyle="--",
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

    total_dim = 2 ** (cfg.N + 1)
    print(f"Total Hilbert-space dimension = 2^(N+1) = {total_dim}")

    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    omega_list = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_omegas)

    qfi_matrix = np.zeros((len(omega_list), len(tlist)))

    print(f"Computing bath-only QFI for {len(omega_list)} Omega values...")

    for i, Omega in enumerate(omega_list):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Processed {i + 1}/{len(omega_list)} Omegas")

        bath_rhos_plus = evolve_and_get_bath_rhos(
            Omega=Omega,
            J=cfg.J_nominal + cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            gamma_bath=cfg.gamma_bath,
        )

        bath_rhos_minus = evolve_and_get_bath_rhos(
            Omega=Omega,
            J=cfg.J_nominal - cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            gamma_bath=cfg.gamma_bath,
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
    print(f"Optimal Omega = {optimal_omega:.6f}")
    print(f"Minimum bath-only QCRB = {min_qcrb_per_omega[optimal_idx]:.6e}")
    print(f"Optimal time at best Omega = {optimal_times[optimal_idx]:.6f}")

    plot_results(
        tlist=tlist,
        omega_list=omega_list,
        qcrb_matrix=qcrb_matrix,
        min_qcrb_per_omega=min_qcrb_per_omega,
        optimal_times=optimal_times,
        output_figure=cfg.output_figure,
    )


if __name__ == "__main__":
    main()