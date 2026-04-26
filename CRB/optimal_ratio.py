import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SimulationConfig:
    N: int = 10
    J_nominal: float = 1.0
    dJ: float = 1e-4

    t_min: float = 0.01
    t_max: float = 100.0
    n_steps: int = 120

    omega_min: float = 0.0
    omega_max: float = 50.0
    n_omegas: int = 50

    num_model: int = 1   # 1 = exact H1, 2 = effective H2
    qfi_tol: float = 1e-12
    qcrb_eps: float = 1e-15

    # Fixed beta; scan gamma directly
    beta: float = 1.0

    gamma_min: float = 0.05
    gamma_max: float = 0.25
    n_gammas: int = 12

    output_figure: str = "min_qcrb_vs_gamma.png"


# ============================================================
# Operator / state construction
# ============================================================

def build_spin_operators(N: int):
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
    plus_state_central = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    plus_state_bath = qt.spin_coherent(S_spin, np.pi / 2, 0)
    return qt.tensor(plus_state_central, plus_state_bath)


def build_hamiltonian(Omega: float, J: float, N: int, num_model: int) -> qt.Qobj:
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
    beta: float = 1.0,     # coefficient for s_x collapse
    gamma: float = 0.3,    # coefficient for s_z collapse
    num_model: int = 1,
):
    ops = build_spin_operators(N)

    H = build_hamiltonian(Omega, J, N, num_model)
    psi0 = build_initial_state(ops["S_spin"])

    collapse_ops = []
    if beta > 0:
        collapse_ops.append(np.sqrt(beta) * ops["sx_s"])
    if gamma > 0:
        collapse_ops.append(np.sqrt(gamma) * ops["sz_s"])

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
# Gamma scan (beta fixed at 1)
# ============================================================

def compute_min_qcrb_for_gamma(cfg: SimulationConfig, gamma: float):
    beta = cfg.beta

    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    omega_list = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_omegas)

    qfi_matrix = np.zeros((len(omega_list), len(tlist)))

    for i, Omega in enumerate(omega_list):
        bath_rhos_plus = get_bath_density_matrices(
            Omega=Omega,
            J=cfg.J_nominal + cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            beta=beta,
            gamma=gamma,
            num_model=cfg.num_model,
        )

        bath_rhos_minus = get_bath_density_matrices(
            Omega=Omega,
            J=cfg.J_nominal - cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            beta=beta,
            gamma=gamma,
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

    # Normalise by the minimum QCRB at Omega = 0.
    omega0_idx = int(np.argmin(np.abs(omega_list)))
    qcrb_at_omega0 = np.min(qcrb_matrix[omega0_idx, :])
    if qcrb_at_omega0 < cfg.qcrb_eps:
        qcrb_at_omega0 = cfg.qcrb_eps  # guard against zero
    qcrb_matrix_norm = qcrb_matrix / qcrb_at_omega0

    flat_idx = np.argmin(qcrb_matrix_norm)
    omega_idx, time_idx = np.unravel_index(flat_idx, qcrb_matrix_norm.shape)

    return {
        "gamma": gamma,
        "beta": beta,
        "min_qcrb_norm": qcrb_matrix_norm[omega_idx, time_idx],
        "min_qcrb": qcrb_matrix[omega_idx, time_idx],
        "qcrb_at_omega0": qcrb_at_omega0,
        "optimal_omega": omega_list[omega_idx],
        "optimal_time": tlist[time_idx],
    }


def plot_gamma_scan(results, output_figure):
    gammas = np.array([r["gamma"] for r in results])
    min_qcrb_norm = np.array([r["min_qcrb_norm"] for r in results])
    min_qcrb = np.array([r["min_qcrb"] for r in results])
    qcrb_omega0 = np.array([r["qcrb_at_omega0"] for r in results])

    best_idx = np.argmin(min_qcrb_norm)
    best_gamma = gammas[best_idx]
    best_qcrb_norm = min_qcrb_norm[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(gammas, min_qcrb_norm, marker="o")
    plt.axhline(1.0, color="gray", linestyle=":", label=r"QCRB$(\Omega{=}0)$ reference")
    plt.axvline(best_gamma, color="red", linestyle="--", label=fr"Best $\gamma$ = {best_gamma:.3g}")
    plt.xlabel(r"$\gamma$")
    plt.ylabel(r"$\min_{\Omega,t}\,\mathrm{QCRB}\;/\;\mathrm{QCRB}(\Omega{=}0)$")
    plt.title(r"Normalised minimal QCRB vs $\gamma$ ($\beta=1$ fixed)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_figure, bbox_inches="tight")
    plt.show()

    print("\n=== Best gamma result ===")
    print(f"Best gamma:                     {best_gamma:.6g}")
    print(f"Normalised minimum QCRB:        {best_qcrb_norm:.6e}")
    print(f"Absolute minimum QCRB:          {min_qcrb[best_idx]:.6e}")
    print(f"QCRB at Omega=0 (reference):    {qcrb_omega0[best_idx]:.6e}")


def main():
    cfg = SimulationConfig()

    gamma_list = np.linspace(cfg.gamma_min, cfg.gamma_max, cfg.n_gammas)

    results = []
    print(f"Scanning {len(gamma_list)} gamma values (beta = {cfg.beta}) ...")

    for i, gamma in enumerate(gamma_list):
        print(f"[{i+1:2d}/{len(gamma_list)}] gamma = {gamma:.5g}")
        res = compute_min_qcrb_for_gamma(cfg, gamma)
        results.append(res)
        print(
            f"    min QCRB (norm) = {res['min_qcrb_norm']:.6e}, "
            f"QCRB(Omega=0) = {res['qcrb_at_omega0']:.6e}, "
            f"Omega* = {res['optimal_omega']:.4f}, "
            f"t* = {res['optimal_time']:.4f}"
        )

    plot_gamma_scan(results, cfg.output_figure)


if __name__ == "__main__":
    main()