import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

try:
    from CRB.crb_core import save_plot
except ModuleNotFoundError:  # Allow: python CRB/Analysis_of_analyitical_dicky_vs_N.py
    from crb_core import save_plot

# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SimulationConfig:
    # Sweep values for N (number of bath qubits)
    N_values: tuple = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    
    # Lindblad dissipation parameters (on the central spin)
    gamma: float = 0.1
    beta: float = 0.0

    J_nominal: float = 1.0
    dJ: float = 1e-3

    t_min: float = 0.01
    t_max: float = 120.0
    n_steps: int = 300

    # Dead time per shot (state preparation + measurement)
    t_overhead: float = 5.0

    # Scan range for Omega_0 (transverse field)
    # Omega_max should be high enough to capture the optimum for larger N.
    omega_min: float = 0.0
    omega_max: float = 120.0
    n_omegas: int = 30

    qfi_tol: float = 1e-12
    qcrb_eps: float = 1e-15

    output_figure: str = "qfi_vs_N_analysis.png"

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

# ============================================================
# Block-diagonal solver for global and reduced density matrices
# ============================================================

def get_density_matrices(
    Omega_0: float,
    J: float,
    tlist: np.ndarray,
    N: int = 10,
    gamma: float = 1.0,
    beta: float = 1.0,
):
    S_spin = N / 2.0
    dim_bath = int(2 * S_spin + 1)

    # S_z eigenvalues in the same basis/ordering as qt.jmat / qt.spin_coherent.
    s_vals = 2.0 * np.real(np.diag(qt.jmat(S_spin, "z").full()))
    chi = qt.spin_coherent(S_spin, np.pi / 2, 0).full().ravel()

    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    I4 = np.eye(4, dtype=complex)

    # Central spin starts in |+>
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    r0 = np.outer(plus, plus.conj()).reshape(4)

    L_diss = beta * (np.kron(sx, sx.T) - I4) + gamma * (np.kron(sz, sz.T) - I4)

    n_times = len(tlist)
    # blocks[t, i, j] is the 2 x 2 central-spin operator associated with
    # |i><j| in the bath basis.  Keeping the blocks lets us construct the
    # global state and both reduced states from the same evolution.
    blocks = np.zeros((n_times, dim_bath, dim_bath, 2, 2), dtype=complex)

    for i in range(dim_bath):
        H_i = Omega_0 * sx + J * s_vals[i] * sz
        for j in range(i, dim_bath):
            H_j = Omega_0 * sx + J * s_vals[j] * sz
            L = -1j * (np.kron(H_i, I2) - np.kron(I2, H_j.T)) + L_diss

            evals, V = np.linalg.eig(L)
            c0 = np.linalg.solve(V, r0) * (chi[i] * np.conj(chi[j]))
            rt = V @ (np.exp(np.outer(evals, tlist)) * c0[:, None])  # 4 x n_times

            central_blocks = rt.T.reshape(n_times, 2, 2)
            blocks[:, i, j, :, :] = central_blocks
            if j != i:
                blocks[:, j, i, :, :] = central_blocks.conj().transpose(0, 2, 1)

    bath = np.trace(blocks, axis1=3, axis2=4)
    central = np.einsum("tiiab->tab", blocks)

    # Convert rho[a, i, b, j] to the tensor-product matrix ordering
    # (central, bath) used by QuTiP.
    global_states = blocks.transpose(0, 3, 1, 4, 2).reshape(
        n_times, 2 * dim_bath, 2 * dim_bath
    )

    return (
        [global_states[k] for k in range(n_times)],
        [bath[k] for k in range(n_times)],
        [central[k] for k in range(n_times)],
    )


def get_bath_density_matrices(
    Omega_0: float,
    J: float,
    tlist: np.ndarray,
    N: int = 10,
    gamma: float = 1.0,
    beta: float = 1.0,
):
    """Backward-compatible bath-only wrapper around the full block solver."""
    _, bath, _ = get_density_matrices(Omega_0, J, tlist, N, gamma, beta)
    return bath

# ============================================================
# QFI trajectory calculation
# ============================================================

def qfi_from_rho_and_drho(
    rho: np.ndarray, drho: np.ndarray, tol: float = 1e-12
) -> tuple[float, np.ndarray]:
    # Hermitize numerically
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)

    evals, evecs = np.linalg.eigh(rho)

    # Clean tiny negative eigenvalues from numerical noise
    evals = np.real(evals)
    evals[np.abs(evals) < tol] = 0.0

    M = evecs.conj().T @ drho @ evecs
    denom = evals[:, None] + evals[None, :]
    inv2 = np.zeros_like(denom)
    mask = denom > tol
    inv2[mask] = 2.0 / denom[mask]

    qfi = np.sum(np.abs(M) ** 2 * inv2)
    W = M * inv2
    L = evecs @ W @ evecs.conj().T  # SLD back in the original basis

    return float(np.real(qfi)), L

def compute_bath_qfi_trajectory(
    bath_rhos_plus,
    bath_rhos_minus,
    dJ: float,
    tol: float = 1e-12,
):
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


def compute_qfi_trajectory(rhos_plus, rhos_minus, dJ: float, tol: float = 1e-12):
    """Compute a QFI trajectory when only the QFI values are required."""
    qfi_t = np.zeros(len(rhos_plus))
    for k, (rho_plus, rho_minus) in enumerate(zip(rhos_plus, rhos_minus)):
        rho = 0.5 * (rho_plus + rho_minus)
        drho = (rho_plus - rho_minus) / (2.0 * dJ)
        qfi_t[k], _ = qfi_from_rho_and_drho(rho, drho, tol=tol)
    return qfi_t

# ============================================================
# Plotting function
# ============================================================

def plot_vs_N_results(
    N_values: np.ndarray,
    min_qcrb_norm: np.ndarray,
    min_qcrb_unnorm: np.ndarray,
    opt_omega_norm: np.ndarray,
    opt_omega_unnorm: np.ndarray,
    opt_t_norm: np.ndarray,
    opt_t_unnorm: np.ndarray,
    opt_quadrature_angle_norm: np.ndarray,
    fq_bath_over_global: np.ndarray,
    fq_central_over_global: np.ndarray,
    output_figure: str,
    cfg: SimulationConfig,
):
    # Function to fit power-law y = a * N^b
    def fit_power_law(x, y):
        log_x = np.log(x)
        log_y = np.log(y)
        b, log_a = np.polyfit(log_x, log_y, 1)
        return np.exp(log_a), b

    fig, axes = plt.subplots(3, 2, figsize=(16, 17))
    axes = axes.flatten()

    # --- Plot 1: Time-normalized QCRB Sensitivity ---
    axes[0].plot(
        N_values,
        min_qcrb_norm,
        marker="o",
        linestyle="",
        color="tab:blue",
        label=r"Data: $\min_{t,\Omega}\sqrt{(t+t_{\mathrm{oh}})/F_Q}$",
    )
    # Fit scaling
    try:
        a_n, b_n = fit_power_law(N_values, min_qcrb_norm)
        fit_y_n = a_n * (N_values ** b_n)
        axes[0].plot(
            N_values,
            fit_y_n,
            linestyle="--",
            color="black",
            label=f"Fit: ${a_n:.2f} \\times N^{{{b_n:.2f}}}$",
        )
    except Exception as e:
        print(f"Error fitting normalized QCRB scaling: {e}")

    axes[0].set_xlabel("Number of Bath Qubits ($N$)")
    axes[0].set_ylabel(r"Time-normalized QCRB $\min_{t,\Omega}\sqrt{(t+t_{\mathrm{oh}})/F_Q}$")
    axes[0].set_title(r"Time-normalized QCRB vs N (Optimal $\Omega$, $t$)")
    axes[0].grid(True, linestyle=":")
    axes[0].legend()
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    # --- Plot 2: Unnormalized QCRB Sensitivity / Inverse QFI ---
    # We will plot 1/\sqrt{F_Q} on left axis and 1/F_Q on right axis
    line_unnorm = axes[1].plot(
        N_values,
        min_qcrb_unnorm,
        marker="s",
        linestyle="",
        color="tab:orange",
        label=r"Data: $\min_{t,\Omega} 1/\sqrt{F_Q}$",
    )
    
    # Fit scaling of unnormalized QCRB (which is 1/\sqrt{F_Q})
    try:
        a_un, b_un = fit_power_law(N_values, min_qcrb_unnorm)
        fit_y_un = a_un * (N_values ** b_un)
        axes[1].plot(
            N_values,
            fit_y_un,
            linestyle="--",
            color="black",
            label=f"Fit (1/sqrt): ${a_un:.2f} \\times N^{{{b_un:.2f}}}$",
        )
    except Exception as e:
        print(f"Error fitting unnormalized QCRB scaling: {e}")

    axes[1].set_xlabel("Number of Bath Qubits ($N$)")
    axes[1].set_ylabel(r"Unnormalized QCRB $\min_{t,\Omega} 1/\sqrt{F_Q}$", color="tab:orange")
    axes[1].tick_params(axis="y", labelcolor="tab:orange")
    
    # Twin axis for Inverse QFI (1/F_Q)
    twin_ax = axes[1].twinx()
    min_inv_qfi = min_qcrb_unnorm ** 2
    twin_ax.plot(
        N_values,
        min_inv_qfi,
        marker="d",
        linestyle="",
        color="tab:red",
        label=r"Inverse QFI: $\min_{t,\Omega} 1/F_Q$",
    )
    
    # Fit scaling of 1/F_Q
    try:
        a_iq, b_iq = fit_power_law(N_values, min_inv_qfi)
        twin_ax.plot(
            N_values,
            a_iq * (N_values ** b_iq),
            linestyle="-.",
            color="darkred",
            label=f"Fit (1/FQ): ${a_iq:.2f} \\times N^{{{b_iq:.2f}}}$",
        )
    except Exception as e:
        print(f"Error fitting inverse QFI scaling: {e}")

    twin_ax.set_ylabel(r"Inverse QFI $\min_{t,\Omega} 1/F_Q$", color="tab:red")
    twin_ax.tick_params(axis="y", labelcolor="tab:red")
    
    axes[1].set_title(r"Unnormalized QCRB & Inverse QFI vs N (Optimal $\Omega$, $t$)")
    axes[1].grid(True, linestyle=":")
    
    # Combine legends from axes[1] and twin_ax
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = twin_ax.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    twin_ax.set_yscale("log")

    # --- Plot 3: Optimal Transverse Field Omega ---
    axes[2].plot(
        N_values,
        opt_omega_norm,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label=r"Optimal $\Omega$ (Normalized)",
    )
    axes[2].plot(
        N_values,
        opt_omega_unnorm,
        marker="s",
        linestyle="--",
        color="tab:orange",
        label=r"Optimal $\Omega$ (Unnormalized)",
    )
    axes[2].set_xlabel("Number of Bath Qubits ($N$)")
    axes[2].set_ylabel(r"Optimal Transverse Field $\Omega^*$")
    axes[2].set_title(r"Optimal Transverse Field $\Omega^*$ vs N")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    # --- Plot 4: Optimal Measurement Time t* and Quadrature Angle ---
    axes[3].plot(
        N_values,
        opt_t_norm,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label="Optimal Time $t^*$ (Normalized)",
    )
    axes[3].plot(
        N_values,
        opt_t_unnorm,
        marker="s",
        linestyle="--",
        color="tab:orange",
        label="Optimal Time $t^*$ (Unnormalized)",
    )
    
    axes[3].set_xlabel("Number of Bath Qubits ($N$)")
    axes[3].set_ylabel("Optimal Measurement Time $t^*$", color="black")
    axes[3].tick_params(axis="y", labelcolor="black")
    
    # Twin axis for Quadrature Angle
    twin_ax4 = axes[3].twinx()
    twin_ax4.plot(
        N_values,
        opt_quadrature_angle_norm,
        marker="^",
        linestyle=":",
        color="green",
        label="Opt. Quadrature Angle (Normalized)",
    )
    twin_ax4.set_ylabel("Angle (radians)", color="green")
    twin_ax4.tick_params(axis="y", labelcolor="green")
    
    axes[3].set_title(r"Optimal Measurement Time $t^*$ and Quadrature vs N")
    axes[3].grid(True, linestyle=":")
    
    handles3, labels3 = axes[3].get_legend_handles_labels()
    handles4, labels4 = twin_ax4.get_legend_handles_labels()
    axes[3].legend(handles3 + handles4, loc="upper right")

    # --- Plot 5: Fraction of global QFI accessible in each subsystem ---
    axes[4].plot(
        N_values,
        fq_bath_over_global,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label=r"$F_Q^{\mathrm{bath}}/F_Q^{\mathrm{global}}$",
    )
    axes[4].plot(
        N_values,
        fq_central_over_global,
        marker="s",
        linestyle="--",
        color="tab:orange",
        label=r"$F_Q^{\mathrm{central}}/F_Q^{\mathrm{global}}$",
    )
    axes[4].set_xlabel("Number of Bath Qubits ($N$)")
    axes[4].set_ylabel("Fraction of Global QFI")
    axes[4].set_title("Subsystem QFI Fractions at the Time-normalized Bath Optimum")
    axes[4].set_ylim(0.0, 1.05)
    axes[4].grid(True, linestyle=":")
    axes[4].legend()

    # The sixth panel is intentionally unused.
    fig.delaxes(axes[5])

    plt.tight_layout()
    save_plot(
        fig,
        output_figure,
        metadata={
            "config": cfg,
            "N_values": N_values,
            "normalized_qcrb_minima": min_qcrb_norm,
            "unnormalized_qcrb_minima": min_qcrb_unnorm,
            "optimal_omega_normalized": opt_omega_norm,
            "optimal_omega_unnormalized": opt_omega_unnorm,
            "optimal_time_normalized": opt_t_norm,
            "optimal_time_unnormalized": opt_t_unnorm,
            "optimal_quadrature_angles": opt_quadrature_angle_norm,
            "bath_qfi_fraction": fq_bath_over_global,
            "central_qfi_fraction": fq_central_over_global,
        },
        script_path=__file__,
        bbox_inches="tight",
    )
    plt.show()

# ============================================================
# Main function
# ============================================================

def main():
    cfg = SimulationConfig()

    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    t_cycle = tlist + cfg.t_overhead

    # Results arrays
    opt_omega_norm = []
    opt_omega_unnorm = []
    min_qcrb_norm = []
    min_qcrb_unnorm = []
    opt_t_norm = []
    opt_t_unnorm = []
    opt_quadrature_angle_norm = []
    fq_bath_over_global = []
    fq_central_over_global = []

    print(f"Running sweep over N in {cfg.N_values}...")

    for N in cfg.N_values:
        print(f"\n========================================")
        print(f"Processing N = {N} ...")
        print(f"========================================")
        
        # Grid of Omega values
        omega_list = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_omegas)
        
        qfi_matrix = np.zeros((len(omega_list), len(tlist)))
        qfi_global_matrix = np.zeros_like(qfi_matrix)
        qfi_central_matrix = np.zeros_like(qfi_matrix)
        L_all = []  # L_all[i] = list of SLD matrices over time for Omega i
        
        # S_spin and spin operators for quadrature calculation
        S_spin = N / 2.0
        Jy = qt.jmat(S_spin, "y").full() * 2.0
        Jz = qt.jmat(S_spin, "z").full() * 2.0
        Jyz = Jy @ Jz + Jz @ Jy
        Jy2_minus_Jz2 = Jy @ Jy - Jz @ Jz

        for i, Omega in enumerate(omega_list):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Omega index {i + 1}/{len(omega_list)}: Omega = {Omega:.2f}")
                
            global_rhos_plus, bath_rhos_plus, central_rhos_plus = get_density_matrices(
                Omega_0=Omega,
                J=cfg.J_nominal + cfg.dJ,
                tlist=tlist,
                N=N,
                gamma=cfg.gamma,
                beta=cfg.beta,
            )

            global_rhos_minus, bath_rhos_minus, central_rhos_minus = get_density_matrices(
                Omega_0=Omega,
                J=cfg.J_nominal - cfg.dJ,
                tlist=tlist,
                N=N,
                gamma=cfg.gamma,
                beta=cfg.beta,
            )

            qfi_t, L_t, _, _ = compute_bath_qfi_trajectory(
                bath_rhos_plus=bath_rhos_plus,
                bath_rhos_minus=bath_rhos_minus,
                dJ=cfg.dJ,
                tol=cfg.qfi_tol,
            )

            qfi_matrix[i, :] = qfi_t
            qfi_global_matrix[i, :] = compute_qfi_trajectory(
                global_rhos_plus, global_rhos_minus, cfg.dJ, tol=cfg.qfi_tol
            )
            qfi_central_matrix[i, :] = compute_qfi_trajectory(
                central_rhos_plus, central_rhos_minus, cfg.dJ, tol=cfg.qfi_tol
            )
            L_all.append(L_t)

        # 1. Time-normalized QCRB optimization
        qcrb_matrix = np.sqrt(t_cycle[None, :] / (qfi_matrix + cfg.qcrb_eps))
        min_qcrb_per_omega = np.min(qcrb_matrix, axis=1)
        opt_omega_idx = np.argmin(min_qcrb_per_omega)
        opt_omega_n = omega_list[opt_omega_idx]
        
        opt_t_idx = np.argmin(qcrb_matrix[opt_omega_idx, :])
        opt_t_n = tlist[opt_t_idx]
        
        min_qcrb_norm.append(min_qcrb_per_omega[opt_omega_idx])
        opt_omega_norm.append(opt_omega_n)
        opt_t_norm.append(opt_t_n)

        # 2. Unnormalized QCRB optimization
        qcrb_unnormalized_matrix = 1.0 / np.sqrt(qfi_matrix + cfg.qcrb_eps)
        min_qcrb_unnormalized_per_omega = np.min(qcrb_unnormalized_matrix, axis=1)
        opt_omega_unnorm_idx = np.argmin(min_qcrb_unnormalized_per_omega)
        opt_omega_unn = omega_list[opt_omega_unnorm_idx]
        
        opt_t_unnorm_idx = np.argmin(qcrb_unnormalized_matrix[opt_omega_unnorm_idx, :])
        opt_t_unn = tlist[opt_t_unnorm_idx]
        
        min_qcrb_unnorm.append(min_qcrb_unnormalized_per_omega[opt_omega_unnorm_idx])
        opt_omega_unnorm.append(opt_omega_unn)
        opt_t_unnorm.append(opt_t_unn)

        # 3. Quadrature angle at the time-normalized optimal point
        L_opt = L_all[opt_omega_idx][opt_t_idx]
        cyz = np.real(np.trace(L_opt @ Jyz))
        cy2z2 = np.real(np.trace(L_opt @ Jy2_minus_Jz2))
        opt_quad = 0.5 * np.arctan2(cyz, cy2z2)
        opt_quadrature_angle_norm.append(opt_quad)

        # Compare all three QFIs at exactly the same operating point.  This is
        # the point selected by the existing time-normalized bath-QFI analysis.
        fq_global_opt = qfi_global_matrix[opt_omega_idx, opt_t_idx]
        if fq_global_opt > cfg.qfi_tol:
            fq_bath_over_global.append(
                qfi_matrix[opt_omega_idx, opt_t_idx] / fq_global_opt
            )
            fq_central_over_global.append(
                qfi_central_matrix[opt_omega_idx, opt_t_idx] / fq_global_opt
            )
        else:
            fq_bath_over_global.append(np.nan)
            fq_central_over_global.append(np.nan)

        print(f"Done N={N}:")
        print(f"  Normalized QCRB:   opt_Omega={opt_omega_n:.2f}, opt_t={opt_t_n:.2f}, min_QCRB={min_qcrb_norm[-1]:.6e}")
        print(f"  Unnormalized QCRB: opt_Omega={opt_omega_unn:.2f}, opt_t={opt_t_unn:.2f}, min_QCRB={min_qcrb_unnorm[-1]:.6e}")
        print(f"  FQ fractions at normalized optimum: bath/global={fq_bath_over_global[-1]:.6f}, central/global={fq_central_over_global[-1]:.6f}")

    # Plot results
    plot_vs_N_results(
        N_values=np.array(cfg.N_values),
        min_qcrb_norm=np.array(min_qcrb_norm),
        min_qcrb_unnorm=np.array(min_qcrb_unnorm),
        opt_omega_norm=np.array(opt_omega_norm),
        opt_omega_unnorm=np.array(opt_omega_unnorm),
        opt_t_norm=np.array(opt_t_norm),
        opt_t_unnorm=np.array(opt_t_unnorm),
        opt_quadrature_angle_norm=np.array(opt_quadrature_angle_norm),
        fq_bath_over_global=np.array(fq_bath_over_global),
        fq_central_over_global=np.array(fq_central_over_global),
        output_figure=cfg.output_figure,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()
