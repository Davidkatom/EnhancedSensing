"""Bath-only QFI analysis for the collective Dicke-like model."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

try:
    from CRB.crb_core import (
        SimulationConfig as BaseSimulationConfig,
        build_bath_operators,
        build_hamiltonian,
        build_initial_state,
        build_spin_operators,
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        compute_qcrb_matrices,
        get_bath_density_matrices,
        observable_moment_fisher,
        observable_projective_fisher,
        qfi_from_rho_and_drho,
    )
except ModuleNotFoundError:  # Allow: python CRB/Analysis_of_analyitical_dicky.py
    from crb_core import (
        SimulationConfig as BaseSimulationConfig,
        build_bath_operators,
        build_hamiltonian,
        build_initial_state,
        build_spin_operators,
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        compute_qcrb_matrices,
        get_bath_density_matrices,
        observable_moment_fisher,
        observable_projective_fisher,
        qfi_from_rho_and_drho,
    )


@dataclass(frozen=True)
class SimulationConfig(BaseSimulationConfig):
    """Dicke-analysis sweep and output configuration.

    The system-size and noise knobs below are inherited from
    ``BaseSimulationConfig`` (``crb_core.py``) and re-declared here so every
    control lives in one place.  Other inherited fields
    (``J_nominal``, ``dJ``, ``t_min``, ``t_max``, ``n_steps``, ``t_overhead``,
    tolerances) can be overridden the same way.
    """

    # --- System size and central-spin noise (inherited defaults surfaced) ---
    N: int = 10            # number of bath spins (bath Hilbert dim = N + 1)
    gamma: float = 0.0    # central-spin sigma_z DEPHASING rate
    beta: float = 0.0     # central-spin sigma_x BIT-FLIP rate -- this is NOISE,
    #                       not the probe angle.  Set to 0.0 for a noiseless run.

    # --- Transverse-drive (Omega_0) sweep ---
    omega_min: float = 0.0
    omega_max: float = 40.0
    n_omegas: int = 20

    # --- Initial bath probe angle ---
    # Spin-N/2 coherent state at polar angle probe_beta_deg from +z (the paper's
    # angle beta; 90 deg = equatorial +x, the prior default).  The central spin
    # stays in |+x>; only the bath is tilted.  THIS field is the probe-angle knob.
    probe_beta_deg: float = 45.0

    output_figure: str = "bath_only_qfi_analysis.png"


def plot_qfi_results(
    tlist: np.ndarray,
    omega_list: np.ndarray,
    min_qcrb_per_omega: np.ndarray,
    min_qcrb_unnormalized_per_omega: np.ndarray,
    min_ccrb_per_omega: np.ndarray,
    min_ccrb_unnormalized_per_omega: np.ndarray,
    optimal_times: np.ndarray,
    opt_quadrature_angles: np.ndarray,
    output_figure: str,
    probe_beta_deg: float,
) -> None:
    """Plot QCRB minima, optimal times, and optimal quadrature angles."""
    optimal_idx = int(np.argmin(min_qcrb_per_omega))
    optimal_omega = omega_list[optimal_idx]
    unnormalized_optimal_idx = int(
        np.argmin(min_qcrb_unnormalized_per_omega)
    )
    unnormalized_optimal_omega = omega_list[unnormalized_optimal_idx]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    normalized_line = axes[0].plot(
        omega_list,
        min_qcrb_per_omega,
        marker="o",
        linestyle="-",
        color="tab:blue",
        label=r"Time-normalized: $\min_t\sqrt{(t+t_{\mathrm{oh}})/F_Q}$",
    )
    normalized_optimum_line = axes[0].axvline(
        optimal_omega,
        linestyle="--",
        color="tab:blue",
        alpha=0.65,
        label=rf"Normalized optimum $\Omega$ = {optimal_omega:.2f}",
    )
    axes[0].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[0].set_ylabel(
        r"Time-normalized QCRB $\min_t\sqrt{(t+t_{\mathrm{oh}})/F_Q}$",
        color="tab:blue",
    )
    axes[0].tick_params(axis="y", labelcolor="tab:blue")

    unnormalized_axis = axes[0].twinx()
    unnormalized_line = unnormalized_axis.plot(
        omega_list,
        min_qcrb_unnormalized_per_omega,
        marker="s",
        linestyle="-",
        color="tab:orange",
        label=r"Unnormalized: $\min_t 1/\sqrt{F_Q}$",
    )
    unnormalized_optimum_line = unnormalized_axis.axvline(
        unnormalized_optimal_omega,
        linestyle=":",
        color="tab:orange",
        alpha=0.75,
        label=(
            rf"Unnormalized optimum $\Omega$ = "
            f"{unnormalized_optimal_omega:.2f}"
        ),
    )
    unnormalized_axis.set_ylabel(
        r"Unnormalized QCRB $\min_t 1/\sqrt{F_Q}$",
        color="tab:orange",
    )
    unnormalized_axis.tick_params(axis="y", labelcolor="tab:orange")
    axes[0].set_title("Time-normalized and unnormalized bath-only QCRB")
    axes[0].grid(True, linestyle=":")

    first_panel_handles = (
        normalized_line
        + [normalized_optimum_line]
        + unnormalized_line
        + [unnormalized_optimum_line]
    )
    axes[0].legend(
        first_panel_handles,
        [handle.get_label() for handle in first_panel_handles],
        fontsize=8,
    )

    axes[1].plot(
        omega_list,
        optimal_times,
        marker="s",
        linestyle="",
        color="darkorange",
        label=r"Data $t^*(\Omega)$",
    )

    def fit_func(omega: np.ndarray, intercept: float, slope: float) -> np.ndarray:
        return intercept + slope * omega

    try:
        fit_parameters, _ = curve_fit(
            fit_func,
            omega_list,
            optimal_times,
            p0=[optimal_times[0], 0.0],
            maxfev=10000,
        )
        intercept, slope = fit_parameters
        omega_dense = np.linspace(min(omega_list), max(omega_list), 200)
        axes[1].plot(
            omega_dense,
            fit_func(omega_dense, intercept, slope),
            linestyle="-",
            color="black",
            label=(
                r"Fit: $T=b+c\Omega$"
                + "\n"
                + f"b={intercept:.3f}, c={slope:.3e}"
            ),
        )
    except Exception as error:
        print(f"Curve fitting failed: {error}")

    axes[1].axvline(
        optimal_omega,
        color="red",
        linestyle="--",
        label=rf"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[1].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[1].set_ylabel(r"Optimal time $t^*$")
    axes[1].set_title(r"Optimal measurement time $t^*$ vs $\Omega$")
    axes[1].grid(True, linestyle=":")
    axes[1].legend()

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
        label=rf"Optimal $\Omega$ = {optimal_omega:.2f}",
    )
    axes[2].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[2].set_ylabel("Angle (radians)")
    axes[2].set_title(r"Optimal Quadrature Angle vs $\Omega$")
    axes[2].grid(True, linestyle=":")
    axes[2].legend()

    # Fourth panel: quantum QCRB vs the classical CRB of a projective Jy
    # measurement (log scale).  The mean <Jy> itself is symmetry-pinned to 0, so
    # its error-propagation CRB is infinite; the information a Jy readout carries
    # lives in the Jy distribution/variance, which this projective bound uses.
    axes[3].semilogy(
        omega_list,
        min_qcrb_unnormalized_per_omega,
        marker="s",
        markersize=4,
        linestyle="-",
        color="tab:orange",
        label=r"Quantum: $\min_t 1/\sqrt{F_Q}$",
    )
    axes[3].semilogy(
        omega_list,
        min_ccrb_unnormalized_per_omega,
        marker="+",
        markersize=7,
        linestyle="--",
        color="tab:orange",
        label=r"Classical $\langle J_y\rangle$: $\min_t 1/\sqrt{F_{\mathrm{cl}}}$",
    )
    axes[3].semilogy(
        omega_list,
        min_qcrb_per_omega,
        marker="o",
        markersize=4,
        linestyle="-",
        color="tab:blue",
        label=r"Quantum: $\min_t\sqrt{(t+t_{\mathrm{oh}})/F_Q}$",
    )
    axes[3].semilogy(
        omega_list,
        min_ccrb_per_omega,
        marker="x",
        markersize=5,
        linestyle="--",
        color="tab:blue",
        label=r"Classical $\langle J_y\rangle$: $\min_t\sqrt{(t+t_{\mathrm{oh}})/F_{\mathrm{cl}}}$",
    )
    axes[3].set_xlabel(r"Transverse Field ($\Omega$)")
    axes[3].set_ylabel(r"CRB $\delta J$ (log scale)")
    axes[3].set_title(
        r"Quantum QCRB vs classical $\langle J_y\rangle$-readout CRB"
    )
    axes[3].grid(True, which="both", linestyle=":")
    axes[3].legend(fontsize=7)

    fig.suptitle(
        rf"Bath probe: coherent state at $\beta={probe_beta_deg:g}^\circ$ "
        r"from $+z$ (central spin $|{+}x\rangle$)",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_figure, bbox_inches="tight")
    plt.show()


def main() -> None:
    cfg = SimulationConfig()
    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    omega_list = np.linspace(cfg.omega_min, cfg.omega_max, cfg.n_omegas)

    qfi_matrix = np.zeros((len(omega_list), len(tlist)))
    fisher_jy_moment_matrix = np.zeros((len(omega_list), len(tlist)))
    fisher_jy_proj_matrix = np.zeros((len(omega_list), len(tlist)))
    sld_trajectories: list[list[np.ndarray]] = []

    bath_operators = build_bath_operators(cfg.N)
    Jy = bath_operators["Jy"]
    Jz = bath_operators["Jz"]
    probe_state = coherent_bath_state(cfg.N, np.deg2rad(cfg.probe_beta_deg))

    print(
        f"Initial bath probe: coherent state at beta={cfg.probe_beta_deg:g} deg "
        "from +z (central spin |+x>)"
    )
    print(f"Computing bath-only QFI for {len(omega_list)} values of Omega_0...")
    for index, Omega_0 in enumerate(omega_list):
        if (index + 1) % 5 == 0 or index == 0:
            print(f"Processed {index + 1}/{len(omega_list)} Omega_0 values")

        bath_rhos_plus = get_bath_density_matrices(
            Omega_0=Omega_0,
            J=cfg.J_nominal + cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            gamma=cfg.gamma,
            beta=cfg.beta,
            bath_state=probe_state,
        )
        bath_rhos_minus = get_bath_density_matrices(
            Omega_0=Omega_0,
            J=cfg.J_nominal - cfg.dJ,
            tlist=tlist,
            N=cfg.N,
            gamma=cfg.gamma,
            beta=cfg.beta,
            bath_state=probe_state,
        )
        qfi_t, sld_t, rho_t, drho_t = compute_bath_qfi_trajectory(
            bath_rhos_plus=bath_rhos_plus,
            bath_rhos_minus=bath_rhos_minus,
            dJ=cfg.dJ,
            tol=cfg.qfi_tol,
        )
        qfi_matrix[index, :] = qfi_t
        sld_trajectories.append(sld_t)

        # Classical Fisher information for a Jy readout, reusing the QFI's rho
        # and d(rho)/dJ.  Two versions: the MEAN <Jy> (error propagation) and a
        # full PROJECTIVE Jy measurement (all outcomes / the Jy distribution).
        fisher_jy_moment_matrix[index, :] = [
            observable_moment_fisher(rho_t[k], drho_t[k], Jy, var_floor=cfg.qfi_tol)
            for k in range(len(tlist))
        ]
        fisher_jy_proj_matrix[index, :] = [
            observable_projective_fisher(rho_t[k], drho_t[k], Jy, tol=cfg.qfi_tol)
            for k in range(len(tlist))
        ]

    qcrb_matrix, qcrb_unnormalized_matrix, optimal_time_idx = (
        compute_qcrb_matrices(
            qfi_matrix=qfi_matrix,
            tlist=tlist,
            t_overhead=cfg.t_overhead,
            qcrb_eps=cfg.qcrb_eps,
        )
    )
    min_qcrb_per_omega = np.min(qcrb_matrix, axis=1)
    min_qcrb_unnormalized_per_omega = np.min(
        qcrb_unnormalized_matrix, axis=1
    )
    optimal_times = tlist[optimal_time_idx]

    # Classical Jy-measurement CRB (projective / full distribution), minimized
    # over t with the same normalization as the QCRB.
    ccrb_matrix, ccrb_unnormalized_matrix, _ = compute_qcrb_matrices(
        qfi_matrix=fisher_jy_proj_matrix,
        tlist=tlist,
        t_overhead=cfg.t_overhead,
        qcrb_eps=cfg.qcrb_eps,
    )
    min_ccrb_per_omega = np.min(ccrb_matrix, axis=1)
    min_ccrb_unnormalized_per_omega = np.min(ccrb_unnormalized_matrix, axis=1)
    # Drop drives where even the full Jy measurement carries no J-signal.
    no_jy_signal = fisher_jy_proj_matrix.max(axis=1) < 1e-9
    min_ccrb_per_omega[no_jy_signal] = np.nan
    min_ccrb_unnormalized_per_omega[no_jy_signal] = np.nan

    # The MEAN <Jy> is symmetry-pinned to 0 (V = sigma_x (x) exp(-i pi Jx)
    # commutes with H and sends Jy -> -Jy), so its error-propagation CRB is
    # infinite; record the residual to confirm the cancellation numerically.
    max_moment_fisher = float(fisher_jy_moment_matrix.max())

    optimal_idx = int(np.argmin(min_qcrb_per_omega))
    optimal_omega = omega_list[optimal_idx]

    Jyz = Jy @ Jz + Jz @ Jy
    Jy2_minus_Jz2 = Jy @ Jy - Jz @ Jz
    opt_quadrature_angles = np.zeros(len(omega_list))
    for index, time_index in enumerate(optimal_time_idx):
        optimal_sld = sld_trajectories[index][time_index]
        cyz = np.real(np.trace(optimal_sld @ Jyz))
        cy2z2 = np.real(np.trace(optimal_sld @ Jy2_minus_Jz2))
        opt_quadrature_angles[index] = 0.5 * np.arctan2(cyz, cy2z2)

    print("\nDone.")
    print(f"Optimal Omega_0 (bath-only QFI criterion): {optimal_omega:.6f}")
    print(
        "Minimum bath-only QCRB sensitivity: "
        f"{min_qcrb_per_omega[optimal_idx]:.6e}"
    )
    unnormalized_optimal_idx = int(
        np.argmin(min_qcrb_unnormalized_per_omega)
    )
    print(
        "Minimum unnormalized bath-only QCRB: "
        f"{min_qcrb_unnormalized_per_omega[unnormalized_optimal_idx]:.6e} "
        f"at Omega_0={omega_list[unnormalized_optimal_idx]:.6f}"
    )
    print(f"Optimal time at best Omega_0: {optimal_times[optimal_idx]:.6f}")

    if np.all(np.isnan(min_ccrb_unnormalized_per_omega)):
        print("Classical <Jy> readout: no J-signal at any swept Omega_0.")
    else:
        best_cl_idx = int(np.nanargmin(min_ccrb_unnormalized_per_omega))
        print(
            "Best classical <Jy>-readout CRB (unnormalized): "
            f"{min_ccrb_unnormalized_per_omega[best_cl_idx]:.6e} "
            f"at Omega_0={omega_list[best_cl_idx]:.6f} "
            f"(quantum there: {min_qcrb_unnormalized_per_omega[best_cl_idx]:.6e})"
        )

    plot_qfi_results(
        tlist=tlist,
        omega_list=omega_list,
        min_qcrb_per_omega=min_qcrb_per_omega,
        min_qcrb_unnormalized_per_omega=min_qcrb_unnormalized_per_omega,
        min_ccrb_per_omega=min_ccrb_per_omega,
        min_ccrb_unnormalized_per_omega=min_ccrb_unnormalized_per_omega,
        optimal_times=optimal_times,
        opt_quadrature_angles=opt_quadrature_angles,
        output_figure=cfg.output_figure,
        probe_beta_deg=cfg.probe_beta_deg,
    )


if __name__ == "__main__":
    main()
