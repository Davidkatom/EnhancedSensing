"""How much bath QFI is captured by physically motivated readout bases?

This script uses the exact 4x4 bath-block solver from
``Analysis_of_analyitical_dicky.py``.  For each time it computes the bath SLD
and projects it onto operator spans using the Fisher (SLD) metric,

    G_ij = Tr[rho {O_i, O_j} / 2],  b_i = Tr[O_i d_J rho],
    G alpha = b,                         F_proj = alpha . b.

The three reported readout capabilities are

1. linear: I + {Jx, Jy},
2. second moments: linear + {Jx^2, Jy^2, {Jx,Jy}}, and
3. every moment of one optimized quadrature J_theta.

The identity is included as a physically irrelevant offset.  This makes the
projection invariant under shifting an observable by a constant.  For the
all-moments basis, the numerically stable spectral-projector representation is
used: {I, J_theta, ..., J_theta^N} and the N+1 projectors of J_theta span the
same operator space, but the latter avoids an ill-conditioned Vandermonde
matrix of powers.
"""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize_scalar

try:
    from CRB.Analysis_of_analyitical_dicky import (
        get_bath_density_matrices,
        qfi_from_rho_and_drho,
    )
except ModuleNotFoundError:  # Allow: python CRB/qfi_captured_fraction.py
    from Analysis_of_analyitical_dicky import (
        get_bath_density_matrices,
        qfi_from_rho_and_drho,
    )


@dataclass(frozen=True)
class CapturedFractionConfig:
    # Requested system and drive sweeps.
    N_values: tuple[int, ...] = (4, 8, 12, 20, 30)
    omega_values: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
    # Keep the time-domain small multiples readable while the full drive sweep
    # is summarized separately as a function of Omega/J.
    time_plot_omega_values: tuple[float, ...] = (0.0, 3.0, 12.0)
    noise_cases: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.1, 0.6))

    J_nominal: float = 1.0
    dJ: float = 1e-3
    t_min: float = 0.01
    t_max: float = 30.0
    n_times: int = 121
    t_overhead: float = 5.0

    qfi_tol: float = 1e-12
    min_qfi_for_ratio: float = 1e-12
    ridge: float = 1e-10

    # theta and theta + pi are the same projective readout.  An even number
    # includes theta=pi/2 exactly, which is also used for the Jz physics check.
    n_theta: int = 90
    refine_theta: bool = True
    theta_refine_tol: float = 1e-5

    fraction_warning_tol: float = 1e-6
    jz_fraction_warning: float = 1e-5
    sld_consistency_rtol: float = 1e-7

    print_angle_table: bool = True
    output_prefix: str = "qfi_captured_fraction"


def build_bath_operators(N: int) -> dict[str, np.ndarray]:
    """Collective bath operators, with the same factor-of-two convention as the solver."""
    spin = N / 2.0
    Jx = 2.0 * qt.jmat(spin, "x").full()
    Jy = 2.0 * qt.jmat(spin, "y").full()
    Jz = 2.0 * qt.jmat(spin, "z").full()
    return {"I": np.eye(N + 1), "Jx": Jx, "Jy": Jy, "Jz": Jz}


def build_configured_operator_bases(
    operators: dict[str, np.ndarray],
) -> dict[str, list[np.ndarray]]:
    """Define the finite operator bases in one easy-to-edit location."""
    I = operators["I"]
    Jx = operators["Jx"]
    Jy = operators["Jy"]

    linear = [I, Jx, Jy]
    second_moments = linear + [
        Jx @ Jx,
        Jy @ Jy,
        Jx @ Jy + Jy @ Jx,
    ]
    return {
        "Linear": linear,
        "Second moments": second_moments,
    }


def frobenius_orthonormal_span(
    operators: list[np.ndarray], tol: float = 1e-12
) -> list[np.ndarray]:
    """Return a stable Hermitian basis for the same real operator span."""
    basis: list[np.ndarray] = []
    for operator in operators:
        candidate = 0.5 * (operator + operator.conj().T)
        for previous in basis:
            candidate = candidate - np.real(np.vdot(previous, candidate)) * previous
        norm = np.linalg.norm(candidate, ord="fro")
        if norm > tol:
            basis.append(candidate / norm)
    return basis


def fisher_metric_projection(
    rho: np.ndarray,
    drho: np.ndarray,
    operators: list[np.ndarray],
    ridge: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Project the SLD onto an operator span via regularized Fisher normal equations."""
    stable_basis = frobenius_orthonormal_span(operators)
    K = len(stable_basis)
    if K == 0:
        return 0.0, np.empty(0), np.empty((0, 0)), np.empty(0)

    G = np.empty((K, K), dtype=float)
    b = np.empty(K, dtype=float)
    rho = 0.5 * (rho + rho.conj().T)
    drho = 0.5 * (drho + drho.conj().T)

    rho_times_ops = [rho @ operator for operator in stable_basis]
    for i, operator_i in enumerate(stable_basis):
        b[i] = float(np.real(np.trace(operator_i @ drho)))
        for j in range(i, K):
            # Cyclicity gives Re Tr[rho O_i O_j] for Hermitian rho, O_i, O_j.
            value = float(np.real(np.trace(rho_times_ops[i] @ stable_basis[j])))
            G[i, j] = value
            G[j, i] = value

    G = 0.5 * (G + G.T)
    scale = max(float(np.max(np.abs(np.diag(G)))), np.finfo(float).tiny)
    G_regularized = G + ridge * scale * np.eye(K)
    try:
        alpha = np.linalg.solve(G_regularized, b)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(G_regularized, b, rcond=None)[0]

    F_projected = float(np.dot(alpha, b))
    return max(F_projected, 0.0), alpha, G, b


def quadrature_projector_rows(
    Jy: np.ndarray, Jz: np.ndarray, theta_grid: np.ndarray
) -> np.ndarray:
    """Flattened spectral projectors for every candidate J_theta measurement."""
    dimension = Jy.shape[0]
    rows = np.empty(
        (len(theta_grid), dimension, dimension * dimension), dtype=np.complex128
    )
    for angle_index, theta in enumerate(theta_grid):
        J_theta = math.cos(theta) * Jy + math.sin(theta) * Jz
        _, eigenvectors = np.linalg.eigh(J_theta)
        # Row k is vec(P_k), arranged so row @ vec(rho.T) = Tr(P_k rho).
        for outcome in range(dimension):
            vector = eigenvectors[:, outcome]
            projector = np.outer(vector, vector.conj())
            rows[angle_index, outcome] = projector.reshape(-1)
    return rows


def all_moments_fisher_for_angles(
    rho: np.ndarray,
    drho: np.ndarray,
    projector_rows: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fisher projection onto all moments of J_theta for every theta.

    In the spectral-projector basis the Fisher Gram matrix is diag(p_k) and
    b_k = d_J p_k.  Solving the regularized normal equations is therefore
    alpha_k = d_J p_k / (p_k + ridge), followed by F_proj = alpha . b.
    """
    n_angles, dimension, _ = projector_rows.shape
    flat_rows = projector_rows.reshape(n_angles * dimension, -1)
    probabilities = np.real(flat_rows @ rho.T.reshape(-1)).reshape(
        n_angles, dimension
    )
    derivatives = np.real(flat_rows @ drho.T.reshape(-1)).reshape(
        n_angles, dimension
    )

    if np.min(probabilities) < -1e-9:
        warnings.warn(
            f"Projective probability became negative: min={np.min(probabilities):.3e}",
            RuntimeWarning,
        )
    probabilities = np.maximum(probabilities, 0.0)

    # max(diag(G)) is the same scale convention used by the generic solver.
    scales = np.maximum(np.max(probabilities, axis=1), np.finfo(float).tiny)
    denominators = probabilities + ridge * scales[:, None]
    return np.sum(derivatives * derivatives / denominators, axis=1)


def safe_fraction(projected: float, qfi: float, minimum_qfi: float) -> float:
    return projected / qfi if qfi > minimum_qfi else np.nan


def canonical_projective_angle(theta: float) -> float:
    """Represent equivalent axes theta and theta+pi in [-pi/2, pi/2)."""
    return float((theta + np.pi / 2.0) % np.pi - np.pi / 2.0)


def flag_fraction_violation(
    fraction: float,
    label: str,
    N: int,
    omega: float,
    gamma: float,
    beta: float,
    time: float,
    tolerance: float,
) -> None:
    if np.isfinite(fraction) and fraction > 1.0 + tolerance:
        warnings.warn(
            f"F_proj/F_Q={fraction:.8f} > 1 for {label}: "
            f"N={N}, Omega={omega:g}, gamma={gamma:g}, beta={beta:g}, t={time:g}",
            RuntimeWarning,
        )


def analyze_condition(
    cfg: CapturedFractionConfig,
    N: int,
    omega: float,
    gamma: float,
    beta: float,
    times: np.ndarray,
    operator_bases: dict[str, list[np.ndarray]],
    Jy: np.ndarray,
    Jz: np.ndarray,
    theta_grid: np.ndarray,
    projector_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute QFI and all captured-fraction trajectories for one condition."""
    bath_plus = get_bath_density_matrices(
        Omega_0=omega,
        J=cfg.J_nominal + cfg.dJ,
        tlist=times,
        N=N,
        gamma=gamma,
        beta=beta,
    )
    bath_minus = get_bath_density_matrices(
        Omega_0=omega,
        J=cfg.J_nominal - cfg.dJ,
        tlist=times,
        N=N,
        gamma=gamma,
        beta=beta,
    )

    qfi_values = np.zeros(len(times))
    linear_fractions = np.full(len(times), np.nan)
    second_fractions = np.full(len(times), np.nan)
    quadrature_fractions = np.full(len(times), np.nan)
    jz_fractions = np.full(len(times), np.nan)
    optimal_angles = np.zeros(len(times))
    linear_projected_values = np.zeros(len(times))
    second_projected_values = np.zeros(len(times))
    quadrature_projected_values = np.zeros(len(times))

    jz_index = int(np.argmin(np.abs(theta_grid - np.pi / 2.0)))
    if not np.isclose(theta_grid[jz_index], np.pi / 2.0, atol=1e-13):
        raise ValueError("n_theta must place theta=pi/2 on the grid for the Jz check")

    for time_index, time in enumerate(times):
        rho = 0.5 * (bath_plus[time_index] + bath_minus[time_index])
        drho = (bath_plus[time_index] - bath_minus[time_index]) / (2.0 * cfg.dJ)
        rho = 0.5 * (rho + rho.conj().T)
        drho = 0.5 * (drho + drho.conj().T)

        qfi, sld = qfi_from_rho_and_drho(rho, drho, tol=cfg.qfi_tol)
        qfi_values[time_index] = qfi
        qfi_from_sld = float(np.real(np.trace(sld @ drho)))
        if qfi > cfg.min_qfi_for_ratio and not np.isclose(
            qfi_from_sld, qfi, rtol=cfg.sld_consistency_rtol, atol=cfg.qfi_tol
        ):
            warnings.warn(
                f"SLD consistency failure: Tr(L drho)={qfi_from_sld:.8e}, "
                f"F_Q={qfi:.8e}, N={N}, Omega={omega:g}, t={time:g}",
                RuntimeWarning,
            )

        linear_projected, *_ = fisher_metric_projection(
            rho, drho, operator_bases["Linear"], cfg.ridge
        )
        second_projected, *_ = fisher_metric_projection(
            rho, drho, operator_bases["Second moments"], cfg.ridge
        )
        angle_fisher = all_moments_fisher_for_angles(
            rho, drho, projector_rows, cfg.ridge
        )
        best_angle_index = int(np.argmax(angle_fisher))
        best_angle = float(theta_grid[best_angle_index])
        best_angle_fisher = float(angle_fisher[best_angle_index])

        # The full grid makes the search global; this bounded local step removes
        # grid discretization from theta* and the captured fraction.
        if cfg.refine_theta:
            spacing = float(np.pi / cfg.n_theta)

            def negative_projected_fisher(unwrapped_angle: float) -> float:
                candidate_angle = unwrapped_angle % np.pi
                candidate_rows = quadrature_projector_rows(
                    Jy, Jz, np.array([candidate_angle])
                )
                return -float(
                    all_moments_fisher_for_angles(
                        rho, drho, candidate_rows, cfg.ridge
                    )[0]
                )

            refinement = minimize_scalar(
                negative_projected_fisher,
                bounds=(best_angle - spacing, best_angle + spacing),
                method="bounded",
                options={"xatol": cfg.theta_refine_tol},
            )
            if refinement.success and -float(refinement.fun) >= best_angle_fisher:
                best_angle = float(refinement.x % np.pi)
                best_angle_fisher = -float(refinement.fun)

        linear_fractions[time_index] = safe_fraction(
            linear_projected, qfi, cfg.min_qfi_for_ratio
        )
        second_fractions[time_index] = safe_fraction(
            second_projected, qfi, cfg.min_qfi_for_ratio
        )
        quadrature_fractions[time_index] = safe_fraction(
            best_angle_fisher, qfi, cfg.min_qfi_for_ratio
        )
        jz_fractions[time_index] = safe_fraction(
            float(angle_fisher[jz_index]), qfi, cfg.min_qfi_for_ratio
        )
        optimal_angles[time_index] = canonical_projective_angle(best_angle)
        linear_projected_values[time_index] = linear_projected
        second_projected_values[time_index] = second_projected
        quadrature_projected_values[time_index] = best_angle_fisher

        for label, fraction in (
            ("Linear", linear_fractions[time_index]),
            ("Second moments", second_fractions[time_index]),
            ("Best quadrature, all moments", quadrature_fractions[time_index]),
            ("Jz all moments diagnostic", jz_fractions[time_index]),
        ):
            flag_fraction_violation(
                fraction,
                label,
                N,
                omega,
                gamma,
                beta,
                float(time),
                cfg.fraction_warning_tol,
            )

        if (
            np.isfinite(jz_fractions[time_index])
            and jz_fractions[time_index] > cfg.jz_fraction_warning
        ):
            warnings.warn(
                "Jz-diagonal readout captured unexpectedly large information: "
                f"fraction={jz_fractions[time_index]:.3e}, N={N}, "
                f"Omega={omega:g}, gamma={gamma:g}, beta={beta:g}, t={time:g}. "
                "This violates the [Jz,H]=0 physics check and should be investigated.",
                RuntimeWarning,
            )

    return {
        "qfi": qfi_values,
        "Linear": linear_fractions,
        "Second moments": second_fractions,
        "Best quadrature, all moments": quadrature_fractions,
        "Jz diagnostic": jz_fractions,
        "theta": optimal_angles,
        "Linear F_proj": linear_projected_values,
        "Second moments F_proj": second_projected_values,
        "Best quadrature F_proj": quadrature_projected_values,
    }


def print_optimal_angles(
    N: int,
    omega: float,
    gamma: float,
    beta: float,
    times: np.ndarray,
    angles: np.ndarray,
) -> None:
    print(
        f"\nOptimal theta*(t): N={N}, Omega/J={omega:g}, "
        f"gamma={gamma:g}, beta={beta:g}"
    )
    print("  " + "  ".join(f"t={time:.3f}: {angle:.6f} rad" for time, angle in zip(times, angles)))


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "gamma",
        "beta",
        "N",
        "omega_over_J",
        "time",
        "qfi",
        "linear_f_projected",
        "second_moments_f_projected",
        "best_quadrature_f_projected",
        "linear_fraction",
        "second_moments_fraction",
        "best_quadrature_all_moments_fraction",
        "jz_all_moments_fraction",
        "theta_star_rad",
        "theta_star_deg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_noise_case(
    cfg: CapturedFractionConfig,
    gamma: float,
    beta: float,
    times: np.ndarray,
    results: dict[tuple[int, float], dict[str, np.ndarray]],
) -> Path:
    n_rows = len(cfg.N_values)
    n_columns = len(cfg.time_plot_omega_values)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(6.2 * n_columns, 3.0 * n_rows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    # Plot each readout basis's projected Fisher information F_proj on top of the
    # full bath QFI F_Q, rather than their ratio.
    styles = {
        "Linear F_proj": dict(color="tab:blue", linestyle="-", label=r"Linear $F_{\rm proj}$"),
        "Second moments F_proj": dict(color="tab:orange", linestyle="--", label=r"Second moments $F_{\rm proj}$"),
        "Best quadrature F_proj": dict(color="tab:green", linestyle="-.", label=r"Best quadrature $F_{\rm proj}$"),
    }
    for row, N in enumerate(cfg.N_values):
        for column, omega in enumerate(cfg.time_plot_omega_values):
            axis = axes[row, column]
            condition = results[(N, omega)]
            axis.plot(
                times,
                condition["qfi"],
                color="black",
                linewidth=2.0,
                label=r"Full SLD $F_Q$",
            )
            for key, style in styles.items():
                axis.plot(times, condition[key], linewidth=1.8, **style)
            axis.set_title(rf"$N={N},\ \Omega/J={omega:g}$")
            axis.grid(True, linestyle=":", alpha=0.55)
            if column == 0:
                axis.set_ylabel(r"Fisher information")
            if row == n_rows - 1:
                axis.set_xlabel(r"Time $t$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.974),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        rf"Bath QFI captured by physical readout bases ($\gamma={gamma:g},\ \beta={beta:g}$)",
        y=0.998,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    suffix = f"gamma_{gamma:g}_beta_{beta:g}".replace(".", "p")
    output_path = Path(f"{cfg.output_prefix}_{suffix}.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_drive_sweep_summary(
    cfg: CapturedFractionConfig,
    gamma: float,
    beta: float,
    times: np.ndarray,
    results: dict[tuple[int, float], dict[str, np.ndarray]],
) -> Path:
    """Compare total and physically extractable information across the drive sweep."""
    fig, axes = plt.subplots(
        3,
        len(cfg.N_values),
        figsize=(4.1 * len(cfg.N_values), 10.0),
        sharex=True,
        squeeze=False,
    )
    omega_values = np.asarray(cfg.omega_values, dtype=float)
    cycle_times = times + cfg.t_overhead

    for column, N in enumerate(cfg.N_values):
        peak_qfi = []
        peak_projected = []
        peak_qfi_rate = []
        peak_projected_rate = []
        fraction_at_physical_optimum = []

        for omega in cfg.omega_values:
            condition = results[(N, omega)]
            qfi = condition["qfi"]
            projected = condition["Best quadrature F_proj"]
            qfi_rate = qfi / cycle_times
            projected_rate = projected / cycle_times
            physical_optimum = int(np.nanargmax(projected_rate))

            peak_qfi.append(float(np.nanmax(qfi)))
            peak_projected.append(float(np.nanmax(projected)))
            peak_qfi_rate.append(float(np.nanmax(qfi_rate)))
            peak_projected_rate.append(float(np.nanmax(projected_rate)))
            fraction_at_physical_optimum.append(
                float(condition["Best quadrature, all moments"][physical_optimum])
            )

        fraction_at_physical_optimum = np.asarray(fraction_at_physical_optimum)
        missing_at_physical_optimum = 1.0 - fraction_at_physical_optimum

        axes[0, column].plot(
            omega_values,
            peak_qfi,
            marker="o",
            color="black",
            label="All bath QFI (SLD limit)",
        )
        axes[0, column].plot(
            omega_values,
            peak_projected,
            marker="s",
            color="tab:green",
            linestyle="--",
            label="Accessible with best single quadrature",
        )
        axes[0, column].set_title(rf"$N={N}$")
        axes[0, column].set_yscale("log")

        axes[1, column].plot(
            omega_values,
            peak_qfi_rate,
            marker="o",
            color="black",
            label="All bath QFI (SLD limit)",
        )
        axes[1, column].plot(
            omega_values,
            peak_projected_rate,
            marker="s",
            color="tab:green",
            linestyle="--",
            label="Accessible with best single quadrature",
        )
        axes[1, column].set_yscale("log")

        axes[2, column].plot(
            omega_values,
            fraction_at_physical_optimum,
            marker="s",
            color="tab:green",
            label=r"Captured at $t^*_{\rm readout}$",
        )
        axes[2, column].plot(
            omega_values,
            missing_at_physical_optimum,
            marker="x",
            color="tab:red",
            linestyle=":",
            label=r"Missing at $t^*_{\rm readout}$",
        )
        axes[2, column].axhline(1.0, color="black", linewidth=1.0, linestyle=":")
        axes[2, column].set_ylim(-0.02, 1.05)
        axes[2, column].set_xlabel(r"Drive $\Omega/J$")

        for row in range(3):
            axes[row, column].grid(True, linestyle=":", alpha=0.55)

    axes[0, 0].set_ylabel(
        "1. ONE-SHOT INFORMATION\n"
        r"$\max_t F$"
    )
    axes[1, 0].set_ylabel(
        "2. INFORMATION RATE\n"
        r"$\max_t F/(t+t_{\rm oh})$"
    )
    axes[2, 0].set_ylabel(
        "3. SAME-TIME COMPOSITION\n"
        r"Fraction at $t^*_{\rm readout}$"
    )

    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    bottom_handles, bottom_labels = axes[2, 0].get_legend_handles_labels()
    legend_handles = top_handles + bottom_handles
    legend_labels = top_labels + bottom_labels
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.941),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Drive sweep: how much bath QFI exists, how much one quadrature extracts, "
        "and what fraction that is\n"
        rf"$\gamma={gamma:g},\ \beta={beta:g}$; "
        rf"$t\in[{times[0]:g},{times[-1]:g}]$; $t_{{\rm oh}}={cfg.t_overhead:g}$",
        y=0.997,
    )
    fig.text(
        0.5,
        0.012,
        "Rows 1-2 optimize the black and green curves independently over time; "
        "their vertical gap is therefore not a captured fraction.  "
        "Row 3 evaluates both at the same time "
        r"$t^*_{\rm readout}=\arg\max_t[F_{\rm proj}/(t+t_{\rm oh})]$.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.045, 1.0, 0.89))
    suffix = f"gamma_{gamma:g}_beta_{beta:g}".replace(".", "p")
    output_path = Path(f"{cfg.output_prefix}_drive_sweep_{suffix}.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    cfg = CapturedFractionConfig()
    if cfg.n_theta % 2 != 0:
        raise ValueError("n_theta must be even so the grid includes theta=pi/2")

    times = np.linspace(cfg.t_min, cfg.t_max, cfg.n_times)
    theta_grid = np.linspace(0.0, np.pi, cfg.n_theta, endpoint=False)
    csv_rows: list[dict[str, float]] = []
    output_figures: list[Path] = []

    for gamma, beta in cfg.noise_cases:
        noise_results: dict[tuple[int, float], dict[str, np.ndarray]] = {}
        for N in cfg.N_values:
            operators = build_bath_operators(N)
            operator_bases = build_configured_operator_bases(operators)
            projector_rows = quadrature_projector_rows(
                operators["Jy"], operators["Jz"], theta_grid
            )

            for omega in cfg.omega_values:
                print(
                    f"Computing N={N}, Omega/J={omega:g}, "
                    f"gamma={gamma:g}, beta={beta:g} ..."
                )
                condition = analyze_condition(
                    cfg,
                    N,
                    omega,
                    gamma,
                    beta,
                    times,
                    operator_bases,
                    operators["Jy"],
                    operators["Jz"],
                    theta_grid,
                    projector_rows,
                )
                noise_results[(N, omega)] = condition

                if cfg.print_angle_table:
                    print_optimal_angles(
                        N, omega, gamma, beta, times, condition["theta"]
                    )

                finite_jz = condition["Jz diagnostic"][
                    np.isfinite(condition["Jz diagnostic"])
                ]
                max_jz = float(np.max(finite_jz)) if finite_jz.size else np.nan
                finite_best = condition["Best quadrature, all moments"][
                    np.isfinite(condition["Best quadrature, all moments"])
                ]
                min_best = float(np.min(finite_best)) if finite_best.size else np.nan
                print(
                    f"  max Jz-only fraction={max_jz:.3e}; "
                    f"minimum best-quadrature fraction={min_best:.4f}; "
                    f"largest missing gap={1.0 - min_best:.4f}"
                )

                for index, time in enumerate(times):
                    theta = float(condition["theta"][index])
                    csv_rows.append(
                        {
                            "gamma": gamma,
                            "beta": beta,
                            "N": N,
                            "omega_over_J": omega / cfg.J_nominal,
                            "time": float(time),
                            "qfi": float(condition["qfi"][index]),
                            "linear_f_projected": float(
                                condition["Linear F_proj"][index]
                            ),
                            "second_moments_f_projected": float(
                                condition["Second moments F_proj"][index]
                            ),
                            "best_quadrature_f_projected": float(
                                condition["Best quadrature F_proj"][index]
                            ),
                            "linear_fraction": float(condition["Linear"][index]),
                            "second_moments_fraction": float(
                                condition["Second moments"][index]
                            ),
                            "best_quadrature_all_moments_fraction": float(
                                condition["Best quadrature, all moments"][index]
                            ),
                            "jz_all_moments_fraction": float(
                                condition["Jz diagnostic"][index]
                            ),
                            "theta_star_rad": theta,
                            "theta_star_deg": float(np.degrees(theta)),
                        }
                    )

        output_figures.append(
            plot_noise_case(cfg, gamma, beta, times, noise_results)
        )
        output_figures.append(
            plot_drive_sweep_summary(cfg, gamma, beta, times, noise_results)
        )

    csv_path = Path(f"{cfg.output_prefix}_data.csv")
    write_csv(csv_path, csv_rows)
    print("\nWrote:")
    for output_figure in output_figures:
        print(f"  {output_figure.resolve()}")
    print(f"  {csv_path.resolve()}")


if __name__ == "__main__":
    main()
