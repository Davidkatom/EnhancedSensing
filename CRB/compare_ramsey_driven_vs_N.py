"""Compare Ramsey and driven bath-sensing protocols as a function of N.

Both protocols use

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x

and estimate the coupling ``J`` from the reduced bath state.  The protocols are

1. Ramsey: a configurable central state and ``|+>_bath^N``.
2. Driven: a configurable central state and ``|0>_bath^N``.

The ``Omega`` and ``omega`` values for every plotted curve are configured in
``PROTOCOLS`` below.  Legend labels are generated from those values, so changing
a drive strength there automatically updates both output figures.

For every N, the fixed-total-time sensitivity

    delta_J * sqrt(T_total) = sqrt((t + t_overhead) / F_Q(t))

is minimized on the configured time grid and plotted.  This accounts for the
fact that a shorter interrogation can be repeated more times.

At each protocol's bath-QFI optimum, the script also compares how much of the
global QFI is locally accessible from the bath and from the central spin,
plotting ``FQ_Bath/FQ_global`` and ``FQ_central/FQ_global`` against N.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

try:
    from CRB.crb_core import (
        build_hamiltonian,
        build_spin_operators,
        central_spin_state,
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        get_bath_density_matrices,
        qfi_from_rho_and_drho,
    )
except ModuleNotFoundError:  # Allow: python CRB/compare_ramsey_driven_vs_N.py
    from crb_core import (
        build_hamiltonian,
        build_spin_operators,
        central_spin_state,
        coherent_bath_state,
        compute_bath_qfi_trajectory,
        get_bath_density_matrices,
        qfi_from_rho_and_drho,
    )


@dataclass(frozen=True)
class ComparisonConfig:
    """Numerical controls for the two-protocol comparison."""

    N_values: tuple[int, ...] = tuple(range(2, 50, 5))

    J_nominal: float = 1.0
    dJ: float = 1e-3

    t_min: float = 0.0
    t_max: float = 1.0
    n_steps: int = 100
    t_overhead: float = 0.0

    gamma: float = 0.0
    beta: float = 0.0

    qfi_tol: float = 1e-12
    output_figure: str = "ramsey_vs_driven_time_normalized_qcrb.png"
    output_fraction_figure: str = "ramsey_vs_driven_fq_fractions.png"


@dataclass(frozen=True)
class Protocol:
    name: str
    Omega: float
    omega: float
    bath_theta: float
    marker: str
    central_theta: float = np.pi / 2.0

    @property
    def legend_label(self) -> str:
        """Return a legend label derived from the simulated parameters."""
        if np.isclose(self.central_theta, 0.0):
            central_state = r"|0\rangle"
        elif np.isclose(self.central_theta, np.pi / 2.0):
            central_state = r"|+\rangle"
        elif np.isclose(self.central_theta, np.pi):
            central_state = r"|1\rangle"
        else:
            central_state = rf"|\theta_c={self.central_theta:g}\rangle"

        if np.isclose(self.bath_theta, 0.0):
            bath_state = r"|0\rangle^{\otimes N}"
        elif np.isclose(self.bath_theta, np.pi / 2.0):
            bath_state = r"|+\rangle^{\otimes N}"
        else:
            bath_state = rf"|\theta={self.bath_theta:g}\rangle^{{\otimes N}}"

        return (
            rf"{self.name}: $\Omega={self.Omega:g},\ \omega={self.omega:g},\ "
            rf"\theta_c={self.central_theta:g},\ "
            rf"{central_state}_{{\rm central}}{bath_state}_{{\rm bath}}$"
        )


# Configure one entry per curve.  Each pair of Omega/omega values is used in
# both figures, and legend_label above displays the values set here.
PROTOCOLS = (
    Protocol(
        Omega=0.0,
        omega=0.0,
        bath_theta=np.pi / 2.0,
        central_theta=np.pi / 2.0,
        marker="o",
        name="Ramsey",
    ),
    Protocol(
        Omega=7.0,
        omega=1.0,
        bath_theta=0.0,
        central_theta=np.pi / 2.0,
        marker="s",
        name="Driven",
    ),
    Protocol(
        Omega=3.0,
        omega=1.0,
        bath_theta=0.0,
        central_theta=np.pi / 2.0,
        marker="^",
        name="Driven",
    ),
)

def optimize_protocol(
    N: int,
    protocol: Protocol,
    tlist: np.ndarray,
    cfg: ComparisonConfig,
) -> tuple[float, float]:
    """Return ``t_opt`` and the optimal fixed-total-time QCRB."""
    bath_state = coherent_bath_state(N, protocol.bath_theta)

    bath_rhos_plus = get_bath_density_matrices(
        Omega_0=protocol.Omega,
        omega=protocol.omega,
        J=cfg.J_nominal + cfg.dJ,
        tlist=tlist,
        N=N,
        gamma=cfg.gamma,
        beta=cfg.beta,
        bath_state=bath_state,
        central_theta=protocol.central_theta,
    )
    bath_rhos_minus = get_bath_density_matrices(
        Omega_0=protocol.Omega,
        omega=protocol.omega,
        J=cfg.J_nominal - cfg.dJ,
        tlist=tlist,
        N=N,
        gamma=cfg.gamma,
        beta=cfg.beta,
        bath_state=bath_state,
        central_theta=protocol.central_theta,
    )
    qfi, _, _, _ = compute_bath_qfi_trajectory(
        bath_rhos_plus=bath_rhos_plus,
        bath_rhos_minus=bath_rhos_minus,
        dJ=cfg.dJ,
        tol=cfg.qfi_tol,
    )

    qcrb = np.full_like(qfi, np.inf, dtype=float)
    informative = qfi > cfg.qfi_tol
    qcrb[informative] = np.sqrt(
        (tlist[informative] + cfg.t_overhead) / qfi[informative]
    )

    optimal_index = int(np.argmin(qcrb))
    qcrb_opt = float(qcrb[optimal_index])
    if not np.isfinite(qcrb_opt):
        return float("nan"), float("inf")

    t_opt = float(tlist[optimal_index])
    return t_opt, qcrb_opt


def evolve_full_state_at_time(
    N: int,
    protocol: Protocol,
    J: float,
    time: float,
    cfg: ComparisonConfig,
) -> qt.Qobj:
    """Return the joint central-spin/bath state at one interrogation time.

    Noiseless evolution is propagated spectrally.  This is exact for the
    time-independent Hamiltonian and avoids ODE step-limit failures at the
    long optimal times used in this sweep.
    """
    if time < 0.0:
        raise ValueError("time must be non-negative")

    bath_state = coherent_bath_state(N, protocol.bath_theta)
    initial_central_state = central_spin_state(protocol.central_theta)
    bath_ket = qt.Qobj(bath_state, dims=[[N + 1], [1]])
    initial_state = qt.tensor(initial_central_state, bath_ket)
    hamiltonian = build_hamiltonian(
        Omega_0=protocol.Omega,
        omega=protocol.omega,
        J=J,
        N=N,
    )

    if time == 0.0:
        return initial_state

    if cfg.beta == 0.0 and cfg.gamma == 0.0:
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian.full())
        initial_vector = initial_state.full().ravel()
        eigenbasis_amplitudes = eigenvectors.conj().T @ initial_vector
        evolved_vector = eigenvectors @ (
            np.exp(-1j * eigenvalues * time) * eigenbasis_amplitudes
        )
        return qt.Qobj(evolved_vector, dims=initial_state.dims)

    collapse_operators = []
    operators = build_spin_operators(N)
    if cfg.beta > 0.0:
        collapse_operators.append(np.sqrt(cfg.beta) * operators["sx_s"])
    if cfg.gamma > 0.0:
        collapse_operators.append(np.sqrt(cfg.gamma) * operators["sz_s"])

    result = qt.mesolve(
        hamiltonian,
        initial_state,
        np.array([0.0, time]),
        c_ops=collapse_operators,
        e_ops=[],
        options={"nsteps": 100_000},
    )
    return result.states[-1]


def qfi_from_parameter_shifted_states(
    rho_plus: np.ndarray,
    rho_minus: np.ndarray,
    cfg: ComparisonConfig,
) -> float:
    """Compute QFI using the same centered finite difference as the bath sweep."""
    rho = 0.5 * (rho_plus + rho_minus)
    drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
    qfi, _ = qfi_from_rho_and_drho(rho, drho, tol=cfg.qfi_tol)
    return qfi


def qfis_at_operating_point(
    N: int,
    protocol: Protocol,
    time: float,
    cfg: ComparisonConfig,
) -> dict[str, float]:
    """Return global, bath, and central-spin QFI at the bath-optimal time."""
    if not np.isfinite(time):
        return {"global": np.nan, "bath": np.nan, "central": np.nan}

    state_plus = evolve_full_state_at_time(
        N=N,
        protocol=protocol,
        J=cfg.J_nominal + cfg.dJ,
        time=time,
        cfg=cfg,
    )
    state_minus = evolve_full_state_at_time(
        N=N,
        protocol=protocol,
        J=cfg.J_nominal - cfg.dJ,
        time=time,
        cfg=cfg,
    )
    global_plus = state_plus.proj() if state_plus.isket else state_plus
    global_minus = state_minus.proj() if state_minus.isket else state_minus

    shifted_states = {
        "global": (global_plus.full(), global_minus.full()),
        "bath": (global_plus.ptrace(1).full(), global_minus.ptrace(1).full()),
        "central": (global_plus.ptrace(0).full(), global_minus.ptrace(0).full()),
    }
    return {
        subsystem: qfi_from_parameter_shifted_states(rho_plus, rho_minus, cfg)
        for subsystem, (rho_plus, rho_minus) in shifted_states.items()
    }


def fit_power_law(
    N_values: np.ndarray,
    qcrb_values: np.ndarray,
) -> tuple[float, float, float]:
    """Fit ``QCRB_opt = A * N**p`` and return ``A``, ``p``, and log-space R²."""
    N_values = np.asarray(N_values, dtype=float)
    qcrb_values = np.asarray(qcrb_values, dtype=float)
    valid = (
        np.isfinite(N_values)
        & np.isfinite(qcrb_values)
        & (N_values > 0.0)
        & (qcrb_values > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        return float("nan"), float("nan"), float("nan")

    log_N = np.log(N_values[valid])
    log_qcrb = np.log(qcrb_values[valid])
    exponent, log_prefactor = np.polyfit(log_N, log_qcrb, 1)
    fitted_log_qcrb = log_prefactor + exponent * log_N

    residual_sum = float(np.sum((log_qcrb - fitted_log_qcrb) ** 2))
    total_sum = float(np.sum((log_qcrb - np.mean(log_qcrb)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0.0 else 1.0
    return float(np.exp(log_prefactor)), float(exponent), r_squared


def plot_comparison(
    results: dict[Protocol, dict[str, np.ndarray]],
    cfg: ComparisonConfig,
) -> Path:
    """Plot the optimized fixed-total-time QCRB against N."""
    figure, axis = plt.subplots(figsize=(9, 6))

    for protocol in PROTOCOLS:
        values = results[protocol]
        data_line, = axis.loglog(
            values["N"],
            values["qcrb_opt"],
            marker=protocol.marker,
            linewidth=1.8,
            label=protocol.legend_label,
        )
        prefactor, exponent, r_squared = fit_power_law(
            values["N"],
            values["qcrb_opt"],
        )
        if np.isfinite(exponent):
            fit_N = np.geomspace(
                float(np.min(values["N"])),
                float(np.max(values["N"])),
                200,
            )
            axis.loglog(
                fit_N,
                prefactor * fit_N**exponent,
                linestyle="--",
                color=data_line.get_color(),
                linewidth=1.5,
                label=(
                    rf"Fit: ${prefactor:.3g}N^{{{exponent:.3f}}}$ "
                    rf"($R^2={r_squared:.3f}$)"
                ),
            )
            print(
                f"{protocol.legend_label}: QCRB_opt = {prefactor:.6g} "
                f"* N^{exponent:.6f}, log-space R^2={r_squared:.6f}"
            )

    axis.set_xlabel(r"Number of bath spins $N$")
    axis.set_ylabel(
        r"$\min_t\sqrt{(t+t_{\mathrm{oh}})/F_Q(t)}$"
    )
    axis.set_title(
        "Fixed-total-time sensitivity, "
        rf"$t_{{\mathrm{{oh}}}}={cfg.t_overhead:g}$, "
        rf"optimized over ${cfg.t_min:g}\leq t\leq{cfg.t_max:g}$"
    )
    axis.grid(True, which="both", linestyle=":", alpha=0.8)
    axis.legend()
    figure.tight_layout()

    output_path = Path(cfg.output_figure)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_fq_fractions(
    results: dict[Protocol, dict[str, np.ndarray]],
    cfg: ComparisonConfig,
) -> Path:
    """Plot subsystem-to-global QFI ratios at each bath-optimal time."""
    figure, axis = plt.subplots(figsize=(10, 6.5))

    for protocol in PROTOCOLS:
        values = results[protocol]
        bath_line, = axis.plot(
            values["N"],
            values["fq_bath_over_global"],
            marker=protocol.marker,
            linewidth=1.8,
            label=(
                protocol.legend_label
                + r": $F_Q^{\mathrm{bath}}/F_Q^{\mathrm{global}}$"
            ),
        )
        axis.plot(
            values["N"],
            values["fq_central_over_global"],
            marker=protocol.marker,
            markerfacecolor="none",
            linestyle="--",
            color=bath_line.get_color(),
            linewidth=1.8,
            label=(
                protocol.legend_label
                + r": $F_Q^{\mathrm{central}}/F_Q^{\mathrm{global}}$"
            ),
        )

    axis.axhline(1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    axis.set_xlabel(r"Number of bath spins $N$")
    axis.set_ylabel(r"Fraction of global QFI")
    axis.set_title(r"QFI fractions at each protocol's bath-optimal $t$")
    axis.set_ylim(bottom=0.0)
    axis.grid(True, linestyle=":", alpha=0.8)
    axis.legend(fontsize="small")
    figure.tight_layout()

    output_path = Path(cfg.output_fraction_figure)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    cfg = ComparisonConfig()
    tlist = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    results: dict[Protocol, dict[str, np.ndarray]] = {}

    for protocol in PROTOCOLS:
        t_opt_values = []
        qcrb_opt_values = []
        fq_bath_over_global_values = []
        fq_central_over_global_values = []

        print(f"\n{protocol.legend_label}")
        for N in cfg.N_values:
            t_opt, qcrb_opt = optimize_protocol(
                N=N,
                protocol=protocol,
                tlist=tlist,
                cfg=cfg,
            )
            t_opt_values.append(t_opt)
            qcrb_opt_values.append(qcrb_opt)
            qfis = qfis_at_operating_point(
                N=N,
                protocol=protocol,
                time=t_opt,
                cfg=cfg,
            )
            if qfis["global"] > cfg.qfi_tol:
                fq_bath_over_global = qfis["bath"] / qfis["global"]
                fq_central_over_global = qfis["central"] / qfis["global"]
            else:
                fq_bath_over_global = np.nan
                fq_central_over_global = np.nan
            fq_bath_over_global_values.append(fq_bath_over_global)
            fq_central_over_global_values.append(fq_central_over_global)
            print(
                f"N={N:3d}: t_opt={t_opt:7.3f}, "
                f"time-normalized QCRB={qcrb_opt:.6e}, "
                f"FQ_bath/FQ_global={fq_bath_over_global:.6f}, "
                f"FQ_central/FQ_global={fq_central_over_global:.6f}"
            )

        results[protocol] = {
            "N": np.asarray(cfg.N_values, dtype=int),
            "t_opt": np.asarray(t_opt_values),
            "qcrb_opt": np.asarray(qcrb_opt_values),
            "fq_bath_over_global": np.asarray(fq_bath_over_global_values),
            "fq_central_over_global": np.asarray(fq_central_over_global_values),
        }

    output_path = plot_comparison(results, cfg)
    fraction_output_path = plot_fq_fractions(results, cfg)
    print(f"\nSaved comparison plot to {output_path}")
    print(f"Saved QFI-fraction plot to {fraction_output_path}")


if __name__ == "__main__":
    main()
