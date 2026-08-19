"""Compare driven and naive-Ramsey sensing trajectories versus time.

The noiseless driven central-spin model is

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x,

where ``S_i = 2 * jmat(N / 2, i)``.  Quantum Fisher information (QFI) for
estimating ``J`` is evaluated by a centered finite difference for the global
state, reduced bath, and reduced central spin.  The reduced bath also gives
the classical Fisher information available from the means of ``S_x``,
``S_y``, and ``S_z``, plus the additive ``S_z``/``S_y`` information for
independent experimental runs.
The nominal trajectory provides the bath-spin expectations and fixed-phase
harmonic fits.  A second figure column shows a naive Ramsey experiment with
the central spin in ``|1>`` and the bath in ``|+>**N`` under zero drives.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import minimize_scalar

try:
    from CRB.crb_core import (
        build_bath_operators,
        build_hamiltonian,
        central_spin_state,
        coherent_bath_state,
        observable_moment_fisher,
        qfi_vectorized,
        save_plot,
    )
except ModuleNotFoundError:  # Allow: python CRB/plot_qfi_and_spin_vs_time.py
    from crb_core import (
        build_bath_operators,
        build_hamiltonian,
        central_spin_state,
        coherent_bath_state,
        observable_moment_fisher,
        qfi_vectorized,
        save_plot,
    )


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Physics, sampling, fitting, and visualization parameters."""

    N: int = 15
    Omega: float = 0.0
    omega: float = 1
    J_nominal: float = 1.0
    dJ: float = 1e-3

    t_min: float = 0.0
    t_max: float = 40
    n_steps: int = 401

    central_theta_rad: float = np.pi / 2.0
    central_phi_rad: float = 0.0
    bath_theta_rad: float = 0.0
    bath_phi_rad: float = 0.0

    ramsey_Omega: float = 0.0
    ramsey_omega: float = 0.0
    ramsey_central_theta_rad: float = np.pi
    ramsey_central_phi_rad: float = 0.0
    ramsey_bath_theta_rad: float = np.pi / 2.0
    ramsey_bath_phi_rad: float = 0.0

    qfi_tol: float = 1e-12
    classical_fisher_variance_floor: float = 1e-12
    fit_frequency_grid_points: int = 5001

    figure_width_in: float = 16.0
    figure_height_in: float = 12.0
    figure_dpi: int = 200
    figure_format: str = "png"
    show_figure: bool = True


@dataclass(frozen=True, slots=True)
class StateReductions:
    """Density matrices for the global system and its two subsystems."""

    global_state: np.ndarray
    bath: np.ndarray
    central: np.ndarray


@dataclass(frozen=True, slots=True)
class Protocol:
    """Hamiltonian drives and initial-state angles for one protocol."""

    name: str
    Omega: float
    omega: float
    central_theta_rad: float
    central_phi_rad: float
    bath_theta_rad: float
    bath_phi_rad: float


@dataclass(frozen=True, slots=True)
class HarmonicFit:
    """Result of a fixed-phase ``A*cos(a*t)`` or ``A*sin(a*t)`` fit."""

    amplitude: float
    angular_frequency: float
    r_squared: float
    values: np.ndarray


def validate_config(cfg: SimulationConfig) -> None:
    """Fail early when a configuration cannot produce a meaningful plot."""
    if cfg.N < 1:
        raise ValueError("N must be a positive integer")
    if not all(
        np.isfinite(value)
        for value in (
            cfg.Omega,
            cfg.omega,
            cfg.J_nominal,
            cfg.dJ,
            cfg.t_min,
            cfg.t_max,
            cfg.central_theta_rad,
            cfg.central_phi_rad,
            cfg.bath_theta_rad,
            cfg.bath_phi_rad,
            cfg.ramsey_Omega,
            cfg.ramsey_omega,
            cfg.ramsey_central_theta_rad,
            cfg.ramsey_central_phi_rad,
            cfg.ramsey_bath_theta_rad,
            cfg.ramsey_bath_phi_rad,
            cfg.qfi_tol,
            cfg.classical_fisher_variance_floor,
            cfg.figure_width_in,
            cfg.figure_height_in,
        )
    ):
        raise ValueError("all floating-point configuration values must be finite")
    if cfg.dJ <= 0.0:
        raise ValueError("dJ must be positive")
    if cfg.t_min < 0.0:
        raise ValueError("t_min must be non-negative")
    if cfg.t_max <= cfg.t_min:
        raise ValueError("t_max must be greater than t_min")
    if cfg.n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if cfg.qfi_tol <= 0.0:
        raise ValueError("qfi_tol must be positive")
    if cfg.classical_fisher_variance_floor <= 0.0:
        raise ValueError("classical_fisher_variance_floor must be positive")
    if cfg.fit_frequency_grid_points < 3:
        raise ValueError("fit_frequency_grid_points must be at least 3")
    if cfg.figure_width_in <= 0.0 or cfg.figure_height_in <= 0.0:
        raise ValueError("figure dimensions must be positive")
    if cfg.figure_dpi <= 0:
        raise ValueError("figure_dpi must be positive")
    if not cfg.figure_format or cfg.figure_format.startswith("."):
        raise ValueError("figure_format must be an extension without a leading dot")
    if re.search(r"[^A-Za-z0-9]", cfg.figure_format):
        raise ValueError("figure_format must contain only letters and digits")


def time_grid(cfg: SimulationConfig) -> np.ndarray:
    """Return the uniformly sampled interrogation-time grid."""
    return np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)


def driven_protocol(cfg: SimulationConfig) -> Protocol:
    """Return the driven protocol encoded by ``cfg``."""
    return Protocol(
        name="Driven",
        Omega=cfg.Omega,
        omega=cfg.omega,
        central_theta_rad=cfg.central_theta_rad,
        central_phi_rad=cfg.central_phi_rad,
        bath_theta_rad=cfg.bath_theta_rad,
        bath_phi_rad=cfg.bath_phi_rad,
    )


def ramsey_protocol(cfg: SimulationConfig) -> Protocol:
    """Return the naive Ramsey protocol encoded by ``cfg``."""
    return Protocol(
        name="Naive Ramsey",
        Omega=cfg.ramsey_Omega,
        omega=cfg.ramsey_omega,
        central_theta_rad=cfg.ramsey_central_theta_rad,
        central_phi_rad=cfg.ramsey_central_phi_rad,
        bath_theta_rad=cfg.ramsey_bath_theta_rad,
        bath_phi_rad=cfg.ramsey_bath_phi_rad,
    )


def build_initial_state(
    cfg: SimulationConfig,
    protocol: Protocol | None = None,
) -> qt.Qobj:
    """Return one protocol's central/bath spin-coherent product state."""
    protocol = driven_protocol(cfg) if protocol is None else protocol
    central_ket = central_spin_state(
        theta=protocol.central_theta_rad,
        phi=protocol.central_phi_rad,
    )
    bath_vector = coherent_bath_state(
        cfg.N,
        theta=protocol.bath_theta_rad,
        phi=protocol.bath_phi_rad,
    )
    bath_ket = qt.Qobj(bath_vector, dims=[[cfg.N + 1], [1]])
    return qt.tensor(central_ket, bath_ket)


def evolve_state_vectors(
    cfg: SimulationConfig,
    J: float,
    times: np.ndarray,
    protocol: Protocol | None = None,
) -> np.ndarray:
    """Return exact pure-state vectors at all ``times`` for one coupling."""
    protocol = driven_protocol(cfg) if protocol is None else protocol
    hamiltonian = build_hamiltonian(
        Omega_0=protocol.Omega,
        omega=protocol.omega,
        J=J,
        N=cfg.N,
    ).full()
    initial_vector = build_initial_state(cfg, protocol).full().ravel()

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    initial_eigenbasis = eigenvectors.conj().T @ initial_vector
    phases = np.exp(-1j * np.outer(eigenvalues, times))
    return (eigenvectors @ (initial_eigenbasis[:, None] * phases)).T


def density_matrices(state_vector: np.ndarray, N: int) -> StateReductions:
    """Trace one pure joint state into global, bath, and central densities."""
    expected_shape = (2 * (N + 1),)
    state_vector = np.asarray(state_vector, dtype=complex)
    if state_vector.shape != expected_shape:
        raise ValueError(
            f"state_vector must have shape {expected_shape}, got {state_vector.shape}"
        )

    amplitudes = state_vector.reshape(2, N + 1)
    return StateReductions(
        global_state=np.outer(state_vector, state_vector.conj()),
        bath=amplitudes.T @ amplitudes.conj(),
        central=amplitudes @ amplitudes.conj().T,
    )


def protocol_fisher_information_trajectories(
    cfg: SimulationConfig,
    times: np.ndarray,
    protocol: Protocol,
    classical_components: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute one protocol's quantum and moment-based classical FI."""
    states_plus = evolve_state_vectors(
        cfg,
        cfg.J_nominal + cfg.dJ,
        times,
        protocol,
    )
    states_minus = evolve_state_vectors(
        cfg,
        cfg.J_nominal - cfg.dJ,
        times,
        protocol,
    )
    qfi = {
        subsystem: np.empty(len(times), dtype=float)
        for subsystem in ("global", "bath", "central")
    }
    density_attributes = {
        "global": "global_state",
        "bath": "bath",
        "central": "central",
    }
    bath_operators = build_bath_operators(cfg.N)
    available_observables = {
        "Sx": bath_operators["Jx"],
        "Sy": bath_operators["Jy"],
        "Sz": bath_operators["Jz"],
    }
    unknown_components = set(classical_components) - set(available_observables)
    if unknown_components:
        names = ", ".join(sorted(unknown_components))
        raise ValueError(f"unknown classical observable components: {names}")
    classical_observables = {
        component: available_observables[component]
        for component in classical_components
    }
    classical_fisher = {
        component: np.empty(len(times), dtype=float)
        for component in classical_observables
    }

    for index, (state_plus, state_minus) in enumerate(
        zip(states_plus, states_minus)
    ):
        plus = density_matrices(state_plus, cfg.N)
        minus = density_matrices(state_minus, cfg.N)
        for subsystem in qfi:
            density_attribute = density_attributes[subsystem]
            rho_plus = getattr(plus, density_attribute)
            rho_minus = getattr(minus, density_attribute)
            rho = 0.5 * (rho_plus + rho_minus)
            drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
            qfi[subsystem][index] = qfi_vectorized(
                rho,
                drho,
                tol=cfg.qfi_tol,
            )
        rho_bath = 0.5 * (plus.bath + minus.bath)
        drho_bath = (plus.bath - minus.bath) / (2.0 * cfg.dJ)
        for component, observable in classical_observables.items():
            classical_fisher[component][index] = observable_moment_fisher(
                rho_bath,
                drho_bath,
                observable,
                var_floor=cfg.classical_fisher_variance_floor,
            )
    return qfi, classical_fisher


def fisher_information_trajectories(
    cfg: SimulationConfig,
    times: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute driven-protocol QFI and spin-observable classical FI."""
    qfi, classical_fisher = protocol_fisher_information_trajectories(
        cfg,
        times,
        driven_protocol(cfg),
        ("Sx", "Sz", "Sy"),
    )
    classical_fisher["Sz+Sy"] = (
        classical_fisher["Sz"] + classical_fisher["Sy"]
    )
    return qfi, classical_fisher


def qfi_trajectories(
    cfg: SimulationConfig,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return only the QFI trajectories for callers that do not need CFI."""
    qfi, _ = fisher_information_trajectories(cfg, times)
    return qfi


def spin_expectation_trajectories(
    cfg: SimulationConfig,
    times: np.ndarray,
    protocol: Protocol | None = None,
    components: tuple[str, ...] = ("Sx", "Sy", "Sz"),
) -> dict[str, np.ndarray]:
    """Compute selected nominal collective-bath spin expectations."""
    protocol = driven_protocol(cfg) if protocol is None else protocol
    states = evolve_state_vectors(cfg, cfg.J_nominal, times, protocol)
    bath_operators = build_bath_operators(cfg.N)
    available_operators = {
        "Sx": bath_operators["Jx"],
        "Sy": bath_operators["Jy"],
        "Sz": bath_operators["Jz"],
    }
    unknown_components = set(components) - set(available_operators)
    if unknown_components:
        names = ", ".join(sorted(unknown_components))
        raise ValueError(f"unknown spin components: {names}")
    operators = {
        component: available_operators[component] for component in components
    }
    expectations = {
        name: np.empty(len(times), dtype=float) for name in operators
    }

    for index, state in enumerate(states):
        rho_bath = density_matrices(state, cfg.N).bath
        for name, operator in operators.items():
            expectations[name][index] = float(
                np.real(np.trace(rho_bath @ operator))
            )
    return expectations


def fit_fixed_phase_harmonic(
    times: np.ndarray,
    values: np.ndarray,
    function_name: str,
    frequency_grid_points: int = 5001,
) -> HarmonicFit:
    """Fit ``A*cos(a*t)`` or ``A*sin(a*t)`` without a frequency guess."""
    if function_name not in {"cos", "sin"}:
        raise ValueError("function_name must be 'cos' or 'sin'")
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("times and values must be one-dimensional with equal shapes")
    if frequency_grid_points < 3:
        raise ValueError("frequency_grid_points must be at least 3")

    sampling_intervals = np.diff(times)
    if np.any(sampling_intervals <= 0.0):
        raise ValueError("times must be strictly increasing")

    basis_function = np.cos if function_name == "cos" else np.sin
    nyquist_angular_frequency = np.pi / float(np.min(sampling_intervals))
    frequency_grid = np.linspace(
        0.0,
        nyquist_angular_frequency,
        frequency_grid_points,
    )
    basis_grid = basis_function(np.outer(frequency_grid, times))
    projections = basis_grid @ values
    norms = np.sum(basis_grid * basis_grid, axis=1)
    residual_sums = np.full_like(frequency_grid, np.inf)
    valid = norms > np.finfo(float).eps
    residual_sums[valid] = (
        np.dot(values, values) - projections[valid] ** 2 / norms[valid]
    )
    best_grid_index = int(np.argmin(residual_sums))
    lower_index = max(best_grid_index - 1, 0)
    upper_index = min(best_grid_index + 1, len(frequency_grid) - 1)

    def residual_sum(angular_frequency: float) -> float:
        basis = basis_function(angular_frequency * times)
        norm = float(np.dot(basis, basis))
        if norm <= np.finfo(float).eps:
            return float("inf")
        projection = float(np.dot(basis, values))
        return float(np.dot(values, values) - projection * projection / norm)

    refinement = minimize_scalar(
        residual_sum,
        bounds=(
            float(frequency_grid[lower_index]),
            float(frequency_grid[upper_index]),
        ),
        method="bounded",
    )
    angular_frequency = (
        float(refinement.x)
        if refinement.success
        else float(frequency_grid[best_grid_index])
    )
    basis = basis_function(angular_frequency * times)
    amplitude = float(np.dot(basis, values) / np.dot(basis, basis))
    fitted_values = amplitude * basis
    residual_total = float(np.sum((values - fitted_values) ** 2))
    centered_total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = (
        1.0 - residual_total / centered_total if centered_total > 0.0 else 1.0
    )
    return HarmonicFit(
        amplitude=amplitude,
        angular_frequency=angular_frequency,
        r_squared=r_squared,
        values=fitted_values,
    )


def fit_spin_expectations(
    times: np.ndarray,
    spin_expectations: dict[str, np.ndarray],
    cfg: SimulationConfig | None = None,
) -> dict[str, HarmonicFit]:
    """Fit ``Sz`` to a cosine and ``Sy`` to a sine."""
    grid_points = (
        cfg.fit_frequency_grid_points if cfg is not None else 5001
    )
    return {
        "Sz": fit_fixed_phase_harmonic(
            times,
            spin_expectations["Sz"],
            "cos",
            grid_points,
        ),
        "Sy": fit_fixed_phase_harmonic(
            times,
            spin_expectations["Sy"],
            "sin",
            grid_points,
        ),
    }


def format_number(value: float | int) -> str:
    """Format a number compactly and deterministically for a filename."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.12g}"


def format_angle(value: float) -> str:
    """Use readable names for common angles and compact numbers otherwise."""
    common_angles = {
        0.0: "0",
        np.pi / 2.0: "pi_2",
        np.pi: "pi",
        2.0 * np.pi: "2pi",
    }
    for angle, label in common_angles.items():
        if np.isclose(value, angle):
            return label
    return format_number(value)


def sanitize_tag(text: str) -> str:
    """Replace characters that are unsafe or awkward in filenames."""
    return re.sub(r"[^A-Za-z0-9._=+-]", "-", text).strip(".-")


def parameter_tags(cfg: SimulationConfig) -> str:
    """Encode every configuration field in compact, stable filename tags."""
    encoded_fields = {
        "N",
        "Omega",
        "omega",
        "J_nominal",
        "dJ",
        "t_min",
        "t_max",
        "n_steps",
        "central_theta_rad",
        "central_phi_rad",
        "bath_theta_rad",
        "bath_phi_rad",
        "ramsey_Omega",
        "ramsey_omega",
        "ramsey_central_theta_rad",
        "ramsey_central_phi_rad",
        "ramsey_bath_theta_rad",
        "ramsey_bath_phi_rad",
        "qfi_tol",
        "classical_fisher_variance_floor",
        "fit_frequency_grid_points",
        "figure_width_in",
        "figure_height_in",
        "figure_dpi",
        "figure_format",
        "show_figure",
    }
    config_fields = {field.name for field in fields(cfg)}
    if encoded_fields != config_fields:
        missing = sorted(config_fields - encoded_fields)
        extra = sorted(encoded_fields - config_fields)
        raise RuntimeError(
            f"filename-tag field mismatch; missing={missing}, extra={extra}"
        )

    tags = (
        f"N={cfg.N}",
        (
            f"D=Om{format_number(cfg.Omega)}-om{format_number(cfg.omega)}-"
            f"tc{format_angle(cfg.central_theta_rad)}-"
            f"pc{format_angle(cfg.central_phi_rad)}-"
            f"tb{format_angle(cfg.bath_theta_rad)}-"
            f"pb{format_angle(cfg.bath_phi_rad)}"
        ),
        (
            f"R=Om{format_number(cfg.ramsey_Omega)}-"
            f"om{format_number(cfg.ramsey_omega)}-"
            f"tc{format_angle(cfg.ramsey_central_theta_rad)}-"
            f"pc{format_angle(cfg.ramsey_central_phi_rad)}-"
            f"tb{format_angle(cfg.ramsey_bath_theta_rad)}-"
            f"pb{format_angle(cfg.ramsey_bath_phi_rad)}"
        ),
        (
            f"J={format_number(cfg.J_nominal)}-"
            f"dJ{format_number(cfg.dJ)}"
        ),
        (
            f"t={format_number(cfg.t_min)}-"
            f"{format_number(cfg.t_max)}-n{cfg.n_steps}"
        ),
        (
            f"tol={format_number(cfg.qfi_tol)}-"
            f"vf{format_number(cfg.classical_fisher_variance_floor)}"
        ),
        f"fit=n{cfg.fit_frequency_grid_points}",
        (
            f"fig={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)}-dpi{cfg.figure_dpi}-"
            f"{cfg.figure_format}-show{int(cfg.show_figure)}"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def output_path(cfg: SimulationConfig) -> Path:
    """Return the parameter-rich filename passed to the shared plot saver."""
    return Path(
        f"driven-vs-ramsey__{parameter_tags(cfg)}.{cfg.figure_format.lower()}"
    )


def plot_trajectories(
    times: np.ndarray,
    qfi: dict[str, np.ndarray],
    classical_fisher: dict[str, np.ndarray],
    spin_expectations: dict[str, np.ndarray],
    spin_fits: dict[str, HarmonicFit],
    ramsey_qfi: dict[str, np.ndarray],
    ramsey_classical_fisher: dict[str, np.ndarray],
    ramsey_spin_expectations: dict[str, np.ndarray],
    cfg: SimulationConfig,
) -> Path:
    """Plot driven trajectories beside the naive Ramsey comparison."""
    figure = plt.figure(figsize=(cfg.figure_width_in, cfg.figure_height_in))
    grid = figure.add_gridspec(4, 2, hspace=0.08, wspace=0.28)
    qfi_axis = figure.add_subplot(grid[0, 0])
    rate_axis = figure.add_subplot(grid[1, 0], sharex=qfi_axis)
    classical_axis = figure.add_subplot(grid[2, 0], sharex=qfi_axis)
    spin_axis = figure.add_subplot(grid[3, 0], sharex=qfi_axis)
    ramsey_information_axis = figure.add_subplot(grid[:2, 1])
    ramsey_spin_axis = figure.add_subplot(
        grid[2:, 1],
        sharex=ramsey_information_axis,
    )

    subsystem_labels = {
        "global": r"$F_Q^{\mathrm{global}}$",
        "bath": r"$F_Q^{\mathrm{bath}}$",
        "central": r"$F_Q^{\mathrm{central}}$",
    }
    for subsystem, label in subsystem_labels.items():
        qfi_axis.plot(times, qfi[subsystem], linewidth=2.0, label=label)
        qfi_per_time = np.divide(
            qfi[subsystem],
            times,
            out=np.full_like(qfi[subsystem], np.nan),
            where=times > 0.0,
        )
        rate_axis.plot(
            times,
            qfi_per_time,
            linewidth=2.0,
            label=rf"{label[:-1]}/t$",
        )

    qfi_axis.set_ylabel(r"Quantum Fisher information $F_Q(t)$")
    qfi_axis.set_title(
        rf"Driven: $\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, "
        rf"$\theta_c={cfg.central_theta_rad:.3g}$, "
        rf"$\theta_b={cfg.bath_theta_rad:.3g}$"
    )
    rate_axis.set_ylabel(r"QFI rate $F_Q(t)/t$")
    for axis in (qfi_axis, rate_axis):
        axis.grid(True, linestyle=":", alpha=0.8)
        axis.legend()

    classical_axis.plot(
        times,
        classical_fisher["Sx"],
        linewidth=2.0,
        label=r"$F_C[\langle S_x\rangle]$",
    )
    classical_axis.plot(
        times,
        classical_fisher["Sz"],
        linewidth=2.0,
        label=r"$F_C[\langle S_z\rangle]$",
    )
    classical_axis.plot(
        times,
        classical_fisher["Sy"],
        linewidth=2.0,
        label=r"$F_C[\langle S_y\rangle]$",
    )
    classical_axis.plot(
        times,
        classical_fisher["Sz+Sy"],
        linewidth=2.0,
        label=(
            r"$F_C[\langle S_z\rangle]+F_C[\langle S_y\rangle]$"
            " (separate runs)"
        ),
    )
    classical_axis.set_ylabel("Classical Fisher information")
    classical_axis.grid(True, linestyle=":", alpha=0.8)
    classical_axis.legend()

    spin_axis.plot(
        times,
        spin_expectations["Sx"],
        linewidth=1.8,
        label=r"$\langle S_x\rangle$",
    )
    sy_line = spin_axis.plot(
        times,
        spin_expectations["Sy"],
        linewidth=1.8,
        label=r"$\langle S_y\rangle$",
    )[0]
    sz_line = spin_axis.plot(
        times,
        spin_expectations["Sz"],
        linewidth=1.8,
        label=r"$\langle S_z\rangle$",
    )[0]
    for component, function_name, color in (
        ("Sz", "cos", sz_line.get_color()),
        ("Sy", "sin", sy_line.get_color()),
    ):
        fit = spin_fits[component]
        spin_axis.plot(
            times,
            fit.values,
            color=color,
            linestyle="--",
            linewidth=2.0,
            label=(
                rf"Fit: ${fit.amplitude:.3g}\{function_name}"
                rf"({fit.angular_frequency:.3g}t)$"
            ),
        )
    spin_axis.set_xlabel(r"Interrogation time $t$")
    spin_axis.set_ylabel("Collective-bath spin expectation")
    spin_axis.grid(True, linestyle=":", alpha=0.8)
    spin_axis.legend()

    ramsey_information_axis.plot(
        times,
        ramsey_qfi["bath"],
        linewidth=2.0,
        label=r"$F_Q^{\mathrm{bath}}$",
    )
    ramsey_information_axis.plot(
        times,
        ramsey_classical_fisher["Sx"],
        linewidth=2.0,
        label=r"$F_C[\langle S_x\rangle]$",
    )
    ramsey_information_axis.plot(
        times,
        ramsey_classical_fisher["Sy"],
        linewidth=2.0,
        label=r"$F_C[\langle S_y\rangle]$",
    )
    ramsey_information_axis.set_ylabel("Fisher information")
    ramsey_information_axis.set_title(
        "Naive Ramsey: "
        rf"$\Omega={cfg.ramsey_Omega:g}$, $\omega={cfg.ramsey_omega:g}$, "
        r"$|1\rangle_c|+\rangle_b^{\otimes N}$"
    )
    ramsey_information_axis.grid(True, linestyle=":", alpha=0.8)
    ramsey_information_axis.legend()

    ramsey_spin_axis.plot(
        times,
        ramsey_spin_expectations["Sx"],
        linewidth=2.0,
        label=r"$\langle S_x\rangle$",
    )
    ramsey_spin_axis.plot(
        times,
        ramsey_spin_expectations["Sy"],
        linewidth=2.0,
        label=r"$\langle S_y\rangle$",
    )
    ramsey_spin_axis.set_xlabel(r"Interrogation time $t$")
    ramsey_spin_axis.set_ylabel("Collective-bath spin expectation")
    ramsey_spin_axis.grid(True, linestyle=":", alpha=0.8)
    ramsey_spin_axis.legend()

    for axis in (qfi_axis, rate_axis, classical_axis):
        axis.tick_params(labelbottom=False)
    ramsey_information_axis.tick_params(labelbottom=False)
    figure.suptitle(rf"$N={cfg.N}$, $J={cfg.J_nominal:g}$")
    figure.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.98)

    path = output_path(cfg)
    path = save_plot(
        figure,
        path,
        metadata={
            "config": cfg,
            "time_values": times,
            "driven_qfi": qfi,
            "driven_classical_fisher": classical_fisher,
            "driven_spin_expectations": spin_expectations,
            "driven_spin_fits": spin_fits,
            "ramsey_qfi": ramsey_qfi,
            "ramsey_classical_fisher": ramsey_classical_fisher,
            "ramsey_spin_expectations": ramsey_spin_expectations,
        },
        script_path=__file__,
        format=cfg.figure_format,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def parse_config(argv: list[str] | None = None) -> SimulationConfig:
    """Parse command-line overrides into the typed configuration."""
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument("--t-min", type=float, default=defaults.t_min)
    parser.add_argument("--t-max", type=float, default=defaults.t_max)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument(
        "--central-theta-rad",
        type=float,
        default=defaults.central_theta_rad,
    )
    parser.add_argument(
        "--central-phi-rad",
        type=float,
        default=defaults.central_phi_rad,
    )
    parser.add_argument(
        "--bath-theta-rad",
        type=float,
        default=defaults.bath_theta_rad,
    )
    parser.add_argument(
        "--bath-phi-rad",
        type=float,
        default=defaults.bath_phi_rad,
    )
    parser.add_argument(
        "--ramsey-Omega",
        dest="ramsey_Omega",
        type=float,
        default=defaults.ramsey_Omega,
    )
    parser.add_argument(
        "--ramsey-omega",
        type=float,
        default=defaults.ramsey_omega,
    )
    parser.add_argument(
        "--ramsey-central-theta-rad",
        type=float,
        default=defaults.ramsey_central_theta_rad,
    )
    parser.add_argument(
        "--ramsey-central-phi-rad",
        type=float,
        default=defaults.ramsey_central_phi_rad,
    )
    parser.add_argument(
        "--ramsey-bath-theta-rad",
        type=float,
        default=defaults.ramsey_bath_theta_rad,
    )
    parser.add_argument(
        "--ramsey-bath-phi-rad",
        type=float,
        default=defaults.ramsey_bath_phi_rad,
    )
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument(
        "--classical-fisher-variance-floor",
        type=float,
        default=defaults.classical_fisher_variance_floor,
    )
    parser.add_argument(
        "--fit-frequency-grid-points",
        type=int,
        default=defaults.fit_frequency_grid_points,
    )
    parser.add_argument("--figure-width-in", type=float, default=defaults.figure_width_in)
    parser.add_argument(
        "--figure-height-in",
        type=float,
        default=defaults.figure_height_in,
    )
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument("--figure-format", default=defaults.figure_format)
    parser.add_argument(
        "--show",
        dest="show_figure",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_figure,
    )
    return SimulationConfig(**vars(parser.parse_args(argv)))


def print_summary(
    times: np.ndarray,
    qfi: dict[str, np.ndarray],
    classical_fisher: dict[str, np.ndarray],
    ramsey_qfi: dict[str, np.ndarray],
    ramsey_classical_fisher: dict[str, np.ndarray],
    spin_fits: dict[str, HarmonicFit],
    path: Path,
) -> None:
    """Print numerical maxima, fit diagnostics, and the output location."""
    for subsystem in ("global", "bath", "central"):
        maximum_index = int(np.argmax(qfi[subsystem]))
        print(
            f"Maximum {subsystem} QFI: {qfi[subsystem][maximum_index]:.6e} "
            f"at t={times[maximum_index]:.6g}"
        )
    for component in ("Sx", "Sz", "Sy", "Sz+Sy"):
        maximum_index = int(np.argmax(classical_fisher[component]))
        if component == "Sz+Sy":
            measurement_label = "additive <Sz> + <Sy>"
        else:
            measurement_label = f"<{component}>"
        print(
            f"Maximum {measurement_label} classical FI: "
            f"{classical_fisher[component][maximum_index]:.6e} "
            f"at t={times[maximum_index]:.6g}"
        )
    ramsey_information = {
        "bath QFI": ramsey_qfi["bath"],
        "<Sx> classical FI": ramsey_classical_fisher["Sx"],
        "<Sy> classical FI": ramsey_classical_fisher["Sy"],
    }
    for label, values in ramsey_information.items():
        maximum_index = int(np.argmax(values))
        print(
            f"Maximum Ramsey {label}: {values[maximum_index]:.6e} "
            f"at t={times[maximum_index]:.6g}"
        )
    for component, amplitude_name, frequency_name in (
        ("Sz", "A", "a"),
        ("Sy", "B", "b"),
    ):
        fit = spin_fits[component]
        print(
            f"<{component}> fit: {amplitude_name}={fit.amplitude:.8g}, "
            f"{frequency_name}={fit.angular_frequency:.8g}, "
            f"R^2={fit.r_squared:.6f}"
        )
    print(f"Saved plot to {path}")


def main(argv: list[str] | None = None) -> Path:
    """Run the configured analysis, save its plot, and return its path."""
    cfg = parse_config(argv)
    validate_config(cfg)
    times = time_grid(cfg)
    qfi, classical_fisher = fisher_information_trajectories(cfg, times)
    spin_expectations = spin_expectation_trajectories(cfg, times)
    spin_fits = fit_spin_expectations(times, spin_expectations, cfg)
    ramsey = ramsey_protocol(cfg)
    ramsey_qfi, ramsey_classical_fisher = (
        protocol_fisher_information_trajectories(
            cfg,
            times,
            ramsey,
            ("Sx", "Sy"),
        )
    )
    ramsey_spin_expectations = spin_expectation_trajectories(
        cfg,
        times,
        ramsey,
        ("Sx", "Sy"),
    )
    path = plot_trajectories(
        times,
        qfi,
        classical_fisher,
        spin_expectations,
        spin_fits,
        ramsey_qfi,
        ramsey_classical_fisher,
        ramsey_spin_expectations,
        cfg,
    )
    print_summary(
        times,
        qfi,
        classical_fisher,
        ramsey_qfi,
        ramsey_classical_fisher,
        spin_fits,
        path,
    )
    return path


if __name__ == "__main__":
    main()
