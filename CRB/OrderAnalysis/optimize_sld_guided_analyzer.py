"""SLD/tangent-guided optimizer for a fixed projective bath-Sx readout.

The protocol is

    |psi0> -- H_s(J), t_s --> -- H_a({u_c,u_b}; J0), T_a --> Sx projectors,

where every analyzer segment is constrained to

    H_a,k = u_c[k] X_c + u_b[k] S_x + J0 Z_c S_z.

In particular, the interaction coefficient is always ``+J0``.  The analyzer
is a J-independent measurement transformation, not an inverse sensing
evolution.  The sensing tangent is calculated with ``scipy.linalg``'s exact
Frechet derivative; finite differences are used only to validate accepted
solutions through probability derivatives and Hellinger curvature.

This module implements the first experiment in the design specification:
select (or configure) one high-bath-QFI-rate sensing time and optimize only the
analyzer controls for a K-segment complexity ladder.  It does not jointly
optimize sensing time and analyzer duration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations_with_replacement, permutations
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.special import logsumexp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from CRB.crb_core import (  # noqa: E402
    build_bath_operators,
    build_spin_operators,
    central_spin_state,
    coherent_bath_state,
    qfi_from_rho_and_drho,
    save_plot,
)


@dataclass(frozen=True, slots=True)
class AnalyzerConfig:
    """Physics, robust-control, validation, and output configuration."""

    N: int = 15
    Omega: float = 2.5
    omega: float = 1.0
    J0: float = 1.0

    sensing_time: float | None = None
    sensing_time_min: float = 0.05
    sensing_time_max: float = 30.0
    n_sensing_times: int = 121

    analyzer_duration: float = 5.0
    segment_counts: tuple[int, ...] = (1, 2, 4, 8)
    central_control_max: float = 5.0
    bath_control_max: float = 2.0

    robust_fraction: float = 0.05
    n_training_J: int = 5
    n_heldout_J: int = 21
    softmin_sharpness: float = 10.0

    n_random_seeds: int = 2
    random_seed: int = 20260809
    random_seed_scale: float = 0.15
    optimizer_max_iterations: int = 150
    optimizer_ftol: float = 1e-10
    optimizer_gtol: float = 1e-7
    power_penalty: float = 1e-4
    roughness_penalty: float = 1e-4

    probability_floor: float = 1e-12
    probability_trust_threshold: float = 1e-10
    qfi_tolerance: float = 1e-12
    covariance_rtol: float = 1e-10
    validation_atol: float = 1e-9
    validation_rtol: float = 2e-6
    hellinger_step: float = 1e-4
    hellinger_relative_tolerance: float = 0.10
    derivative_relative_tolerance: float = 2e-3

    ramsey_time_min: float = 0.05
    ramsey_time_max: float = 30.0
    n_ramsey_times: int = 81
    time_overhead: float = 0.0

    output_directory: str = "graphs/optimize_sld_guided_analyzer"
    figure_format: str = "png"
    figure_dpi: int = 200
    show_figure: bool = False
    make_plots: bool = True
    save_data: bool = True
    run_tests: bool = True
    progress: bool = True

    @property
    def robust_half_width(self) -> float:
        return abs(self.J0) * self.robust_fraction


@dataclass(frozen=True, slots=True)
class SystemOperators:
    """Dense joint and bath operators in the symmetric representation."""

    central_x: np.ndarray
    bath_x_joint: np.ndarray
    interaction: np.ndarray
    bath_x: np.ndarray
    bath_y: np.ndarray
    bath_z: np.ndarray
    sx_eigenvalues: np.ndarray
    sx_eigenvectors: np.ndarray


@dataclass(frozen=True, slots=True)
class ProjectiveStatistics:
    probabilities: np.ndarray
    derivatives: np.ndarray
    fisher: float
    regularized_fisher: float
    trusted_bins: np.ndarray


@dataclass(frozen=True, slots=True)
class InformationMetrics:
    global_qfi: float
    bath_qfi: float
    projective_sx_fi: float
    regularized_sx_fi: float
    mean_sx_fi: float
    bath_transfer_efficiency: float
    measurement_efficiency: float
    total_readout_efficiency: float
    probabilities: np.ndarray
    probability_derivatives: np.ndarray


@dataclass(frozen=True, slots=True)
class FiniteDifferenceCheck:
    tangent_fisher: float
    hellinger_fisher: float
    half_step_hellinger_fisher: float
    quarter_step_hellinger_fisher: float
    derivative_relative_error: float
    fisher_relative_error: float
    hellinger_convergence_error: float
    trusted: bool


@dataclass(frozen=True, slots=True)
class ControlProblem:
    K: int
    sensed_states: np.ndarray
    sensed_tangents: np.ndarray
    ramsey_rates: np.ndarray
    operators: SystemOperators
    config: AnalyzerConfig


@dataclass(slots=True)
class OptimizedAnalyzer:
    K: int
    controls_central: np.ndarray
    controls_bath: np.ndarray
    optimizer_objective: float
    optimizer_success: bool
    optimizer_message: str
    training_J: np.ndarray
    training_advantage: np.ndarray
    training_metrics: list[InformationMetrics]
    training_trust: np.ndarray
    heldout_J: np.ndarray
    heldout_advantage: np.ndarray
    heldout_analyzer_rate: np.ndarray
    heldout_ramsey_rate: np.ndarray
    heldout_bath_qfi_rate: np.ndarray
    heldout_metrics: list[InformationMetrics]
    heldout_trust: np.ndarray
    flow_times: np.ndarray
    flow_global_qfi: np.ndarray
    flow_bath_qfi: np.ndarray
    flow_projective_fi: np.ndarray
    fisher_hierarchy: dict[str, float]
    nominal_bath_sld: np.ndarray
    nominal_bath_sld_eigenvalues: np.ndarray

    @property
    def robust_statistics(self) -> dict[str, float]:
        valid = self.heldout_advantage[np.isfinite(self.heldout_advantage)]
        if len(valid) == 0:
            return {key: float("nan") for key in ("mean", "median", "minimum", "maximum")}
        return {
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
            "minimum": float(np.min(valid)),
            "maximum": float(np.max(valid)),
        }


def validate_config(cfg: AnalyzerConfig) -> None:
    """Validate the first-experiment and numerical configuration."""
    if cfg.N < 1:
        raise ValueError("N must be positive")
    if not np.isfinite(cfg.J0) or cfg.J0 == 0.0:
        raise ValueError("J0 must be finite and nonzero for a fractional robust interval")
    if cfg.sensing_time is not None and cfg.sensing_time <= 0.0:
        raise ValueError("sensing_time must be positive when supplied")
    if (
        cfg.sensing_time_min <= 0.0
        or cfg.sensing_time_max <= cfg.sensing_time_min
        or cfg.n_sensing_times < 2
    ):
        raise ValueError("invalid sensing-time selection grid")
    if cfg.analyzer_duration <= 0.0:
        raise ValueError("analyzer_duration must be positive")
    if not cfg.segment_counts or any(K < 1 for K in cfg.segment_counts):
        raise ValueError("segment_counts must contain positive integers")
    if tuple(sorted(set(cfg.segment_counts))) != cfg.segment_counts:
        raise ValueError("segment_counts must be unique and increasing")
    if cfg.central_control_max <= 0.0 or cfg.bath_control_max <= 0.0:
        raise ValueError("control bounds must be positive")
    if cfg.robust_fraction <= 0.0:
        raise ValueError("robust_fraction must be positive")
    if cfg.n_training_J < 3 or cfg.n_training_J % 2 == 0:
        raise ValueError("n_training_J must be odd and at least 3")
    if cfg.n_heldout_J < cfg.n_training_J:
        raise ValueError("n_heldout_J must be at least n_training_J")
    if cfg.softmin_sharpness <= 0.0:
        raise ValueError("softmin_sharpness must be positive")
    if cfg.n_random_seeds < 0 or cfg.optimizer_max_iterations < 1:
        raise ValueError("invalid optimizer controls")
    if cfg.probability_floor <= 0.0 or cfg.probability_trust_threshold <= 0.0:
        raise ValueError("probability tolerances must be positive")
    if cfg.probability_floor >= cfg.probability_trust_threshold:
        raise ValueError("probability_floor must be below probability_trust_threshold")
    if cfg.hellinger_step <= 0.0:
        raise ValueError("hellinger_step must be positive")
    if cfg.ramsey_time_min <= 0.0 or cfg.ramsey_time_max <= cfg.ramsey_time_min:
        raise ValueError("invalid Ramsey-time grid")
    if cfg.n_ramsey_times < 2 or cfg.time_overhead < 0.0:
        raise ValueError("invalid Ramsey sampling or overhead")
    if cfg.figure_dpi < 1 or not cfg.figure_format.isalnum():
        raise ValueError("invalid figure controls")


def build_system_operators(N: int) -> SystemOperators:
    """Build all physical controls and diagonalize bath Sx exactly once."""
    joint = build_spin_operators(N)
    bath = build_bath_operators(N)
    bath_x = bath["Jx"]
    sx_eigenvalues, sx_eigenvectors = np.linalg.eigh(bath_x)
    return SystemOperators(
        central_x=joint["sx_s"].full(),
        bath_x_joint=joint["Sx_op"].full(),
        interaction=(joint["sz_s"] * joint["Sz_op"]).full(),
        bath_x=bath_x,
        bath_y=bath["Jy"],
        bath_z=bath["Jz"],
        sx_eigenvalues=sx_eigenvalues,
        sx_eigenvectors=sx_eigenvectors,
    )


def initial_joint_state(N: int, *, ramsey: bool = False) -> np.ndarray:
    """Return central ``|+x>`` and either bath ``|0>^N`` or Ramsey ``|+x>^N``."""
    central = central_spin_state(np.pi / 2.0, 0.0).full().ravel()
    bath_theta = np.pi / 2.0 if ramsey else 0.0
    bath = coherent_bath_state(N, theta=bath_theta, phi=0.0)
    return np.kron(central, bath)


def sensing_hamiltonian(
    J: float,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    *,
    ramsey: bool = False,
) -> np.ndarray:
    """Return ``H0 + J V`` using the repository's operator normalization."""
    if ramsey:
        return J * operators.interaction
    return (
        cfg.Omega * operators.central_x
        + cfg.omega * operators.bath_x_joint
        + J * operators.interaction
    )


def sensing_state_and_tangent(
    J: float,
    time: float,
    initial_state: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    *,
    ramsey: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate sensing state and exact local J tangent with a Frechet derivative."""
    hamiltonian = sensing_hamiltonian(J, cfg, operators, ramsey=ramsey)
    generator = -1j * hamiltonian * time
    direction = -1j * operators.interaction * time
    propagator, derivative = expm_frechet(generator, direction, compute_expm=True)
    return propagator @ initial_state, derivative @ initial_state


def sensing_state_only(
    J: float,
    time: float,
    initial_state: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    *,
    ramsey: bool = False,
) -> np.ndarray:
    """Propagate a displaced validation state without calculating a tangent."""
    return expm(-1j * sensing_hamiltonian(J, cfg, operators, ramsey=ramsey) * time) @ initial_state


def density_tangent(
    state: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``rho=|psi><psi|`` and its exact parameter derivative."""
    rho = np.outer(state, state.conj())
    drho = np.outer(tangent, state.conj()) + np.outer(state, tangent.conj())
    return rho, drho


def sensing_tangent_rhs(
    rho: np.ndarray,
    drho: np.ndarray,
    hamiltonian: np.ndarray,
    interaction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Density/tangent RHS with the mandatory sensing information source."""
    rho_rhs = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    tangent_rhs = -1j * (
        hamiltonian @ drho
        - drho @ hamiltonian
        + interaction @ rho
        - rho @ interaction
    )
    return rho_rhs, tangent_rhs


def analyzer_tangent_rhs(
    rho: np.ndarray,
    drho: np.ndarray,
    hamiltonian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Analyzer RHS: the tangent has no J-information source term."""
    return (
        -1j * (hamiltonian @ rho - rho @ hamiltonian),
        -1j * (hamiltonian @ drho - drho @ hamiltonian),
    )


def bath_reduction(
    state: np.ndarray,
    tangent: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace out the central spin from a pure state and its tangent."""
    dimension = N + 1
    amplitudes = state.reshape(2, dimension)
    tangent_amplitudes = tangent.reshape(2, dimension)
    rho = amplitudes.T @ amplitudes.conj()
    drho = (
        tangent_amplitudes.T @ amplitudes.conj()
        + amplitudes.T @ tangent_amplitudes.conj()
    )
    return rho, drho


def validate_density_tangent(
    rho: np.ndarray,
    drho: np.ndarray,
    cfg: AnalyzerConfig,
    label: str,
) -> None:
    """Check trace, Hermiticity, and tangent tracelessness."""
    if not np.allclose(rho, rho.conj().T, atol=cfg.validation_atol, rtol=cfg.validation_rtol):
        raise RuntimeError(f"{label}: rho is not Hermitian")
    if not np.allclose(drho, drho.conj().T, atol=cfg.validation_atol, rtol=cfg.validation_rtol):
        raise RuntimeError(f"{label}: drho is not Hermitian")
    if not np.isclose(np.trace(rho), 1.0, atol=cfg.validation_atol, rtol=cfg.validation_rtol):
        raise RuntimeError(f"{label}: Tr(rho)={np.trace(rho)}")
    if abs(np.trace(drho)) > cfg.validation_atol + cfg.validation_rtol:
        raise RuntimeError(f"{label}: Tr(drho)={np.trace(drho)}")


def pure_state_qfi(state: np.ndarray, tangent: np.ndarray) -> float:
    """Return exact pure-state QFI from a local state tangent."""
    norm = float(np.real(np.vdot(tangent, tangent)))
    gauge = abs(np.vdot(state, tangent)) ** 2
    return max(0.0, 4.0 * (norm - float(gauge)))


def bath_sld(
    rho: np.ndarray,
    drho: np.ndarray,
    tolerance: float,
) -> tuple[float, np.ndarray]:
    """Return reduced-bath QFI and SLD in the density-matrix eigenbasis."""
    return qfi_from_rho_and_drho(rho, drho, tol=tolerance)


def _measurement_amplitudes(
    state: np.ndarray,
    tangent: np.ndarray,
    N: int,
    sx_eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    amplitudes = state.reshape(2, N + 1) @ sx_eigenvectors.conj()
    tangent_amplitudes = tangent.reshape(2, N + 1) @ sx_eigenvectors.conj()
    return amplitudes, tangent_amplitudes


def sx_projective_probabilities(
    state: np.ndarray,
    tangent: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> ProjectiveStatistics:
    """Return full projective-Sx probabilities, tangents, and stable FI."""
    amplitudes, tangent_amplitudes = _measurement_amplitudes(
        state,
        tangent,
        cfg.N,
        operators.sx_eigenvectors,
    )
    probabilities = np.sum(np.abs(amplitudes) ** 2, axis=0).real
    derivatives = 2.0 * np.real(
        np.sum(amplitudes.conj() * tangent_amplitudes, axis=0)
    )
    if not np.isclose(np.sum(probabilities), 1.0, atol=cfg.validation_atol):
        raise RuntimeError(f"projective probabilities sum to {np.sum(probabilities)}")
    if abs(np.sum(derivatives)) > cfg.validation_atol + cfg.validation_rtol:
        raise RuntimeError(f"projective probability tangent sums to {np.sum(derivatives)}")
    trusted_bins = probabilities > cfg.probability_trust_threshold
    fisher = float(np.sum(derivatives[trusted_bins] ** 2 / probabilities[trusted_bins]))
    regularized = float(
        np.sum(derivatives**2 / (probabilities + cfg.probability_floor))
    )
    return ProjectiveStatistics(
        probabilities=probabilities,
        derivatives=derivatives,
        fisher=max(0.0, fisher),
        regularized_fisher=max(0.0, regularized),
        trusted_bins=trusted_bins,
    )


def sx_probabilities_only(
    state: np.ndarray,
    N: int,
    sx_eigenvectors: np.ndarray,
) -> np.ndarray:
    amplitudes = state.reshape(2, N + 1) @ sx_eigenvectors.conj()
    return np.sum(np.abs(amplitudes) ** 2, axis=0).real


def sx_moment_fisher(
    rho: np.ndarray,
    drho: np.ndarray,
    sx: np.ndarray,
    variance_floor: float,
) -> float:
    """Return first-moment/error-propagation Sx Fisher information."""
    mean = float(np.real(np.trace(rho @ sx)))
    derivative = float(np.real(np.trace(drho @ sx)))
    variance = float(np.real(np.trace(rho @ (sx @ sx)))) - mean * mean
    return derivative * derivative / variance if variance > variance_floor else 0.0


def information_metrics(
    state: np.ndarray,
    tangent: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    *,
    label: str,
) -> InformationMetrics:
    """Calculate global/bath ceilings and fixed-Sx readout efficiencies."""
    global_rho, global_drho = density_tangent(state, tangent)
    validate_density_tangent(global_rho, global_drho, cfg, f"{label}/global")
    rho, drho = bath_reduction(state, tangent, cfg.N)
    validate_density_tangent(rho, drho, cfg, f"{label}/bath")
    global_qfi = pure_state_qfi(state, tangent)
    bath_qfi, _sld = bath_sld(rho, drho, cfg.qfi_tolerance)
    projective = sx_projective_probabilities(state, tangent, cfg, operators)
    mean_fi = sx_moment_fisher(rho, drho, operators.bath_x, cfg.probability_floor)
    tolerance = cfg.validation_atol + cfg.validation_rtol * max(1.0, global_qfi)
    if bath_qfi > global_qfi + tolerance:
        raise RuntimeError(f"{label}: FQ_bath={bath_qfi} exceeds FQ_global={global_qfi}")
    if projective.fisher > bath_qfi + tolerance:
        raise RuntimeError(f"{label}: FC_Sx={projective.fisher} exceeds FQ_bath={bath_qfi}")
    if mean_fi > bath_qfi + tolerance:
        raise RuntimeError(f"{label}: mean-Sx FI={mean_fi} exceeds FQ_bath={bath_qfi}")
    bath_efficiency = bath_qfi / global_qfi if global_qfi > cfg.qfi_tolerance else 0.0
    measurement_efficiency = (
        projective.fisher / bath_qfi if bath_qfi > cfg.qfi_tolerance else 0.0
    )
    total_efficiency = (
        projective.fisher / global_qfi if global_qfi > cfg.qfi_tolerance else 0.0
    )
    return InformationMetrics(
        global_qfi=global_qfi,
        bath_qfi=bath_qfi,
        projective_sx_fi=projective.fisher,
        regularized_sx_fi=projective.regularized_fisher,
        mean_sx_fi=mean_fi,
        bath_transfer_efficiency=bath_efficiency,
        measurement_efficiency=measurement_efficiency,
        total_readout_efficiency=total_efficiency,
        probabilities=projective.probabilities,
        probability_derivatives=projective.derivatives,
    )


def analyzer_segment(
    u_c: float,
    u_b: float,
    dt: float,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    *,
    derivatives: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one fixed-sign analyzer segment and optional exact control tangents."""
    # Mandatory physical constraint: this coefficient is +J0 and is not a control.
    interaction_coefficient = cfg.J0
    if interaction_coefficient != cfg.J0:
        raise RuntimeError("analyzer interaction coefficient changed unexpectedly")
    hamiltonian = (
        u_c * operators.central_x
        + u_b * operators.bath_x_joint
        + interaction_coefficient * operators.interaction
    )
    measured_coefficient = float(
        np.real(np.vdot(operators.interaction, hamiltonian))
        / np.real(np.vdot(operators.interaction, operators.interaction))
    )
    if not np.isclose(
        measured_coefficient,
        cfg.J0,
        atol=cfg.validation_atol,
        rtol=cfg.validation_rtol,
    ):
        raise RuntimeError(
            "analyzer interaction must remain +J0: "
            f"measured coefficient={measured_coefficient}, J0={cfg.J0}"
        )
    generator = -1j * hamiltonian * dt
    if not derivatives:
        return expm(generator)
    propagator, derivative_c = expm_frechet(
        generator,
        -1j * operators.central_x * dt,
        compute_expm=True,
    )
    derivative_b = expm_frechet(
        generator,
        -1j * operators.bath_x_joint * dt,
        compute_expm=False,
    )
    return propagator, derivative_c, derivative_b


def control_propagators(
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    *,
    derivatives: bool = False,
) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Construct every analyzer segment with the same positive J0 interaction."""
    if controls_central.shape != controls_bath.shape:
        raise ValueError("central and bath control arrays must have equal shape")
    K = len(controls_central)
    dt = cfg.analyzer_duration / K
    return [
        analyzer_segment(u_c, u_b, dt, cfg, operators, derivatives=derivatives)
        for u_c, u_b in zip(controls_central, controls_bath)
    ]


def propagate_analyzer(
    state: np.ndarray,
    tangent: np.ndarray,
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate state and tangent with no analyzer J-source term."""
    propagators = control_propagators(
        controls_central,
        controls_bath,
        cfg,
        operators,
        derivatives=False,
    )
    output_state = state
    output_tangent = tangent
    for propagator in propagators:
        output_state = propagator @ output_state
        output_tangent = propagator @ output_tangent
    return output_state, output_tangent


def analyzer_unitary(
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> np.ndarray:
    """Return ``W=U_(K-1)...U_0`` for finite-difference validation."""
    dimension = operators.central_x.shape[0]
    unitary = np.eye(dimension, dtype=complex)
    for propagator in control_propagators(
        controls_central,
        controls_bath,
        cfg,
        operators,
        derivatives=False,
    ):
        unitary = propagator @ unitary
    return unitary


def _regularized_fisher_and_control_gradient(
    state: np.ndarray,
    tangent: np.ndarray,
    state_control_tangents: Sequence[np.ndarray],
    tangent_control_tangents: Sequence[np.ndarray],
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> tuple[float, np.ndarray]:
    """Differentiate stabilized projective Sx FI analytically."""
    amplitudes, tangent_amplitudes = _measurement_amplitudes(
        state,
        tangent,
        cfg.N,
        operators.sx_eigenvectors,
    )
    p = np.sum(np.abs(amplitudes) ** 2, axis=0).real
    dp = 2.0 * np.real(np.sum(amplitudes.conj() * tangent_amplitudes, axis=0))
    denominator = p + cfg.probability_floor
    fisher = float(np.sum(dp**2 / denominator))
    gradient = np.empty(len(state_control_tangents), dtype=float)
    for index, (state_q, tangent_q) in enumerate(
        zip(state_control_tangents, tangent_control_tangents)
    ):
        amplitudes_q, tangent_amplitudes_q = _measurement_amplitudes(
            state_q,
            tangent_q,
            cfg.N,
            operators.sx_eigenvectors,
        )
        p_q = 2.0 * np.real(np.sum(amplitudes.conj() * amplitudes_q, axis=0))
        dp_q = 2.0 * np.real(
            np.sum(
                amplitudes_q.conj() * tangent_amplitudes
                + amplitudes.conj() * tangent_amplitudes_q,
                axis=0,
            )
        )
        gradient[index] = np.sum(
            2.0 * dp * dp_q / denominator
            - dp**2 * p_q / denominator**2
        )
    return max(0.0, fisher), gradient


def _control_penalty(
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    cfg: AnalyzerConfig,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return normalized power/roughness penalty and gradients."""
    K = len(controls_central)
    scaled_c = controls_central / cfg.central_control_max
    scaled_b = controls_bath / cfg.bath_control_max
    penalty = cfg.power_penalty * float(np.mean(scaled_c**2 + scaled_b**2))
    grad_c = 2.0 * cfg.power_penalty * controls_central / (
        K * cfg.central_control_max**2
    )
    grad_b = 2.0 * cfg.power_penalty * controls_bath / (
        K * cfg.bath_control_max**2
    )
    if K > 1:
        differences_c = np.diff(scaled_c)
        differences_b = np.diff(scaled_b)
        penalty += cfg.roughness_penalty * float(
            np.mean(differences_c**2 + differences_b**2)
        )
        normalization = K - 1
        for differences, gradient, bound in (
            (differences_c, grad_c, cfg.central_control_max),
            (differences_b, grad_b, cfg.bath_control_max),
        ):
            scale = 2.0 * cfg.roughness_penalty / (normalization * bound)
            gradient[:-1] -= scale * differences
            gradient[1:] += scale * differences
    return penalty, grad_c, grad_b


def control_objective_and_gradient(
    parameters: np.ndarray,
    problem: ControlProblem,
) -> tuple[float, np.ndarray]:
    """Return negative robust log-advantage and its exact GRAPE gradient."""
    K = problem.K
    cfg = problem.config
    controls_central = parameters[:K]
    controls_bath = parameters[K:]
    segment_data = control_propagators(
        controls_central,
        controls_bath,
        cfg,
        problem.operators,
        derivatives=True,
    )
    propagators = [entry[0] for entry in segment_data]
    derivatives_c = [entry[1] for entry in segment_data]
    derivatives_b = [entry[2] for entry in segment_data]

    forward_states = [problem.sensed_states]
    forward_tangents = [problem.sensed_tangents]
    for propagator in propagators:
        forward_states.append(propagator @ forward_states[-1])
        forward_tangents.append(propagator @ forward_tangents[-1])

    dimension = propagators[0].shape[0]
    post_segment: list[np.ndarray] = [np.empty((0, 0)) for _ in range(K)]
    accumulated = np.eye(dimension, dtype=complex)
    for segment in range(K - 1, -1, -1):
        post_segment[segment] = accumulated
        accumulated = accumulated @ propagators[segment]

    n_parameters = 2 * K
    state_control_tangents = np.empty(
        (n_parameters, dimension, problem.sensed_states.shape[1]),
        dtype=complex,
    )
    tangent_control_tangents = np.empty_like(state_control_tangents)
    for segment in range(K):
        for offset, derivative in ((0, derivatives_c[segment]), (K, derivatives_b[segment])):
            parameter_index = offset + segment
            state_control_tangents[parameter_index] = (
                post_segment[segment] @ derivative @ forward_states[segment]
            )
            tangent_control_tangents[parameter_index] = (
                post_segment[segment] @ derivative @ forward_tangents[segment]
            )

    if cfg.sensing_time is None:
        raise RuntimeError("sensing_time has not been resolved")
    total_time = cfg.sensing_time + cfg.analyzer_duration + cfg.time_overhead
    log_advantages = np.empty(problem.sensed_states.shape[1])
    log_gradients = np.empty((problem.sensed_states.shape[1], n_parameters))
    for sample in range(problem.sensed_states.shape[1]):
        fisher, fisher_gradient = _regularized_fisher_and_control_gradient(
            forward_states[-1][:, sample],
            forward_tangents[-1][:, sample],
            state_control_tangents[:, :, sample],
            tangent_control_tangents[:, :, sample],
            cfg,
            problem.operators,
        )
        stabilized_fisher = fisher + cfg.probability_floor
        advantage = stabilized_fisher / (total_time * problem.ramsey_rates[sample])
        log_advantages[sample] = np.log(max(advantage, cfg.probability_floor))
        log_gradients[sample] = fisher_gradient / stabilized_fisher

    alpha = cfg.softmin_sharpness
    weights = np.exp(-alpha * log_advantages - logsumexp(-alpha * log_advantages))
    robust_log_advantage = -(
        logsumexp(-alpha * log_advantages) - np.log(len(log_advantages))
    ) / alpha
    robust_gradient = weights @ log_gradients
    penalty, penalty_c, penalty_b = _control_penalty(
        controls_central,
        controls_bath,
        cfg,
    )
    objective = -robust_log_advantage + penalty
    gradient = -robust_gradient + np.concatenate((penalty_c, penalty_b))
    return float(objective), np.asarray(gradient, dtype=float)


def construct_J_ensemble(cfg: AnalyzerConfig, count: int) -> np.ndarray:
    """Return symmetric true-J values around the programmed J0."""
    return cfg.J0 + np.linspace(-cfg.robust_half_width, cfg.robust_half_width, count)


def select_sensing_time(
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Select one sensing time by maximizing nominal reduced-bath QFI rate."""
    if cfg.sensing_time is not None:
        initial = initial_joint_state(cfg.N)
        state, tangent = sensing_state_and_tangent(
            cfg.J0,
            cfg.sensing_time,
            initial,
            cfg,
            operators,
        )
        rho, drho = bath_reduction(state, tangent, cfg.N)
        qfi, _ = bath_sld(rho, drho, cfg.qfi_tolerance)
        rate = qfi / (cfg.sensing_time + cfg.time_overhead)
        return cfg.sensing_time, np.array([cfg.sensing_time]), np.array([rate])
    times = np.linspace(cfg.sensing_time_min, cfg.sensing_time_max, cfg.n_sensing_times)
    initial = initial_joint_state(cfg.N)
    rates = np.empty(len(times), dtype=float)
    for index, time in enumerate(times):
        state, tangent = sensing_state_and_tangent(
            cfg.J0,
            float(time),
            initial,
            cfg,
            operators,
        )
        rho, drho = bath_reduction(state, tangent, cfg.N)
        qfi, _ = bath_sld(rho, drho, cfg.qfi_tolerance)
        rates[index] = qfi / (time + cfg.time_overhead)
    best = int(np.argmax(rates))
    return float(times[best]), times, rates


def _replace_sensing_time(cfg: AnalyzerConfig, time: float) -> AnalyzerConfig:
    values = asdict(cfg)
    values["sensing_time"] = time
    return AnalyzerConfig(**values)


def _ramsey_rate_at_time(
    J: float,
    time: float,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> float:
    initial = initial_joint_state(cfg.N, ramsey=True)
    state, tangent = sensing_state_and_tangent(
        J,
        time,
        initial,
        cfg,
        operators,
        ramsey=True,
    )
    projective = sx_projective_probabilities(state, tangent, cfg, operators)
    return projective.fisher / (time + cfg.time_overhead)


def optimized_ramsey_rate(
    J: float,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> float:
    """Numerically optimize naive projective-bath-Sx Ramsey information rate."""
    times = np.linspace(cfg.ramsey_time_min, cfg.ramsey_time_max, cfg.n_ramsey_times)
    rates = np.asarray(
        [_ramsey_rate_at_time(J, float(time), cfg, operators) for time in times]
    )
    return float(np.max(rates))


def _cached_ramsey_rate(
    J: float,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    cache: dict[float, float],
) -> float:
    key = float(J)
    if key not in cache:
        cache[key] = optimized_ramsey_rate(key, cfg, operators)
    return cache[key]


def build_training_problem(
    K: int,
    training_J: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    ramsey_cache: dict[float, float],
) -> ControlProblem:
    """Build exact sensed states/tangents for robust ensemble control."""
    if cfg.sensing_time is None:
        raise RuntimeError("sensing time must be resolved before building a problem")
    initial = initial_joint_state(cfg.N)
    pairs = [
        sensing_state_and_tangent(J, cfg.sensing_time, initial, cfg, operators)
        for J in training_J
    ]
    states = np.column_stack([pair[0] for pair in pairs])
    tangents = np.column_stack([pair[1] for pair in pairs])
    ramsey_rates = np.asarray(
        [_cached_ramsey_rate(float(J), cfg, operators, ramsey_cache) for J in training_J]
    )
    if np.any(ramsey_rates <= 0.0):
        raise RuntimeError(f"nonpositive Ramsey rates: {ramsey_rates}")
    return ControlProblem(
        K=K,
        sensed_states=states,
        sensed_tangents=tangents,
        ramsey_rates=ramsey_rates,
        operators=operators,
        config=cfg,
    )


def finite_difference_validation(
    J: float,
    sensing_time: float,
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    tangent_metrics: InformationMetrics,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> FiniteDifferenceCheck:
    """Validate tangent FI with centered dp and Hellinger curvature at two steps."""
    initial = initial_joint_state(cfg.N)
    analyzer = analyzer_unitary(controls_central, controls_bath, cfg, operators)

    def displaced_probabilities(step: float) -> tuple[np.ndarray, np.ndarray]:
        plus = sensing_state_only(J + step, sensing_time, initial, cfg, operators)
        minus = sensing_state_only(J - step, sensing_time, initial, cfg, operators)
        return (
            sx_probabilities_only(analyzer @ plus, cfg.N, operators.sx_eigenvectors),
            sx_probabilities_only(analyzer @ minus, cfg.N, operators.sx_eigenvectors),
        )

    def hellinger(p_plus: np.ndarray, p_minus: np.ndarray, step: float) -> float:
        coefficient = float(
            np.sum(np.sqrt(np.clip(p_plus, 0.0, None) * np.clip(p_minus, 0.0, None)))
        )
        return max(0.0, 2.0 * (1.0 - min(1.0, coefficient)) / step**2)

    step = cfg.hellinger_step
    p_plus, p_minus = displaced_probabilities(step)
    p_plus_half, p_minus_half = displaced_probabilities(step / 2.0)
    p_plus_quarter, p_minus_quarter = displaced_probabilities(step / 4.0)
    dp_finite = (p_plus_quarter - p_minus_quarter) / (step / 2.0)
    derivative_error = float(
        np.linalg.norm(dp_finite - tangent_metrics.probability_derivatives)
        / max(
            cfg.validation_atol,
            np.linalg.norm(tangent_metrics.probability_derivatives),
        )
    )
    fisher_hellinger = hellinger(p_plus, p_minus, step)
    fisher_hellinger_half = hellinger(p_plus_half, p_minus_half, step / 2.0)
    fisher_hellinger_quarter = hellinger(
        p_plus_quarter,
        p_minus_quarter,
        step / 4.0,
    )
    fisher_error = abs(
        tangent_metrics.projective_sx_fi - fisher_hellinger_quarter
    ) / max(
        cfg.validation_atol,
        tangent_metrics.projective_sx_fi,
        fisher_hellinger_quarter,
    )
    convergence_error = abs(fisher_hellinger_quarter - fisher_hellinger_half) / max(
        cfg.validation_atol,
        fisher_hellinger_quarter,
    )
    trusted = (
        derivative_error <= cfg.derivative_relative_tolerance
        and fisher_error <= cfg.hellinger_relative_tolerance
        and convergence_error <= cfg.hellinger_relative_tolerance
    )
    return FiniteDifferenceCheck(
        tangent_fisher=tangent_metrics.projective_sx_fi,
        hellinger_fisher=fisher_hellinger,
        half_step_hellinger_fisher=fisher_hellinger_half,
        quarter_step_hellinger_fisher=fisher_hellinger_quarter,
        derivative_relative_error=derivative_error,
        fisher_relative_error=float(fisher_error),
        hellinger_convergence_error=float(convergence_error),
        trusted=bool(trusted),
    )


def _symmetrized_monomial(
    indices: tuple[int, ...],
    operators: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return the fully symmetrized Hermitian monomial for one index multiset."""
    unique_permutations = sorted(set(permutations(indices)))
    result = np.zeros_like(operators[0])
    for ordering in unique_permutations:
        product = np.eye(operators[0].shape[0], dtype=complex)
        for index in ordering:
            product = product @ operators[index]
        result += product
    result /= len(unique_permutations)
    return 0.5 * (result + result.conj().T)


@lru_cache(maxsize=None)
def _monomial_indices(maximum_order: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        indices
        for order in range(1, maximum_order + 1)
        for indices in combinations_with_replacement(range(3), order)
    )


def basis_moment_fisher(
    rho: np.ndarray,
    drho: np.ndarray,
    basis: Sequence[np.ndarray],
    covariance_rtol: float,
) -> float:
    """Optimize a first-moment observable over a supplied Hermitian basis."""
    count = len(basis)
    means = np.asarray([np.real(np.trace(rho @ operator)) for operator in basis])
    gradient = np.asarray([np.real(np.trace(drho @ operator)) for operator in basis])
    covariance = np.empty((count, count), dtype=float)
    for i in range(count):
        for j in range(i, count):
            moment = np.real(
                np.trace(rho @ (0.5 * (basis[i] @ basis[j] + basis[j] @ basis[i])))
            )
            covariance[i, j] = covariance[j, i] = moment - means[i] * means[j]
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = max(0.0, float(eigenvalues[-1]))
    retained = eigenvalues > covariance_rtol * maximum
    inverse = np.zeros_like(eigenvalues)
    inverse[retained] = 1.0 / eigenvalues[retained]
    pseudoinverse = (eigenvectors * inverse) @ eigenvectors.T
    return max(0.0, float(gradient @ pseudoinverse @ gradient))


def fisher_hierarchy(
    state: np.ndarray,
    tangent: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> dict[str, float]:
    """Compare linear, quadratic, and cubic moment spans with bath QFI."""
    rho, drho = bath_reduction(state, tangent, cfg.N)
    spin_operators = (operators.bath_x, operators.bath_y, operators.bath_z)
    hierarchy: dict[str, float] = {}
    names = {1: "linear", 2: "quadratic", 3: "cubic"}
    for maximum_order in (1, 2, 3):
        basis = [
            _symmetrized_monomial(indices, spin_operators)
            for indices in _monomial_indices(maximum_order)
        ]
        hierarchy[names[maximum_order]] = basis_moment_fisher(
            rho,
            drho,
            basis,
            cfg.covariance_rtol,
        )
    hierarchy["bath_qfi"] = bath_sld(rho, drho, cfg.qfi_tolerance)[0]
    allowed = cfg.validation_atol + cfg.validation_rtol * max(
        1.0,
        hierarchy["bath_qfi"],
    )
    for name in ("linear", "quadratic", "cubic"):
        if hierarchy[name] > hierarchy["bath_qfi"] + allowed:
            raise RuntimeError(
                f"{name} moment hierarchy exceeds bath QFI: "
                f"{hierarchy[name]} > {hierarchy['bath_qfi']}"
            )
    return hierarchy


def _simple_control_seeds(K: int, cfg: AnalyzerConfig) -> list[np.ndarray]:
    """Return physically motivated constant analyzer seeds."""
    combinations = (
        (-cfg.Omega, -cfg.omega),
        (0.0, 0.0),
        (cfg.Omega, -cfg.omega),
        (-cfg.Omega, cfg.omega),
    )
    return [
        np.concatenate((np.full(K, u_c), np.full(K, u_b)))
        for u_c, u_b in combinations
    ]


def _smooth_random_perturbation(
    K: int,
    cfg: AnalyzerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return small low-pass perturbations for a physically smooth seed."""
    central = rng.normal(
        scale=cfg.random_seed_scale * cfg.central_control_max,
        size=K,
    )
    bath = rng.normal(
        scale=cfg.random_seed_scale * cfg.bath_control_max,
        size=K,
    )
    if K > 2:
        kernel = np.array([0.25, 0.5, 0.25])
        central = np.convolve(np.pad(central, 1, mode="edge"), kernel, mode="valid")
        bath = np.convolve(np.pad(bath, 1, mode="edge"), kernel, mode="valid")
    return np.concatenate((central, bath))


def _resample_controls(previous: OptimizedAnalyzer, K: int) -> np.ndarray:
    old_positions = (np.arange(previous.K) + 0.5) / previous.K
    new_positions = (np.arange(K) + 0.5) / K
    central = np.interp(new_positions, old_positions, previous.controls_central)
    bath = np.interp(new_positions, old_positions, previous.controls_bath)
    return np.concatenate((central, bath))


def _bounds(K: int, cfg: AnalyzerConfig) -> list[tuple[float, float]]:
    return (
        [(-cfg.central_control_max, cfg.central_control_max)] * K
        + [(-cfg.bath_control_max, cfg.bath_control_max)] * K
    )


def _state_tangent_at_J(
    J: float,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.sensing_time is None:
        raise RuntimeError("sensing_time has not been resolved")
    return sensing_state_and_tangent(
        J,
        cfg.sensing_time,
        initial_joint_state(cfg.N),
        cfg,
        operators,
    )


def evaluate_controls_at_J(
    J: float,
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    *,
    validate_finite_difference: bool,
) -> tuple[InformationMetrics, FiniteDifferenceCheck | None]:
    """Evaluate one true-J point using the same programmed analyzer."""
    state, tangent = _state_tangent_at_J(J, cfg, operators)
    sensed_global_qfi = pure_state_qfi(state, tangent)
    output_state, output_tangent = propagate_analyzer(
        state,
        tangent,
        controls_central,
        controls_bath,
        cfg,
        operators,
    )
    metrics = information_metrics(
        output_state,
        output_tangent,
        cfg,
        operators,
        label=f"analyzer/J={J:.8g}",
    )
    allowed = cfg.validation_atol + cfg.validation_rtol * max(1.0, sensed_global_qfi)
    if abs(metrics.global_qfi - sensed_global_qfi) > allowed:
        raise RuntimeError(
            "global QFI changed under the J-independent analyzer: "
            f"before={sensed_global_qfi}, after={metrics.global_qfi}"
        )
    check = None
    if validate_finite_difference:
        check = finite_difference_validation(
            J,
            cfg.sensing_time,
            controls_central,
            controls_bath,
            metrics,
            cfg,
            operators,
        )
    return metrics, check


def information_flow(
    J: float,
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure information at sensing end and after every analyzer segment."""
    state, tangent = _state_tangent_at_J(J, cfg, operators)
    K = len(controls_central)
    times = np.linspace(0.0, cfg.analyzer_duration, K + 1)
    global_qfi = np.empty(K + 1)
    bath_qfi = np.empty(K + 1)
    projective_fi = np.empty(K + 1)
    propagators = control_propagators(
        controls_central,
        controls_bath,
        cfg,
        operators,
        derivatives=False,
    )
    for index in range(K + 1):
        metrics = information_metrics(
            state,
            tangent,
            cfg,
            operators,
            label=f"flow/segment={index}",
        )
        global_qfi[index] = metrics.global_qfi
        bath_qfi[index] = metrics.bath_qfi
        projective_fi[index] = metrics.projective_sx_fi
        if index < K:
            state = propagators[index] @ state
            tangent = propagators[index] @ tangent
    tolerance = cfg.validation_atol + cfg.validation_rtol * max(1.0, global_qfi[0])
    if np.max(np.abs(global_qfi - global_qfi[0])) > tolerance:
        raise RuntimeError("global QFI is not constant throughout analyzer")
    return times, global_qfi, bath_qfi, projective_fi


def robust_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _evaluate_solution_grid(
    J_values: np.ndarray,
    controls_central: np.ndarray,
    controls_bath: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    ramsey_cache: dict[float, float],
) -> tuple[
    list[InformationMetrics],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate rates, efficiencies, and trust on a true-J grid."""
    metrics_list: list[InformationMetrics] = []
    analyzer_rates = np.empty(len(J_values))
    ramsey_rates = np.empty(len(J_values))
    bath_qfi_rates = np.empty(len(J_values))
    advantages = np.empty(len(J_values))
    trusted = np.empty(len(J_values), dtype=bool)
    total_time = cfg.sensing_time + cfg.analyzer_duration + cfg.time_overhead
    for index, J in enumerate(J_values):
        metrics, check = evaluate_controls_at_J(
            float(J),
            controls_central,
            controls_bath,
            cfg,
            operators,
            validate_finite_difference=True,
        )
        metrics_list.append(metrics)
        analyzer_rates[index] = metrics.projective_sx_fi / total_time
        bath_qfi_rates[index] = metrics.bath_qfi / total_time
        ramsey_rates[index] = _cached_ramsey_rate(
            float(J), cfg, operators, ramsey_cache
        )
        advantages[index] = analyzer_rates[index] / ramsey_rates[index]
        trusted[index] = bool(check is not None and check.trusted)
    return (
        metrics_list,
        analyzer_rates,
        ramsey_rates,
        bath_qfi_rates,
        advantages,
        trusted,
    )


def optimize_analyzer(
    K: int,
    training_J: np.ndarray,
    heldout_J: np.ndarray,
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    ramsey_cache: dict[float, float],
    previous: OptimizedAnalyzer | None,
) -> OptimizedAnalyzer:
    """Run multi-start L-BFGS-B and accept the best trusted robust analyzer."""
    problem = build_training_problem(K, training_J, cfg, operators, ramsey_cache)
    seeds = _simple_control_seeds(K, cfg)
    if previous is not None:
        seeds.append(_resample_controls(previous, K))
    seed_scores = [control_objective_and_gradient(seed, problem)[0] for seed in seeds]
    best_simple = seeds[int(np.argmin(seed_scores))]
    rng = np.random.default_rng(cfg.random_seed + K)
    for _ in range(cfg.n_random_seeds):
        noise = _smooth_random_perturbation(K, cfg, rng)
        seeds.append(np.clip(best_simple + noise, [b[0] for b in _bounds(K, cfg)], [b[1] for b in _bounds(K, cfg)]))

    candidates = []
    for seed_index, seed in enumerate(seeds):
        result = minimize(
            control_objective_and_gradient,
            seed,
            args=(problem,),
            method="L-BFGS-B",
            jac=True,
            bounds=_bounds(K, cfg),
            options={
                "maxiter": cfg.optimizer_max_iterations,
                "ftol": cfg.optimizer_ftol,
                "gtol": cfg.optimizer_gtol,
            },
        )
        candidates.append(result)
        if cfg.progress:
            print(
                f"K={K} seed {seed_index + 1}/{len(seeds)}: "
                f"objective={result.fun:.8g}, success={result.success}"
            )

    candidates.sort(key=lambda candidate: float(candidate.fun))
    accepted: tuple[Any, Any] | None = None
    for candidate in candidates:
        central = np.asarray(candidate.x[:K])
        bath = np.asarray(candidate.x[K:])
        training_evaluation = _evaluate_solution_grid(
            training_J,
            central,
            bath,
            cfg,
            operators,
            ramsey_cache,
        )
        if np.all(training_evaluation[-1]):
            accepted = candidate, training_evaluation
            break
    if accepted is None:
        trust_counts = []
        for candidate in candidates:
            central = np.asarray(candidate.x[:K])
            bath = np.asarray(candidate.x[K:])
            evaluation = _evaluate_solution_grid(
                training_J,
                central,
                bath,
                cfg,
                operators,
                ramsey_cache,
            )
            trust_counts.append(int(np.count_nonzero(evaluation[-1])))
        raise RuntimeError(
            f"K={K}: rejected every optimized candidate; trusted training "
            f"points per candidate={trust_counts}/{len(training_J)}"
        )

    candidate, training_evaluation = accepted
    controls_central = np.asarray(candidate.x[:K])
    controls_bath = np.asarray(candidate.x[K:])
    (
        training_metrics,
        training_rates,
        training_ramsey,
        _training_bath_rates,
        training_advantage,
        training_trust,
    ) = training_evaluation
    heldout_evaluation = _evaluate_solution_grid(
        heldout_J,
        controls_central,
        controls_bath,
        cfg,
        operators,
        ramsey_cache,
    )
    (
        heldout_metrics,
        heldout_rates,
        heldout_ramsey,
        heldout_bath_rates,
        heldout_advantage,
        heldout_trust,
    ) = heldout_evaluation
    # Untrusted support-changing/numerically singular points are never used to
    # claim a robust advantage, but remain visible as red crosses in plots.
    heldout_advantage = heldout_advantage.copy()
    heldout_advantage[~heldout_trust] = np.nan
    flow = information_flow(
        cfg.J0,
        controls_central,
        controls_bath,
        cfg,
        operators,
    )
    nominal_state, nominal_tangent = _state_tangent_at_J(cfg.J0, cfg, operators)
    final_state, final_tangent = propagate_analyzer(
        nominal_state,
        nominal_tangent,
        controls_central,
        controls_bath,
        cfg,
        operators,
    )
    hierarchy = fisher_hierarchy(final_state, final_tangent, cfg, operators)
    final_rho, final_drho = bath_reduction(final_state, final_tangent, cfg.N)
    _final_qfi, final_sld = bath_sld(final_rho, final_drho, cfg.qfi_tolerance)
    return OptimizedAnalyzer(
        K=K,
        controls_central=controls_central,
        controls_bath=controls_bath,
        optimizer_objective=float(candidate.fun),
        optimizer_success=bool(candidate.success),
        optimizer_message=str(candidate.message),
        training_J=training_J.copy(),
        training_advantage=training_advantage,
        training_metrics=training_metrics,
        training_trust=training_trust,
        heldout_J=heldout_J.copy(),
        heldout_advantage=heldout_advantage,
        heldout_analyzer_rate=heldout_rates,
        heldout_ramsey_rate=heldout_ramsey,
        heldout_bath_qfi_rate=heldout_bath_rates,
        heldout_metrics=heldout_metrics,
        heldout_trust=heldout_trust,
        flow_times=flow[0],
        flow_global_qfi=flow[1],
        flow_bath_qfi=flow[2],
        flow_projective_fi=flow[3],
        fisher_hierarchy=hierarchy,
        nominal_bath_sld=final_sld,
        nominal_bath_sld_eigenvalues=np.linalg.eigvalsh(final_sld),
    )


def baseline_diagnostics(
    cfg: AnalyzerConfig,
    operators: SystemOperators,
) -> dict[str, Any]:
    """Compare sensing end with the allowed constant reversed-drive analyzer."""
    state, tangent = _state_tangent_at_J(cfg.J0, cfg, operators)
    sensing_metrics = information_metrics(
        state,
        tangent,
        cfg,
        operators,
        label="baseline/sensing-end",
    )
    sensing_hierarchy = fisher_hierarchy(state, tangent, cfg, operators)
    sensing_rho, sensing_drho = bath_reduction(state, tangent, cfg.N)
    _sensing_qfi, sensing_sld = bath_sld(
        sensing_rho,
        sensing_drho,
        cfg.qfi_tolerance,
    )
    reversed_central = np.array([-cfg.Omega])
    reversed_bath = np.array([-cfg.omega])
    analyzed_state, analyzed_tangent = propagate_analyzer(
        state,
        tangent,
        reversed_central,
        reversed_bath,
        cfg,
        operators,
    )
    analyzer_metrics = information_metrics(
        analyzed_state,
        analyzed_tangent,
        cfg,
        operators,
        label="baseline/reversed-drive-analyzer",
    )
    analyzer_hierarchy = fisher_hierarchy(
        analyzed_state,
        analyzed_tangent,
        cfg,
        operators,
    )
    analyzer_rho, analyzer_drho = bath_reduction(
        analyzed_state,
        analyzed_tangent,
        cfg.N,
    )
    _analyzer_qfi, analyzer_sld = bath_sld(
        analyzer_rho,
        analyzer_drho,
        cfg.qfi_tolerance,
    )
    finite_difference = finite_difference_validation(
        cfg.J0,
        cfg.sensing_time,
        reversed_central,
        reversed_bath,
        analyzer_metrics,
        cfg,
        operators,
    )
    return {
        "sensing_metrics": sensing_metrics,
        "sensing_hierarchy": sensing_hierarchy,
        "sensing_sld": sensing_sld,
        "sensing_sld_eigenvalues": np.linalg.eigvalsh(sensing_sld),
        "constant_analyzer_metrics": analyzer_metrics,
        "constant_analyzer_hierarchy": analyzer_hierarchy,
        "constant_analyzer_sld": analyzer_sld,
        "constant_analyzer_sld_eigenvalues": np.linalg.eigvalsh(analyzer_sld),
        "constant_analyzer_finite_difference": finite_difference,
    }


def run_numerical_tests(
    cfg: AnalyzerConfig,
    operators: SystemOperators,
    problem: ControlProblem,
) -> None:
    """Check exact sensing tangents and one analytic control-gradient direction."""
    initial = initial_joint_state(cfg.N)
    state, tangent = sensing_state_and_tangent(
        cfg.J0,
        cfg.sensing_time,
        initial,
        cfg,
        operators,
    )
    step = cfg.hellinger_step / 2.0
    state_plus = sensing_state_only(
        cfg.J0 + step,
        cfg.sensing_time,
        initial,
        cfg,
        operators,
    )
    state_minus = sensing_state_only(
        cfg.J0 - step,
        cfg.sensing_time,
        initial,
        cfg,
        operators,
    )
    finite_tangent = (state_plus - state_minus) / (2.0 * step)
    tangent_error = np.linalg.norm(finite_tangent - tangent) / max(
        cfg.validation_atol,
        np.linalg.norm(tangent),
    )
    if tangent_error > cfg.derivative_relative_tolerance:
        raise AssertionError(f"Frechet sensing tangent check failed: {tangent_error}")

    K = problem.K
    parameters = np.concatenate((np.full(K, -cfg.Omega), np.full(K, -cfg.omega)))
    objective, gradient = control_objective_and_gradient(parameters, problem)
    rng = np.random.default_rng(cfg.random_seed + 1000 + K)
    direction = rng.normal(size=2 * K)
    direction /= np.linalg.norm(direction)
    control_step = 2e-6
    objective_plus = control_objective_and_gradient(
        parameters + control_step * direction,
        problem,
    )[0]
    objective_minus = control_objective_and_gradient(
        parameters - control_step * direction,
        problem,
    )[0]
    finite_directional = (objective_plus - objective_minus) / (2.0 * control_step)
    analytic_directional = float(gradient @ direction)
    gradient_error = abs(finite_directional - analytic_directional) / max(
        cfg.validation_atol,
        abs(finite_directional),
        abs(analytic_directional),
    )
    if gradient_error > 2e-4:
        raise AssertionError(
            "analytic analyzer-control gradient check failed: "
            f"finite={finite_directional}, analytic={analytic_directional}, "
            f"relative_error={gradient_error}"
        )

    # A J-independent unitary must preserve global QFI exactly.
    output_state, output_tangent = propagate_analyzer(
        state,
        tangent,
        parameters[:K],
        parameters[K:],
        cfg,
        operators,
    )
    before = pure_state_qfi(state, tangent)
    after = pure_state_qfi(output_state, output_tangent)
    if not np.isclose(before, after, atol=cfg.validation_atol, rtol=cfg.validation_rtol):
        raise AssertionError(f"analyzer changed global QFI: {before} -> {after}")
    if not np.isfinite(objective):
        raise AssertionError("control objective is nonfinite")
    print(
        "numerical tests: Frechet sensing tangent, exact GRAPE gradient, "
        "and analyzer QFI conservation passed"
    )


def _output_directory(cfg: AnalyzerConfig) -> Path:
    path = Path(cfg.output_directory)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _number_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def _base_tag(cfg: AnalyzerConfig) -> str:
    return (
        f"N{cfg.N}_Om{_number_tag(cfg.Omega)}_om{_number_tag(cfg.omega)}_"
        f"J0{_number_tag(cfg.J0)}_ts{_number_tag(cfg.sensing_time)}_"
        f"Ta{_number_tag(cfg.analyzer_duration)}"
    )


def _save_analyzer_figure(
    figure: Any,
    filename: str,
    cfg: AnalyzerConfig,
    plot_name: str,
    result: Any,
) -> Path:
    """Save one optimizer figure with its configuration and result metadata."""
    return save_plot(
        figure,
        filename,
        metadata={"config": cfg, "plot": plot_name, "result": result},
        script_path=__file__,
        format=cfg.figure_format,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )


def plot_control(
    result: OptimizedAnalyzer,
    cfg: AnalyzerConfig,
) -> Path:
    dt = cfg.analyzer_duration / result.K
    edges = np.arange(result.K + 1) * dt
    central = np.r_[result.controls_central, result.controls_central[-1]]
    bath = np.r_[result.controls_bath, result.controls_bath[-1]]
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.step(edges, central, where="post", label=r"$u_c(t)$", linewidth=2)
    axis.step(edges, bath, where="post", label=r"$u_b(t)$", linewidth=2)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Analyzer time")
    axis.set_ylabel("Control amplitude")
    axis.set_title(
        rf"K={result.K} optimized analyzer; coupling fixed at "
        rf"$+J_0 Z_cS_z$, $J_0={cfg.J0:g}$"
    )
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    path = _save_analyzer_figure(
        figure,
        f"control_K{result.K}_{_base_tag(cfg)}.{cfg.figure_format}",
        cfg,
        "optimized_control_waveforms",
        result,
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def plot_robustness(result: OptimizedAnalyzer, cfg: AnalyzerConfig) -> Path:
    offsets = result.heldout_J - cfg.J0
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    axes[0, 0].plot(offsets, result.heldout_analyzer_rate, marker="o")
    axes[0, 0].set_ylabel(r"$F_C^{S_x}/(t_s+T_a+t_{oh})$")
    axes[0, 0].set_title("Analyzer classical-FI rate")
    axes[0, 1].plot(offsets, result.heldout_ramsey_rate, marker="o", color="C1")
    axes[0, 1].set_ylabel("Optimized Ramsey FI rate")
    axes[0, 1].set_title("Numerical Ramsey benchmark")
    raw_advantage = result.heldout_analyzer_rate / result.heldout_ramsey_rate
    axes[1, 0].plot(offsets, result.heldout_advantage, marker="o", color="C2")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("Analyzer / Ramsey")
    axes[1, 1].plot(offsets, result.heldout_bath_qfi_rate, marker="o", color="C3")
    axes[1, 1].set_ylabel(r"$F_Q^{bath}/(t_s+T_a+t_{oh})$")
    for axis in axes.flat:
        axis.set_xlabel(r"$J-J_0$")
        axis.grid(True, linestyle=":", alpha=0.7)
    untrusted = ~result.heldout_trust
    if np.any(untrusted):
        plotted_values = (
            result.heldout_analyzer_rate,
            result.heldout_ramsey_rate,
            raw_advantage,
            result.heldout_bath_qfi_rate,
        )
        for axis, y_values in zip(axes.flat, plotted_values):
            axis.scatter(offsets[untrusted], y_values[untrusted], marker="x", color="red", zorder=5)
    figure.suptitle(f"Robustness of the same K={result.K} analyzer")
    figure.tight_layout()
    path = _save_analyzer_figure(
        figure,
        f"robustness_K{result.K}_{_base_tag(cfg)}.{cfg.figure_format}",
        cfg,
        "heldout_robustness",
        result,
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def plot_information_flow(result: OptimizedAnalyzer, cfg: AnalyzerConfig) -> Path:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(result.flow_times, result.flow_global_qfi, marker="o", label=r"$F_Q^{global}$")
    axis.plot(result.flow_times, result.flow_bath_qfi, marker="o", label=r"$F_Q^{bath}$")
    axis.plot(result.flow_times, result.flow_projective_fi, marker="o", label=r"$F_C^{S_x}$")
    axis.set_xlabel("Analyzer elapsed time (0 = sensing end)")
    axis.set_ylabel("Information")
    axis.set_title(
        rf"Information flow, K={result.K}; analyzer coupling $+J_0 Z_cS_z$"
    )
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    path = _save_analyzer_figure(
        figure,
        f"information_flow_K{result.K}_{_base_tag(cfg)}.{cfg.figure_format}",
        cfg,
        "information_flow",
        result,
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def plot_efficiency(result: OptimizedAnalyzer, cfg: AnalyzerConfig) -> Path:
    bath_efficiency = np.divide(
        result.flow_bath_qfi,
        result.flow_global_qfi,
        out=np.zeros_like(result.flow_bath_qfi),
        where=result.flow_global_qfi > cfg.qfi_tolerance,
    )
    measurement_efficiency = np.divide(
        result.flow_projective_fi,
        result.flow_bath_qfi,
        out=np.zeros_like(result.flow_projective_fi),
        where=result.flow_bath_qfi > cfg.qfi_tolerance,
    )
    total_efficiency = np.divide(
        result.flow_projective_fi,
        result.flow_global_qfi,
        out=np.zeros_like(result.flow_projective_fi),
        where=result.flow_global_qfi > cfg.qfi_tolerance,
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(result.flow_times, bath_efficiency, marker="o", label=r"$F_Q^{bath}/F_Q^{global}$")
    axis.plot(result.flow_times, measurement_efficiency, marker="o", label=r"$F_C^{S_x}/F_Q^{bath}$")
    axis.plot(result.flow_times, total_efficiency, marker="o", label=r"$F_C^{S_x}/F_Q^{global}$")
    axis.set_xlabel("Analyzer elapsed time")
    axis.set_ylabel("Efficiency")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(f"Information-transfer/readout efficiencies, K={result.K}")
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    path = _save_analyzer_figure(
        figure,
        f"efficiency_K{result.K}_{_base_tag(cfg)}.{cfg.figure_format}",
        cfg,
        "information_efficiencies",
        result,
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def plot_complexity(results: Sequence[OptimizedAnalyzer], cfg: AnalyzerConfig) -> Path:
    K_values = np.asarray([result.K for result in results])
    robust_efficiency = np.asarray(
        [
            np.nanmin(
                [
                    metrics.total_readout_efficiency if trusted else np.nan
                    for metrics, trusted in zip(result.heldout_metrics, result.heldout_trust)
                ]
            )
            for result in results
        ]
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.8))
    axis.plot(K_values, robust_efficiency, marker="o", linewidth=2)
    axis.set_xscale("log", base=2)
    axis.set_xticks(K_values, labels=[str(K) for K in K_values])
    axis.set_xlabel("Number of analyzer segments K")
    axis.set_ylabel(r"Worst trusted $F_C^{S_x}/F_Q^{global}$")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Analyzer control-complexity ladder")
    axis.grid(True, linestyle=":", alpha=0.7)
    figure.tight_layout()
    path = _save_analyzer_figure(
        figure,
        f"complexity_{_base_tag(cfg)}.{cfg.figure_format}",
        cfg,
        "control_complexity_ladder",
        results,
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def plot_results(
    results: Sequence[OptimizedAnalyzer],
    cfg: AnalyzerConfig,
) -> list[Path]:
    paths: list[Path] = []
    for result in results:
        paths.extend(
            (
                plot_control(result, cfg),
                plot_robustness(result, cfg),
                plot_information_flow(result, cfg),
                plot_efficiency(result, cfg),
            )
        )
    paths.append(plot_complexity(results, cfg))
    return paths


def _metrics_summary(metrics: InformationMetrics) -> dict[str, float]:
    return {
        "global_qfi": metrics.global_qfi,
        "bath_qfi": metrics.bath_qfi,
        "projective_sx_fi": metrics.projective_sx_fi,
        "regularized_sx_fi": metrics.regularized_sx_fi,
        "mean_sx_fi": metrics.mean_sx_fi,
        "bath_transfer_efficiency": metrics.bath_transfer_efficiency,
        "measurement_efficiency": metrics.measurement_efficiency,
        "total_readout_efficiency": metrics.total_readout_efficiency,
    }


def _finite_difference_summary(check: FiniteDifferenceCheck) -> dict[str, Any]:
    return asdict(check)


def save_results(
    results: Sequence[OptimizedAnalyzer],
    baseline: dict[str, Any],
    sensing_selection_times: np.ndarray,
    sensing_selection_rates: np.ndarray,
    cfg: AnalyzerConfig,
) -> list[Path]:
    """Save controls, robustness arrays, information flow, and JSON summary."""
    output = _output_directory(cfg)
    paths: list[Path] = []
    for result in results:
        path = output / f"analyzer_K{result.K}_{_base_tag(cfg)}.npz"
        np.savez_compressed(
            path,
            controls_central=result.controls_central,
            controls_bath=result.controls_bath,
            training_J=result.training_J,
            training_advantage=result.training_advantage,
            training_trust=result.training_trust,
            heldout_J=result.heldout_J,
            heldout_advantage=result.heldout_advantage,
            heldout_analyzer_rate=result.heldout_analyzer_rate,
            heldout_ramsey_rate=result.heldout_ramsey_rate,
            heldout_bath_qfi_rate=result.heldout_bath_qfi_rate,
            heldout_trust=result.heldout_trust,
            flow_times=result.flow_times,
            flow_global_qfi=result.flow_global_qfi,
            flow_bath_qfi=result.flow_bath_qfi,
            flow_projective_fi=result.flow_projective_fi,
            nominal_bath_sld=result.nominal_bath_sld,
            nominal_bath_sld_eigenvalues=result.nominal_bath_sld_eigenvalues,
        )
        paths.append(path)

    baseline_sld_path = output / f"baseline_sld_{_base_tag(cfg)}.npz"
    np.savez_compressed(
        baseline_sld_path,
        sensing_end_sld=baseline["sensing_sld"],
        sensing_end_sld_eigenvalues=baseline["sensing_sld_eigenvalues"],
        constant_analyzer_sld=baseline["constant_analyzer_sld"],
        constant_analyzer_sld_eigenvalues=baseline[
            "constant_analyzer_sld_eigenvalues"
        ],
    )
    paths.append(baseline_sld_path)

    summary = {
        "config": asdict(cfg),
        "physical_constraint": "Every analyzer segment uses +J0 * Z_c S_z.",
        "analyzer_tangent_source": "none (J-independent analyzer)",
        "sensing_time_selection": {
            "selected_time": cfg.sensing_time,
            "criterion": "maximum nominal reduced-bath QFI / sensing time",
            "candidate_times": sensing_selection_times.tolist(),
            "candidate_rates": sensing_selection_rates.tolist(),
        },
        "baseline": {
            "sensing_end": _metrics_summary(baseline["sensing_metrics"]),
            "sensing_end_fisher_hierarchy": baseline["sensing_hierarchy"],
            "sensing_end_sld_eigenvalues": baseline[
                "sensing_sld_eigenvalues"
            ].tolist(),
            "constant_reversed_drive_analyzer": _metrics_summary(
                baseline["constant_analyzer_metrics"]
            ),
            "constant_analyzer_fisher_hierarchy": baseline[
                "constant_analyzer_hierarchy"
            ],
            "constant_analyzer_sld_eigenvalues": baseline[
                "constant_analyzer_sld_eigenvalues"
            ].tolist(),
            "constant_analyzer_finite_difference": _finite_difference_summary(
                baseline["constant_analyzer_finite_difference"]
            ),
        },
        "optimized_analyzers": [
            {
                "K": result.K,
                "controls_central": result.controls_central.tolist(),
                "controls_bath": result.controls_bath.tolist(),
                "optimizer_objective": result.optimizer_objective,
                "optimizer_success": result.optimizer_success,
                "optimizer_message": result.optimizer_message,
                "robust_advantage": result.robust_statistics,
                "training_all_trusted": bool(np.all(result.training_trust)),
                "heldout_trusted_fraction": float(np.mean(result.heldout_trust)),
                "nominal_final_metrics": _metrics_summary(
                    result.heldout_metrics[int(np.argmin(np.abs(result.heldout_J - cfg.J0)))]
                ),
                "fisher_hierarchy": result.fisher_hierarchy,
                "nominal_bath_sld_eigenvalues": (
                    result.nominal_bath_sld_eigenvalues.tolist()
                ),
            }
            for result in results
        ],
    }
    summary_path = output / f"summary_{_base_tag(cfg)}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.append(summary_path)
    return paths


def print_baseline(baseline: dict[str, Any], cfg: AnalyzerConfig) -> None:
    print("\nBaseline diagnostics at J0")
    for label, key in (
        ("Sensing end (no analyzer)", "sensing_metrics"),
        ("Constant (-Omega,-omega,+J0) analyzer", "constant_analyzer_metrics"),
    ):
        metrics: InformationMetrics = baseline[key]
        print(
            f"  {label}: FQ_global={metrics.global_qfi:.10g}, "
            f"FQ_bath={metrics.bath_qfi:.10g}, "
            f"FC_projective_Sx={metrics.projective_sx_fi:.10g}, "
            f"FC_mean_Sx={metrics.mean_sx_fi:.10g}"
        )
        print(
            "    efficiencies: "
            f"bath-transfer={metrics.bath_transfer_efficiency:.6g}, "
            f"measurement={metrics.measurement_efficiency:.6g}, "
            f"total={metrics.total_readout_efficiency:.6g}"
        )
    print(f"  sensing hierarchy: {baseline['sensing_hierarchy']}")
    print(f"  constant-analyzer hierarchy: {baseline['constant_analyzer_hierarchy']}")
    print(
        "  bath-SLD spectral ranges: sensing="
        f"[{baseline['sensing_sld_eigenvalues'][0]:.6g}, "
        f"{baseline['sensing_sld_eigenvalues'][-1]:.6g}], analyzer="
        f"[{baseline['constant_analyzer_sld_eigenvalues'][0]:.6g}, "
        f"{baseline['constant_analyzer_sld_eigenvalues'][-1]:.6g}]"
    )
    check: FiniteDifferenceCheck = baseline["constant_analyzer_finite_difference"]
    print(
        "  constant-analyzer tangent/Hellinger validation: "
        f"trusted={check.trusted}, FI_tangent={check.tangent_fisher:.10g}, "
        f"FI_Hellinger={check.quarter_step_hellinger_fisher:.10g}"
    )
    print(
        rf"  Analyzer interaction is constrained to +J0 Z_c S_z, J0={cfg.J0:g}"
    )


def print_result(result: OptimizedAnalyzer) -> None:
    statistics = result.robust_statistics
    nominal_index = int(np.argmin(np.abs(result.heldout_J - result.training_J[len(result.training_J) // 2])))
    nominal = result.heldout_metrics[nominal_index]
    print(f"\nOptimized analyzer K={result.K}")
    print(
        f"  robust analyzer/Ramsey advantage: min={statistics['minimum']:.8g}, "
        f"median={statistics['median']:.8g}, mean={statistics['mean']:.8g}, "
        f"max={statistics['maximum']:.8g}"
    )
    print(
        f"  heldout trusted fraction={np.mean(result.heldout_trust):.3f}; "
        f"optimizer success={result.optimizer_success} ({result.optimizer_message})"
    )
    print(
        "  nominal efficiencies: "
        f"bath-transfer={nominal.bath_transfer_efficiency:.6g}, "
        f"measurement={nominal.measurement_efficiency:.6g}, "
        f"total={nominal.total_readout_efficiency:.6g}"
    )
    print(f"  Fisher hierarchy: {result.fisher_hierarchy}")
    print(
        "  nominal bath-SLD spectral range="
        f"[{result.nominal_bath_sld_eigenvalues[0]:.6g}, "
        f"{result.nominal_bath_sld_eigenvalues[-1]:.6g}]"
    )
    print(f"  u_c={np.array2string(result.controls_central, precision=6)}")
    print(f"  u_b={np.array2string(result.controls_bath, precision=6)}")


def _parse_segment_counts(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("segment counts must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one segment count is required")
    return values


def parse_config(argv: list[str] | None = None) -> AnalyzerConfig:
    defaults = AnalyzerConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--J0", type=float, default=defaults.J0)
    parser.add_argument("--sensing-time", type=float, default=defaults.sensing_time)
    parser.add_argument("--sensing-time-min", type=float, default=defaults.sensing_time_min)
    parser.add_argument("--sensing-time-max", type=float, default=defaults.sensing_time_max)
    parser.add_argument("--n-sensing-times", type=int, default=defaults.n_sensing_times)
    parser.add_argument("--analyzer-duration", type=float, default=defaults.analyzer_duration)
    parser.add_argument(
        "--segment-counts",
        type=_parse_segment_counts,
        default=defaults.segment_counts,
        help="comma-separated control-complexity ladder, e.g. 1,2,4,8",
    )
    parser.add_argument("--central-control-max", type=float, default=defaults.central_control_max)
    parser.add_argument("--bath-control-max", type=float, default=defaults.bath_control_max)
    parser.add_argument("--robust-fraction", type=float, default=defaults.robust_fraction)
    parser.add_argument("--n-training-J", type=int, default=defaults.n_training_J)
    parser.add_argument("--n-heldout-J", type=int, default=defaults.n_heldout_J)
    parser.add_argument("--softmin-sharpness", type=float, default=defaults.softmin_sharpness)
    parser.add_argument("--n-random-seeds", type=int, default=defaults.n_random_seeds)
    parser.add_argument("--random-seed", type=int, default=defaults.random_seed)
    parser.add_argument("--optimizer-max-iterations", type=int, default=defaults.optimizer_max_iterations)
    parser.add_argument("--power-penalty", type=float, default=defaults.power_penalty)
    parser.add_argument("--roughness-penalty", type=float, default=defaults.roughness_penalty)
    parser.add_argument("--probability-floor", type=float, default=defaults.probability_floor)
    parser.add_argument(
        "--probability-trust-threshold",
        type=float,
        default=defaults.probability_trust_threshold,
    )
    parser.add_argument("--hellinger-step", type=float, default=defaults.hellinger_step)
    parser.add_argument(
        "--hellinger-relative-tolerance",
        type=float,
        default=defaults.hellinger_relative_tolerance,
    )
    parser.add_argument("--ramsey-time-min", type=float, default=defaults.ramsey_time_min)
    parser.add_argument("--ramsey-time-max", type=float, default=defaults.ramsey_time_max)
    parser.add_argument("--n-ramsey-times", type=int, default=defaults.n_ramsey_times)
    parser.add_argument("--time-overhead", type=float, default=defaults.time_overhead)
    parser.add_argument("--output-directory", default=defaults.output_directory)
    parser.add_argument("--figure-format", default=defaults.figure_format)
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument(
        "--show",
        dest="show_figure",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_figure,
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=defaults.make_plots,
    )
    parser.add_argument(
        "--save-data",
        action=argparse.BooleanOptionalAction,
        default=defaults.save_data,
    )
    parser.add_argument(
        "--run-tests",
        action=argparse.BooleanOptionalAction,
        default=defaults.run_tests,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=defaults.progress,
    )
    return AnalyzerConfig(**vars(parser.parse_args(argv)))


def run_experiment(cfg: AnalyzerConfig) -> tuple[list[OptimizedAnalyzer], dict[str, Any], list[Path]]:
    """Run the fixed-sensing-time analyzer complexity experiment."""
    validate_config(cfg)
    operators = build_system_operators(cfg.N)
    selected_time, selection_times, selection_rates = select_sensing_time(cfg, operators)
    cfg = _replace_sensing_time(cfg, selected_time)
    print(f"Selected sensing time ts={selected_time:.12g} from bath-QFI-rate criterion")

    training_J = construct_J_ensemble(cfg, cfg.n_training_J)
    heldout_J = construct_J_ensemble(cfg, cfg.n_heldout_J)
    ramsey_cache: dict[float, float] = {}
    baseline = baseline_diagnostics(cfg, operators)
    print_baseline(baseline, cfg)

    first_problem = build_training_problem(
        cfg.segment_counts[0],
        training_J,
        cfg,
        operators,
        ramsey_cache,
    )
    if cfg.run_tests:
        run_numerical_tests(cfg, operators, first_problem)

    results: list[OptimizedAnalyzer] = []
    previous: OptimizedAnalyzer | None = None
    for K in cfg.segment_counts:
        if cfg.progress:
            print(f"\nOptimizing K={K} fixed-sign analyzer...")
        result = optimize_analyzer(
            K,
            training_J,
            heldout_J,
            cfg,
            operators,
            ramsey_cache,
            previous,
        )
        results.append(result)
        previous = result
        print_result(result)

    paths: list[Path] = []
    if cfg.make_plots:
        paths.extend(plot_results(results, cfg))
    if cfg.save_data:
        paths.extend(
            save_results(
                results,
                baseline,
                selection_times,
                selection_rates,
                cfg,
            )
        )
    for path in paths:
        print(f"Saved: {path}")
    return results, baseline, paths


def main(argv: list[str] | None = None) -> list[OptimizedAnalyzer]:
    cfg = parse_config(argv)
    results, _baseline, _paths = run_experiment(cfg)
    return results


if __name__ == "__main__":
    main()
