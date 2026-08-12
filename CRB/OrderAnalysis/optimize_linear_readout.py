"""Optimize a linear collective-bath readout after one sensing/decoding pair.

For every point on a rectangular ``(t_s, t_d)`` grid this script applies

    U_dec(J0, t_d) U_sense(J, t_s)

to a central ``|+x>`` state and a bath ``|0>^N`` state.  The centered
finite-difference states use ``J +/- dJ`` only in the sensing Hamiltonian;
the decoder coupling ``J0`` is fixed for all three trajectories.

The best first-moment Fisher information over every real linear collective
spin observable is found analytically from the full covariance matrix,

    max_n (n.T g)^2 / (n.T Gamma n) = g.T Gamma^+ g.

The reported optimization target is exclusively the information rate

    R_lin(t_s, t_d) = F_lin_max(t_s, t_d) / (t_s + t_d).

The zero-duration point is excluded.  The default run saves the complete
scan, a JSON summary, and five rate-focused figures.  Use ``--include-qfi``
to add the optional reduced-bath QFI ceiling and accessibility heatmap.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from CRB.OrderAnalysis.Echo_phase_cycling import (  # noqa: E402
    apply_propagator,
    reduced_bath_density_matrix,
)
from CRB.crb_core import (  # noqa: E402
    build_bath_operators,
    build_hamiltonian,
    central_spin_state,
    coherent_bath_state,
    qfi_vectorized,
    save_plot,
)


Spectrum = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class LinearReadoutConfig:
    """Physical, grid, validation, output, and plotting controls."""

    N: int = 15
    Omega: float = 2.5
    omega: float = 1.0
    J_nominal: float = 1.0
    J0: float | None = None
    dJ: float = 1e-4

    ts_min: float = 0.0
    ts_max: float = 30.0
    n_ts: int = 151
    td_min: float = 0.0
    td_max: float = 5.0
    n_td: int = 101

    covariance_rtol: float = 1e-10
    variance_atol: float = 1e-12
    qfi_tol: float = 1e-12
    validation_atol: float = 1e-9
    validation_rtol: float = 2e-6
    finite_difference_rtol: float = 1e-2

    include_qfi: bool = False
    run_tests: bool = True
    run_convergence: bool = True
    random_test_directions: int = 100
    random_seed: int = 20260809

    output_directory: str = "graphs/optimize_linear_readout"
    figure_format: str = "png"
    figure_dpi: int = 200
    colormap: str = "viridis"
    make_plots: bool = True
    save_data: bool = True
    show_figure: bool = False
    progress: bool = True

    @property
    def decoder_coupling(self) -> float:
        """Use the nominal coupling unless an explicit decoder value is set."""
        return self.J_nominal if self.J0 is None else self.J0


@dataclass(frozen=True, slots=True)
class LinearReadoutScan:
    """All scalar and directional fields on the ``(t_s, t_d)`` grid."""

    ts_values: np.ndarray
    td_values: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    rate: np.ndarray
    rate_x: np.ndarray
    rate_y: np.ndarray
    rate_z: np.ndarray
    eta: np.ndarray
    covariance_eigenvalues: np.ndarray
    covariance_condition: np.ndarray
    covariance_rank: np.ndarray
    accessibility_fraction: np.ndarray
    ramsey_rate: np.ndarray


def validate_config(cfg: LinearReadoutConfig) -> None:
    """Reject invalid physics, grids, tolerances, and output controls."""
    finite = {
        "Omega": cfg.Omega,
        "omega": cfg.omega,
        "J_nominal": cfg.J_nominal,
        "decoder_coupling": cfg.decoder_coupling,
        "dJ": cfg.dJ,
        "ts_min": cfg.ts_min,
        "ts_max": cfg.ts_max,
        "td_min": cfg.td_min,
        "td_max": cfg.td_max,
        "covariance_rtol": cfg.covariance_rtol,
        "variance_atol": cfg.variance_atol,
        "qfi_tol": cfg.qfi_tol,
        "validation_atol": cfg.validation_atol,
        "validation_rtol": cfg.validation_rtol,
        "finite_difference_rtol": cfg.finite_difference_rtol,
    }
    nonfinite = [name for name, value in finite.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError(f"configuration values must be finite: {nonfinite}")
    if cfg.N < 1:
        raise ValueError("N must be positive")
    if cfg.dJ <= 0.0:
        raise ValueError("dJ must be positive")
    if cfg.ts_min < 0.0 or cfg.ts_max <= cfg.ts_min or cfg.n_ts < 2:
        raise ValueError("require 0 <= ts_min < ts_max and n_ts >= 2")
    if cfg.td_min < 0.0 or cfg.td_max <= cfg.td_min or cfg.n_td < 2:
        raise ValueError("require 0 <= td_min < td_max and n_td >= 2")
    for name in (
        "covariance_rtol",
        "variance_atol",
        "qfi_tol",
        "validation_atol",
        "validation_rtol",
        "finite_difference_rtol",
    ):
        if getattr(cfg, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if cfg.random_test_directions < 1:
        raise ValueError("random_test_directions must be positive")
    if cfg.figure_dpi < 1:
        raise ValueError("figure_dpi must be positive")
    if not cfg.figure_format.isalnum():
        raise ValueError("figure_format must be alphanumeric without a leading dot")
    if cfg.colormap not in plt.colormaps():
        raise ValueError(f"unknown matplotlib colormap: {cfg.colormap!r}")


def initial_joint_state(cfg: LinearReadoutConfig) -> np.ndarray:
    """Return exactly ``|+x>_central tensor |0>_bath^N``."""
    central = central_spin_state(np.pi / 2.0, 0.0).full().ravel()
    bath = coherent_bath_state(cfg.N, theta=0.0, phi=0.0)
    return np.kron(central, bath)


def spectral_hamiltonian(cfg: LinearReadoutConfig, J: float) -> Spectrum:
    """Diagonalize ``Omega X_c + J Z_c S_z + omega S_x``."""
    matrix = build_hamiltonian(cfg.Omega, J, cfg.N, cfg.omega).full()
    return np.linalg.eigh(matrix)


def spectral_decoder_hamiltonian(cfg: LinearReadoutConfig) -> Spectrum:
    """Diagonalize the fixed signal-preserving decoder once."""
    matrix = build_hamiltonian(
        -cfg.Omega,
        cfg.decoder_coupling,
        cfg.N,
        -cfg.omega,
    ).full()
    return np.linalg.eigh(matrix)


def apply_propagator_many(
    state: np.ndarray,
    spectrum: Spectrum,
    times: np.ndarray,
) -> np.ndarray:
    """Apply one fixed spectral propagator at every requested time.

    The returned array has shape ``(state_dimension, number_of_times)``.
    """
    eigenvalues, eigenvectors = spectrum
    coefficients = eigenvectors.conj().T @ state
    phases = np.exp(-1j * np.outer(eigenvalues, np.asarray(times, dtype=float)))
    return eigenvectors @ (coefficients[:, None] * phases)


def _real_trace(matrix: np.ndarray, label: str, atol: float) -> float:
    """Return a trace's real part after checking its imaginary residual."""
    value = np.trace(matrix)
    scale = max(1.0, abs(float(np.real(value))))
    if abs(float(np.imag(value))) > atol * scale:
        raise RuntimeError(
            f"non-negligible imaginary residual in {label}: {value!r}"
        )
    return float(np.real(value))


def collective_mean_vector(
    rho: np.ndarray,
    operators: Iterable[np.ndarray],
    imaginary_atol: float = 1e-9,
) -> np.ndarray:
    """Return ``(<Sx>, <Sy>, <Sz>)`` with reality checks."""
    return np.asarray(
        [
            _real_trace(rho @ operator, f"<{axis}>", imaginary_atol)
            for operator, axis in zip(operators, ("Sx", "Sy", "Sz"))
        ],
        dtype=float,
    )


def collective_derivative_vector(
    drho: np.ndarray,
    operators: Iterable[np.ndarray],
    imaginary_atol: float = 1e-9,
) -> np.ndarray:
    """Return ``d_J(<Sx>, <Sy>, <Sz>)`` with reality checks."""
    return np.asarray(
        [
            _real_trace(drho @ operator, f"d_J<{axis}>", imaginary_atol)
            for operator, axis in zip(operators, ("Sx", "Sy", "Sz"))
        ],
        dtype=float,
    )


def collective_covariance_matrix(
    rho: np.ndarray,
    operators: Iterable[np.ndarray],
    imaginary_atol: float = 1e-9,
) -> np.ndarray:
    """Construct the complete real symmetric collective-spin covariance."""
    ops = tuple(operators)
    means = collective_mean_vector(rho, ops, imaginary_atol)
    covariance = np.empty((3, 3), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            symmetrized = 0.5 * (ops[i] @ ops[j] + ops[j] @ ops[i])
            moment = _real_trace(
                rho @ symmetrized,
                f"symmetrized covariance moment ({i}, {j})",
                imaginary_atol,
            )
            covariance[i, j] = covariance[j, i] = moment - means[i] * means[j]
    return 0.5 * (covariance + covariance.T)


def _stable_linear_optimum(
    gradient: np.ndarray,
    covariance: np.ndarray,
    covariance_rtol: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Return pseudoinverse optimum and covariance diagnostics."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = max(0.0, float(eigenvalues[-1]))
    cutoff = covariance_rtol * maximum
    retained = eigenvalues > cutoff
    inverse_values = np.zeros(3, dtype=float)
    inverse_values[retained] = 1.0 / eigenvalues[retained]
    pseudoinverse = (eigenvectors * inverse_values) @ eigenvectors.T
    fisher = max(0.0, float(gradient @ pseudoinverse @ gradient))
    raw_direction = pseudoinverse @ gradient
    norm = float(np.linalg.norm(raw_direction))
    if norm > 0.0:
        direction = raw_direction / norm
        # Fix the physically irrelevant sign for reproducible direction plots.
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction = -direction
    else:
        direction = np.array([1.0, 0.0, 0.0])
    rank = int(np.count_nonzero(retained))
    condition = (
        maximum / float(np.min(eigenvalues[retained]))
        if rank == 3
        else float("inf")
    )
    return fisher, direction, eigenvalues, pseudoinverse, rank, condition


def _directional_fisher(
    gradient: np.ndarray,
    covariance: np.ndarray,
    direction: np.ndarray,
    variance_atol: float,
) -> float:
    """Evaluate the error-propagation FI for one real direction."""
    derivative = float(direction @ gradient)
    variance = float(direction @ covariance @ direction)
    if variance <= variance_atol:
        if derivative * derivative <= variance_atol:
            return 0.0
        return float("inf")
    return derivative * derivative / variance


def optimal_linear_moment_fisher(
    rho: np.ndarray,
    drho: np.ndarray,
    operators: Iterable[np.ndarray],
    *,
    covariance_rtol: float = 1e-10,
    variance_atol: float = 1e-12,
    validation_atol: float = 1e-9,
    validation_rtol: float = 2e-6,
) -> dict[str, Any]:
    """Optimize over every linear collective-spin measurement direction."""
    ops = tuple(np.asarray(operator, dtype=complex) for operator in operators)
    if len(ops) != 3:
        raise ValueError("operators must contain exactly Sx, Sy, and Sz")
    gradient = collective_derivative_vector(drho, ops, validation_atol)
    covariance = collective_covariance_matrix(rho, ops, validation_atol)
    (
        fisher,
        direction,
        eigenvalues,
        pseudoinverse,
        rank,
        condition,
    ) = _stable_linear_optimum(gradient, covariance, covariance_rtol)

    psd_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues[0]) < -validation_atol * psd_scale:
        raise RuntimeError(f"covariance is not positive semidefinite: {eigenvalues}")

    nx, ny, nz = direction
    theta = float(np.arccos(np.clip(nz, -1.0, 1.0)))
    phi = float(np.arctan2(ny, nx))

    # Reconstruct S_opt explicitly, independently of the 3x3 calculation.
    optimal_observable = sum(component * op for component, op in zip(direction, ops))
    optimal_mean = _real_trace(rho @ optimal_observable, "<S_opt>", validation_atol)
    derivative = _real_trace(drho @ optimal_observable, "d_J<S_opt>", validation_atol)
    second_moment = _real_trace(
        rho @ (optimal_observable @ optimal_observable),
        "<S_opt^2>",
        validation_atol,
    )
    direct_variance = second_moment - optimal_mean * optimal_mean
    direct_check = (
        derivative * derivative / direct_variance
        if direct_variance > variance_atol
        else 0.0
    )
    if not np.isclose(
        direct_check,
        fisher,
        atol=validation_atol,
        rtol=validation_rtol,
    ):
        raise RuntimeError(
            "direct S_opt reconstruction disagrees with pseudoinverse result: "
            f"direct={direct_check:.16g}, analytic={fisher:.16g}, "
            f"eigenvalues={eigenvalues}"
        )

    axis_fisher = np.asarray(
        [
            _directional_fisher(gradient, covariance, axis, variance_atol)
            for axis in np.eye(3)
        ]
    )
    if not np.all(np.isfinite(axis_fisher)):
        raise RuntimeError(
            "a Cartesian observable has negligible variance but a nonzero "
            f"finite-difference slope: F_xyz={axis_fisher}"
        )
    if np.any(
        axis_fisher
        > fisher + validation_atol + validation_rtol * np.maximum(1.0, axis_fisher)
    ):
        raise RuntimeError(
            "optimized FI is smaller than a Cartesian-axis FI: "
            f"F_opt={fisher}, F_xyz={axis_fisher}"
        )

    return {
        "fisher": fisher,
        "direction": direction,
        "theta": theta,
        "phi": phi,
        "gradient": gradient,
        "covariance": covariance,
        "covariance_eigenvalues": eigenvalues,
        "covariance_pseudoinverse": pseudoinverse,
        "covariance_rank": rank,
        "covariance_condition": condition,
        "direct_check": direct_check,
        "axis_fisher": axis_fisher,
    }


def _validate_reduced_states(
    rho: np.ndarray,
    rho_plus: np.ndarray,
    rho_minus: np.ndarray,
    drho: np.ndarray,
    cfg: LinearReadoutConfig,
) -> None:
    """Check normalization, trace derivative, and Hermiticity."""
    for label, state in (
        ("rho", rho),
        ("rho_plus", rho_plus),
        ("rho_minus", rho_minus),
    ):
        if not np.allclose(
            state,
            state.conj().T,
            atol=cfg.validation_atol,
            rtol=cfg.validation_rtol,
        ):
            raise RuntimeError(f"{label} is not Hermitian")
        if not np.isclose(
            np.trace(state),
            1.0,
            atol=cfg.validation_atol,
            rtol=cfg.validation_rtol,
        ):
            raise RuntimeError(f"{label} is not normalized: trace={np.trace(state)}")
    if not np.allclose(
        drho,
        drho.conj().T,
        atol=cfg.validation_atol,
        rtol=cfg.validation_rtol,
    ):
        raise RuntimeError("drho is not Hermitian")
    if abs(np.trace(drho)) > cfg.validation_atol + cfg.validation_rtol:
        raise RuntimeError(f"drho is not traceless: trace={np.trace(drho)}")


def evaluate_ts_td_point(
    sensed_nominal: np.ndarray,
    sensed_plus: np.ndarray,
    sensed_minus: np.ndarray,
    td: float,
    decoder_spectrum: Spectrum,
    operators: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: LinearReadoutConfig,
) -> dict[str, Any]:
    """Evaluate one point, always using the same fixed decoder spectrum."""
    decoded = tuple(
        apply_propagator(state, *decoder_spectrum, td)
        for state in (sensed_nominal, sensed_plus, sensed_minus)
    )
    rho, rho_plus, rho_minus = (
        reduced_bath_density_matrix(state, cfg.N) for state in decoded
    )
    drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
    _validate_reduced_states(rho, rho_plus, rho_minus, drho, cfg)
    result = optimal_linear_moment_fisher(
        rho,
        drho,
        operators,
        covariance_rtol=cfg.covariance_rtol,
        variance_atol=cfg.variance_atol,
        validation_atol=cfg.validation_atol,
        validation_rtol=cfg.validation_rtol,
    )
    result.update({"rho": rho, "drho": drho})
    if cfg.include_qfi:
        bath_qfi = qfi_vectorized(rho, drho, tol=cfg.qfi_tol)
        allowed = cfg.validation_atol + cfg.validation_rtol * max(1.0, bath_qfi)
        if result["fisher"] > bath_qfi + allowed:
            raise RuntimeError(
                "linear moment FI exceeds bath QFI: "
                f"F_lin={result['fisher']}, F_Q={bath_qfi}"
            )
        result["bath_qfi"] = bath_qfi
    else:
        result["bath_qfi"] = float("nan")
    return result


def scan_ts_td(cfg: LinearReadoutConfig) -> LinearReadoutScan:
    """Run the rectangular scan with only four main eigendecompositions."""
    validate_config(cfg)
    ts_values = np.linspace(cfg.ts_min, cfg.ts_max, cfg.n_ts)
    td_values = np.linspace(cfg.td_min, cfg.td_max, cfg.n_td)
    initial_state = initial_joint_state(cfg)

    # These are the only four Hamiltonian diagonalizations in the main scan.
    sensing_spectra = (
        spectral_hamiltonian(cfg, cfg.J_nominal),
        spectral_hamiltonian(cfg, cfg.J_nominal + cfg.dJ),
        spectral_hamiltonian(cfg, cfg.J_nominal - cfg.dJ),
    )
    decoder_spectrum = spectral_decoder_hamiltonian(cfg)
    sensed_trajectories = tuple(
        apply_propagator_many(initial_state, spectrum, ts_values)
        for spectrum in sensing_spectra
    )

    bath_operators = build_bath_operators(cfg.N)
    operators = tuple(bath_operators[key] for key in ("Jx", "Jy", "Jz"))
    shape = (cfg.n_ts, cfg.n_td)
    fisher = np.empty(shape)
    direction = np.empty((*shape, 3))
    angles = np.empty((*shape, 2))
    axis_fisher = np.empty((*shape, 3))
    covariance_eigenvalues = np.empty((*shape, 3))
    covariance_condition = np.empty(shape)
    covariance_rank = np.empty(shape, dtype=np.int8)
    bath_qfi = np.full(shape, np.nan)

    progress_stride = max(1, cfg.n_ts // 10)
    for its, ts in enumerate(ts_values):
        sensed_states = tuple(trajectory[:, its] for trajectory in sensed_trajectories)
        decoded_trajectories = tuple(
            apply_propagator_many(state, decoder_spectrum, td_values)
            for state in sensed_states
        )
        for itd, _td in enumerate(td_values):
            rho, rho_plus, rho_minus = (
                reduced_bath_density_matrix(trajectory[:, itd], cfg.N)
                for trajectory in decoded_trajectories
            )
            drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
            _validate_reduced_states(rho, rho_plus, rho_minus, drho, cfg)
            point = optimal_linear_moment_fisher(
                rho,
                drho,
                operators,
                covariance_rtol=cfg.covariance_rtol,
                variance_atol=cfg.variance_atol,
                validation_atol=cfg.validation_atol,
                validation_rtol=cfg.validation_rtol,
            )
            fisher[its, itd] = point["fisher"]
            direction[its, itd] = point["direction"]
            angles[its, itd] = point["theta"], point["phi"]
            axis_fisher[its, itd] = point["axis_fisher"]
            covariance_eigenvalues[its, itd] = point["covariance_eigenvalues"]
            covariance_condition[its, itd] = point["covariance_condition"]
            covariance_rank[its, itd] = point["covariance_rank"]
            if cfg.include_qfi:
                value = qfi_vectorized(rho, drho, tol=cfg.qfi_tol)
                allowed = cfg.validation_atol + cfg.validation_rtol * max(1.0, value)
                if point["fisher"] > value + allowed:
                    raise RuntimeError(
                        f"F_lin > F_Q at ts={ts}, td={_td}: "
                        f"{point['fisher']} > {value}"
                    )
                bath_qfi[its, itd] = value
        if cfg.progress and (
            its == 0 or its == cfg.n_ts - 1 or (its + 1) % progress_stride == 0
        ):
            print(f"scan: completed {its + 1}/{cfg.n_ts} sensing times")

    total_time = ts_values[:, None] + td_values[None, :]
    rate = np.full(shape, np.nan)
    axis_rate = np.full((*shape, 3), np.nan)
    # The zero-duration protocol has no defined information rate.
    valid_rate = total_time > 0.0
    rate[valid_rate] = fisher[valid_rate] / total_time[valid_rate]
    axis_rate[valid_rate] = (
        axis_fisher[valid_rate] / total_time[valid_rate][:, None]
    )
    eta = np.full(shape, np.inf)
    informative = valid_rate & (fisher > 0.0)
    eta[informative] = np.sqrt(total_time[informative] / fisher[informative])
    accessibility = np.full(shape, np.nan)
    valid_qfi = bath_qfi > cfg.qfi_tol
    accessibility[valid_qfi] = fisher[valid_qfi] / bath_qfi[valid_qfi]

    ramsey_rate = calculate_ramsey_benchmark(cfg, ts_values)
    return LinearReadoutScan(
        ts_values=ts_values,
        td_values=td_values,
        nx=direction[..., 0],
        ny=direction[..., 1],
        nz=direction[..., 2],
        theta=angles[..., 0],
        phi=angles[..., 1],
        rate=rate,
        rate_x=axis_rate[..., 0],
        rate_y=axis_rate[..., 1],
        rate_z=axis_rate[..., 2],
        eta=eta,
        covariance_eigenvalues=covariance_eigenvalues,
        covariance_condition=covariance_condition,
        covariance_rank=covariance_rank,
        accessibility_fraction=accessibility,
        ramsey_rate=ramsey_rate,
    )


def calculate_ramsey_benchmark(
    cfg: LinearReadoutConfig,
    times: np.ndarray,
) -> np.ndarray:
    """Numerically reproduce the repository's undriven bath Ramsey protocol.

    Ramsey uses ``Omega=omega=0``, central ``|+x>``, and bath ``|+x>^N``.
    Its reduced-bath QFI is calculated with the same scaled ``S_z`` operator,
    centered finite difference, and total-time convention as the main scan.
    """
    central = central_spin_state(np.pi / 2.0, 0.0).full().ravel()
    bath = coherent_bath_state(cfg.N, theta=np.pi / 2.0, phi=0.0)
    initial = np.kron(central, bath)

    def spectrum(J: float) -> Spectrum:
        hamiltonian = build_hamiltonian(0.0, J, cfg.N, 0.0).full()
        return np.linalg.eigh(hamiltonian)

    trajectories = tuple(
        apply_propagator_many(initial, spectrum(J), times)
        for J in (
            cfg.J_nominal,
            cfg.J_nominal + cfg.dJ,
            cfg.J_nominal - cfg.dJ,
        )
    )
    qfi = np.empty(len(times), dtype=float)
    for index in range(len(times)):
        rho, rho_plus, rho_minus = (
            reduced_bath_density_matrix(trajectory[:, index], cfg.N)
            for trajectory in trajectories
        )
        drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
        qfi[index] = qfi_vectorized(rho, drho, tol=cfg.qfi_tol)
    rate = np.full(len(times), np.nan)
    valid = times > 0.0
    rate[valid] = qfi[valid] / times[valid]
    return rate


def find_optima(scan: LinearReadoutScan) -> dict[str, dict[str, Any]]:
    """Locate the maximum of ``F_lin_max / (t_s + t_d)``."""
    rate_flat = int(np.nanargmax(scan.rate))

    def describe(flat_index: int, objective: np.ndarray) -> dict[str, Any]:
        its, itd = np.unravel_index(flat_index, objective.shape)
        return {
            "index": (int(its), int(itd)),
            "value": float(objective[its, itd]),
            "rate": float(scan.rate[its, itd]),
            "eta": float(scan.eta[its, itd]),
            "ts": float(scan.ts_values[its]),
            "td": float(scan.td_values[itd]),
            "direction": np.array(
                [scan.nx[its, itd], scan.ny[its, itd], scan.nz[its, itd]]
            ),
            "theta": float(scan.theta[its, itd]),
            "phi": float(scan.phi[its, itd]),
            "covariance_eigenvalues": scan.covariance_eigenvalues[its, itd],
            "covariance_condition": float(scan.covariance_condition[its, itd]),
        }

    return {"rate": describe(rate_flat, scan.rate)}


def _fresh_point(
    cfg: LinearReadoutConfig,
    ts: float,
    td: float,
    dJ: float | None = None,
) -> dict[str, Any]:
    """Re-evaluate a selected point (used only by validation tests)."""
    step = cfg.dJ if dJ is None else dJ
    local_values = asdict(cfg)
    local_values["dJ"] = step
    local_values["progress"] = False
    local_cfg = LinearReadoutConfig(**local_values)
    initial = initial_joint_state(local_cfg)
    spectra = (
        spectral_hamiltonian(local_cfg, local_cfg.J_nominal),
        spectral_hamiltonian(local_cfg, local_cfg.J_nominal + step),
        spectral_hamiltonian(local_cfg, local_cfg.J_nominal - step),
    )
    sensed = tuple(
        apply_propagator(initial, *spectrum, ts) for spectrum in spectra
    )
    operators_dict = build_bath_operators(local_cfg.N)
    operators = tuple(operators_dict[key] for key in ("Jx", "Jy", "Jz"))
    return evaluate_ts_td_point(
        sensed[0],
        sensed[1],
        sensed[2],
        td,
        spectral_decoder_hamiltonian(local_cfg),
        operators,
        local_cfg,
    )


def finite_difference_convergence(
    cfg: LinearReadoutConfig,
    points: Iterable[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    """Re-run selected rate points with ``dJ``, ``dJ/2``, and ``dJ/4``."""
    reports: list[dict[str, Any]] = []
    for label, ts, td in points:
        evaluations = [
            _fresh_point(cfg, ts, td, cfg.dJ / divisor)
            for divisor in (1.0, 2.0, 4.0)
        ]
        total_time = ts + td
        if total_time <= 0.0:
            raise ValueError("finite-difference rate check requires ts + td > 0")
        values = np.asarray(
            [point["fisher"] / total_time for point in evaluations]
        )
        directions = np.asarray([point["direction"] for point in evaluations])
        relative_change = abs(values[-1] - values[-2]) / max(
            cfg.validation_atol,
            abs(values[-1]),
        )
        alignment = abs(float(directions[-1] @ directions[-2]))
        converged = relative_change <= cfg.finite_difference_rtol
        if values[-1] > cfg.validation_atol:
            converged = converged and (
                1.0 - alignment <= cfg.finite_difference_rtol
            )
        reports.append(
            {
                "label": label,
                "ts": ts,
                "td": td,
                "dJ_values": [cfg.dJ, cfg.dJ / 2.0, cfg.dJ / 4.0],
                "rate_values": values.tolist(),
                "directions": directions.tolist(),
                "finest_relative_change": float(relative_change),
                "finest_direction_alignment": alignment,
                "converged": bool(converged),
            }
        )
    return reports


def run_numerical_tests(
    cfg: LinearReadoutConfig,
    scan: LinearReadoutScan,
    optima: dict[str, dict[str, Any]],
) -> None:
    """Run the requested unit-style analytic and representative-state tests."""
    # Test 1: a deliberately x-optimal covariance/gradient pair.
    synthetic_gradient = np.array([2.0, 0.0, 0.0])
    synthetic_covariance = np.diag([4.0, 9.0, 16.0])
    synthetic = _stable_linear_optimum(
        synthetic_gradient,
        synthetic_covariance,
        cfg.covariance_rtol,
    )
    if not np.isclose(synthetic[0], 1.0) or not np.allclose(
        np.abs(synthetic[1]),
        [1.0, 0.0, 0.0],
    ):
        raise AssertionError("x-optimal analytic test failed")

    representative = {
        (max(1, cfg.n_ts // 3), cfg.n_td // 3),
        (cfg.n_ts // 2, cfg.n_td // 2),
        tuple(optima["rate"]["index"]),
    }
    rng = np.random.default_rng(cfg.random_seed)
    theta_grid = np.linspace(0.0, np.pi, 181)
    phi_grid = np.linspace(-np.pi, np.pi, 361, endpoint=False)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    brute_directions = np.column_stack(
        (
            (np.sin(theta_mesh) * np.cos(phi_mesh)).ravel(),
            (np.sin(theta_mesh) * np.sin(phi_mesh)).ravel(),
            np.cos(theta_mesh).ravel(),
        )
    )

    for its, itd in representative:
        point = _fresh_point(
            cfg,
            float(scan.ts_values[its]),
            float(scan.td_values[itd]),
        )
        gradient = point["gradient"]
        covariance = point["covariance"]
        optimum = point["fisher"]

        random_directions = rng.normal(size=(cfg.random_test_directions, 3))
        random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
        random_values = np.asarray(
            [
                _directional_fisher(
                    gradient,
                    covariance,
                    direction,
                    cfg.variance_atol,
                )
                for direction in random_directions
            ]
        )
        allowed = cfg.validation_atol + cfg.validation_rtol * max(1.0, optimum)
        if np.nanmax(random_values) > optimum + allowed:
            raise AssertionError("a random direction exceeded the analytic optimum")

        derivatives = brute_directions @ gradient
        variances = np.einsum(
            "ni,ij,nj->n",
            brute_directions,
            covariance,
            brute_directions,
        )
        brute_values = np.zeros_like(derivatives)
        valid = variances > cfg.variance_atol
        brute_values[valid] = derivatives[valid] ** 2 / variances[valid]
        brute_maximum = float(np.max(brute_values))
        if brute_maximum > optimum + allowed:
            raise AssertionError("spherical scan exceeded the analytic optimum")
        if optimum > cfg.validation_atol:
            grid_gap = (optimum - brute_maximum) / optimum
            if grid_gap > 2e-2:
                raise AssertionError(
                    f"spherical scan did not approach optimum (gap={grid_gap})"
                )
    print("numerical tests: analytic, random-direction, spherical, PSD, and QFI checks passed")


def _output_directory(cfg: LinearReadoutConfig) -> Path:
    path = Path(cfg.output_directory)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tag(cfg: LinearReadoutConfig) -> str:
    def number(value: float) -> str:
        return f"{value:.6g}".replace("-", "m").replace(".", "p")

    return (
        f"N{cfg.N}_Om{number(cfg.Omega)}_om{number(cfg.omega)}_"
        f"J{number(cfg.J_nominal)}_J0{number(cfg.decoder_coupling)}"
    )


def _heatmap(
    axis: plt.Axes,
    scan: LinearReadoutScan,
    values: np.ndarray,
    label: str,
    cfg: LinearReadoutConfig,
) -> None:
    mesh = axis.pcolormesh(
        scan.ts_values,
        scan.td_values,
        values.T,
        shading="auto",
        cmap=cfg.colormap,
    )
    axis.set_xlabel(r"Sensing time $t_s$")
    axis.set_ylabel(r"Decoding time $t_d$")
    plt.colorbar(mesh, ax=axis, label=label)


def plot_heatmaps(
    scan: LinearReadoutScan,
    optima: dict[str, dict[str, Any]],
    cfg: LinearReadoutConfig,
) -> list[Path]:
    """Create the five required figures and optional accessibility heatmap."""
    tag = _tag(cfg)
    paths: list[Path] = []

    def save_figure(figure: Any, filename: str, plot_name: str) -> Path:
        return save_plot(
            figure,
            filename,
            metadata={
                "config": cfg,
                "plot": plot_name,
                "optima": optima,
                "scan": scan,
            },
            script_path=__file__,
            format=cfg.figure_format,
            dpi=cfg.figure_dpi,
            bbox_inches="tight",
        )

    rate = optima["rate"]
    positive = scan.rate > 0.0
    log_rate = np.full_like(scan.rate, np.nan)
    log_rate[positive] = np.log10(scan.rate[positive])
    figure, axis = plt.subplots(figsize=(10.5, 7.0))
    _heatmap(axis, scan, log_rate, r"$\log_{10}R_{\rm lin}$", cfg)
    axis.plot(rate["ts"], rate["td"], "r*", markersize=14)
    nx, ny, nz = rate["direction"]
    axis.set_title(
        rf"$N={cfg.N}$, $\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, "
        rf"$J={cfg.J_nominal:g}$, $J_0={cfg.decoder_coupling:g}$" "\n"
        rf"max $R_{{\rm lin}}={rate['rate']:.6g}$ at "
        rf"$(t_s,t_d)=({rate['ts']:.6g},{rate['td']:.6g})$, "
        rf"$n=({nx:.4f},{ny:.4f},{nz:.4f})$"
    )
    figure.tight_layout()
    paths.append(
        save_figure(
            figure,
            f"figure1_log_rate_{tag}.{cfg.figure_format}",
            "log_information_rate_heatmap",
        )
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 7.0))
    _heatmap(axis, scan, scan.rate, r"$R_{\rm lin}$", cfg)
    axis.plot(rate["ts"], rate["td"], "r*", markersize=14)
    nx, ny, nz = rate["direction"]
    axis.set_title(
        rf"max $R_{{\rm lin}}={rate['rate']:.6g}$ at "
        rf"$(t_s,t_d)=({rate['ts']:.6g},{rate['td']:.6g})$" "\n"
        rf"$n=({nx:.4f},{ny:.4f},{nz:.4f})$, "
        rf"$\theta={rate['theta']:.5f}$, $\phi={rate['phi']:.5f}$"
    )
    figure.tight_layout()
    paths.append(
        save_figure(
            figure,
            f"figure2_rate_{tag}.{cfg.figure_format}",
            "information_rate_heatmap",
        )
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 7.0))
    finite_eta = np.where(np.isfinite(scan.eta), scan.eta, np.nan)
    _heatmap(axis, scan, finite_eta, r"$\eta_{\rm lin}$", cfg)
    axis.plot(rate["ts"], rate["td"], "r*", markersize=14)
    axis.set_title(
        rf"Minimum $\eta_{{\rm lin}}={rate['eta']:.6g}$ at "
        rf"$(t_s,t_d)=({rate['ts']:.6g},{rate['td']:.6g})$"
    )
    figure.tight_layout()
    paths.append(
        save_figure(
            figure,
            f"figure3_eta_{tag}.{cfg.figure_format}",
            "eta_heatmap",
        )
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)

    its = int(rate["index"][0])
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    axis.plot(scan.td_values, scan.rate[its], label=r"$R_{\rm lin}^{\max}$", lw=2.2)
    axis.plot(scan.td_values, scan.rate_x[its], label=r"$R_x$")
    axis.plot(scan.td_values, scan.rate_y[its], label=r"$R_y$")
    axis.plot(scan.td_values, scan.rate_z[its], label=r"$R_z$")
    axis.set_xlabel(r"Decoding time $t_d$")
    axis.set_ylabel(r"Information rate $F/(t_s+t_d)$")
    axis.set_title(rf"Linear-readout rate comparison at $t_s={rate['ts']:.6g}$")
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    paths.append(
        save_figure(
            figure,
            f"figure4_axes_{tag}.{cfg.figure_format}",
            "readout_axis_rate_comparison",
        )
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    axis.plot(scan.td_values, scan.nx[its], label=r"$n_x$")
    axis.plot(scan.td_values, scan.ny[its], label=r"$n_y$")
    axis.plot(scan.td_values, scan.nz[its], label=r"$n_z$")
    axis.set_xlabel(r"Decoding time $t_d$")
    axis.set_ylabel("Optimal direction component")
    axis.set_ylim(-1.05, 1.05)
    axis.set_title(rf"Optimal measurement direction at $t_s={rate['ts']:.6g}$")
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    paths.append(
        save_figure(
            figure,
            f"figure5_direction_{tag}.{cfg.figure_format}",
            "optimal_measurement_direction",
        )
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)

    if cfg.include_qfi:
        figure, axis = plt.subplots(figsize=(10.5, 7.0))
        _heatmap(
            axis,
            scan,
            scan.accessibility_fraction,
            r"$F_{\rm lin}^{\max}/F_Q^{\rm bath}$",
            cfg,
        )
        axis.set_title("Fraction of reduced-bath QFI accessible to a linear moment")
        figure.tight_layout()
        paths.append(
            save_figure(
                figure,
                f"figure6_accessibility_{tag}.{cfg.figure_format}",
                "qfi_accessibility_heatmap",
            )
        )
        if cfg.show_figure:
            plt.show()
        plt.close(figure)
    return paths


def save_results(
    scan: LinearReadoutScan,
    optima: dict[str, dict[str, Any]],
    convergence: list[dict[str, Any]],
    cfg: LinearReadoutConfig,
) -> tuple[Path, Path]:
    """Persist rate-focused grid fields plus a human-readable summary."""
    output = _output_directory(cfg)
    tag = _tag(cfg)
    data_path = output / f"linear_readout_scan_{tag}.npz"
    np.savez_compressed(
        data_path,
        ts_values=scan.ts_values,
        td_values=scan.td_values,
        rate=scan.rate,
        rate_x=scan.rate_x,
        rate_y=scan.rate_y,
        rate_z=scan.rate_z,
        eta=scan.eta,
        nx=scan.nx,
        ny=scan.ny,
        nz=scan.nz,
        theta=scan.theta,
        phi=scan.phi,
        covariance_eigenvalues=scan.covariance_eigenvalues,
        covariance_condition=scan.covariance_condition,
        covariance_rank=scan.covariance_rank,
        accessibility_fraction=scan.accessibility_fraction,
        ramsey_rate=scan.ramsey_rate,
    )
    best_ramsey_index = int(np.nanargmax(scan.ramsey_rate))
    best_ramsey = float(scan.ramsey_rate[best_ramsey_index])

    def serializable_optimum(point: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in point.items()
        }

    summary = {
        "config": asdict(cfg),
        "resolved_J0": cfg.decoder_coupling,
        "rate_optimum": serializable_optimum(optima["rate"]),
        "best_ramsey_rate": best_ramsey,
        "best_ramsey_time": float(scan.ts_values[best_ramsey_index]),
        "advantage_lin": float(optima["rate"]["rate"] / best_ramsey)
        if best_ramsey > 0.0
        else None,
        "finite_difference_convergence": convergence,
    }
    summary_path = output / f"linear_readout_summary_{tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return data_path, summary_path


def print_summary(
    scan: LinearReadoutScan,
    optima: dict[str, dict[str, Any]],
    convergence: list[dict[str, Any]],
    paths: Iterable[Path],
) -> None:
    """Print the scientific comparison and numerical diagnostics."""
    rate = optima["rate"]
    best_ramsey_index = int(np.nanargmax(scan.ramsey_rate))
    best_ramsey = float(scan.ramsey_rate[best_ramsey_index])
    advantage = rate["rate"] / best_ramsey if best_ramsey > 0.0 else np.nan
    print("\nInformation-rate optimum, R_lin = F_lin_max / (ts + td)")
    print(
        f"  R_lin_max={rate['rate']:.12g}, eta={rate['eta']:.12g}, "
        f"ts={rate['ts']:.12g}, td={rate['td']:.12g}"
    )
    print(
        f"  n={np.array2string(rate['direction'], precision=7)}, "
        f"theta={rate['theta']:.12g}, phi={rate['phi']:.12g}"
    )
    print(
        "  covariance eigenvalues="
        f"{np.array2string(rate['covariance_eigenvalues'], precision=7)}, "
        f"condition={rate['covariance_condition']:.6g}"
    )
    print("Ramsey benchmark")
    print(
        f"  best_Ramsey={best_ramsey:.12g} at "
        f"ts={scan.ts_values[best_ramsey_index]:.12g}"
    )
    print(f"  advantage_lin={advantage:.12g}")
    for report in convergence:
        print(
            f"dJ convergence [{report['label']}]: "
            f"R={np.array2string(np.asarray(report['rate_values']), precision=8)}, "
            f"fine relative change={report['finest_relative_change']:.3g}, "
            f"direction alignment={report['finest_direction_alignment']:.8f}, "
            f"converged={report['converged']}"
        )
    rank_deficient = int(np.count_nonzero(scan.covariance_rank < 3))
    print(f"Rank-deficient/nearly-null covariance points: {rank_deficient}/{scan.rate.size}")
    if rank_deficient:
        its, itd = np.argwhere(scan.covariance_rank < 3)[0]
        print(
            "  representative nearly-null point: "
            f"ts={scan.ts_values[its]:.12g}, td={scan.td_values[itd]:.12g}, "
            "eigenvalues="
            f"{np.array2string(scan.covariance_eigenvalues[its, itd], precision=7)}, "
            f"condition={scan.covariance_condition[its, itd]}"
        )
    for path in paths:
        print(f"Saved: {path}")


def parse_config(argv: list[str] | None = None) -> LinearReadoutConfig:
    defaults = LinearReadoutConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--J0", type=float, default=defaults.J0)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument("--ts-min", type=float, default=defaults.ts_min)
    parser.add_argument("--ts-max", type=float, default=defaults.ts_max)
    parser.add_argument("--n-ts", type=int, default=defaults.n_ts)
    parser.add_argument("--td-min", type=float, default=defaults.td_min)
    parser.add_argument("--td-max", type=float, default=defaults.td_max)
    parser.add_argument("--n-td", type=int, default=defaults.n_td)
    parser.add_argument(
        "--covariance-rtol",
        type=float,
        default=defaults.covariance_rtol,
    )
    parser.add_argument("--variance-atol", type=float, default=defaults.variance_atol)
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument(
        "--finite-difference-rtol",
        type=float,
        default=defaults.finite_difference_rtol,
    )
    parser.add_argument(
        "--include-qfi",
        action=argparse.BooleanOptionalAction,
        default=defaults.include_qfi,
    )
    parser.add_argument(
        "--run-tests",
        action=argparse.BooleanOptionalAction,
        default=defaults.run_tests,
    )
    parser.add_argument(
        "--run-convergence",
        action=argparse.BooleanOptionalAction,
        default=defaults.run_convergence,
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
        "--show",
        dest="show_figure",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_figure,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=defaults.progress,
    )
    parser.add_argument("--output-directory", default=defaults.output_directory)
    parser.add_argument("--figure-format", default=defaults.figure_format)
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument("--colormap", default=defaults.colormap)
    return LinearReadoutConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> LinearReadoutScan:
    """Run, validate, save, plot, and summarize the nested optimization."""
    cfg = parse_config(argv)
    scan = scan_ts_td(cfg)
    optima = find_optima(scan)
    if cfg.run_tests:
        run_numerical_tests(cfg, scan, optima)
    convergence: list[dict[str, Any]] = []
    if cfg.run_convergence:
        rate_coordinates = (
            float(optima["rate"]["ts"]),
            float(optima["rate"]["td"]),
        )
        points = [("rate optimum", *rate_coordinates)]
        convergence = finite_difference_convergence(cfg, points)
    paths: list[Path] = []
    if cfg.make_plots:
        paths.extend(plot_heatmaps(scan, optima, cfg))
    if cfg.save_data:
        paths.extend(save_results(scan, optima, convergence, cfg))
    print_summary(scan, optima, convergence, paths)
    return scan


if __name__ == "__main__":
    main()
