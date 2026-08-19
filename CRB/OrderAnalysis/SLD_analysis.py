"""Analyze the reduced-bath SLD throughout the echo protocol.

The preparation/sensing/decoding trajectory matches
``plot_echo_bath_classical_fi_vs_time.py``.  At every sampled protocol time,
the reduced bath state ``rho_b`` and its finite-difference derivative with
respect to the true coupling ``J`` define the symmetric logarithmic derivative

    partial_J rho_b = (L_J rho_b + rho_b L_J) / 2.

The script attempts to reconstruct ``L_J`` using Hermitian operators generated
by noncommutative products of ``S_x`` and ``S_y``.  For a word ``W`` it includes

    H[W] = (W + W^dagger) / 2,
    K[W] = (W - W^dagger) / (2 i),

up to a configurable maximum word order.  Candidate operators are Frobenius
normalized and linearly dependent candidates are removed while their explicit
word labels are retained.  Reconstruction uses the SLD-weighted Fisher metric,
so the captured information is the squared SLD norm within the chosen span.

Separate figures show the SLD spectrum, the Fisher-information reconstruction
hierarchy, the time-dependent word coefficients, and dimensionless local
sensitivities of the SLD to ``J``, ``Omega``, and ``omega``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from itertools import product
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from CRB.OrderAnalysis.Echo_first_order import (  # noqa: E402
    initial_joint_state,
    spectral_decoder_hamiltonian,
    spectral_hamiltonian,
)
from CRB.OrderAnalysis.Echo_phase_cycling import (  # noqa: E402
    apply_propagator,
    reduced_bath_density_matrix,
)
from CRB.OrderAnalysis.plot_phase_cycling import (  # noqa: E402
    format_angle,
    format_number,
    sanitize_tag,
)
from CRB.crb_core import (  # noqa: E402
    build_bath_operators,
    fisher_metric_decomposition,
    qfi_from_rho_and_drho,
    save_plot,
)


@dataclass(frozen=True, slots=True)
class SLDAnalysisConfig:
    """Physical, protocol, decomposition, sensitivity, and plot controls."""

    N: int = 15
    Omega: float = 4.2
    omega: float = 2.0
    J_nominal: float = 1.0
    J_estimate: float = 1.0
    dJ: float = 1e-4


    preparation_time: float = 0
    n_preparation_times: int = 300
    sensing_time: float = 0.5
    n_sense_times: int = 300
    decode_time: float = 0.001
    n_decode_times: int = 2
    num_of_cycles: int = 1

    central_theta_rad: float = np.pi / 2.0
    central_phi_rad: float = 0.0
    bath_theta_rad: float = 0.0
    bath_phi_rad: float = 0.0

    qfi_tol: float = 1e-12
    basis_max_order: int = 3
    basis_independence_tol: float = 1e-10
    fisher_metric_rtol: float = 1e-10
    max_coefficient_curves: int = 10

    run_parameter_sensitivity: bool = True
    sensitivity_parameters: tuple[str, ...] = (
        "J_nominal",
        "Omega",
        "omega",
    )
    sensitivity_relative_step: float = 1e-3
    sensitivity_absolute_step: float = 1e-5
    sensitivity_scale_floor: float = 1.0
    sensitivity_log_y: bool = True

    figure_width_in: float = 12.0
    figure_height_in: float = 9.0
    figure_dpi: int = 200
    figure_format: str = "png"
    colormap: str = "viridis"
    coefficient_colormap: str = "coolwarm"
    show_figure: bool = True


@dataclass(frozen=True, slots=True)
class SLDProtocolTrajectory:
    """Reduced-bath state, tangent, SLD, and QFI along one protocol."""

    protocol_times: np.ndarray
    bath_states: tuple[np.ndarray, ...]
    bath_derivatives: tuple[np.ndarray, ...]
    slds: tuple[np.ndarray, ...]
    bath_qfi: np.ndarray
    preparation_end_index: int
    sensing_end_indices: np.ndarray
    cycle_end_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class SLDAnalysisResult:
    """SLD spectrum, word reconstruction, and parameter sensitivities."""

    protocol_times: np.ndarray
    bath_qfi: np.ndarray
    sld_eigenvalues: np.ndarray
    sld_frobenius_norm: np.ndarray
    basis_labels: tuple[str, ...]
    basis_orders: np.ndarray
    coefficients: np.ndarray
    captured_qfi_by_order: np.ndarray
    captured_fraction_by_order: np.ndarray
    full_reconstruction_frobenius_error: np.ndarray
    parameter_sensitivities: dict[str, np.ndarray]
    sensitivity_steps: dict[str, float]
    preparation_end_index: int
    sensing_end_indices: np.ndarray
    cycle_end_indices: np.ndarray


def validate_config(cfg: SLDAnalysisConfig) -> None:
    """Reject invalid physical, numerical, basis, and plot inputs."""
    finite_values = {
        "Omega": cfg.Omega,
        "omega": cfg.omega,
        "J_nominal": cfg.J_nominal,
        "J_estimate": cfg.J_estimate,
        "dJ": cfg.dJ,
        "preparation_time": cfg.preparation_time,
        "sensing_time": cfg.sensing_time,
        "decode_time": cfg.decode_time,
        "central_theta_rad": cfg.central_theta_rad,
        "central_phi_rad": cfg.central_phi_rad,
        "bath_theta_rad": cfg.bath_theta_rad,
        "bath_phi_rad": cfg.bath_phi_rad,
        "qfi_tol": cfg.qfi_tol,
        "basis_independence_tol": cfg.basis_independence_tol,
        "fisher_metric_rtol": cfg.fisher_metric_rtol,
        "sensitivity_relative_step": cfg.sensitivity_relative_step,
        "sensitivity_absolute_step": cfg.sensitivity_absolute_step,
        "sensitivity_scale_floor": cfg.sensitivity_scale_floor,
        "figure_width_in": cfg.figure_width_in,
        "figure_height_in": cfg.figure_height_in,
    }
    nonfinite = [name for name, value in finite_values.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError(f"configuration values must be finite: {nonfinite}")
    if cfg.N < 1:
        raise ValueError("N must be positive")
    if cfg.dJ <= 0.0:
        raise ValueError("dJ must be positive")
    if cfg.preparation_time < 0.0:
        raise ValueError("preparation_time must be non-negative")
    if cfg.sensing_time <= 0.0 or cfg.decode_time <= 0.0:
        raise ValueError("sensing_time and decode_time must be positive")
    if cfg.n_preparation_times < 2:
        raise ValueError("n_preparation_times must be at least 2")
    if cfg.n_sense_times < 2 or cfg.n_decode_times < 2:
        raise ValueError("sense/decode sample counts must be at least 2")
    if cfg.num_of_cycles < 1:
        raise ValueError("num_of_cycles must be at least 1")
    if cfg.qfi_tol <= 0.0:
        raise ValueError("qfi_tol must be positive")
    if cfg.basis_max_order < 1:
        raise ValueError("basis_max_order must be at least 1")
    if cfg.basis_independence_tol <= 0.0 or cfg.fisher_metric_rtol <= 0.0:
        raise ValueError("basis tolerances must be positive")
    if cfg.max_coefficient_curves < 1:
        raise ValueError("max_coefficient_curves must be positive")
    if cfg.sensitivity_relative_step <= 0.0:
        raise ValueError("sensitivity_relative_step must be positive")
    if cfg.sensitivity_absolute_step <= 0.0:
        raise ValueError("sensitivity_absolute_step must be positive")
    if cfg.sensitivity_scale_floor <= 0.0:
        raise ValueError("sensitivity_scale_floor must be positive")
    allowed_sensitivity_parameters = {"J_nominal", "Omega", "omega"}
    unknown = sorted(set(cfg.sensitivity_parameters) - allowed_sensitivity_parameters)
    if unknown:
        raise ValueError(f"unknown sensitivity parameters: {unknown}")
    if len(set(cfg.sensitivity_parameters)) != len(cfg.sensitivity_parameters):
        raise ValueError("sensitivity_parameters must not contain duplicates")
    if cfg.run_parameter_sensitivity and not cfg.sensitivity_parameters:
        raise ValueError("sensitivity_parameters cannot be empty when enabled")
    if cfg.figure_width_in <= 0.0 or cfg.figure_height_in <= 0.0:
        raise ValueError("figure dimensions must be positive")
    if cfg.figure_dpi <= 0:
        raise ValueError("figure_dpi must be positive")
    if not cfg.figure_format or cfg.figure_format.startswith("."):
        raise ValueError("figure_format must omit the leading dot")
    if re.search(r"[^A-Za-z0-9]", cfg.figure_format):
        raise ValueError("figure_format must contain only letters and digits")
    for name, colormap in {
        "colormap": cfg.colormap,
        "coefficient_colormap": cfg.coefficient_colormap,
    }.items():
        if colormap not in plt.colormaps():
            raise ValueError(f"unknown {name}: {colormap!r}")


def protocol_time_grids(
    cfg: SLDAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the preparation, sensing, and decoding time grids."""
    preparation_times = (
        np.array([0.0])
        if cfg.preparation_time == 0.0
        else np.linspace(0.0, cfg.preparation_time, cfg.n_preparation_times)
    )
    sense_times = np.linspace(0.0, cfg.sensing_time, cfg.n_sense_times)
    decode_times = np.linspace(0.0, cfg.decode_time, cfg.n_decode_times)
    return preparation_times, sense_times, decode_times


def compute_sld_protocol(cfg: SLDAnalysisConfig) -> SLDProtocolTrajectory:
    """Compute the reduced-bath SLD throughout the piecewise protocol."""
    preparation_times, sense_times, decode_times = protocol_time_grids(cfg)
    initial_state = initial_joint_state(cfg)
    sensing_spectra = (
        spectral_hamiltonian(cfg, cfg.J_nominal),
        spectral_hamiltonian(cfg, cfg.J_nominal + cfg.dJ),
        spectral_hamiltonian(cfg, cfg.J_nominal - cfg.dJ),
    )
    decoder_eigenvalues, decoder_eigenvectors = spectral_decoder_hamiltonian(cfg)

    protocol_times: list[float] = []
    bath_states: list[np.ndarray] = []
    bath_derivatives: list[np.ndarray] = []
    slds: list[np.ndarray] = []
    bath_qfi: list[float] = []
    sensing_end_indices: list[int] = []
    cycle_end_indices: list[int] = []

    def record_sample(time: float, states: tuple[np.ndarray, ...]) -> None:
        rho = reduced_bath_density_matrix(states[0], cfg.N)
        rho_plus = reduced_bath_density_matrix(states[1], cfg.N)
        rho_minus = reduced_bath_density_matrix(states[2], cfg.N)
        drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
        qfi, sld = qfi_from_rho_and_drho(rho, drho, tol=cfg.qfi_tol)
        protocol_times.append(time)
        bath_states.append(rho)
        bath_derivatives.append(drho)
        slds.append(sld)
        bath_qfi.append(qfi)

    prepared_states: tuple[np.ndarray, ...] = (
        initial_state,
        initial_state,
        initial_state,
    )
    for preparation_time in preparation_times:
        prepared_states = tuple(
            apply_propagator(
                initial_state,
                eigenvalues,
                eigenvectors,
                preparation_time,
            )
            for eigenvalues, eigenvectors in sensing_spectra
        )
        record_sample(preparation_time, prepared_states)
    preparation_end_index = len(protocol_times) - 1

    cycle_start_states = prepared_states
    cycle_duration = cfg.sensing_time + cfg.decode_time
    for cycle_index in range(cfg.num_of_cycles):
        cycle_start_time = cfg.preparation_time + cycle_index * cycle_duration
        sensed_states = cycle_start_states
        for sense_time in sense_times[1:]:
            sensed_states = tuple(
                apply_propagator(
                    cycle_start_state,
                    eigenvalues,
                    eigenvectors,
                    sense_time,
                )
                for cycle_start_state, (eigenvalues, eigenvectors) in zip(
                    cycle_start_states,
                    sensing_spectra,
                )
            )
            record_sample(cycle_start_time + sense_time, sensed_states)
        sensing_end_indices.append(len(protocol_times) - 1)

        decoded_states = sensed_states
        for decode_time in decode_times[1:]:
            decoded_states = tuple(
                apply_propagator(
                    sensed_state,
                    decoder_eigenvalues,
                    decoder_eigenvectors,
                    decode_time,
                )
                for sensed_state in sensed_states
            )
            record_sample(
                cycle_start_time + cfg.sensing_time + decode_time,
                decoded_states,
            )
        cycle_end_indices.append(len(protocol_times) - 1)
        cycle_start_states = decoded_states

    return SLDProtocolTrajectory(
        protocol_times=np.asarray(protocol_times, dtype=float),
        bath_states=tuple(bath_states),
        bath_derivatives=tuple(bath_derivatives),
        slds=tuple(slds),
        bath_qfi=np.asarray(bath_qfi, dtype=float),
        preparation_end_index=preparation_end_index,
        sensing_end_indices=np.asarray(sensing_end_indices, dtype=int),
        cycle_end_indices=np.asarray(cycle_end_indices, dtype=int),
    )


def xy_word_basis(
    spin_x: np.ndarray,
    spin_y: np.ndarray,
    maximum_order: int,
    independence_tol: float,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray]:
    """Build independent, normalized Hermitian ``S_x/S_y`` word operators."""
    generators = {"x": spin_x, "y": spin_y}
    dimension = spin_x.shape[0]
    labels: list[str] = []
    operators: list[np.ndarray] = []
    orders: list[int] = []
    orthonormal_residuals: list[np.ndarray] = []

    def add_candidate(
        label: str,
        candidate: np.ndarray,
        order: int,
        reference_norm: float,
    ) -> None:
        hermitian = 0.5 * (candidate + candidate.conj().T)
        norm = np.linalg.norm(hermitian, ord="fro")
        if norm <= independence_tol * max(reference_norm, 1.0):
            return
        normalized = hermitian / norm
        residual = normalized.copy()
        for previous in orthonormal_residuals:
            residual -= np.real(np.vdot(previous, residual)) * previous
        residual_norm = np.linalg.norm(residual, ord="fro")
        if residual_norm <= independence_tol:
            return
        labels.append(label)
        operators.append(normalized)
        orders.append(order)
        orthonormal_residuals.append(residual / residual_norm)

    identity = np.eye(dimension, dtype=complex)
    add_candidate("I", identity, 0, np.linalg.norm(identity, ord="fro"))
    for order in range(1, maximum_order + 1):
        for letters in product(("x", "y"), repeat=order):
            word = np.eye(dimension, dtype=complex)
            for letter in letters:
                word = word @ generators[letter]
            word_norm = np.linalg.norm(word, ord="fro")
            readable_word = "".join(f"S_{letter}" for letter in letters)
            add_candidate(
                f"H[{readable_word}]",
                0.5 * (word + word.conj().T),
                order,
                word_norm,
            )
            add_candidate(
                f"K[{readable_word}]",
                (word - word.conj().T) / (2.0j),
                order,
                word_norm,
            )
    return tuple(labels), tuple(operators), np.asarray(orders, dtype=int)


def parameter_sensitivity_step(value: float, cfg: SLDAnalysisConfig) -> float:
    """Return a stable symmetric finite-difference step for one parameter."""
    return max(
        abs(value) * cfg.sensitivity_relative_step,
        cfg.sensitivity_absolute_step,
    )


def weighted_operator_norm(rho: np.ndarray, operator: np.ndarray) -> float:
    """Return ``sqrt((A,A)_rho)`` for a Hermitian operator ``A``."""
    squared_norm = float(np.real(np.trace(rho @ operator @ operator)))
    return float(np.sqrt(max(squared_norm, 0.0)))


def compute_parameter_sensitivities(
    nominal: SLDProtocolTrajectory,
    cfg: SLDAnalysisConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Compare local SLD changes under ``J``, ``Omega``, and ``omega``."""
    if not cfg.run_parameter_sensitivity:
        return {}, {}
    sensitivities: dict[str, np.ndarray] = {}
    steps: dict[str, float] = {}
    for parameter in cfg.sensitivity_parameters:
        value = float(getattr(cfg, parameter))
        step = parameter_sensitivity_step(value, cfg)
        plus = compute_sld_protocol(replace(cfg, **{parameter: value + step}))
        minus = compute_sld_protocol(replace(cfg, **{parameter: value - step}))
        if not np.array_equal(plus.protocol_times, nominal.protocol_times):
            raise RuntimeError(f"{parameter} sensitivity changed the time grid")
        scale = max(abs(value), cfg.sensitivity_scale_floor)
        values = np.full_like(nominal.bath_qfi, np.nan)
        for index, (rho, sld, sld_plus, sld_minus, qfi) in enumerate(
            zip(
                nominal.bath_states,
                nominal.slds,
                plus.slds,
                minus.slds,
                nominal.bath_qfi,
            )
        ):
            if qfi <= cfg.qfi_tol:
                continue
            scaled_derivative = scale * (sld_plus - sld_minus) / (2.0 * step)
            values[index] = weighted_operator_norm(rho, scaled_derivative) / np.sqrt(
                qfi
            )
        sensitivities[parameter] = values
        steps[parameter] = step
    return sensitivities, steps


def analyze_sld(
    trajectory: SLDProtocolTrajectory,
    cfg: SLDAnalysisConfig,
) -> SLDAnalysisResult:
    """Decompose every SLD and evaluate parameter sensitivity trajectories."""
    bath_operators = build_bath_operators(cfg.N)
    labels, basis, basis_orders = xy_word_basis(
        bath_operators["Jx"],
        bath_operators["Jy"],
        cfg.basis_max_order,
        cfg.basis_independence_tol,
    )
    n_times = len(trajectory.protocol_times)
    n_basis = len(basis)
    dimension = cfg.N + 1
    coefficients = np.full((n_basis, n_times), np.nan)
    captured_qfi = np.zeros((cfg.basis_max_order, n_times), dtype=float)
    captured_fraction = np.full_like(captured_qfi, np.nan)
    reconstruction_error = np.full(n_times, np.nan)
    sld_eigenvalues = np.empty((dimension, n_times), dtype=float)
    sld_frobenius_norm = np.empty(n_times, dtype=float)

    for time_index, (rho, drho, sld, qfi) in enumerate(
        zip(
            trajectory.bath_states,
            trajectory.bath_derivatives,
            trajectory.slds,
            trajectory.bath_qfi,
        )
    ):
        sld_eigenvalues[:, time_index] = np.linalg.eigvalsh(sld)
        sld_norm = np.linalg.norm(sld, ord="fro")
        sld_frobenius_norm[time_index] = sld_norm
        full_reconstruction = np.zeros_like(sld)
        for order in range(1, cfg.basis_max_order + 1):
            selected_indices = np.flatnonzero(basis_orders <= order)
            selected_basis = [basis[index] for index in selected_indices]
            (
                projected_qfi,
                selected_coefficients,
                reconstructed,
                _,
                _,
            ) = fisher_metric_decomposition(
                rho,
                drho,
                selected_basis,
                rtol=cfg.fisher_metric_rtol,
            )
            captured_qfi[order - 1, time_index] = projected_qfi
            if qfi > cfg.qfi_tol:
                captured_fraction[order - 1, time_index] = projected_qfi / qfi
            if order == cfg.basis_max_order:
                coefficients[selected_indices, time_index] = selected_coefficients
                full_reconstruction = reconstructed
        if sld_norm > cfg.qfi_tol:
            reconstruction_error[time_index] = (
                np.linalg.norm(sld - full_reconstruction, ord="fro") / sld_norm
            )

    sensitivities, sensitivity_steps = compute_parameter_sensitivities(
        trajectory,
        cfg,
    )
    return SLDAnalysisResult(
        protocol_times=trajectory.protocol_times,
        bath_qfi=trajectory.bath_qfi,
        sld_eigenvalues=sld_eigenvalues,
        sld_frobenius_norm=sld_frobenius_norm,
        basis_labels=labels,
        basis_orders=basis_orders,
        coefficients=coefficients,
        captured_qfi_by_order=captured_qfi,
        captured_fraction_by_order=captured_fraction,
        full_reconstruction_frobenius_error=reconstruction_error,
        parameter_sensitivities=sensitivities,
        sensitivity_steps=sensitivity_steps,
        preparation_end_index=trajectory.preparation_end_index,
        sensing_end_indices=trajectory.sensing_end_indices,
        cycle_end_indices=trajectory.cycle_end_indices,
    )


def parameter_tags(cfg: SLDAnalysisConfig) -> str:
    """Encode every configuration field in stable output filenames."""
    encoded_fields = {
        "N",
        "Omega",
        "omega",
        "J_nominal",
        "J_estimate",
        "dJ",
        "preparation_time",
        "n_preparation_times",
        "sensing_time",
        "n_sense_times",
        "decode_time",
        "n_decode_times",
        "num_of_cycles",
        "central_theta_rad",
        "central_phi_rad",
        "bath_theta_rad",
        "bath_phi_rad",
        "qfi_tol",
        "basis_max_order",
        "basis_independence_tol",
        "fisher_metric_rtol",
        "max_coefficient_curves",
        "run_parameter_sensitivity",
        "sensitivity_parameters",
        "sensitivity_relative_step",
        "sensitivity_absolute_step",
        "sensitivity_scale_floor",
        "sensitivity_log_y",
        "figure_width_in",
        "figure_height_in",
        "figure_dpi",
        "figure_format",
        "colormap",
        "coefficient_colormap",
        "show_figure",
    }
    config_fields = {field.name for field in fields(cfg)}
    if encoded_fields != config_fields:
        missing = sorted(config_fields - encoded_fields)
        extra = sorted(encoded_fields - config_fields)
        raise RuntimeError(
            f"filename-tag field mismatch; missing={missing}, extra={extra}"
        )
    sensitivity_codes = {
        "J_nominal": "J",
        "Omega": "O",
        "omega": "w",
    }
    colormap_codes = {
        "viridis": "v",
        "coolwarm": "cw",
    }
    tags = (
        f"N{cfg.N}",
        (
            f"H=O{format_number(cfg.Omega)}-w{format_number(cfg.omega)}-"
            f"J{format_number(cfg.J_nominal)}-J0{format_number(cfg.J_estimate)}-"
            f"d{format_number(cfg.dJ)}"
        ),
        (
            f"t=p{format_number(cfg.preparation_time)},{cfg.n_preparation_times}-"
            f"s{format_number(cfg.sensing_time)},{cfg.n_sense_times}-"
            f"d{format_number(cfg.decode_time)},{cfg.n_decode_times}-"
            f"c{cfg.num_of_cycles}"
        ),
        (
            f"a=c{format_angle(cfg.central_theta_rad)},"
            f"{format_angle(cfg.central_phi_rad)}-"
            f"b{format_angle(cfg.bath_theta_rad)},"
            f"{format_angle(cfg.bath_phi_rad)}"
        ),
        (
            f"b=o{cfg.basis_max_order}-i{format_number(cfg.basis_independence_tol)}-"
            f"m{format_number(cfg.fisher_metric_rtol)}-"
            f"k{cfg.max_coefficient_curves}-q{format_number(cfg.qfi_tol)}"
        ),
        (
            f"s=e{int(cfg.run_parameter_sensitivity)}-p"
            f"{''.join(sensitivity_codes[name] for name in cfg.sensitivity_parameters) or 'n'}-"
            f"r{format_number(cfg.sensitivity_relative_step)}-"
            f"a{format_number(cfg.sensitivity_absolute_step)}-"
            f"f{format_number(cfg.sensitivity_scale_floor)}-"
            f"l{int(cfg.sensitivity_log_y)}"
        ),
        (
            f"f={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)},{cfg.figure_dpi},"
            f"{cfg.figure_format},"
            f"{colormap_codes.get(cfg.colormap, cfg.colormap)},"
            f"{colormap_codes.get(cfg.coefficient_colormap, cfg.coefficient_colormap)},"
            f"{int(cfg.show_figure)}"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def sample_coordinates(result: SLDAnalysisResult) -> np.ndarray:
    """Return chronological sample indices that keep short stages visible."""
    return np.arange(len(result.protocol_times), dtype=float)


def mark_protocol_boundaries(axis: plt.Axes, result: SLDAnalysisResult) -> None:
    """Mark preparation, sensing, and cycle boundaries on a sample-index axis."""
    axis.axvline(
        result.preparation_end_index,
        color="black",
        linestyle="-.",
        linewidth=1.1,
    )
    for index in result.sensing_end_indices:
        axis.axvline(index, color="black", linestyle="--", linewidth=1.0)
    for index in result.cycle_end_indices[:-1]:
        axis.axvline(index, color="black", linestyle=":", linewidth=0.9)


def save_figure(
    figure: plt.Figure,
    label: str,
    result: SLDAnalysisResult,
    cfg: SLDAnalysisConfig,
) -> Path:
    """Save and close one parameter-rich analysis figure."""
    path = Path(
        f"{label}__{parameter_tags(cfg)}.{cfg.figure_format.lower()}"
    )
    path = save_plot(
        figure,
        path,
        metadata={"config": cfg, "result": result, "plot_label": label},
        script_path=__file__,
        format=cfg.figure_format,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def plot_sld_spectrum(
    result: SLDAnalysisResult,
    cfg: SLDAnalysisConfig,
) -> Path:
    """Plot the instantaneous SLD eigenvalues and norms over protocol samples."""
    x = sample_coordinates(result)
    figure, (spectrum_axis, norm_axis) = plt.subplots(
        2,
        1,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    colors = plt.colormaps[cfg.colormap](
        np.linspace(0.05, 0.95, result.sld_eigenvalues.shape[0])
    )
    for eigenvalues, color in zip(result.sld_eigenvalues, colors):
        spectrum_axis.plot(x, eigenvalues, color=color, linewidth=1.0, alpha=0.85)
    spectrum_axis.axhline(0.0, color="black", linewidth=0.8)
    spectrum_axis.set_ylabel(r"Eigenvalues of $L_J$")
    spectrum_axis.grid(True, linestyle=":", alpha=0.7)
    norm_axis.plot(
        x,
        result.sld_frobenius_norm,
        color="tab:blue",
        linewidth=2.0,
        label=r"$\|L_J\|_F$",
    )
    qfi_axis = norm_axis.twinx()
    qfi_axis.plot(
        x,
        result.bath_qfi,
        color="tab:purple",
        linestyle="--",
        linewidth=1.7,
        label=r"$F_Q^{\rm bath}$",
    )
    norm_axis.set_ylabel(r"$\|L_J\|_F$")
    qfi_axis.set_ylabel(r"$F_Q^{\rm bath}$", color="tab:purple")
    qfi_axis.tick_params(axis="y", colors="tab:purple")
    norm_axis.set_xlabel("Chronological protocol sample index")
    norm_axis.grid(True, linestyle=":", alpha=0.7)
    handles, labels = norm_axis.get_legend_handles_labels()
    qfi_handles, qfi_labels = qfi_axis.get_legend_handles_labels()
    norm_axis.legend(handles + qfi_handles, labels + qfi_labels)
    for axis in (spectrum_axis, norm_axis):
        mark_protocol_boundaries(axis, result)
    figure.suptitle(
        r"Time-dependent reduced-bath SLD spectrum at $J="
        rf"{cfg.J_nominal:g}$; $N={cfg.N}$, $\Omega={cfg.Omega:g}$, "
        rf"$\omega={cfg.omega:g}$"
    )
    figure.tight_layout()
    return save_figure(figure, "sld-spectrum-vs-time", result, cfg)


def plot_reconstruction_hierarchy(
    result: SLDAnalysisResult,
    cfg: SLDAnalysisConfig,
) -> Path:
    """Plot QFI captured by successively higher ``S_x/S_y`` word orders."""
    x = sample_coordinates(result)
    figure, axes = plt.subplots(
        3,
        1,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
        sharex=True,
    )
    colors = plt.colormaps[cfg.colormap](
        np.linspace(0.15, 0.9, cfg.basis_max_order)
    )
    axes[0].plot(
        x,
        result.bath_qfi,
        color="black",
        linewidth=2.1,
        label=r"$F_Q^{\rm bath}$",
    )
    for order, color in enumerate(colors, start=1):
        axes[0].plot(
            x,
            result.captured_qfi_by_order[order - 1],
            color=color,
            linewidth=1.5,
            label=rf"Words through order {order}",
        )
        axes[1].plot(
            x,
            result.captured_fraction_by_order[order - 1],
            color=color,
            linewidth=1.7,
            label=rf"Order $\leq {order}$",
        )
    axes[0].set_ylabel("Fisher information")
    axes[0].legend(ncol=2)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_ylabel(r"Captured $F_Q$ fraction")
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(ncol=2)
    axes[2].plot(
        x,
        result.full_reconstruction_frobenius_error,
        color="tab:red",
        linewidth=1.8,
    )
    axes[2].set_ylabel(r"$\|L-L_{\rm rec}\|_F/\|L\|_F$")
    axes[2].set_xlabel("Chronological protocol sample index")
    axes[2].set_ylim(bottom=0.0)
    for axis in axes:
        mark_protocol_boundaries(axis, result)
        axis.grid(True, linestyle=":", alpha=0.7)
    figure.suptitle(
        r"SLD reconstruction from Hermitian $S_x/S_y$ products; "
        rf"maximum word order $={cfg.basis_max_order}$"
    )
    figure.tight_layout()
    return save_figure(figure, "sld-xy-reconstruction", result, cfg)


def plot_word_coefficients(
    result: SLDAnalysisResult,
    cfg: SLDAnalysisConfig,
) -> Path:
    """Plot all decomposition coefficients and the largest RMS components."""
    x = sample_coordinates(result)
    figure, (heatmap_axis, curve_axis) = plt.subplots(
        2,
        1,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 1.0)},
    )
    finite_coefficients = np.where(np.isfinite(result.coefficients), result.coefficients, 0.0)
    maximum_absolute = float(np.max(np.abs(finite_coefficients)))
    color_limit = maximum_absolute if maximum_absolute > 0.0 else 1.0
    image = heatmap_axis.imshow(
        finite_coefficients,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(x[0], x[-1], -0.5, len(result.basis_labels) - 0.5),
        cmap=cfg.coefficient_colormap,
        vmin=-color_limit,
        vmax=color_limit,
    )
    heatmap_axis.set_ylabel("Word-basis index")
    figure.colorbar(image, ax=heatmap_axis, label="Fisher-metric coefficient")

    rms = np.sqrt(np.mean(finite_coefficients**2, axis=1))
    top_count = min(cfg.max_coefficient_curves, len(rms))
    top_indices = np.argsort(rms)[-top_count:][::-1]
    colors = plt.colormaps[cfg.colormap](np.linspace(0.05, 0.95, top_count))
    for index, color in zip(top_indices, colors):
        curve_axis.plot(
            x,
            result.coefficients[index],
            color=color,
            linewidth=1.4,
            label=f"{index}: {result.basis_labels[index]}",
        )
    curve_axis.axhline(0.0, color="black", linewidth=0.8)
    curve_axis.set_xlabel("Chronological protocol sample index")
    curve_axis.set_ylabel("Coefficient")
    curve_axis.legend(ncol=2, fontsize="small")
    for axis in (heatmap_axis, curve_axis):
        mark_protocol_boundaries(axis, result)
    curve_axis.grid(True, linestyle=":", alpha=0.7)
    figure.suptitle(
        "Time dependence of the SLD decomposition coefficients\n"
        r"Each displayed $H[W]$ or $K[W]$ operator is Frobenius normalized"
    )
    figure.tight_layout()
    return save_figure(figure, "sld-xy-word-coefficients", result, cfg)


def plot_parameter_sensitivities(
    result: SLDAnalysisResult,
    cfg: SLDAnalysisConfig,
) -> Path | None:
    """Plot dimensionless local SLD sensitivities to Hamiltonian parameters."""
    if not result.parameter_sensitivities:
        return None
    x = sample_coordinates(result)
    figure, axis = plt.subplots(
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
    )
    colors = plt.colormaps[cfg.colormap](
        np.linspace(0.1, 0.9, len(result.parameter_sensitivities))
    )
    display_names = {
        "J_nominal": r"$J$",
        "Omega": r"$\Omega$",
        "omega": r"$\omega$",
    }
    for (parameter, values), color in zip(
        result.parameter_sensitivities.items(),
        colors,
    ):
        axis.plot(
            x,
            values,
            color=color,
            linewidth=1.8,
            label=(
                rf"Sensitivity to {display_names[parameter]} "
                rf"($\delta={result.sensitivity_steps[parameter]:.3g}$)"
            ),
        )
    if cfg.sensitivity_log_y:
        axis.set_yscale("log")
    else:
        axis.set_ylim(bottom=0.0)
    mark_protocol_boundaries(axis, result)
    axis.set_xlabel("Chronological protocol sample index")
    axis.set_ylabel(
        r"$p_{\rm scale}\|\partial_p L_J\|_\rho/\sqrt{F_Q^{\rm bath}}$"
    )
    axis.set_title(
        "Which Hamiltonian parameter changes the bath SLD most?\n"
        "State-weighted, dimensionless local sensitivities"
    )
    axis.grid(True, which="both", linestyle=":", alpha=0.7)
    axis.legend()
    figure.tight_layout()
    return save_figure(figure, "sld-parameter-sensitivity", result, cfg)


def plot_analysis(
    result: SLDAnalysisResult,
    cfg: SLDAnalysisConfig,
) -> tuple[Path, ...]:
    """Create and save all SLD analysis figures."""
    paths: list[Path] = [
        plot_sld_spectrum(result, cfg),
        plot_reconstruction_hierarchy(result, cfg),
        plot_word_coefficients(result, cfg),
    ]
    sensitivity_path = plot_parameter_sensitivities(result, cfg)
    if sensitivity_path is not None:
        paths.append(sensitivity_path)
    return tuple(paths)


def parse_config(argv: list[str] | None = None) -> SLDAnalysisConfig:
    """Parse command-line overrides into the typed configuration."""
    defaults = SLDAnalysisConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--J0", dest="J_estimate", type=float, default=defaults.J_estimate)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument(
        "--prep-time",
        "--t-p",
        dest="preparation_time",
        type=float,
        default=defaults.preparation_time,
    )
    parser.add_argument(
        "--n-preparation-times",
        type=int,
        default=defaults.n_preparation_times,
    )
    parser.add_argument(
        "--sense-time",
        "--t-s",
        dest="sensing_time",
        type=float,
        default=defaults.sensing_time,
    )
    parser.add_argument("--n-sense-times", type=int, default=defaults.n_sense_times)
    parser.add_argument(
        "--decode-time",
        "--t-d",
        dest="decode_time",
        type=float,
        default=defaults.decode_time,
    )
    parser.add_argument("--n-decode-times", type=int, default=defaults.n_decode_times)
    parser.add_argument("--num-of-cycles", type=int, default=defaults.num_of_cycles)
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
    parser.add_argument("--bath-theta-rad", type=float, default=defaults.bath_theta_rad)
    parser.add_argument("--bath-phi-rad", type=float, default=defaults.bath_phi_rad)
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument("--basis-max-order", type=int, default=defaults.basis_max_order)
    parser.add_argument(
        "--basis-independence-tol",
        type=float,
        default=defaults.basis_independence_tol,
    )
    parser.add_argument(
        "--fisher-metric-rtol",
        type=float,
        default=defaults.fisher_metric_rtol,
    )
    parser.add_argument(
        "--max-coefficient-curves",
        type=int,
        default=defaults.max_coefficient_curves,
    )
    parser.add_argument(
        "--run-parameter-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=defaults.run_parameter_sensitivity,
    )
    parser.add_argument(
        "--sensitivity-parameters",
        nargs="+",
        choices=("J_nominal", "Omega", "omega"),
        default=list(defaults.sensitivity_parameters),
    )
    parser.add_argument(
        "--sensitivity-relative-step",
        type=float,
        default=defaults.sensitivity_relative_step,
    )
    parser.add_argument(
        "--sensitivity-absolute-step",
        type=float,
        default=defaults.sensitivity_absolute_step,
    )
    parser.add_argument(
        "--sensitivity-scale-floor",
        type=float,
        default=defaults.sensitivity_scale_floor,
    )
    parser.add_argument(
        "--sensitivity-log-y",
        action=argparse.BooleanOptionalAction,
        default=defaults.sensitivity_log_y,
    )
    parser.add_argument("--figure-width-in", type=float, default=defaults.figure_width_in)
    parser.add_argument("--figure-height-in", type=float, default=defaults.figure_height_in)
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument("--figure-format", default=defaults.figure_format)
    parser.add_argument("--colormap", default=defaults.colormap)
    parser.add_argument(
        "--coefficient-colormap",
        default=defaults.coefficient_colormap,
    )
    parser.add_argument(
        "--show",
        dest="show_figure",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_figure,
    )
    values = vars(parser.parse_args(argv))
    values["sensitivity_parameters"] = tuple(values["sensitivity_parameters"])
    return SLDAnalysisConfig(**values)


def print_summary(
    result: SLDAnalysisResult,
    paths: tuple[Path, ...],
    cfg: SLDAnalysisConfig,
) -> None:
    """Print reconstruction quality and the dominant parameter sensitivity."""
    print(f"SLD samples: {len(result.protocol_times)}")
    print(f"Independent Sx/Sy word operators: {len(result.basis_labels)}")
    print(f"Maximum bath QFI: {np.max(result.bath_qfi):.12g}")
    diagnostic_indices = {
        "preparation end": result.preparation_end_index,
        "protocol end": len(result.protocol_times) - 1,
    }
    for label, index in diagnostic_indices.items():
        print(f"At {label} (tau={result.protocol_times[index]:.12g}):")
        for order in range(1, cfg.basis_max_order + 1):
            fraction = result.captured_fraction_by_order[order - 1, index]
            print(f"  order <= {order} captured FQ fraction: {fraction:.12g}")
        print(
            "  full-order relative Frobenius error: "
            f"{result.full_reconstruction_frobenius_error[index]:.12g}"
        )
    if result.parameter_sensitivities:
        rms_sensitivities: dict[str, float] = {}
        for parameter, values in result.parameter_sensitivities.items():
            finite = values[np.isfinite(values)]
            rms = float(np.sqrt(np.mean(finite**2))) if finite.size else np.nan
            rms_sensitivities[parameter] = rms
            print(f"RMS dimensionless SLD sensitivity to {parameter}: {rms:.12g}")
        finite_rms = {
            name: value
            for name, value in rms_sensitivities.items()
            if np.isfinite(value)
        }
        if finite_rms:
            dominant = max(finite_rms, key=finite_rms.get)
            print(f"Largest RMS SLD sensitivity: {dominant}")
    for path in paths:
        print(f"Saved {path}")


def main(argv: list[str] | None = None) -> tuple[Path, ...]:
    """Run the SLD trajectory, decomposition, sensitivity, and plots."""
    cfg = parse_config(argv)
    validate_config(cfg)
    trajectory = compute_sld_protocol(cfg)
    result = analyze_sld(trajectory, cfg)
    paths = plot_analysis(result, cfg)
    print_summary(result, paths, cfg)
    return paths


if __name__ == "__main__":
    main()
