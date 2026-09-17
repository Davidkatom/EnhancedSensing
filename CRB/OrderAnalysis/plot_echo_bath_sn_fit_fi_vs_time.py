"""Fit the reduced-bath SLD to a single collective-spin readout ``S_n``.

The preparation/sensing/decoding trajectory matches
``plot_echo_bath_classical_fi_vs_time.py``.  At every sampled protocol time the
reduced bath state ``rho_b`` and its finite-difference derivative with respect
to the true coupling ``J`` define the symmetric logarithmic derivative ``L_J``
through

    partial_J rho_b = (L_J rho_b + rho_b L_J) / 2.

This script assumes the SLD is captured by a *single* collective-spin
direction,

    S_n = x S_x + y S_y + z S_z,

and finds the best fit.  The fit is the Fisher-metric (SLD-weighted)
projection of ``L_J`` onto the span of the collective-spin operators, solved by
:func:`CRB.crb_core.fisher_metric_decomposition`.  The identity is included in
the fit basis so that the projection subtracts the state mean; with that term
the captured Fisher information equals the linear-moment classical Fisher
information of reading out the single observable ``<S_n>`` through error
propagation,

    F_C^bath[S_n] = (partial_J <S_n>)^2 / Var(S_n),

maximized over the readout direction ``(x, y, z)``.  The identity coefficient is
only the mean offset; the reported direction is the traceless
``S_n = x S_x + y S_y + z S_z`` part.

The plot shows ``F_C^bath[S_n]`` versus protocol time with the reduced-bath
quantum Fisher information ``F_Q^bath`` overlaid as the measurement-independent
single-copy benchmark.  A lower panel shows the best-fit unit direction
``(n_x, n_y, n_z)`` of ``S_n``, i.e. how the optimal collective-spin readout
axis rotates through the protocol.

As an independent check the classical moment Fisher information of the
recovered ``S_n`` is also evaluated with
:func:`CRB.crb_core.observable_moment_fisher`; it agrees with the captured
projection value whenever the mean is subtracted.

The state is not reset between any stages.  Preparation and sensing use the
same Hamiltonian and the same true ``J``; their names only distinguish the
one-time initial interval from the sensing intervals inside the cycle.  The
decoder alone holds ``J0`` fixed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
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
    observable_moment_fisher,
    qfi_from_rho_and_drho,
)
from CRB.smart_save import save_plot  # noqa: E402


@dataclass(frozen=True, slots=True)
class EchoBathSnFitConfig:
    """Physics, piecewise-time sampling, state, fit, and plot controls."""

    N: int = 15
    Omega: float = 2.5
    omega: float = 1.0
    J_nominal: float = 1.0
    J_estimate: float = 1.0
    dJ: float = 1e-4

    preparation_time: float = 1.0
    n_preparation_times: int = 300
    sensing_time: float = 0.01
    n_sense_times: int = 300
    decode_time: float = 0.01
    n_decode_times: int = 300
    num_of_cycles: int = 1

    central_theta_rad: float = np.pi / 2.0
    central_phi_rad: float = 0.0
    bath_theta_rad: float = 0.0
    bath_phi_rad: float = 0.0

    subtract_mean: bool = True
    classical_fisher_variance_floor: float = 1e-12
    fisher_metric_rtol: float = 1e-10
    qfi_tol: float = 1e-12
    direction_norm_floor: float = 1e-12

    figure_width_in: float = 11.0
    figure_height_in: float = 9.0
    figure_dpi: int = 200
    figure_format: str = "png"
    colormap: str = "viridis"
    log_y: bool = False
    show_figure: bool = True


@dataclass(frozen=True, slots=True)
class EchoBathSnFitResult:
    """Protocol time, bath QFI, best-fit ``S_n`` FI, and readout direction."""

    protocol_times: np.ndarray
    preparation_times: np.ndarray
    sense_times: np.ndarray
    decode_elapsed_times: np.ndarray
    bath_qfi: np.ndarray
    fc_bath_sn: np.ndarray
    sn_moment_fi: np.ndarray
    captured_fraction: np.ndarray
    sn_coefficients: np.ndarray
    sn_direction: np.ndarray
    mean_offset: np.ndarray
    preparation_end_index: int
    sensing_end_indices: np.ndarray
    cycle_end_indices: np.ndarray


def validate_config(cfg: EchoBathSnFitConfig) -> None:
    """Fail early for invalid physical, numerical, sampling, or plot inputs."""
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
        "classical_fisher_variance_floor": cfg.classical_fisher_variance_floor,
        "fisher_metric_rtol": cfg.fisher_metric_rtol,
        "qfi_tol": cfg.qfi_tol,
        "direction_norm_floor": cfg.direction_norm_floor,
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
    if cfg.sensing_time <= 0.0:
        raise ValueError("sensing_time must be positive")
    if cfg.decode_time <= 0.0:
        raise ValueError("decode_time must be positive")
    if cfg.n_preparation_times < 2:
        raise ValueError("n_preparation_times must be at least 2")
    if cfg.n_sense_times < 2:
        raise ValueError("n_sense_times must be at least 2")
    if cfg.n_decode_times < 2:
        raise ValueError("n_decode_times must be at least 2")
    if cfg.num_of_cycles < 1:
        raise ValueError("num_of_cycles must be at least 1")
    if cfg.classical_fisher_variance_floor <= 0.0:
        raise ValueError("classical_fisher_variance_floor must be positive")
    if cfg.fisher_metric_rtol <= 0.0:
        raise ValueError("fisher_metric_rtol must be positive")
    if cfg.qfi_tol <= 0.0:
        raise ValueError("qfi_tol must be positive")
    if cfg.direction_norm_floor <= 0.0:
        raise ValueError("direction_norm_floor must be positive")
    if cfg.figure_width_in <= 0.0 or cfg.figure_height_in <= 0.0:
        raise ValueError("figure dimensions must be positive")
    if cfg.figure_dpi <= 0:
        raise ValueError("figure_dpi must be positive")
    if not cfg.figure_format or cfg.figure_format.startswith("."):
        raise ValueError("figure_format must be an extension without a leading dot")
    if re.search(r"[^A-Za-z0-9]", cfg.figure_format):
        raise ValueError("figure_format must contain only letters and digits")
    if cfg.colormap not in plt.colormaps():
        raise ValueError(f"unknown matplotlib colormap: {cfg.colormap!r}")


def sn_fit_observables(
    nominal_state: np.ndarray,
    plus_state: np.ndarray,
    minus_state: np.ndarray,
    bath_operators: dict[str, np.ndarray],
    cfg: EchoBathSnFitConfig,
) -> tuple[float, float, float, float, float, float, float]:
    """Return bath QFI, best-fit ``S_n`` FI, and its readout coefficients.

    The reduced bath state and its ``J`` derivative are formed by symmetric
    finite differences.  The SLD is projected onto the collective-spin span
    ``span{I, S_x, S_y, S_z}`` (the identity is dropped when
    ``cfg.subtract_mean`` is false); the projection coefficients give the best
    fit ``S_n = x S_x + y S_y + z S_z`` and the captured Fisher information.
    """
    rho_bath = reduced_bath_density_matrix(nominal_state, cfg.N)
    rho_plus = reduced_bath_density_matrix(plus_state, cfg.N)
    rho_minus = reduced_bath_density_matrix(minus_state, cfg.N)
    drho_bath = (rho_plus - rho_minus) / (2.0 * cfg.dJ)

    bath_qfi, _sld = qfi_from_rho_and_drho(rho_bath, drho_bath, tol=cfg.qfi_tol)

    identity = bath_operators["I"]
    spin_x = bath_operators["Jx"]
    spin_y = bath_operators["Jy"]
    spin_z = bath_operators["Jz"]
    fit_basis = (
        [identity, spin_x, spin_y, spin_z]
        if cfg.subtract_mean
        else [spin_x, spin_y, spin_z]
    )
    captured_qfi, coefficients, _reconstructed, _metric, _derivative = (
        fisher_metric_decomposition(
            rho_bath,
            drho_bath,
            fit_basis,
            rtol=cfg.fisher_metric_rtol,
        )
    )
    spin_coefficients = coefficients[1:] if cfg.subtract_mean else coefficients
    mean_offset = float(coefficients[0]) if cfg.subtract_mean else 0.0
    x, y, z = (
        float(spin_coefficients[0]),
        float(spin_coefficients[1]),
        float(spin_coefficients[2]),
    )

    sn_operator = x * spin_x + y * spin_y + z * spin_z
    sn_moment_fi = observable_moment_fisher(
        rho_bath,
        drho_bath,
        sn_operator,
        var_floor=cfg.classical_fisher_variance_floor,
    )
    return captured_qfi, sn_moment_fi, bath_qfi, x, y, z, mean_offset


def run_echo_bath_sn_fit(cfg: EchoBathSnFitConfig) -> EchoBathSnFitResult:
    """Evaluate the best-fit ``S_n`` readout through preparation and cycles."""
    preparation_times = (
        np.array([0.0])
        if cfg.preparation_time == 0.0
        else np.linspace(
            0.0,
            cfg.preparation_time,
            cfg.n_preparation_times,
        )
    )
    sense_times = np.linspace(0.0, cfg.sensing_time, cfg.n_sense_times)
    decode_elapsed_times = np.linspace(0.0, cfg.decode_time, cfg.n_decode_times)

    initial_state = initial_joint_state(cfg)
    sensing_spectra = (
        spectral_hamiltonian(cfg, cfg.J_nominal),
        spectral_hamiltonian(cfg, cfg.J_nominal + cfg.dJ),
        spectral_hamiltonian(cfg, cfg.J_nominal - cfg.dJ),
    )
    decoder_spectrum = spectral_decoder_hamiltonian(cfg)
    bath_operators = build_bath_operators(cfg.N)

    protocol_times: list[float] = []
    bath_qfi: list[float] = []
    fc_bath_sn: list[float] = []
    sn_moment_fi: list[float] = []
    sn_coefficients: list[tuple[float, float, float]] = []
    mean_offset: list[float] = []
    sensing_end_indices: list[int] = []
    cycle_end_indices: list[int] = []

    def record_sample(time: float, states: tuple[np.ndarray, ...]) -> None:
        """Append the best-fit ``S_n`` observables for one state triplet."""
        (
            captured_qfi,
            moment_fi,
            qfi_bath,
            x,
            y,
            z,
            offset,
        ) = sn_fit_observables(
            states[0],
            states[1],
            states[2],
            bath_operators,
            cfg,
        )
        protocol_times.append(time)
        fc_bath_sn.append(captured_qfi)
        sn_moment_fi.append(moment_fi)
        bath_qfi.append(qfi_bath)
        sn_coefficients.append((x, y, z))
        mean_offset.append(offset)

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
    decoder_eigenvalues, decoder_eigenvectors = decoder_spectrum
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
        for decode_elapsed_time in decode_elapsed_times[1:]:
            decoded_states = tuple(
                apply_propagator(
                    sensed_state,
                    decoder_eigenvalues,
                    decoder_eigenvectors,
                    decode_elapsed_time,
                )
                for sensed_state in sensed_states
            )
            record_sample(
                cycle_start_time + cfg.sensing_time + decode_elapsed_time,
                decoded_states,
            )
        cycle_end_indices.append(len(protocol_times) - 1)
        cycle_start_states = decoded_states

    bath_qfi_array = np.asarray(bath_qfi, dtype=float)
    fc_bath_sn_array = np.asarray(fc_bath_sn, dtype=float)
    coefficient_array = np.asarray(sn_coefficients, dtype=float)
    direction_norm = np.linalg.norm(coefficient_array, axis=1)
    direction = np.full_like(coefficient_array, np.nan)
    valid_direction = direction_norm > cfg.direction_norm_floor
    direction[valid_direction] = (
        coefficient_array[valid_direction]
        / direction_norm[valid_direction, None]
    )
    captured_fraction = np.full_like(bath_qfi_array, np.nan)
    informative = bath_qfi_array > cfg.qfi_tol
    captured_fraction[informative] = (
        fc_bath_sn_array[informative] / bath_qfi_array[informative]
    )

    return EchoBathSnFitResult(
        protocol_times=np.asarray(protocol_times, dtype=float),
        preparation_times=preparation_times,
        sense_times=sense_times,
        decode_elapsed_times=decode_elapsed_times,
        bath_qfi=bath_qfi_array,
        fc_bath_sn=fc_bath_sn_array,
        sn_moment_fi=np.asarray(sn_moment_fi, dtype=float),
        captured_fraction=captured_fraction,
        sn_coefficients=coefficient_array,
        sn_direction=direction,
        mean_offset=np.asarray(mean_offset, dtype=float),
        preparation_end_index=preparation_end_index,
        sensing_end_indices=np.asarray(sensing_end_indices, dtype=int),
        cycle_end_indices=np.asarray(cycle_end_indices, dtype=int),
    )


def parameter_tags(cfg: EchoBathSnFitConfig) -> str:
    """Encode every configuration field in stable, compact filename tags."""
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
        "subtract_mean",
        "classical_fisher_variance_floor",
        "fisher_metric_rtol",
        "qfi_tol",
        "direction_norm_floor",
        "figure_width_in",
        "figure_height_in",
        "figure_dpi",
        "figure_format",
        "colormap",
        "log_y",
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
        f"N{cfg.N}",
        (
            f"H=O{format_number(cfg.Omega)}-w{format_number(cfg.omega)}-"
            f"J{format_number(cfg.J_nominal)}-J0{format_number(cfg.J_estimate)}-"
            f"dJ{format_number(cfg.dJ)}"
        ),
        (
            f"time=tp{format_number(cfg.preparation_time)}-"
            f"np{cfg.n_preparation_times}-"
            f"ts{format_number(cfg.sensing_time)}-ns{cfg.n_sense_times}-"
            f"td{format_number(cfg.decode_time)}-nd{cfg.n_decode_times}-"
            f"cyc{cfg.num_of_cycles}"
        ),
        (
            f"state=tc{format_angle(cfg.central_theta_rad)}-"
            f"pc{format_angle(cfg.central_phi_rad)}-"
            f"tb{format_angle(cfg.bath_theta_rad)}-"
            f"pb{format_angle(cfg.bath_phi_rad)}"
        ),
        (
            f"fit=sm{int(cfg.subtract_mean)}-"
            f"vf{format_number(cfg.classical_fisher_variance_floor)}-"
            f"mr{format_number(cfg.fisher_metric_rtol)}-"
            f"qt{format_number(cfg.qfi_tol)}-"
            f"nf{format_number(cfg.direction_norm_floor)}"
        ),
        (
            f"fig={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)}-d{cfg.figure_dpi}-"
            f"{cfg.figure_format}-{cfg.colormap}-log{int(cfg.log_y)}-"
            f"s{int(cfg.show_figure)}"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def output_path(cfg: EchoBathSnFitConfig) -> Path:
    """Return the parameter-rich filename passed to the shared plot saver."""
    return Path(
        f"bath-sn-fit-classical-fi-vs-time__{parameter_tags(cfg)}."
        f"{cfg.figure_format.lower()}"
    )


def _shade_protocol(axis: plt.Axes, cfg: EchoBathSnFitConfig, label: bool) -> None:
    """Shade preparation, sensing, and decoding stages on one time axis."""
    cycle_duration = cfg.sensing_time + cfg.decode_time
    if cfg.preparation_time > 0.0:
        axis.axvspan(
            0.0,
            cfg.preparation_time,
            color="tab:gray",
            alpha=0.08,
            label=(
                r"Preparation under $H_{\rm sense}(J)$" if label else "_nolegend_"
            ),
        )
        axis.axvline(
            cfg.preparation_time,
            color="black",
            linestyle="-.",
            linewidth=1.2,
        )
    for cycle_index in range(cfg.num_of_cycles):
        cycle_start = cfg.preparation_time + cycle_index * cycle_duration
        sensing_end = cycle_start + cfg.sensing_time
        cycle_end = cycle_start + cycle_duration
        axis.axvspan(
            cycle_start,
            sensing_end,
            color="tab:blue",
            alpha=0.055,
            label=(
                r"Sensing under $H_{\rm sense}(J)$"
                if label and cycle_index == 0
                else "_nolegend_"
            ),
        )
        axis.axvspan(
            sensing_end,
            cycle_end,
            color="tab:red",
            alpha=0.045,
            label=(
                r"Decoding under $H_{\rm dec}(J_0)$"
                if label and cycle_index == 0
                else "_nolegend_"
            ),
        )
        axis.axvline(
            sensing_end,
            color="black",
            linestyle="--",
            linewidth=1.1,
        )
        if cycle_index < cfg.num_of_cycles - 1:
            axis.axvline(
                cycle_end,
                color="black",
                linestyle=":",
                linewidth=1.0,
            )


def plot_echo_bath_sn_fit(
    result: EchoBathSnFitResult,
    cfg: EchoBathSnFitConfig,
) -> Path:
    """Plot ``F_C^bath[S_n]`` with ``F_Q^bath`` and the readout direction."""
    figure, (fisher_axis, direction_axis) = plt.subplots(
        2,
        1,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    colormap = plt.colormaps[cfg.colormap]

    fisher_axis.plot(
        result.protocol_times,
        result.fc_bath_sn,
        color=colormap(0.25),
        linewidth=2.2,
        label=r"$F_C^{\rm bath}[S_{\hat n}]$ (best-fit readout)",
    )
    fisher_axis.plot(
        result.protocol_times,
        result.bath_qfi,
        color="tab:purple",
        linestyle=":",
        linewidth=2.0,
        label=r"$F_Q^{\rm bath}$",
    )
    _shade_protocol(fisher_axis, cfg, label=True)
    fisher_axis.set_ylabel("Fisher information for $J$")
    if cfg.log_y:
        fisher_axis.set_yscale("log")
    else:
        fisher_axis.set_ylim(bottom=0.0)
    fisher_axis.grid(True, linestyle=":", alpha=0.8)
    fisher_axis.legend(ncol=2)

    direction_curves = (
        (result.sn_direction[:, 0], r"$n_x$", colormap(0.18)),
        (result.sn_direction[:, 1], r"$n_y$", colormap(0.52)),
        (result.sn_direction[:, 2], r"$n_z$", colormap(0.86)),
    )
    for values, label, color in direction_curves:
        direction_axis.plot(
            result.protocol_times,
            values,
            color=color,
            linewidth=1.8,
            label=label,
        )
    direction_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    _shade_protocol(direction_axis, cfg, label=False)

    cycle_duration = cfg.sensing_time + cfg.decode_time
    protocol_end = cfg.preparation_time + cfg.num_of_cycles * cycle_duration
    direction_axis.set_xlim(0.0, protocol_end)
    direction_axis.set_ylim(-1.05, 1.05)
    direction_axis.set_xlabel(r"Protocol time $\tau$")
    direction_axis.set_ylabel(r"Best-fit $\hat n$")
    direction_axis.grid(True, linestyle=":", alpha=0.8)
    direction_axis.legend(ncol=3, loc="upper left")

    figure.suptitle(
        r"Reduced-bath SLD fit to a single collective-spin readout "
        r"$S_{\hat n}=x S_x+y S_y+z S_z$"
        "\n"
        rf"Decoder $(-\Omega,+J_0,-\omega)$; $N={cfg.N}$, $J={cfg.J_nominal:g}$, "
        rf"$J_0={cfg.J_estimate:g}$, $t_p={cfg.preparation_time:g}$, "
        rf"$t_s={cfg.sensing_time:g}$, $t_d={cfg.decode_time:g}$, "
        rf"$n_{{\rm cycles}}={cfg.num_of_cycles}$"
    )
    figure.tight_layout()

    path = save_plot(
        figure,
        output_path(cfg),
        metadata={"config": cfg, "result": result},
        script_path=__file__,
        format=cfg.figure_format,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def parse_config(argv: list[str] | None = None) -> EchoBathSnFitConfig:
    """Parse command-line overrides into the typed configuration."""
    defaults = EchoBathSnFitConfig()
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
    parser.add_argument(
        "--n-sense-times",
        type=int,
        default=defaults.n_sense_times,
    )
    parser.add_argument(
        "--decode-time",
        "--t-d",
        dest="decode_time",
        type=float,
        default=defaults.decode_time,
    )
    parser.add_argument(
        "--n-decode-times",
        type=int,
        default=defaults.n_decode_times,
    )
    parser.add_argument(
        "--num-of-cycles",
        type=int,
        default=defaults.num_of_cycles,
    )
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
        "--subtract-mean",
        action=argparse.BooleanOptionalAction,
        default=defaults.subtract_mean,
        help="include the identity in the fit basis (mean-subtracted readout)",
    )
    parser.add_argument(
        "--classical-fisher-variance-floor",
        type=float,
        default=defaults.classical_fisher_variance_floor,
    )
    parser.add_argument(
        "--fisher-metric-rtol",
        type=float,
        default=defaults.fisher_metric_rtol,
    )
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument(
        "--direction-norm-floor",
        type=float,
        default=defaults.direction_norm_floor,
    )
    parser.add_argument("--figure-width-in", type=float, default=defaults.figure_width_in)
    parser.add_argument(
        "--figure-height-in",
        type=float,
        default=defaults.figure_height_in,
    )
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument("--figure-format", default=defaults.figure_format)
    parser.add_argument("--colormap", default=defaults.colormap)
    parser.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=defaults.log_y,
    )
    parser.add_argument(
        "--show",
        dest="show_figure",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_figure,
    )
    return EchoBathSnFitConfig(**vars(parser.parse_args(argv)))


def print_summary(
    result: EchoBathSnFitResult,
    path: Path,
    cfg: EchoBathSnFitConfig,
) -> None:
    """Print extrema, capture fractions, and the fit-consistency residual."""
    cycle_duration = cfg.sensing_time + cfg.decode_time
    protocol_duration = cfg.preparation_time + cfg.num_of_cycles * cycle_duration
    print(f"Subtract mean in fit: {cfg.subtract_mean}")
    print(f"Preparation duration: {cfg.preparation_time:.12g}")
    print(f"Number of sense/decode cycles: {cfg.num_of_cycles}")
    print(
        "Total protocol-time range: "
        f"0 <= tau <= {protocol_duration:.12g}"
    )
    final_sensing_end_index = int(result.sensing_end_indices[-1])
    maximum_qfi_index = int(np.argmax(result.bath_qfi))
    print(
        f"Maximum F_Q^bath: {result.bath_qfi[maximum_qfi_index]:.12g} "
        f"at tau={result.protocol_times[maximum_qfi_index]:.12g}"
    )
    maximum_sn_index = int(np.argmax(result.fc_bath_sn))
    print(
        "Maximum F_C^bath[S_n]: "
        f"{result.fc_bath_sn[maximum_sn_index]:.12g} "
        f"at tau={result.protocol_times[maximum_sn_index]:.12g}"
    )
    print(
        "  at final-cycle sensing boundary: "
        f"{result.fc_bath_sn[final_sensing_end_index]:.12g}; "
        f"at protocol end: {result.fc_bath_sn[-1]:.12g}"
    )
    finite_fraction = result.captured_fraction[
        np.isfinite(result.captured_fraction)
    ]
    if finite_fraction.size:
        print(
            "Captured fraction F_C^bath[S_n]/F_Q^bath: "
            f"max={np.max(finite_fraction):.12g}, "
            f"min={np.min(finite_fraction):.12g}"
        )
    informative = result.bath_qfi > cfg.qfi_tol
    consistency = np.abs(result.fc_bath_sn - result.sn_moment_fi)
    finite_consistency = consistency[informative & np.isfinite(consistency)]
    if finite_consistency.size:
        print(
            "Max |captured - moment-Fisher(S_n)| over informative samples: "
            f"{np.max(finite_consistency):.12g}"
        )
    best_direction = result.sn_direction[maximum_sn_index]
    if np.all(np.isfinite(best_direction)):
        print(
            "Best-fit direction at the F_C^bath[S_n] maximum: "
            f"n=({best_direction[0]:.6g}, {best_direction[1]:.6g}, "
            f"{best_direction[2]:.6g})"
        )
    print(f"Saved bath S_n-fit classical-FI timeline to {path}")


def main(argv: list[str] | None = None) -> Path:
    """Run the trajectory, save the figure, and return its path."""
    cfg = parse_config(argv)
    validate_config(cfg)
    result = run_echo_bath_sn_fit(cfg)
    path = plot_echo_bath_sn_fit(result, cfg)
    print_summary(result, path, cfg)
    return path


if __name__ == "__main__":
    main()
