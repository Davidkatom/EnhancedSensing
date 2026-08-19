"""Plot bath linear-moment FI through preparation, sensing, and decoding.

The protocol begins with one preparation segment,

    0 <= tau <= t_p:
        |psi(tau)> = exp[-i H_sense(J) tau] |psi_0>.

It is followed by ``num_of_cycles`` repetitions of sensing and decoding.
During each sensing segment,

    |psi> -> exp[-i H_sense(J) t_s] |psi>,

with

    H_sense(J) = Omega sigma_x + J sigma_z S_z + omega S_x.

and during each decoding segment,

    |psi> -> exp[-i H_dec(J0) t_d] |psi>,

where the fixed-estimate decoder keeps the coupling sign unchanged,

    H_dec(J0) = -Omega sigma_x + J0 sigma_z S_z - omega S_x.

At each time, the central spin is traced out.  The plotted quantities are the
classical Fisher information accessible from the linear collective-bath
moments ``<S_x>``, ``<S_y>``, and ``<S_z>`` through error propagation,

    F_C[<S_i>] = (partial_J <S_i>)^2 / Var(S_i).

The plot also includes the Fisher information from the full projective
``S_x`` outcome distribution and the sum of the full projective Fisher
information from three separate ``S_x``, ``S_y``, and ``S_z`` experiments,

    F_C^proj(S_x) + F_C^proj(S_y) + F_C^proj(S_z).

The reduced-bath quantum Fisher information ``F_Q^bath`` is shown as the
single-copy measurement-independent benchmark.

At each sampled time, the symmetric logarithmic derivative (SLD) is obtained
from the reduced bath state and its ``J`` derivative.  The plot includes the
classical Fisher information of the SLD's projective eigenbasis measurement,
``F_C[SLD]``.

A separate figure shows the fraction of bath QFI captured by the complete
projective ``S_x`` outcome distribution.  Its score operator is

    L_Sx = sum_m [(partial_J p_m) / p_m] Pi_m^x,

and the plotted identity is

    (L_Sx, L_Sx)_rho / (L, L)_rho = F_C^proj(S_x) / F_Q^bath.

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
    observable_moment_fisher,
    observable_projective_fisher,
    observable_projective_score,
    qfi_from_rho_and_drho,
    save_plot,
)


@dataclass(frozen=True, slots=True)
class EchoBathClassicalFIConfig:
    """Physics, piecewise-time sampling, state, numerical, and plot controls."""

    N: int = 15
    Omega: float = 2.5
    omega: float = 1
    J_nominal: float = 1
    J_estimate: float = 0.95
    dJ: float = 1e-4

    preparation_time: float = 4
    n_preparation_times: int = 300
    sensing_time: float = 1
    n_sense_times: int = 300
    decode_time: float = 1
    n_decode_times: int = 300
    num_of_cycles: int = 2

    central_theta_rad: float = np.pi / 2.0
    central_phi_rad: float = 0.0
    bath_theta_rad: float = 0.0
    bath_phi_rad: float = 0.0

    classical_fisher_variance_floor: float = 1e-2
    projective_fisher_probability_tol: float = 1e-2
    qfi_tol: float = 1e-12

    figure_width_in: float = 11.0
    figure_height_in: float = 9.0
    figure_dpi: int = 200
    figure_format: str = "png"
    colormap: str = "viridis"
    log_y: bool = False
    show_figure: bool = True


@dataclass(frozen=True, slots=True)
class EchoBathClassicalFIResult:
    """Protocol time, bath QFI/classical FI, and ``S_x`` diagnostics."""

    protocol_times: np.ndarray
    preparation_times: np.ndarray
    sense_times: np.ndarray
    decode_elapsed_times: np.ndarray
    bath_qfi: np.ndarray
    classical_fi_sld: np.ndarray
    classical_fi_x: np.ndarray
    classical_fi_y: np.ndarray
    classical_fi_z: np.ndarray
    classical_fi_x_projective: np.ndarray
    classical_fi_y_projective: np.ndarray
    classical_fi_z_projective: np.ndarray
    classical_fi_projective_sum: np.ndarray
    sld_sx_projective_fraction: np.ndarray
    sx_projective_fi_over_qfi: np.ndarray
    spin_x_expectation: np.ndarray
    spin_x_derivative: np.ndarray
    preparation_end_index: int
    sensing_end_indices: np.ndarray
    cycle_end_indices: np.ndarray


def validate_config(cfg: EchoBathClassicalFIConfig) -> None:
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
        "projective_fisher_probability_tol": cfg.projective_fisher_probability_tol,
        "qfi_tol": cfg.qfi_tol,
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
    if cfg.projective_fisher_probability_tol <= 0.0:
        raise ValueError("projective_fisher_probability_tol must be positive")
    if cfg.qfi_tol <= 0.0:
        raise ValueError("qfi_tol must be positive")
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


def fisher_and_spin_observables(
    nominal_state: np.ndarray,
    plus_state: np.ndarray,
    minus_state: np.ndarray,
    bath_observables: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: EchoBathClassicalFIConfig,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Return bath QFI, SLD/axis classical FI, and ``<S_x>`` data."""
    rho_bath = reduced_bath_density_matrix(nominal_state, cfg.N)
    rho_plus = reduced_bath_density_matrix(plus_state, cfg.N)
    rho_minus = reduced_bath_density_matrix(minus_state, cfg.N)
    drho_bath = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
    classical_values = tuple(
        observable_moment_fisher(
            rho_bath,
            drho_bath,
            observable,
            var_floor=cfg.classical_fisher_variance_floor,
        )
        for observable in bath_observables
    )
    classical_fi_x_projective, spin_x_score = observable_projective_score(
        rho_bath,
        drho_bath,
        bath_observables[0],
        tol=cfg.projective_fisher_probability_tol,
    )
    other_projective_values = tuple(
        observable_projective_fisher(
            rho_bath,
            drho_bath,
            observable,
            tol=cfg.projective_fisher_probability_tol,
        )
        for observable in bath_observables[1:]
    )
    projective_values = (
        classical_fi_x_projective,
        *other_projective_values,
    )
    bath_qfi, sld = qfi_from_rho_and_drho(
        rho_bath,
        drho_bath,
        tol=cfg.qfi_tol,
    )
    classical_fi_sld = observable_projective_fisher(
        rho_bath,
        drho_bath,
        sld,
        tol=cfg.qfi_tol,
    )
    spin_x = bath_observables[0]
    spin_x_expectation = float(np.real(np.trace(rho_bath @ spin_x)))
    spin_x_derivative = float(np.real(np.trace(drho_bath @ spin_x)))
    spin_x_score_norm = float(
        np.real(np.trace(rho_bath @ (spin_x_score @ spin_x_score)))
    )
    sld_norm = float(np.real(np.trace(rho_bath @ (sld @ sld))))
    score_sld_inner_product = 0.5 * np.trace(
        rho_bath @ (spin_x_score @ sld + sld @ spin_x_score)
    )
    if spin_x_score_norm > cfg.qfi_tol and sld_norm > cfg.qfi_tol:
        sld_sx_projective_fraction = float(
            np.abs(score_sld_inner_product) ** 2
            / (spin_x_score_norm * sld_norm)
        )
    else:
        sld_sx_projective_fraction = np.nan
    if bath_qfi > cfg.qfi_tol:
        sx_projective_fi_over_qfi = classical_fi_x_projective / bath_qfi
    else:
        sx_projective_fi_over_qfi = np.nan
    return (
        *classical_values,
        *projective_values,
        bath_qfi,
        classical_fi_sld,
        sld_sx_projective_fraction,
        sx_projective_fi_over_qfi,
        spin_x_expectation,
        spin_x_derivative,
    )


def run_echo_bath_classical_fi(
    cfg: EchoBathClassicalFIConfig,
) -> EchoBathClassicalFIResult:
    """Evaluate observables through preparation then sense/decode cycles."""
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
    bath_observables = (
        bath_operators["Jx"],
        bath_operators["Jy"],
        bath_operators["Jz"],
    )
    protocol_times: list[float] = []
    bath_qfi: list[float] = []
    classical_fi_sld: list[float] = []
    classical_fi_x: list[float] = []
    classical_fi_x_projective: list[float] = []
    classical_fi_y: list[float] = []
    classical_fi_z: list[float] = []
    classical_fi_y_projective: list[float] = []
    classical_fi_z_projective: list[float] = []
    sld_sx_projective_fraction: list[float] = []
    sx_projective_fi_over_qfi: list[float] = []
    spin_x_expectation: list[float] = []
    spin_x_derivative: list[float] = []
    sensing_end_indices: list[int] = []
    cycle_end_indices: list[int] = []

    def record_sample(time: float, states: tuple[np.ndarray, ...]) -> None:
        """Append the observables for one nominal/plus/minus state triplet."""
        (
            fi_x,
            fi_y,
            fi_z,
            fi_x_projective,
            fi_y_projective,
            fi_z_projective,
            qfi_bath,
            fi_sld,
            sx_projective_fraction,
            sx_projective_ratio,
            mean_x,
            derivative_x,
        ) = fisher_and_spin_observables(
            states[0],
            states[1],
            states[2],
            bath_observables,
            cfg,
        )
        protocol_times.append(time)
        classical_fi_x.append(fi_x)
        classical_fi_x_projective.append(fi_x_projective)
        classical_fi_y.append(fi_y)
        classical_fi_z.append(fi_z)
        classical_fi_y_projective.append(fi_y_projective)
        classical_fi_z_projective.append(fi_z_projective)
        bath_qfi.append(qfi_bath)
        classical_fi_sld.append(fi_sld)
        sld_sx_projective_fraction.append(sx_projective_fraction)
        sx_projective_fi_over_qfi.append(sx_projective_ratio)
        spin_x_expectation.append(mean_x)
        spin_x_derivative.append(derivative_x)

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
        record_sample(
            preparation_time,
            prepared_states,
        )
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

    classical_fi_x_projective_array = np.asarray(
        classical_fi_x_projective,
        dtype=float,
    )
    classical_fi_y_projective_array = np.asarray(
        classical_fi_y_projective,
        dtype=float,
    )
    classical_fi_z_projective_array = np.asarray(
        classical_fi_z_projective,
        dtype=float,
    )
    return EchoBathClassicalFIResult(
        protocol_times=np.asarray(protocol_times, dtype=float),
        preparation_times=preparation_times,
        sense_times=sense_times,
        decode_elapsed_times=decode_elapsed_times,
        bath_qfi=np.asarray(bath_qfi, dtype=float),
        classical_fi_sld=np.asarray(classical_fi_sld, dtype=float),
        classical_fi_x=np.asarray(classical_fi_x, dtype=float),
        classical_fi_y=np.asarray(classical_fi_y, dtype=float),
        classical_fi_z=np.asarray(classical_fi_z, dtype=float),
        classical_fi_x_projective=classical_fi_x_projective_array,
        classical_fi_y_projective=classical_fi_y_projective_array,
        classical_fi_z_projective=classical_fi_z_projective_array,
        classical_fi_projective_sum=(
            classical_fi_x_projective_array
            + classical_fi_y_projective_array
            + classical_fi_z_projective_array
        ),
        sld_sx_projective_fraction=np.asarray(
            sld_sx_projective_fraction,
            dtype=float,
        ),
        sx_projective_fi_over_qfi=np.asarray(
            sx_projective_fi_over_qfi,
            dtype=float,
        ),
        spin_x_expectation=np.asarray(spin_x_expectation, dtype=float),
        spin_x_derivative=np.asarray(spin_x_derivative, dtype=float),
        preparation_end_index=preparation_end_index,
        sensing_end_indices=np.asarray(sensing_end_indices, dtype=int),
        cycle_end_indices=np.asarray(cycle_end_indices, dtype=int),
    )


def parameter_tags(cfg: EchoBathClassicalFIConfig) -> str:
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
        "classical_fisher_variance_floor",
        "projective_fisher_probability_tol",
        "qfi_tol",
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
            f"num=vf{format_number(cfg.classical_fisher_variance_floor)}-"
            f"pt{format_number(cfg.projective_fisher_probability_tol)}-"
            f"qt{format_number(cfg.qfi_tol)}"
        ),
        (
            f"fig={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)}-d{cfg.figure_dpi}-"
            f"{cfg.figure_format}-{cfg.colormap}-log{int(cfg.log_y)}-"
            f"s{int(cfg.show_figure)}"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def output_path(cfg: EchoBathClassicalFIConfig) -> Path:
    """Return the parameter-rich filename passed to the shared plot saver."""
    return Path(
        f"bath-qfi-sld-and-spin-projective-fi__{parameter_tags(cfg)}."
        f"{cfg.figure_format.lower()}"
    )


def sld_sx_projection_output_path(cfg: EchoBathClassicalFIConfig) -> Path:
    """Return the parameter-rich path for the full-Sx SLD projection plot."""
    return Path(
        f"bath-sld-full-sx-projective-fraction__{parameter_tags(cfg)}."
        f"{cfg.figure_format.lower()}"
    )


def plot_echo_bath_classical_fi(
    result: EchoBathClassicalFIResult,
    cfg: EchoBathClassicalFIConfig,
) -> Path:
    """Plot bath QFI, SLD/spin classical FI, and ``<S_x>`` diagnostics."""
    figure, (fisher_axis, spin_axis) = plt.subplots(
        2,
        1,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
        sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    colormap = plt.colormaps[cfg.colormap]
    curves = (
        (
            result.classical_fi_x,
            r"$F_C[\langle S_x\rangle]$",
            colormap(0.18),
            "-",
        ),
        (
            result.classical_fi_y,
            r"$F_C[\langle S_y\rangle]$",
            colormap(0.52),
            "-",
        ),
        (
            result.classical_fi_z,
            r"$F_C[\langle S_z\rangle]$",
            colormap(0.86),
            "-",
        ),
        (
            result.classical_fi_x_projective,
            r"$F_C^{\rm proj}(S_x)$",
            "tab:red",
            "--",
        ),
        (
            result.classical_fi_projective_sum,
            (
                r"$F_C^{\rm proj}(S_x)+F_C^{\rm proj}(S_y)"
                r"+F_C^{\rm proj}(S_z)$"
            ),
            "black",
            "-.",
        ),
        (
            result.classical_fi_sld,
            r"$F_C[\mathrm{SLD}]$",
            "tab:cyan",
            "-",
        ),
        (
            result.bath_qfi,
            r"$F_Q^{\rm bath}$",
            "tab:purple",
            ":",
        ),
    )
    for values, label, color, linestyle in curves:
        fisher_axis.plot(
            result.protocol_times,
            values,
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
            label=label,
        )

    cycle_duration = cfg.sensing_time + cfg.decode_time
    protocol_end = cfg.preparation_time + cfg.num_of_cycles * cycle_duration
    if cfg.preparation_time > 0.0:
        for axis in (fisher_axis, spin_axis):
            axis.axvspan(
                0.0,
                cfg.preparation_time,
                color="tab:gray",
                alpha=0.08,
                label=(
                    r"Preparation under $H_{\rm sense}(J)$"
                    if axis is fisher_axis
                    else "_nolegend_"
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
        for axis in (fisher_axis, spin_axis):
            axis.axvspan(
                cycle_start,
                sensing_end,
                color="tab:blue",
                alpha=0.055,
                label=(
                    r"Sensing under $H_{\rm sense}(J)$"
                    if axis is fisher_axis and cycle_index == 0
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
                    if axis is fisher_axis and cycle_index == 0
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

    fisher_axis.set_ylabel("Fisher information for $J$")
    if cfg.log_y:
        fisher_axis.set_yscale("log")
    else:
        fisher_axis.set_ylim(bottom=0.0)
    fisher_axis.grid(True, linestyle=":", alpha=0.8)
    fisher_axis.legend(ncol=2)

    spin_axis.plot(
        result.protocol_times,
        result.spin_x_expectation,
        color="tab:blue",
        linewidth=2.0,
        label=r"$\langle S_x\rangle$",
    )
    spin_derivative_axis = spin_axis.twinx()
    spin_derivative_axis.plot(
        result.protocol_times,
        result.spin_x_derivative,
        color="tab:orange",
        linewidth=1.8,
        linestyle="--",
        label=r"$\partial_J\langle S_x\rangle$",
    )
    spin_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    spin_axis.set_xlim(0.0, protocol_end)
    spin_axis.set_xlabel(r"Protocol time $\tau$")
    spin_axis.set_ylabel(r"$\langle S_x\rangle$")
    spin_derivative_axis.set_ylabel(
        r"$\partial_J\langle S_x\rangle$",
        color="tab:orange",
    )
    spin_derivative_axis.tick_params(axis="y", colors="tab:orange")
    spin_axis.grid(True, linestyle=":", alpha=0.8)
    spin_handles, spin_labels = spin_axis.get_legend_handles_labels()
    derivative_handles, derivative_labels = (
        spin_derivative_axis.get_legend_handles_labels()
    )
    spin_axis.legend(
        spin_handles + derivative_handles,
        spin_labels + derivative_labels,
        loc="upper left",
    )

    figure.suptitle(
        "Reduced-bath quantum and classical Fisher information\n"
        r"Decoder $(-\Omega,+J_0,-\omega)$; "
        rf"$N={cfg.N}$, $J={cfg.J_nominal:g}$, $J_0={cfg.J_estimate:g}$, "
        rf"$t_p={cfg.preparation_time:g}$, $t_s={cfg.sensing_time:g}$, "
        rf"$t_d={cfg.decode_time:g}$, "
        rf"$n_{{\rm cycles}}={cfg.num_of_cycles}$"
    )
    figure.tight_layout()

    path = output_path(cfg)
    path = save_plot(
        figure,
        path,
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


def plot_sld_sx_projection(
    result: EchoBathClassicalFIResult,
    cfg: EchoBathClassicalFIConfig,
) -> Path:
    """Plot the full ``S_x`` projective-score/SLD overlap identity."""
    figure, axis = plt.subplots(
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
    )
    axis.plot(
        result.protocol_times,
        result.sld_sx_projective_fraction,
        color="tab:blue",
        linewidth=2.2,
        label=(
            r"$|(L_{S_x},L_J)_\rho|^2/"
            r"[(L_{S_x},L_{S_x})_\rho(L_J,L_J)_\rho]$"
        ),
    )
    axis.plot(
        result.protocol_times,
        result.sx_projective_fi_over_qfi,
        color="tab:orange",
        linewidth=1.8,
        linestyle="--",
        label=r"$F_C^{\rm proj}(S_x)/F_Q^{\rm bath}$",
    )


    cycle_duration = cfg.sensing_time + cfg.decode_time
    protocol_end = cfg.preparation_time + cfg.num_of_cycles * cycle_duration
    if cfg.preparation_time > 0.0:
        axis.axvspan(
            0.0,
            cfg.preparation_time,
            color="tab:gray",
            alpha=0.08,
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
        )
        axis.axvspan(
            sensing_end,
            cycle_end,
            color="tab:red",
            alpha=0.045,
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

    finite_values = np.concatenate(
        (
            result.sld_sx_projective_fraction[
                np.isfinite(result.sld_sx_projective_fraction)
            ],
            result.sx_projective_fi_over_qfi[
                np.isfinite(result.sx_projective_fi_over_qfi)
            ],
        )
    )
    upper_limit = 1.05
    if finite_values.size:
        upper_limit = max(upper_limit, 1.05 * float(np.max(finite_values)))
    axis.set_xlim(0.0, protocol_end)
    # axis.set_ylim(0.0, upper_limit)
    axis.set_xlabel(r"Protocol time $\tau$")
    axis.set_ylabel("Fraction of bath QFI captured")
    axis.set_title(
        r"Full projective $S_x$ score alignment with the bath SLD $L_J$"
        "\n"
        rf"SLD evaluated at $J={cfg.J_nominal:g}$; "
        rf"$N={cfg.N}$, $t_p={cfg.preparation_time:g}$, "
        rf"$t_s={cfg.sensing_time:g}$, $t_d={cfg.decode_time:g}$"
    )
    axis.grid(True, linestyle=":", alpha=0.8)
    axis.legend()
    figure.tight_layout()

    path = sld_sx_projection_output_path(cfg)
    path = save_plot(
        figure,
        path,
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


def parse_config(
    argv: list[str] | None = None,
) -> EchoBathClassicalFIConfig:
    """Parse command-line overrides into the typed configuration."""
    defaults = EchoBathClassicalFIConfig()
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
        "--classical-fisher-variance-floor",
        type=float,
        default=defaults.classical_fisher_variance_floor,
    )
    parser.add_argument(
        "--projective-fisher-probability-tol",
        type=float,
        default=defaults.projective_fisher_probability_tol,
    )
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
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
    return EchoBathClassicalFIConfig(**vars(parser.parse_args(argv)))


def print_summary(
    result: EchoBathClassicalFIResult,
    path: Path,
    sld_projection_path: Path,
    cfg: EchoBathClassicalFIConfig,
) -> None:
    """Print extrema plus final-cycle boundary and endpoint diagnostics."""
    cycle_duration = cfg.sensing_time + cfg.decode_time
    protocol_duration = (
        cfg.preparation_time + cfg.num_of_cycles * cycle_duration
    )
    print(f"Preparation duration: {cfg.preparation_time:.12g}")
    print(f"Number of sense/decode cycles: {cfg.num_of_cycles}")
    print(f"Sensing duration per cycle: {cfg.sensing_time:.12g}")
    print(f"Decoding duration per cycle: {cfg.decode_time:.12g}")
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
    print(
        "  at final-cycle sensing boundary: "
        f"{result.bath_qfi[final_sensing_end_index]:.12g}; "
        f"at protocol end: {result.bath_qfi[-1]:.12g}"
    )
    maximum_sld_fi_index = int(np.argmax(result.classical_fi_sld))
    print(
        "Maximum F_C[SLD]: "
        f"{result.classical_fi_sld[maximum_sld_fi_index]:.12g} "
        f"at tau={result.protocol_times[maximum_sld_fi_index]:.12g}"
    )
    print(
        "  at final-cycle sensing boundary: "
        f"{result.classical_fi_sld[final_sensing_end_index]:.12g}; "
        f"at protocol end: {result.classical_fi_sld[-1]:.12g}"
    )
    print(
        "Maximum |F_Q^bath - F_C[SLD]|: "
        f"{np.max(np.abs(result.bath_qfi - result.classical_fi_sld)):.12g}"
    )
    projection_identity_error = np.abs(
        result.sld_sx_projective_fraction
        - result.sx_projective_fi_over_qfi
    )
    finite_projection_error = projection_identity_error[
        np.isfinite(projection_identity_error)
    ]
    if finite_projection_error.size:
        print(
            "Maximum full-Sx SLD-projection identity error: "
            f"{np.max(finite_projection_error):.12g}"
        )
    trajectories = {
        "<Sx>": result.classical_fi_x,
        "<Sy>": result.classical_fi_y,
        "<Sz>": result.classical_fi_z,
        "projective Sx": result.classical_fi_x_projective,
        "projective Sy": result.classical_fi_y_projective,
        "projective Sz": result.classical_fi_z_projective,
        "projective Sx + Sy + Sz": result.classical_fi_projective_sum,
    }
    for measurement, values in trajectories.items():
        maximum_index = int(np.argmax(values))
        print(
            f"Maximum F_C[{measurement}]: {values[maximum_index]:.12g} "
            f"at tau={result.protocol_times[maximum_index]:.12g}"
        )
        print(
            "  at final-cycle sensing boundary: "
            f"{values[final_sensing_end_index]:.12g}; "
            f"at protocol end: {values[-1]:.12g}"
        )
    print(
        "Nominal <Sx>: "
        f"initial={result.spin_x_expectation[0]:.12g}, "
        "after preparation="
        f"{result.spin_x_expectation[result.preparation_end_index]:.12g}, "
        f"final={result.spin_x_expectation[-1]:.12g}"
    )
    maximum_slope_index = int(np.argmax(np.abs(result.spin_x_derivative)))
    print(
        "Maximum |d_J <Sx>|: "
        f"{abs(result.spin_x_derivative[maximum_slope_index]):.12g} "
        f"at tau={result.protocol_times[maximum_slope_index]:.12g}"
    )
    print(f"Saved bath classical-FI timeline to {path}")
    print(
        "Saved bath SLD/full-Sx projection timeline to "
        f"{sld_projection_path}"
    )


def main(argv: list[str] | None = None) -> Path:
    """Run the trajectory, save both figures, and return the main path."""
    cfg = parse_config(argv)
    validate_config(cfg)
    result = run_echo_bath_classical_fi(cfg)
    path = plot_echo_bath_classical_fi(result, cfg)
    sld_projection_path = plot_sld_sx_projection(result, cfg)
    print_summary(result, path, sld_projection_path, cfg)
    return path


if __name__ == "__main__":
    main()
