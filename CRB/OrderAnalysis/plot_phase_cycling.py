"""Plot a phase-cycling signal and its multiple-quantum coherences.

The driven central-spin model is

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x,

with the repository convention ``S_i = 2 * jmat(N / 2, i)``.  MQC order is
reported in conventional single-spin-flip units using ``M_z = S_z / 2``.
After evolving the initial product state, the script applies

    R_z(phi) = exp(-i * phi * M_z)

to the reduced bath state and evaluates the Hilbert--Schmidt return signal

    S(phi) = Tr[rho_B R_z(phi) rho_B R_z(phi)^dagger].

Its Fourier coefficients are the multiple-quantum-coherence intensities

    I_k = sum_{m - m' = k} |rho_B[m, m']|^2.

The script also estimates the coupling derivative by a centered difference and
plots its coherence-order weights

    D_k = sum_{m - m' = k} |partial_J rho_B[m, m']|^2.

Both distributions have orders between ``-N`` and ``N``.  Comparing their
spread tests whether high-order coherences also carry parameter sensitivity.
``D_k`` is a derivative-support diagnostic, not an additive decomposition of
QFI, whose eigenvalue weighting is evaluated separately and reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.crb_core import (
        build_bath_operators,
        coherent_bath_state,
        evolve_bath_density_matrix_noiseless,
        qfi_vectorized,
    )
except ModuleNotFoundError:  # Allow: python CRB/plot_phase_cycling.py
    from crb_core import (
        build_bath_operators,
        coherent_bath_state,
        evolve_bath_density_matrix_noiseless,
        qfi_vectorized,
    )


@dataclass(frozen=True, slots=True)
class PhaseCyclingConfig:
    """Physics, phase sampling, numerical, and visualization parameters."""

    N: int = 15
    Omega: float = 2.5
    omega: float = 1.0
    J_nominal: float = 1.0
    dJ: float = 1e-5
    interrogation_time: float = 30

    central_theta_rad: float = np.pi / 2.0
    bath_theta_rad: float = 0.0
    bath_phi_rad: float = 0.0

    n_phase_samples: int = 129
    normalize_signal: bool = True
    normalize_derivative_spectrum: bool = True
    spectrum_noise_floor: float = 1e-12
    qfi_tol: float = 1e-12

    figure_width_in: float = 10.0
    figure_height_in: float = 8.0
    figure_dpi: int = 200
    figure_format: str = "png"
    colormap: str = "viridis"
    show_figure: bool = True


@dataclass(frozen=True, slots=True)
class PhaseCyclingResult:
    """Reduced state, phase signal, and exact/Fourier MQC intensities."""

    bath_density_matrix: np.ndarray
    bath_density_derivative: np.ndarray
    phases_rad: np.ndarray
    signal: np.ndarray
    coherence_orders: np.ndarray
    exact_intensities: np.ndarray
    fourier_intensities: np.ndarray
    derivative_intensities: np.ndarray
    purity: float
    derivative_norm_squared: float
    bath_qfi: float


def validate_config(cfg: PhaseCyclingConfig) -> None:
    """Fail early when the configuration cannot resolve the MQC spectrum."""
    if cfg.N < 1:
        raise ValueError("N must be a positive integer")
    floating_values = (
        cfg.Omega,
        cfg.omega,
        cfg.J_nominal,
        cfg.dJ,
        cfg.interrogation_time,
        cfg.central_theta_rad,
        cfg.bath_theta_rad,
        cfg.bath_phi_rad,
        cfg.spectrum_noise_floor,
        cfg.qfi_tol,
        cfg.figure_width_in,
        cfg.figure_height_in,
    )
    if not all(np.isfinite(value) for value in floating_values):
        raise ValueError("all floating-point configuration values must be finite")
    if cfg.interrogation_time < 0.0:
        raise ValueError("interrogation_time must be non-negative")
    if cfg.dJ <= 0.0:
        raise ValueError("dJ must be positive")
    minimum_phase_samples = 2 * cfg.N + 1
    if cfg.n_phase_samples < minimum_phase_samples:
        raise ValueError(
            "n_phase_samples must be at least 2*N + 1 "
            f"({minimum_phase_samples} for N={cfg.N}) to avoid aliasing"
        )
    if cfg.spectrum_noise_floor <= 0.0:
        raise ValueError("spectrum_noise_floor must be positive")
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
        raise ValueError(f"unknown Matplotlib colormap: {cfg.colormap}")


def evolve_reduced_bath_state(
    cfg: PhaseCyclingConfig,
    J: float | None = None,
) -> np.ndarray:
    """Return the exact noiseless reduced bath state for one coupling value."""
    coupling = cfg.J_nominal if J is None else J
    bath_state = coherent_bath_state(
        cfg.N,
        theta=cfg.bath_theta_rad,
        phi=cfg.bath_phi_rad,
    )
    return evolve_bath_density_matrix_noiseless(
        Omega_0=cfg.Omega,
        omega=cfg.omega,
        J=coupling,
        time=cfg.interrogation_time,
        N=cfg.N,
        bath_state=bath_state,
        central_theta=cfg.central_theta_rad,
    )


def magnetization_eigenvalues(cfg: PhaseCyclingConfig) -> np.ndarray:
    """Return ``M_z=S_z/2`` eigenvalues in single-spin-flip order units."""
    operator = build_bath_operators(cfg.N)["Jz"]
    eigenvalues = 0.5 * np.real(np.diag(operator))
    order_differences = eigenvalues[:, None] - eigenvalues[None, :]
    if not np.allclose(order_differences, np.rint(order_differences)):
        raise RuntimeError("collective magnetization differences are not integers")
    return eigenvalues


def coherence_order_squared_norms(
    matrix: np.ndarray,
    magnetizations: np.ndarray,
    orders: np.ndarray,
) -> np.ndarray:
    """Group a matrix's squared element magnitudes by coherence order."""
    order_matrix = np.rint(
        magnetizations[:, None] - magnetizations[None, :]
    ).astype(int)
    weights = np.abs(matrix) ** 2
    return np.asarray(
        [float(np.sum(weights[order_matrix == order])) for order in orders],
        dtype=float,
    )


def phase_cycling_signal(
    rho_bath: np.ndarray,
    magnetizations: np.ndarray,
    phases_rad: np.ndarray,
) -> np.ndarray:
    """Return ``Tr[rho R_z(phi) rho R_z(phi)^dagger]`` for all phases."""
    order_matrix = magnetizations[:, None] - magnetizations[None, :]
    weights = np.abs(rho_bath) ** 2
    phase_factors = np.exp(1j * order_matrix[..., None] * phases_rad)
    return np.sum(weights[..., None] * phase_factors, axis=(0, 1))


def fourier_intensities(
    signal: np.ndarray,
    n_phase_samples: int,
    orders: np.ndarray,
    noise_floor: float,
) -> np.ndarray:
    """Recover the configured coherence orders from uniformly sampled phases."""
    coefficients = np.fft.fft(signal) / n_phase_samples
    fft_orders = np.rint(
        np.fft.fftfreq(n_phase_samples, d=1.0 / n_phase_samples)
    ).astype(int)
    coefficient_by_order = dict(zip(fft_orders, coefficients))
    recovered = np.asarray(
        [coefficient_by_order.get(int(order), 0.0).real for order in orders],
        dtype=float,
    )
    recovered[np.abs(recovered) < noise_floor] = 0.0
    return recovered


def run_phase_cycling(cfg: PhaseCyclingConfig) -> PhaseCyclingResult:
    """Recover the coherence and coupling-derivative order distributions."""
    rho_bath = evolve_reduced_bath_state(cfg)
    rho_plus = evolve_reduced_bath_state(cfg, cfg.J_nominal + cfg.dJ)
    rho_minus = evolve_reduced_bath_state(cfg, cfg.J_nominal - cfg.dJ)
    rho_for_qfi = 0.5 * (rho_plus + rho_minus)
    drho_bath = (rho_plus - rho_minus) / (2.0 * cfg.dJ)

    magnetizations = magnetization_eigenvalues(cfg)
    orders = np.arange(-cfg.N, cfg.N + 1, dtype=int)
    phases = np.linspace(0.0, 2.0 * np.pi, cfg.n_phase_samples, endpoint=False)
    signal = phase_cycling_signal(rho_bath, magnetizations, phases)
    purity = float(np.real(np.trace(rho_bath @ rho_bath)))
    exact = coherence_order_squared_norms(rho_bath, magnetizations, orders)
    derivative = coherence_order_squared_norms(
        drho_bath,
        magnetizations,
        orders,
    )
    derivative_norm_squared = float(np.sum(derivative))
    bath_qfi = qfi_vectorized(rho_for_qfi, drho_bath, tol=cfg.qfi_tol)

    if cfg.normalize_signal:
        if purity <= cfg.spectrum_noise_floor:
            raise RuntimeError("bath purity is too small to normalize the signal")
        signal = signal / purity
        exact = exact / purity

    if cfg.normalize_derivative_spectrum:
        if derivative_norm_squared <= cfg.spectrum_noise_floor:
            raise RuntimeError(
                "density-derivative norm is too small to normalize D_k"
            )
        derivative = derivative / derivative_norm_squared

    recovered = fourier_intensities(
        signal,
        cfg.n_phase_samples,
        orders,
        cfg.spectrum_noise_floor,
    )
    return PhaseCyclingResult(
        bath_density_matrix=rho_bath,
        bath_density_derivative=drho_bath,
        phases_rad=phases,
        signal=signal,
        coherence_orders=orders,
        exact_intensities=exact,
        fourier_intensities=recovered,
        derivative_intensities=derivative,
        purity=purity,
        derivative_norm_squared=derivative_norm_squared,
        bath_qfi=bath_qfi,
    )


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


def parameter_tags(cfg: PhaseCyclingConfig) -> str:
    """Encode every configuration field in stable, readable filename tags."""
    tag_names = {
        "N": "N",
        "Omega": "Om",
        "omega": "om",
        "J_nominal": "J",
        "dJ": "dJ",
        "interrogation_time": "t",
        "central_theta_rad": "tc",
        "bath_theta_rad": "tb",
        "bath_phi_rad": "pb",
        "n_phase_samples": "nphi",
        "normalize_signal": "normI",
        "normalize_derivative_spectrum": "normD",
        "spectrum_noise_floor": "floor",
        "qfi_tol": "qtol",
        "figure_width_in": "fw",
        "figure_height_in": "fh",
        "figure_dpi": "dpi",
        "figure_format": "fmt",
        "colormap": "cmap",
        "show_figure": "show",
    }
    angle_fields = {
        "central_theta_rad",
        "bath_theta_rad",
        "bath_phi_rad",
    }
    tags: list[str] = []
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if field.name in angle_fields:
            formatted = format_angle(float(value))
        elif isinstance(value, bool):
            formatted = str(int(value))
        elif isinstance(value, (float, int, np.floating, np.integer)):
            formatted = format_number(value)
        else:
            formatted = str(value)
        tags.append(f"{tag_names[field.name]}={formatted}")
    return "__".join(sanitize_tag(tag) for tag in tags)


def output_path(cfg: PhaseCyclingConfig) -> Path:
    """Return the traceable ``graphs/<script-stem>/`` figure path."""
    repository_root = Path(__file__).resolve().parent.parent
    directory = repository_root / "graphs" / Path(__file__).stem
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (
        f"mqc-spectrum__{parameter_tags(cfg)}.{cfg.figure_format.lower()}"
    )


def plot_phase_cycling(
    result: PhaseCyclingResult,
    cfg: PhaseCyclingConfig,
) -> Path:
    """Plot the phase signal and exact/Fourier coherence-order spectrum."""
    figure, (signal_axis, spectrum_axis) = plt.subplots(
        2,
        1,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
    )

    signal_axis.plot(
        result.phases_rad,
        result.signal.real,
        linewidth=2.0,
        color="tab:blue",
    )
    signal_axis.set_xlim(0.0, 2.0 * np.pi)
    signal_axis.set_xlabel(r"Phase-cycling angle $\phi$")
    signal_axis.set_ylabel(
        r"Normalized return signal $S(\phi)/S(0)$"
        if cfg.normalize_signal
        else r"Return signal $S(\phi)$"
    )
    signal_axis.grid(True, linestyle=":", alpha=0.8)

    colormap = plt.colormaps[cfg.colormap]
    coherence_color = colormap(0.25)
    derivative_color = colormap(0.78)
    spectrum_axis.bar(
        result.coherence_orders,
        result.exact_intensities,
        width=0.8,
        color=coherence_color,
        alpha=0.5,
        label=(
            r"$I_k/\sum_j I_j$"
            if cfg.normalize_signal
            else r"$I_k=\sum_{m-m'=k}|\rho_{m,m'}|^2$"
        ),
    )
    spectrum_axis.plot(
        result.coherence_orders,
        result.derivative_intensities,
        color=derivative_color,
        marker="o",
        markersize=4.0,
        linewidth=2.0,
        label=(
            r"$D_k/\sum_j D_j$"
            if cfg.normalize_derivative_spectrum
            else r"$D_k=\sum_{m-m'=k}|\partial_J\rho_{m,m'}|^2$"
        ),
    )
    spectrum_axis.scatter(
        result.coherence_orders,
        result.fourier_intensities,
        color="black",
        marker="x",
        s=28,
        label=r"$I_k$ recovered from phase-cycle FFT",
        zorder=3,
    )
    spectrum_axis.set_xlabel(r"Coherence order $k=m-m'$")
    spectrum_axis.set_ylabel("Normalized order weight")
    if not (cfg.normalize_signal and cfg.normalize_derivative_spectrum):
        spectrum_axis.set_ylabel("Coherence-order weight")
    spectrum_axis.set_xlim(-cfg.N - 0.75, cfg.N + 0.75)
    spectrum_axis.grid(True, axis="y", linestyle=":", alpha=0.8)
    spectrum_axis.legend()

    figure.suptitle(
        "Collective-bath phase cycling\n"
        rf"$N={cfg.N}$, $J={cfg.J_nominal:g}$, "
        rf"$\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, "
        rf"$t={cfg.interrogation_time:g}$, "
        rf"$F_Q^{{\mathrm{{bath}}}}={result.bath_qfi:.4g}$"
    )
    figure.tight_layout()

    path = output_path(cfg)
    figure.savefig(
        path,
        format=cfg.figure_format,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )
    if cfg.show_figure:
        plt.show()
    plt.close(figure)
    return path


def parse_config(argv: list[str] | None = None) -> PhaseCyclingConfig:
    """Parse command-line overrides into the typed configuration."""
    defaults = PhaseCyclingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument(
        "--time",
        dest="interrogation_time",
        type=float,
        default=defaults.interrogation_time,
    )
    parser.add_argument(
        "--central-theta-rad",
        type=float,
        default=defaults.central_theta_rad,
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
        "--n-phase-samples",
        type=int,
        default=defaults.n_phase_samples,
    )
    parser.add_argument(
        "--normalize-signal",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize_signal,
    )
    parser.add_argument(
        "--normalize-derivative-spectrum",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize_derivative_spectrum,
    )
    parser.add_argument(
        "--spectrum-noise-floor",
        type=float,
        default=defaults.spectrum_noise_floor,
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
        "--show",
        dest="show_figure",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_figure,
    )
    return PhaseCyclingConfig(**vars(parser.parse_args(argv)))


def print_summary(result: PhaseCyclingResult, path: Path, cfg: PhaseCyclingConfig) -> None:
    """Print QFI, coherence/sensitivity spans, diagnostics, and output path."""
    reconstruction_error = float(
        np.max(np.abs(result.exact_intensities - result.fourier_intensities))
    )
    populated = result.coherence_orders[
        result.exact_intensities > cfg.spectrum_noise_floor
    ]
    derivative_populated = result.coherence_orders[
        result.derivative_intensities > cfg.spectrum_noise_floor
    ]
    maximum_order = int(np.max(np.abs(populated))) if len(populated) else 0
    maximum_derivative_order = (
        int(np.max(np.abs(derivative_populated)))
        if len(derivative_populated)
        else 0
    )
    edge_mask = np.abs(result.coherence_orders) == cfg.N
    edge_coherence_weight = float(np.sum(result.exact_intensities[edge_mask]))
    edge_derivative_weight = float(
        np.sum(result.derivative_intensities[edge_mask])
    )
    imaginary_residual = float(np.max(np.abs(result.signal.imag)))
    print(f"Bath purity Tr(rho_B^2): {result.purity:.12g}")
    print(f"Bath QFI for J: {result.bath_qfi:.12g}")
    print(
        "Derivative norm Tr[(d_J rho_B)^2]: "
        f"{result.derivative_norm_squared:.12g}"
    )
    print(f"Largest I_k order above floor: |k|={maximum_order}")
    print(f"Largest D_k order above floor: |k|={maximum_derivative_order}")
    print(f"Total I_k weight at |k|=N: {edge_coherence_weight:.6e}")
    print(f"Total D_k weight at |k|=N: {edge_derivative_weight:.6e}")
    print(f"Maximum FFT reconstruction error: {reconstruction_error:.3e}")
    print(f"Maximum imaginary signal residual: {imaginary_residual:.3e}")
    print(f"Saved phase-cycling plot to {path}")


def main(argv: list[str] | None = None) -> Path:
    """Run the configured phase cycle, save its plot, and return its path."""
    cfg = parse_config(argv)
    validate_config(cfg)
    result = run_phase_cycling(cfg)
    path = plot_phase_cycling(result, cfg)
    print_summary(result, path, cfg)
    return path


if __name__ == "__main__":
    main()
