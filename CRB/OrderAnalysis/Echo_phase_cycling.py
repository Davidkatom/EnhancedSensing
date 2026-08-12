"""Phase-cycle the reduced bath state after an imperfect Hamiltonian echo.

The global protocol is

    |psi_0> --U(J)--> --U(J0)^dagger--> |psi_after>,

where ``U(J) = exp(-i H(J) t)`` and ``J0`` is the coupling estimate used for
the reverse evolution.  The analyzed state is

    rho_B_after(J; J0) = Tr_central[|psi_after><psi_after|].

The script then performs the same MQC analysis as ``plot_phase_cycling.py``:

    I_k = sum_(m-m'=k) |rho_B_after[m,m']|^2,
    D_k = sum_(m-m'=k) |partial_J rho_B_after[m,m']|^2,

with ``partial_J`` evaluated by a centered difference while ``J0`` remains
fixed.  Phase cycling is applied to ``rho_B_after`` itself, so the FFT
reconstructs its ``I_k`` distribution rather than a Loschmidt return signal.
It also evaluates the classical Fisher information of a projective collective
``S_z`` measurement from the full Dicke-basis population distribution,

    F_C^(Z) = sum_(m : p_m > tol) (partial_J p_m)^2 / p_m.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from CRB.OrderAnalysis.plot_phase_cycling import (  # noqa: E402
    PhaseCyclingConfig,
    coherence_order_squared_norms,
    format_angle,
    format_number,
    fourier_intensities,
    magnetization_eigenvalues,
    phase_cycling_signal,
    sanitize_tag,
    validate_config as validate_phase_cycling_config,
)
from CRB.crb_core import (  # noqa: E402
    build_hamiltonian,
    central_spin_state,
    coherent_bath_state,
    observable_projective_fisher,
    qfi_vectorized,
    save_plot,
)


@dataclass(frozen=True, slots=True)
class EchoPhaseCyclingConfig(PhaseCyclingConfig):
    """Phase-cycling parameters plus the reverse-evolution estimate."""

    J_estimate: float = -0.99
    figure_height_in: float = 10.0


@dataclass(frozen=True, slots=True)
class EchoPhaseCyclingResult:
    """Post-echo bath state, derivative, phase trace, and order spectra."""

    bath_density_matrix: np.ndarray
    bath_density_derivative: np.ndarray
    phases_rad: np.ndarray
    signal: np.ndarray
    coherence_orders: np.ndarray
    coherence_intensities: np.ndarray
    fourier_intensities: np.ndarray
    derivative_intensities: np.ndarray
    dicke_populations: np.ndarray
    dicke_population_derivatives: np.ndarray
    purity: float
    derivative_norm_squared: float
    bath_qfi: float
    dicke_population_fi: float


def validate_config(cfg: EchoPhaseCyclingConfig) -> None:
    """Validate inherited MQC controls and the echo coupling estimate."""
    validate_phase_cycling_config(cfg)
    if not np.isfinite(cfg.J_estimate):
        raise ValueError("J_estimate must be finite")


def initial_joint_state(cfg: EchoPhaseCyclingConfig) -> np.ndarray:
    """Return the configured central-spin/bath product-state vector."""
    central = central_spin_state(cfg.central_theta_rad).full().ravel()
    bath = coherent_bath_state(
        cfg.N,
        theta=cfg.bath_theta_rad,
        phi=cfg.bath_phi_rad,
    )
    return np.kron(central, bath)


def spectral_hamiltonian(
    cfg: EchoPhaseCyclingConfig,
    J: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues/eigenvectors of ``H(J)`` from the shared core."""
    hamiltonian = build_hamiltonian(
        Omega_0=cfg.Omega,
        omega=cfg.omega,
        J=J,
        N=cfg.N,
    ).full()
    return np.linalg.eigh(hamiltonian)


def apply_propagator(
    state: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    time: float,
) -> np.ndarray:
    """Apply ``exp(-i H time)`` using an existing spectral decomposition."""
    eigenbasis_state = eigenvectors.conj().T @ state
    return eigenvectors @ (
        np.exp(-1j * eigenvalues * time) * eigenbasis_state
    )


def reduced_bath_density_matrix(state: np.ndarray, N: int) -> np.ndarray:
    """Trace the central spin from one pure joint-state vector."""
    expected_shape = (2 * (N + 1),)
    if state.shape != expected_shape:
        raise ValueError(f"state must have shape {expected_shape}, got {state.shape}")
    amplitudes = state.reshape(2, N + 1)
    return amplitudes.T @ amplitudes.conj()


def post_echo_bath_state(
    cfg: EchoPhaseCyclingConfig,
    J: float,
    initial_state: np.ndarray,
    reverse_spectrum: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return ``rho_B`` after ``U(J)`` followed by fixed ``U(J0)^dagger``."""
    forward_eigenvalues, forward_eigenvectors = spectral_hamiltonian(cfg, J)
    forward_state = apply_propagator(
        initial_state,
        forward_eigenvalues,
        forward_eigenvectors,
        cfg.interrogation_time,
    )
    reverse_eigenvalues, reverse_eigenvectors = reverse_spectrum
    after_echo = apply_propagator(
        forward_state,
        reverse_eigenvalues,
        reverse_eigenvectors,
        -cfg.interrogation_time,
    )
    return reduced_bath_density_matrix(after_echo, cfg.N)


def run_echo_phase_cycling(
    cfg: EchoPhaseCyclingConfig,
) -> EchoPhaseCyclingResult:
    """Calculate post-echo order spectra, QFI, Dicke FI, and phase FFT."""
    initial_state = initial_joint_state(cfg)
    reverse_spectrum = spectral_hamiltonian(cfg, cfg.J_estimate)
    rho_bath = post_echo_bath_state(
        cfg,
        cfg.J_nominal,
        initial_state,
        reverse_spectrum,
    )
    rho_plus = post_echo_bath_state(
        cfg,
        cfg.J_nominal + cfg.dJ,
        initial_state,
        reverse_spectrum,
    )
    rho_minus = post_echo_bath_state(
        cfg,
        cfg.J_nominal - cfg.dJ,
        initial_state,
        reverse_spectrum,
    )
    drho_bath = (rho_plus - rho_minus) / (2.0 * cfg.dJ)

    magnetizations = magnetization_eigenvalues(cfg)
    dicke_populations = np.real(np.diag(rho_bath))
    dicke_population_derivatives = np.real(np.diag(drho_bath))
    dicke_population_fi = observable_projective_fisher(
        rho_bath,
        drho_bath,
        np.diag(magnetizations),
        tol=cfg.qfi_tol,
    )
    orders = np.arange(-cfg.N, cfg.N + 1, dtype=int)
    phases = np.linspace(0.0, 2.0 * np.pi, cfg.n_phase_samples, endpoint=False)
    signal = phase_cycling_signal(rho_bath, magnetizations, phases)
    purity = float(np.real(np.trace(rho_bath @ rho_bath)))
    coherence = coherence_order_squared_norms(
        rho_bath,
        magnetizations,
        orders,
    )
    derivative = coherence_order_squared_norms(
        drho_bath,
        magnetizations,
        orders,
    )
    derivative_norm_squared = float(np.sum(derivative))
    bath_qfi = qfi_vectorized(rho_bath, drho_bath, tol=cfg.qfi_tol)

    if cfg.normalize_signal:
        if purity <= cfg.spectrum_noise_floor:
            raise RuntimeError("post-echo bath purity is too small to normalize")
        signal = signal / purity
        coherence = coherence / purity
    if cfg.normalize_derivative_spectrum:
        if derivative_norm_squared <= cfg.spectrum_noise_floor:
            raise RuntimeError(
                "post-echo density derivative is too small to normalize D_k"
            )
        derivative = derivative / derivative_norm_squared

    recovered = fourier_intensities(
        signal,
        cfg.n_phase_samples,
        orders,
        cfg.spectrum_noise_floor,
    )
    return EchoPhaseCyclingResult(
        bath_density_matrix=rho_bath,
        bath_density_derivative=drho_bath,
        phases_rad=phases,
        signal=signal,
        coherence_orders=orders,
        coherence_intensities=coherence,
        fourier_intensities=recovered,
        derivative_intensities=derivative,
        dicke_populations=dicke_populations,
        dicke_population_derivatives=dicke_population_derivatives,
        purity=purity,
        derivative_norm_squared=derivative_norm_squared,
        bath_qfi=bath_qfi,
        dicke_population_fi=dicke_population_fi,
    )


def parameter_tags(cfg: EchoPhaseCyclingConfig) -> str:
    """Encode every field in compact grouped tags suitable for Windows."""
    encoded_fields = {
        "N",
        "Omega",
        "omega",
        "J_nominal",
        "dJ",
        "interrogation_time",
        "central_theta_rad",
        "bath_theta_rad",
        "bath_phi_rad",
        "n_phase_samples",
        "normalize_signal",
        "normalize_derivative_spectrum",
        "spectrum_noise_floor",
        "qfi_tol",
        "figure_width_in",
        "figure_height_in",
        "figure_dpi",
        "figure_format",
        "colormap",
        "show_figure",
        "J_estimate",
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
            f"H=Om{format_number(cfg.Omega)}-om{format_number(cfg.omega)}-"
            f"J{format_number(cfg.J_nominal)}-"
            f"J0={format_number(cfg.J_estimate)}-"
            f"dJ{format_number(cfg.dJ)}-"
            f"t{format_number(cfg.interrogation_time)}"
        ),
        (
            f"state=tc{format_angle(cfg.central_theta_rad)}-"
            f"tb{format_angle(cfg.bath_theta_rad)}-"
            f"pb{format_angle(cfg.bath_phi_rad)}"
        ),
        (
            f"cycle=n{cfg.n_phase_samples}-nI{int(cfg.normalize_signal)}-"
            f"nD{int(cfg.normalize_derivative_spectrum)}-"
            f"floor{format_number(cfg.spectrum_noise_floor)}-"
            f"qtol{format_number(cfg.qfi_tol)}"
        ),
        (
            f"fig={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)}-dpi{cfg.figure_dpi}-"
            f"{cfg.figure_format}-{cfg.colormap}-show{int(cfg.show_figure)}"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def output_path(cfg: EchoPhaseCyclingConfig) -> Path:
    """Return the parameter-rich filename passed to the shared plot saver."""
    return Path(
        f"post-echo-mqc__{parameter_tags(cfg)}.{cfg.figure_format.lower()}"
    )


def plot_echo_phase_cycling(
    result: EchoPhaseCyclingResult,
    cfg: EchoPhaseCyclingConfig,
) -> Path:
    """Plot phase cycling, order spectra, and post-echo Fisher information."""
    figure, (signal_axis, spectrum_axis, information_axis) = plt.subplots(
        3,
        1,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
        gridspec_kw={"height_ratios": (1.0, 1.0, 0.75)},
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
        r"Normalized post-echo signal $S(\phi)/S(0)$"
        if cfg.normalize_signal
        else r"Post-echo phase-cycling signal $S(\phi)$"
    )
    signal_axis.ticklabel_format(style="plain", axis="y", useOffset=False)
    signal_axis.grid(True, linestyle=":", alpha=0.8)

    colormap = plt.colormaps[cfg.colormap]
    spectrum_axis.bar(
        result.coherence_orders,
        result.coherence_intensities,
        width=0.8,
        color=colormap(0.25),
        alpha=0.5,
        label=(
            r"Post-echo $I_k/\sum_jI_j$"
            if cfg.normalize_signal
            else r"Post-echo $I_k$"
        ),
    )
    spectrum_axis.plot(
        result.coherence_orders,
        result.derivative_intensities,
        color=colormap(0.78),
        marker="o",
        markersize=4.0,
        linewidth=2.0,
        label=(
            r"Post-echo $D_k/\sum_jD_j$"
            if cfg.normalize_derivative_spectrum
            else r"Post-echo $D_k$"
        ),
    )
    spectrum_axis.scatter(
        result.coherence_orders,
        result.fourier_intensities,
        color="black",
        marker="x",
        s=28,
        label=r"Post-echo $I_k$ recovered by FFT",
        zorder=3,
    )
    spectrum_axis.set_xlabel(r"Coherence order $k=m-m'$")
    spectrum_axis.set_ylabel("Normalized order weight")
    if not (cfg.normalize_signal and cfg.normalize_derivative_spectrum):
        spectrum_axis.set_ylabel("Coherence-order weight")
    spectrum_axis.set_xlim(-cfg.N - 0.75, cfg.N + 0.75)
    spectrum_axis.grid(True, axis="y", linestyle=":", alpha=0.8)
    spectrum_axis.legend()

    information_bars = information_axis.bar(
        (r"$F_Q$", r"$F_C^{(Z)}$"),
        (result.bath_qfi, result.dicke_population_fi),
        color=("tab:purple", "tab:orange"),
        width=0.6,
    )
    information_axis.bar_label(
        information_bars,
        labels=(
            f"{result.bath_qfi:.4g}",
            f"{result.dicke_population_fi:.4g}",
        ),
        padding=3,
    )
    maximum_information = max(result.bath_qfi, result.dicke_population_fi)
    information_axis.set_ylim(
        0.0,
        1.15 * maximum_information if maximum_information > 0.0 else 1.0,
    )
    information_axis.set_ylabel("Fisher information")
    information_axis.set_title("Post-echo information for estimating $J$")
    information_axis.grid(True, axis="y", linestyle=":", alpha=0.8)

    figure.suptitle(
        "Post-echo collective-bath phase cycling\n"
        rf"$N={cfg.N}$, $J={cfg.J_nominal:g}$, $J_0={cfg.J_estimate:g}$, "
        rf"$\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, "
        rf"$t={cfg.interrogation_time:g}$"
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


def parse_config(argv: list[str] | None = None) -> EchoPhaseCyclingConfig:
    """Parse command-line overrides into the typed echo configuration."""
    defaults = EchoPhaseCyclingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--J0", dest="J_estimate", type=float, default=defaults.J_estimate)
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
    return EchoPhaseCyclingConfig(**vars(parser.parse_args(argv)))


def print_summary(
    result: EchoPhaseCyclingResult,
    path: Path,
    cfg: EchoPhaseCyclingConfig,
) -> None:
    """Print post-echo QFI, coherence spans, diagnostics, and output path."""
    reconstruction_error = float(
        np.max(
            np.abs(
                result.coherence_intensities - result.fourier_intensities
            )
        )
    )
    populated_i = result.coherence_orders[
        result.coherence_intensities > cfg.spectrum_noise_floor
    ]
    populated_d = result.coherence_orders[
        result.derivative_intensities > cfg.spectrum_noise_floor
    ]
    maximum_i = int(np.max(np.abs(populated_i))) if len(populated_i) else 0
    maximum_d = int(np.max(np.abs(populated_d))) if len(populated_d) else 0
    edge_mask = np.abs(result.coherence_orders) == cfg.N
    edge_i = float(np.sum(result.coherence_intensities[edge_mask]))
    edge_d = float(np.sum(result.derivative_intensities[edge_mask]))
    imaginary_residual = float(np.max(np.abs(result.signal.imag)))
    print(f"True coupling J: {cfg.J_nominal:.12g}")
    print(f"Reverse-evolution estimate J0: {cfg.J_estimate:.12g}")
    print(f"Post-echo bath purity: {result.purity:.12g}")
    print(f"Post-echo bath QFI for J: {result.bath_qfi:.12g}")
    print(
        "Post-echo Dicke-population classical FI F_C^(Z): "
        f"{result.dicke_population_fi:.12g}"
    )
    print(
        "Post-echo derivative norm Tr[(d_J rho_B)^2]: "
        f"{result.derivative_norm_squared:.12g}"
    )
    print(f"Largest post-echo I_k order above floor: |k|={maximum_i}")
    print(f"Largest post-echo D_k order above floor: |k|={maximum_d}")
    print(f"Total post-echo I_k weight at |k|=N: {edge_i:.6e}")
    print(f"Total post-echo D_k weight at |k|=N: {edge_d:.6e}")
    print(f"Maximum FFT reconstruction error: {reconstruction_error:.3e}")
    print(f"Maximum imaginary signal residual: {imaginary_residual:.3e}")
    print(f"Saved post-echo phase-cycling plot to {path}")


def main(argv: list[str] | None = None) -> Path:
    """Analyze the post-echo state, save the plot, and return its path."""
    cfg = parse_config(argv)
    validate_config(cfg)
    result = run_echo_phase_cycling(cfg)
    path = plot_echo_phase_cycling(result, cfg)
    print_summary(result, path, cfg)
    return path


if __name__ == "__main__":
    main()
