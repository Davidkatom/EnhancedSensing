"""Compare post-decoding bath QFI with three collective-spin readouts.

The protocol is

    |psi_0> --exp[-i H(J) t_s]--> --exp[-i H_dec(J0) t_d]--> |psi_out>,

where ``J`` is the true coupling and the decoder uses the fixed estimate
``J0``.  For ``H(J) = Omega sigma_x + J sigma_z S_z + omega S_x``, the
decoder Hamiltonian is

    H_dec(J0) = -Omega sigma_x + J0 sigma_z S_z - omega S_x.

Thus the single-spin and bath-drive terms are reversed, while the interaction
keeps its original sign and only ``J`` is replaced by ``J0``.  At every
decoding time ``t_d``, the script traces out the central spin and evaluates
the reduced-bath quantum Fisher information for ``J``.

It also evaluates the classical Fisher information accessible from the three
linear collective-spin moments ``<S_x>``, ``<S_y>``, and ``<S_z>`` via error
propagation.  These noncommuting readouts are treated as separate experimental
settings, so their information is added exactly as requested.  The plotted
quantity is

    F_Q / (F_C[<S_x>] + F_C[<S_y>] + F_C[<S_z>]).
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
    build_hamiltonian,
    central_spin_state,
    coherent_bath_state,
    observable_moment_fisher,
    qfi_vectorized,
)


@dataclass(frozen=True, slots=True)
class EchoFirstOrderConfig:
    """Physics, decoding sweep, numerical, state, and plot parameters."""

    N: int = 15
    Omega: float = 2.5
    omega: float = 1.0
    J_nominal: float = 1.0
    J_estimate: float = 1.0
    dJ: float = 1e-4

    sensing_time: float = 30.0
    decode_time_min: float = 0.0
    decode_time_max: float | None = 10
    n_decode_times: int = 301

    central_theta_rad: float = np.pi / 2.0
    central_phi_rad: float = 0.0
    bath_theta_rad: float = 0.0
    bath_phi_rad: float = 0.0

    qfi_tol: float = 1e-12
    classical_fisher_variance_floor: float = 1e-12
    ratio_denominator_floor: float = 1e-15

    figure_width_in: float = 10.0
    figure_height_in: float = 6.0
    figure_dpi: int = 200
    figure_format: str = "png"
    colormap: str = "viridis"
    show_figure: bool = True


@dataclass(frozen=True, slots=True)
class EchoFirstOrderResult:
    """Fisher-information trajectories over the decoding-time sweep."""

    decode_times: np.ndarray
    bath_qfi: np.ndarray
    classical_fi_x: np.ndarray
    classical_fi_y: np.ndarray
    classical_fi_z: np.ndarray
    classical_fi_sum: np.ndarray
    qfi_to_classical_sum_ratio: np.ndarray


def resolved_decode_time_max(cfg: EchoFirstOrderConfig) -> float:
    """Return the configured maximum, or ``2 t_s`` for the automatic range."""
    if cfg.decode_time_max is not None:
        return cfg.decode_time_max
    return 2.0 * cfg.sensing_time


def validate_config(cfg: EchoFirstOrderConfig) -> None:
    """Reject invalid physical, numerical, sweep, and plotting parameters."""
    finite_values = {
        "Omega": cfg.Omega,
        "omega": cfg.omega,
        "J_nominal": cfg.J_nominal,
        "J_estimate": cfg.J_estimate,
        "dJ": cfg.dJ,
        "sensing_time": cfg.sensing_time,
        "decode_time_min": cfg.decode_time_min,
        "central_theta_rad": cfg.central_theta_rad,
        "central_phi_rad": cfg.central_phi_rad,
        "bath_theta_rad": cfg.bath_theta_rad,
        "bath_phi_rad": cfg.bath_phi_rad,
        "qfi_tol": cfg.qfi_tol,
        "classical_fisher_variance_floor": cfg.classical_fisher_variance_floor,
        "ratio_denominator_floor": cfg.ratio_denominator_floor,
        "figure_width_in": cfg.figure_width_in,
        "figure_height_in": cfg.figure_height_in,
    }
    if cfg.decode_time_max is not None:
        finite_values["decode_time_max"] = cfg.decode_time_max
    nonfinite = [name for name, value in finite_values.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError(f"configuration values must be finite: {nonfinite}")
    if cfg.N < 1:
        raise ValueError("N must be positive")
    if cfg.dJ <= 0.0:
        raise ValueError("dJ must be positive")
    if cfg.sensing_time < 0.0:
        raise ValueError("sensing_time must be non-negative")
    if cfg.decode_time_min < 0.0:
        raise ValueError("decode_time_min must be non-negative")
    if resolved_decode_time_max(cfg) <= cfg.decode_time_min:
        raise ValueError("decode_time_max must be greater than decode_time_min")
    if cfg.n_decode_times < 2:
        raise ValueError("n_decode_times must be at least 2")
    if cfg.qfi_tol <= 0.0:
        raise ValueError("qfi_tol must be positive")
    if cfg.classical_fisher_variance_floor <= 0.0:
        raise ValueError("classical_fisher_variance_floor must be positive")
    if cfg.ratio_denominator_floor <= 0.0:
        raise ValueError("ratio_denominator_floor must be positive")
    if cfg.figure_width_in <= 0.0 or cfg.figure_height_in <= 0.0:
        raise ValueError("figure dimensions must be positive")
    if cfg.figure_dpi <= 0:
        raise ValueError("figure_dpi must be positive")
    if not cfg.figure_format.strip():
        raise ValueError("figure_format must not be empty")
    if cfg.colormap not in plt.colormaps():
        raise ValueError(f"unknown matplotlib colormap: {cfg.colormap!r}")


def initial_joint_state(cfg: EchoFirstOrderConfig) -> np.ndarray:
    """Return the configured central-spin/bath product-state vector."""
    central = central_spin_state(
        cfg.central_theta_rad,
        cfg.central_phi_rad,
    ).full().ravel()
    bath = coherent_bath_state(
        cfg.N,
        theta=cfg.bath_theta_rad,
        phi=cfg.bath_phi_rad,
    )
    return np.kron(central, bath)


def spectral_hamiltonian(
    cfg: EchoFirstOrderConfig,
    J: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the eigendecomposition of the shared Hamiltonian ``H(J)``."""
    hamiltonian = build_hamiltonian(
        Omega_0=cfg.Omega,
        omega=cfg.omega,
        J=J,
        N=cfg.N,
    ).full()
    return np.linalg.eigh(hamiltonian)


def spectral_decoder_hamiltonian(
    cfg: EchoFirstOrderConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize ``-Omega sigma_x + J0 sigma_z S_z - omega S_x``."""
    decoder_hamiltonian = build_hamiltonian(
        Omega_0=-cfg.Omega,
        omega=-cfg.omega,
        J=cfg.J_estimate,
        N=cfg.N,
    ).full()
    return np.linalg.eigh(decoder_hamiltonian)


def sensed_state(
    cfg: EchoFirstOrderConfig,
    J: float,
    initial_state: np.ndarray,
) -> np.ndarray:
    """Apply the sensing evolution ``exp[-i H(J) t_s]`` once."""
    eigenvalues, eigenvectors = spectral_hamiltonian(cfg, J)
    return apply_propagator(
        initial_state,
        eigenvalues,
        eigenvectors,
        cfg.sensing_time,
    )


def decoded_bath_state(
    sensed: np.ndarray,
    decode_time: float,
    decoder_spectrum: tuple[np.ndarray, np.ndarray],
    N: int,
) -> np.ndarray:
    """Apply ``exp[-i H_dec(J0) t_d]`` and trace out the central spin."""
    eigenvalues, eigenvectors = decoder_spectrum
    decoded = apply_propagator(
        sensed,
        eigenvalues,
        eigenvectors,
        decode_time,
    )
    return reduced_bath_density_matrix(decoded, N)


def run_echo_first_order(cfg: EchoFirstOrderConfig) -> EchoFirstOrderResult:
    """Calculate QFI and three linear-moment classical FI trajectories."""
    decode_times = np.linspace(
        cfg.decode_time_min,
        resolved_decode_time_max(cfg),
        cfg.n_decode_times,
    )
    initial_state = initial_joint_state(cfg)
    decoder_spectrum = spectral_decoder_hamiltonian(cfg)
    sensed_nominal = sensed_state(cfg, cfg.J_nominal, initial_state)
    sensed_plus = sensed_state(cfg, cfg.J_nominal + cfg.dJ, initial_state)
    sensed_minus = sensed_state(cfg, cfg.J_nominal - cfg.dJ, initial_state)

    bath_operators = build_bath_operators(cfg.N)
    measurement_operators = (
        bath_operators["Jx"],
        bath_operators["Jy"],
        bath_operators["Jz"],
    )
    n_times = len(decode_times)
    bath_qfi = np.empty(n_times, dtype=float)
    classical_fi_x = np.empty(n_times, dtype=float)
    classical_fi_y = np.empty(n_times, dtype=float)
    classical_fi_z = np.empty(n_times, dtype=float)

    for index, decode_time in enumerate(decode_times):
        rho_bath = decoded_bath_state(
            sensed_nominal,
            decode_time,
            decoder_spectrum,
            cfg.N,
        )
        rho_plus = decoded_bath_state(
            sensed_plus,
            decode_time,
            decoder_spectrum,
            cfg.N,
        )
        rho_minus = decoded_bath_state(
            sensed_minus,
            decode_time,
            decoder_spectrum,
            cfg.N,
        )
        drho_bath = (rho_plus - rho_minus) / (2.0 * cfg.dJ)

        bath_qfi[index] = qfi_vectorized(
            rho_bath,
            drho_bath,
            tol=cfg.qfi_tol,
        )
        classical_values = tuple(
            observable_moment_fisher(
                rho_bath,
                drho_bath,
                observable,
                var_floor=cfg.classical_fisher_variance_floor,
            )
            for observable in measurement_operators
        )
        classical_fi_x[index], classical_fi_y[index], classical_fi_z[index] = (
            classical_values
        )

    classical_fi_sum = classical_fi_x + classical_fi_y + classical_fi_z
    ratio = np.full(n_times, np.nan, dtype=float)
    valid = classical_fi_sum > cfg.ratio_denominator_floor
    ratio[valid] = bath_qfi[valid] / classical_fi_sum[valid]

    return EchoFirstOrderResult(
        decode_times=decode_times,
        bath_qfi=bath_qfi,
        classical_fi_x=classical_fi_x,
        classical_fi_y=classical_fi_y,
        classical_fi_z=classical_fi_z,
        classical_fi_sum=classical_fi_sum,
        qfi_to_classical_sum_ratio=ratio,
    )


def parameter_tags(cfg: EchoFirstOrderConfig) -> str:
    """Encode every configuration field into compact, stable filename tags."""
    encoded_fields = {
        "N",
        "Omega",
        "omega",
        "J_nominal",
        "J_estimate",
        "dJ",
        "sensing_time",
        "decode_time_min",
        "decode_time_max",
        "n_decode_times",
        "central_theta_rad",
        "central_phi_rad",
        "bath_theta_rad",
        "bath_phi_rad",
        "qfi_tol",
        "classical_fisher_variance_floor",
        "ratio_denominator_floor",
        "figure_width_in",
        "figure_height_in",
        "figure_dpi",
        "figure_format",
        "colormap",
        "show_figure",
    }
    config_fields = {field.name for field in fields(cfg)}
    if encoded_fields != config_fields:
        missing = sorted(config_fields - encoded_fields)
        extra = sorted(encoded_fields - config_fields)
        raise RuntimeError(
            f"filename-tag field mismatch; missing={missing}, extra={extra}"
        )
    decode_max_tag = (
        "auto2ts"
        if cfg.decode_time_max is None
        else format_number(cfg.decode_time_max)
    )
    tags = (
        f"N{cfg.N}",
        (
            f"H=O{format_number(cfg.Omega)}-w{format_number(cfg.omega)}-"
            f"J{format_number(cfg.J_nominal)}-J0{format_number(cfg.J_estimate)}-"
            f"dJ{format_number(cfg.dJ)}"
        ),
        (
            f"time=ts{format_number(cfg.sensing_time)}-"
            f"d{format_number(cfg.decode_time_min)}to{decode_max_tag}-"
            f"n{cfg.n_decode_times}"
        ),
        (
            f"state=tc{format_angle(cfg.central_theta_rad)}-"
            f"pc{format_angle(cfg.central_phi_rad)}-"
            f"tb{format_angle(cfg.bath_theta_rad)}-"
            f"pb{format_angle(cfg.bath_phi_rad)}"
        ),
        (
            f"num=qt{format_number(cfg.qfi_tol)}-"
            f"vf{format_number(cfg.classical_fisher_variance_floor)}-"
            f"rf{format_number(cfg.ratio_denominator_floor)}"
        ),
        (
            f"fig={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)}-d{cfg.figure_dpi}-"
            f"{cfg.figure_format}-{cfg.colormap}-s{int(cfg.show_figure)}"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def output_path(cfg: EchoFirstOrderConfig) -> Path:
    """Return a parameter-rich path under ``graphs/Echo_first_order``."""
    directory = REPOSITORY_ROOT / "graphs" / Path(__file__).stem
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (
        f"keep-J-decoder-qfi-to-xyz-linear-moment-fi__{parameter_tags(cfg)}."
        f"{cfg.figure_format.lower()}"
    )


def plot_echo_first_order(
    result: EchoFirstOrderResult,
    cfg: EchoFirstOrderConfig,
) -> Path:
    """Plot QFI divided by the sum of three linear-moment FI values."""
    figure, axis = plt.subplots(
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
    )
    color = plt.colormaps[cfg.colormap](0.68)
    axis.plot(
        result.decode_times,
        result.qfi_to_classical_sum_ratio,
        color=color,
        linewidth=2.0,
        label=(
            r"$F_Q/(F_C[\langle S_x\rangle]+F_C[\langle S_y\rangle]"
            r"+F_C[\langle S_z\rangle])$"
        ),
    )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    axis.axvline(
        cfg.sensing_time,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label=r"$t_d=t_s$",
    )
    axis.set_xlim(result.decode_times[0], result.decode_times[-1])
    axis.set_xlabel(r"Decoding time $t_d$")
    axis.set_ylabel(
        r"$F_Q/\sum_iF_C[\langle S_i\rangle]$"
    )
    axis.set_title(
        "Information accessible from three linear spin moments\n"
        r"Decoder: $(-\Omega,+J_0,-\omega)$; "
        rf"$N={cfg.N}$, $J={cfg.J_nominal:g}$, $J_0={cfg.J_estimate:g}$, "
        rf"$t_s={cfg.sensing_time:g}$"
    )
    axis.grid(True, linestyle=":", alpha=0.8)
    axis.legend()
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


def parse_config(argv: list[str] | None = None) -> EchoFirstOrderConfig:
    """Parse command-line overrides into the typed configuration."""
    defaults = EchoFirstOrderConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--Omega", type=float, default=defaults.Omega)
    parser.add_argument("--omega", type=float, default=defaults.omega)
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--J0", dest="J_estimate", type=float, default=defaults.J_estimate)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument(
        "--sense-time",
        "--t-s",
        dest="sensing_time",
        type=float,
        default=defaults.sensing_time,
    )
    parser.add_argument(
        "--decode-time-min",
        type=float,
        default=defaults.decode_time_min,
    )
    parser.add_argument(
        "--decode-time-max",
        type=float,
        default=defaults.decode_time_max,
        help="default: 2 * sensing_time",
    )
    parser.add_argument(
        "--n-decode-times",
        type=int,
        default=defaults.n_decode_times,
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
        "--qfi-tol",
        type=float,
        default=defaults.qfi_tol,
    )
    parser.add_argument(
        "--classical-fisher-variance-floor",
        type=float,
        default=defaults.classical_fisher_variance_floor,
    )
    parser.add_argument(
        "--ratio-denominator-floor",
        type=float,
        default=defaults.ratio_denominator_floor,
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
        "--show",
        dest="show_figure",
        action=argparse.BooleanOptionalAction,
        default=defaults.show_figure,
    )
    return EchoFirstOrderConfig(**vars(parser.parse_args(argv)))


def print_summary(
    result: EchoFirstOrderResult,
    path: Path,
    cfg: EchoFirstOrderConfig,
) -> None:
    """Print finite-ratio extrema and values nearest the nominal echo time."""
    finite = np.isfinite(result.qfi_to_classical_sum_ratio)
    print(f"Sensing time t_s: {cfg.sensing_time:.12g}")
    print(
        "Decode-time range: "
        f"[{result.decode_times[0]:.12g}, {result.decode_times[-1]:.12g}]"
    )
    if np.any(finite):
        finite_indices = np.flatnonzero(finite)
        minimum_index = finite_indices[
            np.argmin(result.qfi_to_classical_sum_ratio[finite])
        ]
        maximum_index = finite_indices[
            np.argmax(result.qfi_to_classical_sum_ratio[finite])
        ]
        print(
            "Minimum finite ratio: "
            f"{result.qfi_to_classical_sum_ratio[minimum_index]:.12g} "
            f"at t_d={result.decode_times[minimum_index]:.12g}"
        )
        print(
            "Maximum finite ratio: "
            f"{result.qfi_to_classical_sum_ratio[maximum_index]:.12g} "
            f"at t_d={result.decode_times[maximum_index]:.12g}"
        )
    echo_index = int(np.argmin(np.abs(result.decode_times - cfg.sensing_time)))
    print(f"Nearest sampled echo time t_d: {result.decode_times[echo_index]:.12g}")
    print(f"F_Q at nearest echo time: {result.bath_qfi[echo_index]:.12g}")
    print(
        "F_C[<Sx>] + F_C[<Sy>] + F_C[<Sz>] at nearest echo time: "
        f"{result.classical_fi_sum[echo_index]:.12g}"
    )
    print(
        "Ratio at nearest echo time: "
        f"{result.qfi_to_classical_sum_ratio[echo_index]:.12g}"
    )
    print(f"Saved first-order echo plot to {path}")


def main(argv: list[str] | None = None) -> Path:
    """Run the decoding-time sweep, save its plot, and return the path."""
    cfg = parse_config(argv)
    validate_config(cfg)
    result = run_echo_first_order(cfg)
    path = plot_echo_first_order(result, cfg)
    print_summary(result, path, cfg)
    return path


if __name__ == "__main__":
    main()
