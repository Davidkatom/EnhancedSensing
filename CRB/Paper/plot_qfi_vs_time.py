"""Plot global, bath, and central-spin QFI per interrogation time.

The dynamics and finite-difference QFI are reused from
``CRB.plot_qfi_and_spin_vs_time`` with the same simulation defaults.
The dashed Ramsey benchmark uses zero drives and the initial state
``|1>_central |+>_bath**N``. For the repository convention
``S_z = sum_i sigma_z^(i)``, its exact QFI is ``4*N*t**2``;
central-spin dephasing leaves this protocol unchanged.

Run from the repository root::

    python CRB/Paper/plot_qfi_vs_time.py
    python CRB/Paper/plot_qfi_vs_time.py --gamma 0.1 --t-max 2 --no-show

One figure is saved through smart_save under its configured
``Graphs/plot_qfi_vs_time/`` directory, with all parameters in the filename
and embedded metadata. Times and frequencies use the original model units
with hbar = 1; the estimated parameter is J.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CRB.plot_qfi_and_spin_vs_time import (
    SimulationConfig as TimeConfig,
    driven_protocol,
    format_angle,
    format_number,
    protocol_fisher_information_trajectories,
    sanitize_tag,
    validate_config as validate_time_config,
)
from CRB.smart_save import save_plot


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Physics, time sampling, and single-figure export settings."""

    N: int = 15
    Omega: float = 2.5
    omega: float = 1.0
    J_nominal: float = 1.0
    dJ: float = 1e-3
    gamma: float = 0.0  # L = sqrt(gamma) * sigma_z on the central spin.

    t_min: float = 0.0
    t_max: float = 1.0
    n_steps: int = 401

    central_theta_rad: float = np.pi / 2.0
    central_phi_rad: float = 0.0
    bath_theta_rad: float = 0.0
    bath_phi_rad: float = 0.0
    qfi_tol: float = 1e-12

    figure_width_in: float = 7.0
    figure_height_in: float = 4.5
    figure_dpi: int = 200
    figure_format: str = "png"
    show_figure: bool = True


def time_config(cfg: SimulationConfig) -> TimeConfig:
    """Adapt only the relevant settings to the existing simulation API."""
    return TimeConfig(**asdict(cfg))


def validate_config(cfg: SimulationConfig) -> None:
    """Reuse the original physics validation and check export support."""
    validate_time_config(time_config(cfg))
    if cfg.figure_format not in ("png", "pdf", "svg"):
        raise ValueError("figure_format must be png, pdf, or svg")


def compute_rates(
    cfg: SimulationConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return interrogation times and the four plotted F_Q/t curves."""
    validate_config(cfg)
    times = np.linspace(cfg.t_min, cfg.t_max, cfg.n_steps)
    simulation = time_config(cfg)
    qfi, _ = protocol_fisher_information_trajectories(
        simulation, times, driven_protocol(simulation), classical_components=(),
    )
    # The parameter-independent initial state has F_Q(t) = O(t**2), so
    # F_Q(t)/t has the continuous limit zero at t = 0.
    rates = {
        subsystem: np.divide(
            values, times, out=np.zeros_like(values), where=times > 0.0,
        )
        for subsystem, values in qfi.items()
    }
    rates["ramsey"] = 4.0 * cfg.N * times
    return times, rates


def parameter_tags(cfg: SimulationConfig) -> str:
    """Record every setting in compact, Windows-safe filename tags.

    The time tag is min-max-count; each state tag is theta-phi in radians;
    the figure tag contains width, height, DPI, format, and display mode.
    """
    tags = (
        f"N={cfg.N}",
        f"Omega={format_number(cfg.Omega)}",
        f"omega={format_number(cfg.omega)}",
        f"J={format_number(cfg.J_nominal)}",
        f"dJ={format_number(cfg.dJ)}",
        f"gamma={format_number(cfg.gamma)}",
        f"t={format_number(cfg.t_min)}-{format_number(cfg.t_max)}-n{cfg.n_steps}",
        f"central_rad={format_angle(cfg.central_theta_rad)}-{format_angle(cfg.central_phi_rad)}",
        f"bath_rad={format_angle(cfg.bath_theta_rad)}-{format_angle(cfg.bath_phi_rad)}",
        f"tol={format_number(cfg.qfi_tol)}",
        (
            f"fig={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)}-dpi{cfg.figure_dpi}-"
            f"{cfg.figure_format}-show{int(cfg.show_figure)}"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def plot_rates(
    times: np.ndarray,
    rates: dict[str, np.ndarray],
    cfg: SimulationConfig,
) -> Path:
    """Create one axes with three solid QFI curves and a dashed benchmark."""
    validate_config(cfg)
    figure, axis = plt.subplots(figsize=(cfg.figure_width_in, cfg.figure_height_in))
    try:
        for subsystem in ("global", "bath", "central"):
            axis.plot(
                times, rates[subsystem], linewidth=2.0,
                label=rf"$F_Q^{{\mathrm{{{subsystem}}}}}/t$",
            )
        axis.plot(
            times, rates["ramsey"], color="black", linestyle="--",
            linewidth=1.8, label=r"Ramsey: $4Nt$",
        )
        axis.set_xlabel(r"Interrogation time $t$")
        axis.set_ylabel(r"QFI per interrogation time $F_Q(t)/t$")
        axis.set_title(
            rf"$N={cfg.N}$, $\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, "
            rf"$J={cfg.J_nominal:g}$, $\gamma={cfg.gamma:g}$"
        )
        axis.set_xlim(cfg.t_min, cfg.t_max)
        axis.set_ylim(bottom=0.0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        figure.tight_layout()

        # The image API preserves the full parameter-rich filename and
        # supports paper exports as PNG, PDF, or SVG through smart_save.
        path = save_plot(
            figure,
            f"qfi-per-time__{parameter_tags(cfg)}.{cfg.figure_format}",
            metadata={
                "config": asdict(cfg),
                "estimated_parameter": "J",
                "hamiltonian": "Omega*sigma_x + J*sigma_z*S_z + omega*S_x",
                "ramsey_initial_state": "|1>_central |+>_bath**N",
                "ramsey_qfi_per_time": "4*N*t",
            },
            script_path=__file__,
            format=cfg.figure_format,
            dpi=cfg.figure_dpi,
            bbox_inches="tight",
        )
        if cfg.show_figure:
            plt.show()
        return path
    finally:
        plt.close(figure)


def parse_config(argv: list[str] | None = None) -> SimulationConfig:
    """Expose the dataclass settings as command-line overrides."""
    defaults = SimulationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        if field.name == "show_figure":
            parser.add_argument(
                "--show", dest=field.name, action=argparse.BooleanOptionalAction,
                default=default,
            )
        else:
            flag = "--J" if field.name == "J_nominal" else "--" + field.name.replace("_", "-")
            options = {"choices": ("png", "pdf", "svg")} if field.name == "figure_format" else {}
            parser.add_argument(flag, dest=field.name, type=type(default), default=default, **options)
    cfg = SimulationConfig(**vars(parser.parse_args(argv)))
    validate_config(cfg)
    return cfg


def main(argv: list[str] | None = None) -> Path:
    """Compute the QFI rates, save the single graph, and return its path."""
    cfg = parse_config(argv)
    times, rates = compute_rates(cfg)
    path = plot_rates(times, rates, cfg)
    print(f"Saved QFI per time to {path}")
    return path


if __name__ == "__main__":
    main()
