"""Shared machinery for one-dimensional drive sweeps of the bath and global QFI.

The estimated parameter is ``J`` in

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x,

with initial state ``|theta_c>_central |theta_b>_bath**N``.  One drive is swept
while the other is held fixed; at each point the reduced-bath and global
(central + bath) QFI are evaluated at one fixed interrogation time, both from
the same evolved joint state.

Entry scripts only choose which drive is swept (``CENTRAL_DRIVE`` or
``BATH_DRIVE``) and call :func:`main`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.crb_core import qfi_from_rho_and_drho
except ModuleNotFoundError:  # Allow: python CRB/sweeps/<script>.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from CRB.crb_core import qfi_from_rho_and_drho
from CRB.plot_driven_qfi_map import bounded_uniform_grid
from CRB.plot_qfi_and_spin_vs_time import (
    SimulationConfig as TimeConfig,
    density_matrices,
    evolve_states,
)
from CRB.smart_save import PlotRecord, save_plot


@dataclass(frozen=True, slots=True)
class SweptDrive:
    """Which drive is swept, and how it and the fixed drive are labelled."""

    name: str  # CLI/series name of the swept drive, e.g. "Omega".
    symbol: str  # LaTeX symbol of the swept drive.
    description: str  # Axis label prefix, e.g. "Central-spin drive".
    fixed_name: str
    fixed_symbol: str
    plot_type: str  # Case-distinct folder name for smart_save.

    def drives(self, swept: float, fixed: float) -> tuple[float, float]:
        """Return ``(Omega, omega)`` for one sweep point."""
        return (swept, fixed) if self.name == "Omega" else (fixed, swept)

    @property
    def axis_label(self) -> str:
        return rf"{self.description} ${self.symbol}$"


CENTRAL_DRIVE = SweptDrive(
    name="Omega", symbol=r"\Omega", description="Central-spin drive",
    fixed_name="omega", fixed_symbol=r"\omega",
    plot_type="qfi_vs_central_drive",
)
BATH_DRIVE = SweptDrive(
    name="omega", symbol=r"\omega", description="Bath drive",
    fixed_name="Omega", fixed_symbol=r"\Omega",
    plot_type="qfi_vs_bath_drive",
)


@dataclass(frozen=True, slots=True)
class DriveSweepConfig:
    """Physics, sweep, and figure parameters."""

    N: int = 15
    interrogation_time: float = np.pi/2
    J_nominal: float = 1.0
    dJ: float = 1e-3
    gamma: float = 0.0  # Central-spin dephasing: L = sqrt(gamma) * sigma_z.
    fixed_drive: float = 1.0  # Value of the drive that is not swept.

    sweep_min: float = 0.0
    sweep_max: float = 5.0
    sweep_step: float = 0.02

    central_theta_rad: float = np.pi / 2.0
    bath_theta_rad: float = 0.0
    qfi_tol: float = 1e-12

    figure_width_in: float = 8.0
    figure_height_in: float = 5.0
    figure_dpi: int = 200


def bath_and_global_qfi(
    Omega: float,
    omega: float,
    cfg: DriveSweepConfig,
) -> tuple[float, float]:
    """Return ``(F_Q_bath, F_Q_global)`` at one drive pair."""
    time_cfg = TimeConfig(
        N=cfg.N, Omega=Omega, omega=omega, J_nominal=cfg.J_nominal,
        dJ=cfg.dJ, gamma=cfg.gamma, central_theta_rad=cfg.central_theta_rad,
        bath_theta_rad=cfg.bath_theta_rad, qfi_tol=cfg.qfi_tol,
    )
    times = np.asarray([cfg.interrogation_time])
    plus = density_matrices(
        evolve_states(time_cfg, cfg.J_nominal + cfg.dJ, times)[0], cfg.N,
    )
    minus = density_matrices(
        evolve_states(time_cfg, cfg.J_nominal - cfg.dJ, times)[0], cfg.N,
    )

    def qfi(rho_plus: np.ndarray, rho_minus: np.ndarray) -> float:
        rho = 0.5 * (rho_plus + rho_minus)
        drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
        value, _ = qfi_from_rho_and_drho(rho, drho, tol=cfg.qfi_tol)
        return float(value)

    return (
        qfi(plus.bath, minus.bath),
        qfi(plus.global_state, minus.global_state),
    )


def run_sweep(
    cfg: DriveSweepConfig,
    swept: SweptDrive,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the swept-drive grid with bath and global QFI along it."""
    values = bounded_uniform_grid(cfg.sweep_min, cfg.sweep_max, cfg.sweep_step)
    qfi_bath = np.empty(len(values))
    qfi_global = np.empty(len(values))
    for i, value in enumerate(values):
        Omega, omega = swept.drives(float(value), cfg.fixed_drive)
        qfi_bath[i], qfi_global[i] = bath_and_global_qfi(Omega, omega, cfg)
        print(
            f"[{i + 1:4d}/{len(values)}] {swept.name}={value:.6g}: "
            f"F_Q_bath={qfi_bath[i]:.6e}, F_Q_global={qfi_global[i]:.6e}"
        )
    return values, qfi_bath, qfi_global


def plot_sweep(
    values: np.ndarray,
    qfi_bath: np.ndarray,
    qfi_global: np.ndarray,
    cfg: DriveSweepConfig,
    swept: SweptDrive,
    script_path: str | Path,
) -> PlotRecord:
    """Plot bath and global QFI per unit time versus the swept drive."""
    if cfg.interrogation_time <= 0.0:
        raise ValueError("interrogation_time must be positive to plot F_Q / t")
    rate_global = qfi_global / cfg.interrogation_time
    rate_bath = qfi_bath / cfg.interrogation_time
    ylabel = r"QFI per interrogation time $F_Q / t$"

    figure, axis = plt.subplots(figsize=(cfg.figure_width_in, cfg.figure_height_in))
    axis.plot(values, rate_global, label=r"$F_Q^{\mathrm{global}} / t$")
    axis.plot(values, rate_bath, label=r"$F_Q^{\mathrm{bath}} / t$")
    axis.set_xlabel(swept.axis_label)
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"Bath and global QFI per time versus {swept.description.lower()}\n"
        rf"${swept.fixed_symbol}={cfg.fixed_drive:g}$, $J={cfg.J_nominal:g}$, "
        rf"$N={cfg.N}$, $t={cfg.interrogation_time:g}$, $\gamma={cfg.gamma:g}$"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    figure_path = save_plot(
        figure,
        system="central_spin",
        plot_type=swept.plot_type,
        params={
            **asdict(cfg),
            "swept_drive": swept.name,
            swept.fixed_name: cfg.fixed_drive,
        },
        data={
            "F_Q global / t": (values, rate_global),
            "F_Q bath / t": (values, rate_bath),
        },
        xlabel=swept.axis_label,
        ylabel=ylabel,
        script_path=script_path,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )
    plt.show()
    return figure_path


def parse_config(
    swept: SweptDrive,
    defaults: DriveSweepConfig,
    argv: list[str] | None = None,
    description: str | None = None,
) -> DriveSweepConfig:
    """Parse CLI overrides, naming the sweep flags after the swept drive."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument(
        "--time", dest="interrogation_time", type=float,
        default=defaults.interrogation_time,
    )
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument(
        f"--{swept.fixed_name}", dest="fixed_drive", type=float,
        default=defaults.fixed_drive,
    )
    parser.add_argument(
        f"--{swept.name}-min", dest="sweep_min", type=float,
        default=defaults.sweep_min,
    )
    parser.add_argument(
        f"--{swept.name}-max", dest="sweep_max", type=float,
        default=defaults.sweep_max,
    )
    parser.add_argument(
        f"--{swept.name}-step", dest="sweep_step", type=float,
        default=defaults.sweep_step,
    )
    parser.add_argument(
        "--central-theta-rad", type=float, default=defaults.central_theta_rad,
    )
    parser.add_argument("--bath-theta-rad", type=float, default=defaults.bath_theta_rad)
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    return replace(defaults, **vars(parser.parse_args(argv)))


def main(
    swept: SweptDrive,
    script_path: str | Path,
    defaults: DriveSweepConfig = DriveSweepConfig(),
    argv: list[str] | None = None,
    description: str | None = None,
) -> PlotRecord:
    """Parse, sweep, plot, and save one drive sweep."""
    cfg = parse_config(swept, defaults, argv, description)
    values, qfi_bath, qfi_global = run_sweep(cfg, swept)
    figure_path = plot_sweep(values, qfi_bath, qfi_global, cfg, swept, script_path)
    print(f"Saved QFI sweep to {figure_path}")
    return figure_path
