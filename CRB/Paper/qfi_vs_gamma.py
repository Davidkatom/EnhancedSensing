"""Plot bath QFI versus central-spin dephasing at N = 15, t = 2.

The estimated parameter is ``J`` in

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x,

with central-spin dephasing ``L = sqrt(gamma) * sigma_z``.  Each point reuses
``sweep_helpers.bath_and_global_qfi`` at one fixed interrogation time.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.sweeps.sweep_helpers import DriveSweepConfig, bath_and_global_qfi
except ModuleNotFoundError:  # Allow: python CRB/Paper/qfi_vs_gamma.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from CRB.sweeps.sweep_helpers import DriveSweepConfig, bath_and_global_qfi
from CRB.plot_driven_qfi_map import bounded_uniform_grid
from CRB.plot_qfi_and_spin_vs_time import SimulationConfig as TimeConfig, ramsey_protocol
from CRB.smart_save import PlotRecord, save_plot


@dataclass(frozen=True, slots=True)
class GammaSweepConfig:
    """Fixed physics and the dephasing grid."""

    N: int = 15
    interrogation_time: float = 2.0
    Omega: float = 2.2
    omega: float = 1.0

    gamma_min: float = 0.0
    gamma_max: float = 1.0
    gamma_step: float = 0.01

    figure_width_in: float = 8.0
    figure_height_in: float = 5.0
    figure_dpi: int = 200


def run_sweep(cfg: GammaSweepConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the gamma grid with the driven bath QFI along it."""
    gammas = bounded_uniform_grid(cfg.gamma_min, cfg.gamma_max, cfg.gamma_step)
    base = DriveSweepConfig(N=cfg.N, interrogation_time=cfg.interrogation_time)
    qfi_bath = np.empty(len(gammas))
    for i, gamma in enumerate(gammas):
        qfi_bath[i], _ = bath_and_global_qfi(
            cfg.Omega, cfg.omega, replace(base, gamma=float(gamma)),
        )
        print(f"[{i + 1:4d}/{len(gammas)}] gamma={gamma:.6g}: F_Q_bath={qfi_bath[i]:.6e}")
    return gammas, qfi_bath


def ramsey_bath_qfi(cfg: GammaSweepConfig) -> float:
    """Bath QFI of the naive-Ramsey benchmark at the same N and time.

    The Ramsey central spin starts in a ``sigma_z`` eigenstate and has no
    drive, so ``L = sqrt(gamma) * sigma_z`` leaves it untouched: the benchmark
    is independent of gamma and is evaluated once, without dephasing.
    """
    base = DriveSweepConfig(N=cfg.N, interrogation_time=cfg.interrogation_time)
    qfi, _ = bath_and_global_qfi(
        cfg.Omega, cfg.omega, base, protocol=ramsey_protocol(TimeConfig()),
    )
    return qfi


def plot_sweep(
    gammas: np.ndarray,
    qfi_bath: np.ndarray,
    qfi_ramsey: float,
    cfg: GammaSweepConfig,
) -> PlotRecord:
    """Plot bath QFI versus gamma with the Ramsey line and save via smart_save."""
    xlabel = r"Central-spin dephasing $\gamma$"
    ylabel = r"Bath quantum Fisher information $F_Q^{\mathrm{bath}}$"

    figure, axis = plt.subplots(figsize=(cfg.figure_width_in, cfg.figure_height_in))
    axis.plot(gammas, qfi_bath, label="Driven")
    axis.axhline(qfi_ramsey, color="k", linestyle="--", label="Ramsey benchmark")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(
        "Bath QFI versus dephasing\n"
        rf"$\Omega={cfg.Omega:g}$, $\omega={cfg.omega:g}$, "
        rf"$N={cfg.N}$, $t={cfg.interrogation_time:g}$"
    )
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()

    base = DriveSweepConfig(N=cfg.N, interrogation_time=cfg.interrogation_time)
    figure_path = save_plot(
        figure,
        system="central_spin",
        plot_type="qfi_vs_gamma",
        params={
            **asdict(cfg),
            "J_nominal": base.J_nominal,
            "dJ": base.dJ,
            "central_theta_rad": base.central_theta_rad,
            "bath_theta_rad": base.bath_theta_rad,
            "qfi_tol": base.qfi_tol,
        },
        data={
            "F_Q bath": (gammas, qfi_bath),
            "Ramsey F_Q bath": (gammas, np.full_like(gammas, qfi_ramsey)),
        },
        xlabel=xlabel,
        ylabel=ylabel,
        tags=["paper"],
        script_path=__file__,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )
    plt.show()
    return figure_path


def main() -> PlotRecord:
    cfg = GammaSweepConfig()
    gammas, qfi_bath = run_sweep(cfg)
    qfi_ramsey = ramsey_bath_qfi(cfg)
    print(f"Ramsey F_Q_bath={qfi_ramsey:.6e}")
    figure_path = plot_sweep(gammas, qfi_bath, qfi_ramsey, cfg)
    print(f"Saved QFI vs gamma to {figure_path}")
    return figure_path


if __name__ == "__main__":
    main()
