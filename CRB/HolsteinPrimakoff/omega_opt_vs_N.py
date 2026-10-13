"""Optimal central drive versus N at one readout time: exact, linearized HP, and the LZ 50:50 rule, hbar = 1.

The model is H = Omega X + J Z S_z + omega S_x with S_a = sum_i sigma_a^(i), started in |+>|S_z = N>.
For each N the reduced-bath QFI for J at time t is maximized over Omega in
    * the exact model, through plot_driven_qfi_map.qfi_at_drive_pair (the objective of
      plot_optimized_qfi_vs_N), and
    * the linearized Holstein-Primakoff model of hp_dynamics (rotating frame, boson cutoff 4N + 40).
Both scan Omega = c sqrt(J N omega) for c in [c_min, c_max] and refine the best grid point.

The Landau-Zener 50:50 rule sets the crossing's transition probability exp(-pi Omega^2 / 2 J N omega) to 1/2:
    Omega_LZ = sqrt(2 ln2 J N omega / pi) ~ 0.66 sqrt(J N omega).
The exact bath QFI at Omega_LZ is reported as a fraction of the exact maximum.

Panels: Omega_opt versus N; Omega_opt / sqrt(J N omega); and F_B(Omega_LZ) / F_B,max with F_B,max / N^2.
The default t = 1 reads out after the first crossing at t_c = pi / (4 omega), where the linearized
model is accurate.

Run from the repository root:
    python CRB/HolsteinPrimakoff/omega_opt_vs_N.py
    python CRB/HolsteinPrimakoff/omega_opt_vs_N.py --N-values 10 20 40 --no-hp --no-show

The figure and its complete numerical .plot.json record are saved together by
CRB.smart_save under PhD/Graphs/holstein_primakoff/omega_opt_vs_N.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# Support both direct execution and `python -m CRB.HolsteinPrimakoff.omega_opt_vs_N`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CRB.crb_core import coherent_bath_state
from CRB.HolsteinPrimakoff.hp_dynamics import SimulationConfig as HPConfig, simulate as hp_simulate
from CRB.plot_driven_qfi_map import SweepConfig as MapConfig, qfi_at_drive_pair
from CRB.smart_save import PlotRecord, save_plot

LZ_COEFFICIENT = np.sqrt(2.0 * np.log(2.0) / np.pi)  # Omega_LZ / sqrt(J N omega)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """N values, physics, Omega search, and figure settings.

    The Omega grid is c sqrt(J N omega) for c_points values of c in [c_min, c_max].
    """

    N_values: tuple[int, ...] = (5, 8, 10, 15, 20, 25, 30, 40)
    t: float = 1.0
    J: float = 1.0
    omega: float = 1.0
    dJ: float = 1e-4
    c_min: float = 0.3
    c_max: float = 1.5
    c_points: int = 25
    Omega_tol: float = 1e-3
    hp: bool = True
    size: tuple[float, float] = (15.0, 4.4)
    dpi: int = 160
    show: bool = True

    def __post_init__(self) -> None:
        if not self.N_values or not all(isinstance(n, int) and n >= 1 for n in self.N_values):
            raise ValueError("N_values must be positive integers.")
        for name in ("t", "J", "omega", "dJ", "Omega_tol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 < self.c_min < self.c_max:
            raise ValueError("require 0 < c_min < c_max.")
        if not isinstance(self.c_points, int) or self.c_points < 3:
            raise ValueError("c_points must be an integer of at least 3.")
        if len(self.size) != 2 or not all(np.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("size must contain two finite positive dimensions.")
        if not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer.")


@dataclass(frozen=True, slots=True)
class SweepResult:
    N: np.ndarray
    Omega_lz: np.ndarray
    Omega_exact: np.ndarray
    F_exact_max: np.ndarray
    F_exact_at_lz: np.ndarray
    Omega_hp: np.ndarray  # NaN when the HP model is skipped.
    F_hp_max: np.ndarray


def omega_lz(N: float, J: float, omega: float) -> float:
    """Omega at which the Landau-Zener transition probability of the first crossing is 1/2."""
    return LZ_COEFFICIENT * np.sqrt(J * N * omega)


def exact_bath_qfi(cfg: SweepConfig, N: int, Omega: float) -> float:
    """Exact reduced-bath QFI for J at time t, the objective of plot_optimized_qfi_vs_N."""
    map_cfg = MapConfig(N=N, interrogation_time=cfg.t, J_nominal=cfg.J, dJ=cfg.dJ,
                        central_theta_rad=np.pi / 2.0, bath_theta_rad=0.0, qfi_tol=1e-12)
    return qfi_at_drive_pair(Omega, cfg.omega, map_cfg, coherent_bath_state(N, 0.0))


def hp_bath_qfi(cfg: SweepConfig, N: int, Omega: float) -> float:
    """Linearized-HP reduced-bath QFI for J at time t, from the analytic J-tangent of hp_dynamics."""
    hp_cfg = HPConfig(N=N, Omega=float(Omega), omega=cfg.omega, J=cfg.J, t_pi=cfg.t / np.pi, points=2, show=False)
    return float(hp_simulate(hp_cfg, "HP").qfi_bath[-1])


def maximize(objective: Callable[[float], float], grid: np.ndarray, tol: float) -> tuple[float, float]:
    """Scan the grid, then refine the best point between its neighbours."""
    values = np.array([objective(float(Omega)) for Omega in grid])
    best = int(np.argmax(values))
    fit = minimize_scalar(lambda Omega: -objective(float(Omega)),
                          bounds=(grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)]),
                          method="bounded", options={"xatol": tol})
    if -fit.fun >= values[best]:
        return float(fit.x), float(-fit.fun)
    return float(grid[best]), float(values[best])


def run_sweep(cfg: SweepConfig) -> SweepResult:
    rows = []
    for N in cfg.N_values:
        grid = np.linspace(cfg.c_min, cfg.c_max, cfg.c_points) * np.sqrt(cfg.J * N * cfg.omega)
        lz = omega_lz(N, cfg.J, cfg.omega)
        Omega_exact, F_exact = maximize(lambda Omega: exact_bath_qfi(cfg, N, Omega), grid, cfg.Omega_tol)
        Omega_hp, F_hp = (maximize(lambda Omega: hp_bath_qfi(cfg, N, Omega), grid, cfg.Omega_tol)
                          if cfg.hp else (np.nan, np.nan))
        rows.append((N, lz, Omega_exact, F_exact, exact_bath_qfi(cfg, N, lz), Omega_hp, F_hp))
        print(f"N={N:3d}: Omega_LZ={lz:.3f}  exact Omega_opt={Omega_exact:.3f} (F={F_exact:.1f})  "
              f"HP Omega_opt={Omega_hp:.3f}  F(Omega_LZ)/F_max={rows[-1][4] / F_exact:.4f}", flush=True)
    columns = [np.array(column, dtype=float) for column in zip(*rows)]
    return SweepResult(*columns)


def plot_sweep(cfg: SweepConfig, result: SweepResult) -> PlotRecord:
    N, scale = result.N, np.sqrt(cfg.J * result.N * cfg.omega)
    N_dense = np.linspace(N.min(), N.max(), 200)
    fit = float(np.sum(result.Omega_exact * scale) / np.sum(scale**2))  # least squares Omega = c sqrt(J N omega)
    fraction_at_lz = result.F_exact_at_lz / result.F_exact_max
    figure, (omega_axis, ratio_axis, qfi_axis) = plt.subplots(1, 3, figsize=cfg.size, layout="constrained")

    omega_axis.plot(N_dense, omega_lz(N_dense, cfg.J, cfg.omega), "k--", lw=1.4,
                    label=rf"LZ 50:50, ${LZ_COEFFICIENT:.3f}\sqrt{{JN\omega}}$")
    omega_axis.plot(N_dense, fit * np.sqrt(cfg.J * N_dense * cfg.omega), color="C0", ls=":", lw=1.2,
                    label=rf"fit to exact, ${fit:.3f}\sqrt{{JN\omega}}$")
    omega_axis.plot(N, result.Omega_exact, "o", color="C0", label="exact")
    if cfg.hp:
        omega_axis.plot(N, result.Omega_hp, "x", color="C1", ms=8, mew=1.8, label="linearized HP")
    omega_axis.set(title=r"Bath-optimal central drive", xlabel=r"$N$", ylabel=r"$\Omega_{\rm opt}$")
    omega_axis.legend(fontsize=9)

    ratio_axis.axhline(LZ_COEFFICIENT, color="k", ls="--", lw=1.4, label="LZ 50:50")
    ratio_axis.plot(N, result.Omega_exact / scale, "o-", color="C0", label="exact")
    if cfg.hp:
        ratio_axis.plot(N, result.Omega_hp / scale, "x-", color="C1", ms=8, mew=1.8, label="linearized HP")
    ratio_axis.set(title=r"Prefactor $\Omega_{\rm opt}/\sqrt{JN\omega}$", xlabel=r"$N$",
                   ylabel=r"$\Omega_{\rm opt}/\sqrt{JN\omega}$")
    ratio_axis.legend(fontsize=9)

    qfi_axis.plot(N, fraction_at_lz, "s-", color="C2", label=r"$F_B(\Omega_{\rm LZ})/F_{B,\max}$")
    qfi_axis.plot(N, result.F_exact_max / N**2, "o-", color="C3", label=r"$F_{B,\max}/N^2$")
    qfi_axis.set(title="Exact bath QFI", xlabel=r"$N$", ylabel="Ratio")
    qfi_axis.legend(fontsize=9)
    for axis in (omega_axis, ratio_axis, qfi_axis):
        axis.grid(True, alpha=0.25)
    figure.suptitle(rf"$H=\Omega X+JZS_z+\omega S_x$, $|+\rangle|S_z=N\rangle$, readout $t={cfg.t:g}$, "
                    rf"$J={cfg.J:g}$, $\omega={cfg.omega:g}$ (first crossing at $t_c=\pi/4\omega$)")

    data = {
        "LZ 50:50 Omega": (N, result.Omega_lz),
        "exact: Omega_opt": (N, result.Omega_exact),
        "exact: Omega_opt/sqrt(JN omega)": (N, result.Omega_exact / scale),
        "exact: F_B max": (N, result.F_exact_max),
        "exact: F_B max/N^2": (N, result.F_exact_max / N**2),
        "exact: F_B(Omega_LZ)/F_B max": (N, fraction_at_lz),
    }
    if cfg.hp:
        data["HP: Omega_opt"] = (N, result.Omega_hp)
        data["HP: Omega_opt/sqrt(JN omega)"] = (N, result.Omega_hp / scale)
        data["HP: F_B max"] = (N, result.F_hp_max)
    try:
        record = save_plot(
            figure,
            system="holstein_primakoff",
            plot_type="omega_opt_vs_N",
            params=asdict(cfg),
            data=data,
            name=f"omega-opt-vs-N__t={cfg.t:g}__J={cfg.J:g}__omega={cfg.omega:g}__hp={int(cfg.hp)}",
            xlabel=r"$N$",
            ylabel="Optimal drive / prefactor / QFI ratio",
            notes=(
                "Bath-optimal Omega at one readout time t for H = Omega X + J Z S_z + omega S_x (Pauli sums), "
                "initial |+>|S_z = N>. exact: qfi_at_drive_pair objective; HP: linearized Holstein-Primakoff "
                "(hp_dynamics, cutoff 4N + 40). LZ 50:50: Omega = sqrt(2 ln2 J N omega / pi). "
                "Grid c sqrt(J N omega), c in [c_min, c_max], refined by bounded minimization."
            ),
            metadata={"exact_fit_coefficient": fit, "lz_coefficient": float(LZ_COEFFICIENT)},
            script_path=__file__,
            format="png",
            dpi=cfg.dpi,
            bbox_inches="tight",
        )
        if cfg.show:
            plt.show()
        return record
    finally:
        plt.close(figure)


def main() -> None:
    defaults = SweepConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N-values", type=int, nargs="+", default=list(defaults.N_values))
    parser.add_argument("--t", type=float, default=defaults.t, help="Readout time (default: 1).")
    parser.add_argument("--J", type=float, default=defaults.J)
    parser.add_argument("--omega", "--w", dest="omega", type=float, default=defaults.omega)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument("--c-min", type=float, default=defaults.c_min)
    parser.add_argument("--c-max", type=float, default=defaults.c_max)
    parser.add_argument("--c-points", type=int, default=defaults.c_points)
    parser.add_argument("--Omega-tol", type=float, default=defaults.Omega_tol)
    parser.add_argument("--no-hp", action="store_false", dest="hp", default=defaults.hp,
                        help="Skip the linearized-HP optimization (the slow part at large N).")
    parser.add_argument("--size", nargs=2, type=float, default=defaults.size, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=defaults.dpi)
    parser.add_argument("--no-show", action="store_false", dest="show", default=defaults.show)
    arguments = vars(parser.parse_args())
    arguments["N_values"] = tuple(arguments["N_values"])
    arguments["size"] = tuple(arguments["size"])
    cfg = SweepConfig(**arguments)
    record = plot_sweep(cfg, run_sweep(cfg))
    print(f"Saved numerical data: {record.json_path}")
    print(f"Saved figure: {record.image_path}")


if __name__ == "__main__":
    main()
