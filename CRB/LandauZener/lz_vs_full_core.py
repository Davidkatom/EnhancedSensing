"""Shared core for comparing the optimal Omega of the LZ toy and the full model.

Both models estimate J and share the Pauli convention (no factors of 1/2):

  * full model (CRB.crb_core):  H = Omega X + J Z S_z + omega S_x, S_a = sum_i sigma_a,
    initial state |+>_c |0>_b**N (bath at the north pole);
  * LZ toy (CRB.LandauZener.driven_qubit):  H(t) = Omega X + N J cos(2 omega t) Z,
    the mean-field limit of the full model.  The bath drive omega S_x rotates
    <S_z> = N cos(2 omega t), so the toy modulation frequency is 2 omega.

The LZ figure of merit is the qubit QFI for J (exact sensitivity ODE).  The full
model figure of merit is selected by ``objective``: the reduced-bath QFI (default,
as in plot_optimized_qfi_vs_N.py), the central-spin QFI, or the global QFI.

The optimal Omega of each model maximizes its figure of merit over
[Omega_min, Omega_max].  A grid scan is refined around every resolved local maximum
and both boundary intervals (as in plot_optimized_qfi_vs_N.optimize_Omega), so
multi-peaked landscapes are handled and the best peak is returned.  The result is a
numerical optimum within the configured range and resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CRB.LandauZener.driven_qubit import SimulationConfig as LZConfig, simulate
from CRB.plot_qfi_and_spin_vs_time import (
    SimulationConfig as FullConfig,
    driven_protocol,
    protocol_fisher_information_trajectories,
)

OBJECTIVES = ("bath", "central", "global")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Physics, optimizer, and numerical settings shared by both models."""

    J_nominal: float = 1.0
    dJ: float = 1e-3  # Finite-difference step of the full model only.
    omega: float = 1.0  # Bath drive; the LZ toy modulates at 2 * omega.
    central_theta_rad: float = np.pi / 2.0
    objective: str = "bath"  # Full-model figure of merit: bath, central, or global.
    Omega_min: float = 0.0  # Omega = 0 (no drive) is a genuine candidate at short t.
    Omega_max: float = 12.0
    Omega_points: int = 241
    Omega_tol: float = 1e-4
    maxiter: int = 100
    qfi_tol: float = 1e-12
    lz_rtol: float = 1e-9
    lz_atol: float = 1e-11


def validate_model_config(cfg: ModelConfig) -> None:
    for name in ("J_nominal", "dJ", "omega", "central_theta_rad", "Omega_min",
                 "Omega_max", "Omega_tol", "qfi_tol", "lz_rtol", "lz_atol"):
        if not np.isfinite(getattr(cfg, name)):
            raise ValueError(f"{name} must be finite")
    if cfg.objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if not 0 <= cfg.Omega_min < cfg.Omega_max:
        raise ValueError("require 0 <= Omega_min < Omega_max")
    if cfg.Omega_points < 3:
        raise ValueError("Omega_points must be at least 3")
    for name in ("dJ", "Omega_tol", "maxiter", "qfi_tol", "lz_rtol", "lz_atol"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.omega < 0:
        raise ValueError("omega must be non-negative")


def omega_grid(cfg: ModelConfig) -> np.ndarray:
    return np.linspace(cfg.Omega_min, cfg.Omega_max, cfg.Omega_points)


def lz_qfi_trajectory(
    cfg: ModelConfig, N: int, Omega: float, t_max: float, points: int,
) -> np.ndarray:
    """LZ-toy QFI(J) on ``linspace(0, t_max, points)`` (``points >= 2``)."""
    lz_cfg = LZConfig(
        N=float(N), Omega=float(Omega), omega=2.0 * cfg.omega, J=cfg.J_nominal,
        qfi="J", t_pi=t_max / np.pi, points=points, rtol=cfg.lz_rtol,
        atol=cfg.lz_atol, show=False,
    )
    return simulate(lz_cfg, cfg.central_theta_rad).qfi


def full_qfi_trajectory(
    cfg: ModelConfig, N: int, Omega: float, times: np.ndarray,
) -> np.ndarray:
    """Full-model QFI(J) of the chosen objective at each of ``times``."""
    full_cfg = FullConfig(
        N=N, Omega=float(Omega), omega=cfg.omega, J_nominal=cfg.J_nominal,
        dJ=cfg.dJ, t_max=float(np.max(times)), central_theta_rad=cfg.central_theta_rad,
        bath_theta_rad=0.0, qfi_tol=cfg.qfi_tol, show_figure=False,
    )
    qfi, _ = protocol_fisher_information_trajectories(
        full_cfg, np.asarray(times, dtype=float), driven_protocol(full_cfg),
        classical_components=(),
    )
    return qfi[cfg.objective]


def lz_qfi_at(cfg: ModelConfig, N: int, Omega: float, t: float) -> float:
    return float(lz_qfi_trajectory(cfg, N, Omega, t, points=2)[-1])


def full_qfi_at(cfg: ModelConfig, N: int, Omega: float, t: float) -> float:
    return float(full_qfi_trajectory(cfg, N, Omega, np.asarray([t]))[0])


def refine_maximum(
    objective: Callable[[float], float],
    grid: np.ndarray,
    scan: np.ndarray,
    cfg: ModelConfig,
    label: str,
) -> tuple[float, float]:
    """Refine every local maximum and boundary interval of a scanned objective.

    Returns ``(argmax, max)`` over the scan values and all refined candidates.
    """
    if not np.all(np.isfinite(scan)):
        raise RuntimeError(f"Non-finite scan values for {label}")
    best_index = int(np.argmax(scan))
    best_Omega, best_value = float(grid[best_index]), float(scan[best_index])

    intervals = {(0, 1), (len(grid) - 2, len(grid) - 1)}
    for i in range(1, len(grid) - 1):
        if (scan[i] >= scan[i - 1] and scan[i] >= scan[i + 1]
                and (scan[i] > scan[i - 1] or scan[i] > scan[i + 1])):
            intervals.add((i - 1, i + 1))
    for left, right in sorted(intervals):
        fit = minimize_scalar(
            lambda drive: -objective(float(drive)),
            bounds=(float(grid[left]), float(grid[right])),
            method="bounded",
            options={"xatol": cfg.Omega_tol, "maxiter": cfg.maxiter},
        )
        if not fit.success:
            raise RuntimeError(
                f"Omega refinement failed for {label}: {fit.message}; "
                "increase --maxiter or relax --Omega-tol"
            )
        if -fit.fun > best_value:
            best_Omega, best_value = float(fit.x), float(-fit.fun)
    return best_Omega, best_value


def at_search_boundary(Omega: float, cfg: ModelConfig) -> bool:
    """True when the optimum is pinned to a bound that more range could move.

    Omega = 0 is a physical bound (the sign of Omega is immaterial), so a lower
    bound of 0 is never reported.
    """
    at_upper = abs(Omega - cfg.Omega_max) <= cfg.Omega_tol
    at_lower = cfg.Omega_min > 0.0 and abs(Omega - cfg.Omega_min) <= cfg.Omega_tol
    return at_upper or at_lower


def add_model_arguments(parser) -> None:
    """Add the ModelConfig command-line flags shared by both scripts."""
    defaults = ModelConfig()
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument("--omega", type=float, default=defaults.omega,
                        help="Bath drive; the LZ toy modulates at 2*omega.")
    parser.add_argument("--central-theta-rad", type=float, default=defaults.central_theta_rad)
    parser.add_argument("--objective", choices=OBJECTIVES, default=defaults.objective,
                        help="Full-model QFI maximized over Omega (default: bath).")
    parser.add_argument("--Omega-min", type=float, default=defaults.Omega_min)
    parser.add_argument("--Omega-max", type=float, default=defaults.Omega_max)
    parser.add_argument("--Omega-points", type=int, default=defaults.Omega_points)
    parser.add_argument("--Omega-tol", type=float, default=defaults.Omega_tol)
    parser.add_argument("--maxiter", type=int, default=defaults.maxiter)
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument("--lz-rtol", type=float, default=defaults.lz_rtol)
    parser.add_argument("--lz-atol", type=float, default=defaults.lz_atol)


MODEL_FIELDS = tuple(ModelConfig.__dataclass_fields__)
