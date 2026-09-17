"""Optimize Omega for bath QFI, then compare bath and global QFI versus N.

Combines the fixed-time bath-QFI objective from plot_driven_qfi_map with the
subsystem QFI calculation from plot_qfi_and_spin_vs_time. The estimated
parameter is J in H = Omega*sigma_x + J*sigma_z*S_z + omega*S_x, with the
same collective-spin normalization as crb_core. Frequencies use the existing
model units and time uses their inverse (hbar = 1).

Defaults: N = 1,...,50, t_max = 2, omega = 1, and |+>_c |0>_b**N.
For EACH N, maximize bath QFI over Omega in [0.1, 5]. Scan a 1D grid, refine
every resolved peak and both boundary intervals, and retain the best value
including the endpoints. This is a numerical optimum within the configured
range and resolution; increase --Omega-points to resolve narrower peaks.
Global QFI is evaluated at that SAME bath-optimal Omega. The drives remain
fixed when differentiating with respect to J. No omega or time scan is run.

Run from the repository root:
    python CRB/plot_optimized_qfi_vs_N.py
    python CRB/plot_optimized_qfi_vs_N.py --Omega-max 10 --show

One figure contains three vertically stacked panels: bath/global QFI, their
ratio, and optimal Omega versus N. The figure and its Graph Viewer record are
saved with smart_save under its Google Drive central_spin folders.
A results CSV and scan NPZ (including the full
configuration) are saved under graphs/plot_optimized_qfi_vs_N/. Initial-state
azimuths are zero, as in the drive map. All configurable values and optimal
drives are included in the plot records.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from functools import cache
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

try:
    from CRB.crb_core import coherent_bath_state
    from CRB.smart_save import PlotRecord, save_plot
    from CRB.plot_driven_qfi_map import (
        SweepConfig as MapConfig,
        format_angle,
        format_number,
        qfi_at_drive_pair,
        sanitize_tag,
    )
    from CRB.plot_qfi_and_spin_vs_time import (
        SimulationConfig as TimeConfig,
        driven_protocol,
        protocol_fisher_information_trajectories,
    )
except ModuleNotFoundError:  # Allow: python CRB/plot_optimized_qfi_vs_N.py
    from crb_core import coherent_bath_state
    from smart_save import PlotRecord, save_plot
    from plot_driven_qfi_map import (
        SweepConfig as MapConfig,
        format_angle,
        format_number,
        qfi_at_drive_pair,
        sanitize_tag,
    )
    from plot_qfi_and_spin_vs_time import (
        SimulationConfig as TimeConfig,
        driven_protocol,
        protocol_fisher_information_trajectories,
    )


@dataclass(frozen=True, slots=True)
class OptimizedQFIConfig:
    """Physics, inclusive sweep bounds, numerics, and figure settings."""

    N_min: int = 1
    N_max: int = 50
    t_max: float = 2.0  # Single evaluation time, in inverse-frequency units.
    J_nominal: float = 1.0
    dJ: float = 1e-3
    omega: float = 1.0  # Fixed bath drive, independent of J +/- dJ.
    Omega_min: float = 0.1
    Omega_max: float = 5.0
    Omega_points: int = 101
    Omega_tol: float = 1e-4  # Absolute refinement tolerance in drive units.
    maxiter: int = 100
    central_theta_rad: float = np.pi / 2.0
    bath_theta_rad: float = 0.0
    qfi_tol: float = 1e-12
    figure_width_in: float = 7.0
    figure_height_in: float = 9.0
    figure_dpi: int = 160
    figure_format: str = "png"
    show_figure: bool = False


@dataclass(frozen=True, slots=True)
class SweepResult:
    N_values: np.ndarray
    Omega_grid: np.ndarray
    bath_qfi_scan: np.ndarray
    Omega_opt: np.ndarray
    qfi_bath: np.ndarray
    qfi_global: np.ndarray
    ratio: np.ndarray
    at_boundary: np.ndarray


def validate_config(cfg: OptimizedQFIConfig) -> None:
    for field in fields(cfg):
        value = getattr(cfg, field.name)
        if isinstance(value, (float, int)) and not np.isfinite(value):
            raise ValueError(f"{field.name} must be finite")
    for name in ("N_min", "N_max", "Omega_points", "maxiter", "figure_dpi"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 1 <= cfg.N_min <= cfg.N_max:
        raise ValueError("require 1 <= N_min <= N_max")
    if cfg.t_max < 0:
        raise ValueError("t_max must be non-negative")
    if not 0 <= cfg.Omega_min < cfg.Omega_max:
        raise ValueError("require 0 <= Omega_min < Omega_max")
    if cfg.Omega_points < 3:
        raise ValueError("Omega_points must be at least 3")
    for name in ("dJ", "Omega_tol", "maxiter", "qfi_tol", "figure_width_in",
                 "figure_height_in", "figure_dpi"):
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.figure_format not in ("png", "pdf", "svg"):
        raise ValueError("figure_format must be png, pdf, or svg")


def optimize_Omega(
    N: int, cfg: OptimizedQFIConfig, Omega_grid: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Maximize the map's bath objective along one fixed-omega slice."""
    map_cfg = MapConfig(
        N=N, interrogation_time=cfg.t_max, J_nominal=cfg.J_nominal,
        dJ=cfg.dJ, central_theta_rad=cfg.central_theta_rad,
        bath_theta_rad=cfg.bath_theta_rad, qfi_tol=cfg.qfi_tol,
    )
    bath_state = coherent_bath_state(N, theta=cfg.bath_theta_rad)

    @cache
    def objective(Omega: float) -> float:
        value = qfi_at_drive_pair(Omega, cfg.omega, map_cfg, bath_state)
        if not np.isfinite(value) or value < 0:
            raise RuntimeError(f"Invalid bath QFI at N={N}, Omega={Omega}: {value}")
        return value

    scan = np.asarray([objective(float(drive)) for drive in Omega_grid])
    best_index = int(np.argmax(scan))
    best_Omega = float(Omega_grid[best_index])
    best_qfi = float(scan[best_index])

    # Resolve multiple peaks, rather than assuming the full slice is unimodal.
    intervals = {(0, 1), (len(Omega_grid) - 2, len(Omega_grid) - 1)}
    for i in range(1, len(Omega_grid) - 1):
        if (scan[i] >= scan[i - 1] and scan[i] >= scan[i + 1]
                and (scan[i] > scan[i - 1] or scan[i] > scan[i + 1])):
            intervals.add((i - 1, i + 1))
    for left, right in sorted(intervals):
        fit = minimize_scalar(
            lambda drive: -objective(float(drive)),
            bounds=(float(Omega_grid[left]), float(Omega_grid[right])),
            method="bounded",
            options={"xatol": cfg.Omega_tol, "maxiter": cfg.maxiter},
        )
        if not fit.success:
            raise RuntimeError(
                f"Omega refinement failed for N={N}: {fit.message}; "
                "increase --maxiter or relax --Omega-tol"
            )
        if -fit.fun > best_qfi:
            best_Omega, best_qfi = float(fit.x), float(-fit.fun)
    return best_Omega, scan


def qfi_at_optimum(
    N: int, Omega: float, cfg: OptimizedQFIConfig,
) -> tuple[float, float]:
    """Return bath and global QFI at the same drives and the single time."""
    time_cfg = TimeConfig(
        N=N, Omega=Omega, omega=cfg.omega, J_nominal=cfg.J_nominal,
        dJ=cfg.dJ, t_max=cfg.t_max, central_theta_rad=cfg.central_theta_rad,
        bath_theta_rad=cfg.bath_theta_rad, qfi_tol=cfg.qfi_tol,
    )
    qfi, _ = protocol_fisher_information_trajectories(
        time_cfg, np.asarray([cfg.t_max]), driven_protocol(time_cfg),
        classical_components=(),
    )
    bath, global_qfi = float(qfi["bath"][0]), float(qfi["global"][0])
    if not np.all(np.isfinite([bath, global_qfi])) or min(bath, global_qfi) < 0:
        raise RuntimeError(f"Invalid final QFI for N={N}, Omega={Omega}")
    return bath, global_qfi


def run_sweep(cfg: OptimizedQFIConfig) -> SweepResult:
    validate_config(cfg)
    N_values = np.arange(cfg.N_min, cfg.N_max + 1)
    Omega_grid = np.linspace(cfg.Omega_min, cfg.Omega_max, cfg.Omega_points)
    scans = np.empty((len(N_values), len(Omega_grid)))
    optima, bath, global_qfi = np.empty((3, len(N_values)))
    at_boundary = np.zeros(len(N_values), dtype=bool)
    for i, N in enumerate(N_values):
        optima[i], scans[i] = optimize_Omega(int(N), cfg, Omega_grid)
        bath[i], global_qfi[i] = qfi_at_optimum(int(N), optima[i], cfg)
        at_boundary[i] = min(
            abs(optima[i] - cfg.Omega_min), abs(optima[i] - cfg.Omega_max),
        ) <= cfg.Omega_tol
        suffix = " (at search boundary; consider widening Omega bounds)" if at_boundary[i] else ""
        print(
            f"N={N:2d}: Omega_opt={optima[i]:.8g}, "
            f"FQ_bath={bath[i]:.8g}, FQ_global={global_qfi[i]:.8g}{suffix}",
            flush=True,
        )
    # A zero-information experiment has an undefined fraction, not a zero one.
    ratio = np.divide(bath, global_qfi, out=np.full_like(bath, np.nan),
                      where=global_qfi > cfg.qfi_tol)
    return SweepResult(N_values, Omega_grid, scans, optima, bath, global_qfi,
                       ratio, at_boundary)


def parameter_tags(cfg: OptimizedQFIConfig) -> str:
    """Compact, stable tags covering every configuration field."""
    # Group related fields to keep the default absolute paths below 260 chars.
    groups = (
        (("N_min", "N_max"), f"N={cfg.N_min}-{cfg.N_max}"),
        (("t_max",), f"t={format_number(cfg.t_max)}"),
        (("J_nominal",), f"J={format_number(cfg.J_nominal)}"),
        (("dJ",), f"dJ={format_number(cfg.dJ)}"),
        (("omega",), f"om={format_number(cfg.omega)}"),
        (("Omega_min", "Omega_max", "Omega_points"),
         f"Om={format_number(cfg.Omega_min)}-{format_number(cfg.Omega_max)}-n{cfg.Omega_points}"),
        (("Omega_tol", "maxiter"), f"opt={format_number(cfg.Omega_tol)}-n{cfg.maxiter}"),
        (("central_theta_rad",), f"tc={format_angle(cfg.central_theta_rad)}"),
        (("bath_theta_rad",), f"tb={format_angle(cfg.bath_theta_rad)}"),
        (("qfi_tol",), f"qt={format_number(cfg.qfi_tol)}"),
        (("figure_width_in", "figure_height_in", "figure_dpi", "figure_format", "show_figure"),
         f"fig={format_number(cfg.figure_width_in)}x{format_number(cfg.figure_height_in)}"
         f"-{cfg.figure_dpi}dpi-{cfg.figure_format}-s{int(cfg.show_figure)}"),
    )
    if {name for names, _ in groups for name in names} != {f.name for f in fields(cfg)}:
        raise RuntimeError("Update parameter_tags to encode all configuration fields")
    return "__".join(sanitize_tag(tag) for _, tag in groups)


def output_directory() -> Path:
    return Path(__file__).resolve().parent.parent / "graphs" / Path(__file__).stem


def save_results(result: SweepResult, cfg: OptimizedQFIConfig) -> tuple[Path, Path]:
    directory = output_directory()
    directory.mkdir(parents=True, exist_ok=True)
    tags = parameter_tags(cfg)
    csv_path = directory / f"values__{tags}.csv"
    scan_path = directory / f"scan__{tags}.npz"
    np.savetxt(
        csv_path,
        np.column_stack((result.N_values, result.Omega_opt, result.qfi_bath,
                         result.qfi_global, result.ratio, result.at_boundary)),
        delimiter=",", fmt=("%d", "%.16g", "%.16g", "%.16g", "%.16g", "%d"),
        header="N,Omega_opt,FQ_bath,FQ_global,FQ_bath_over_global,at_boundary",
        comments="",
    )
    np.savez_compressed(
        scan_path, **asdict(result),
        config_json=np.asarray(json.dumps(asdict(cfg), sort_keys=True)),
    )
    return csv_path, scan_path


def plot_results(result: SweepResult, cfg: OptimizedQFIConfig) -> PlotRecord:
    """Save one three-panel figure and all its series through smart_save."""
    params = {
        **asdict(cfg),
        "N_values": result.N_values,
        "Omega_opt": result.Omega_opt,
        "at_boundary": result.at_boundary,
    }
    title = (
        rf"$t={cfg.t_max:g}$, $J={cfg.J_nominal:g}$, $\omega={cfg.omega:g}$"
        "\n" + rf"$\Omega$ optimized for bath QFI in [{cfg.Omega_min:g}, {cfg.Omega_max:g}]"
    )
    figure, axes = plt.subplots(
        3, 1, sharex=True,
        figsize=(cfg.figure_width_in, cfg.figure_height_in),
        layout="constrained",
    )
    qfi_axis, ratio_axis, drive_axis = axes
    qfi_axis.plot(result.N_values, result.qfi_global, "s-", label=r"$F_Q^{\mathrm{global}}$")
    qfi_axis.plot(result.N_values, result.qfi_bath, "o-", label=r"$F_Q^{\mathrm{bath}}$")
    qfi_axis.set_ylabel(r"Quantum Fisher information $F_Q$")
    qfi_axis.set_ylim(bottom=0)
    qfi_axis.legend()
    qfi_axis.set_title("(a) Bath and global QFI", loc="left")

    ratio_axis.plot(result.N_values, result.ratio, "o-", color="tab:green")
    ratio_axis.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ratio_axis.set_ylabel(r"$F_Q^{\mathrm{bath}} / F_Q^{\mathrm{global}}$")
    finite = result.ratio[np.isfinite(result.ratio)]
    ratio_axis.set_ylim(0, max(1.05, float(finite.max()) * 1.05 if finite.size else 1.05))
    ratio_axis.set_title("(b) Fraction accessible from the bath", loc="left")

    drive_axis.plot(result.N_values, result.Omega_opt, "o-", color="tab:purple")
    drive_axis.set_ylabel(r"Optimal central drive $\Omega_{\mathrm{opt}}$")
    drive_axis.set_ylim(bottom=0)
    drive_axis.set_title(r"(c) Bath-optimal $\Omega$", loc="left")
    drive_axis.set_xlabel(r"Number of bath spins $N$")
    tick_indices = np.linspace(
        0, len(result.N_values) - 1, min(len(result.N_values), 20), dtype=int,
    )
    drive_axis.set_xticks(result.N_values[tick_indices])
    for axis in axes:
        axis.grid(True, linestyle=":", alpha=0.6)
    figure.suptitle(title)

    data = {
        "F_Q global": (result.N_values, result.qfi_global),
        "F_Q bath": (result.N_values, result.qfi_bath),
        "F_Q bath / F_Q global": (result.N_values, result.ratio),
        "Unity": (result.N_values, np.ones_like(result.ratio)),
        "Omega_opt": (result.N_values, result.Omega_opt),
    }
    # smart_save's record format has one set of axis labels; retain the panel
    # grouping explicitly alongside the full combined preview and curve data.
    panels = [
        {"ylabel": axis.get_ylabel(), "series": names}
        for axis, names in zip(axes, (
            ["F_Q global", "F_Q bath"],
            ["F_Q bath / F_Q global", "Unity"],
            ["Omega_opt"],
        ))
    ]
    name = f"combined__{parameter_tags(cfg)}"
    try:
        record = save_plot(
            figure,
            system="central_spin",
            plot_type="optimized_qfi_summary_vs_N",
            params=params,
            data=data,
            name=name,
            xlabel=drive_axis.get_xlabel(),
            ylabel="QFI, bath/global QFI, and optimal Omega (separate panels)",
            metadata={"panels": panels},
            notes="Omega maximizes bath QFI for each N; global QFI uses the same drives.",
            script_path=__file__,
            dpi=cfg.figure_dpi,
            bbox_inches="tight",
        )
        # Explorer records always have PNG previews. Preserve optional vector
        # exports through smart_save's legacy API, which embeds their metadata.
        if cfg.figure_format != "png":
            extra_path = save_plot(
                figure,
                f"{name}.{cfg.figure_format}",
                metadata={"parameters": params, "data": data, "panels": panels,
                          "record": str(record)},
                script_path=__file__,
                format=cfg.figure_format,
                dpi=cfg.figure_dpi,
                bbox_inches="tight",
            )
            print(f"Saved {extra_path}")
        if cfg.show_figure:
            plt.show()
    finally:
        plt.close(figure)
    return record


def parse_config(argv: list[str] | None = None) -> OptimizedQFIConfig:
    defaults = OptimizedQFIConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in fields(defaults):
        if field.name == "show_figure":
            parser.add_argument("--show", dest=field.name, action=argparse.BooleanOptionalAction,
                                default=defaults.show_figure)
            continue
        value = getattr(defaults, field.name)
        flag = "--J" if field.name == "J_nominal" else "--" + field.name.replace("_", "-")
        parser.add_argument(flag, dest=field.name, type=type(value), default=value,
                            help=f"{field.name} (default: {value})")
    return OptimizedQFIConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> PlotRecord:
    cfg = parse_config(argv)
    result = run_sweep(cfg)
    data_paths = save_results(result, cfg)
    record = plot_results(result, cfg)
    print(f"Saved {record.json_path}")
    if record.image_path is not None:
        print(f"Saved {record.image_path}")
    for path in data_paths:
        print(f"Saved {path}")
    return record


if __name__ == "__main__":
    main()
