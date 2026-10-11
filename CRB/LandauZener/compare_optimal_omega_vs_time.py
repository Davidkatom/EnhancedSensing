"""Optimal Omega of the Landau-Zener toy versus the full model, as a function of t.

For every interrogation time t in (0, t_max] the central drive Omega is chosen to
maximize the QFI for J in

  * the LZ toy   H(t) = Omega X + N J cos(2 omega t) Z      (single qubit), and
  * the full model H = Omega X + J Z S_z + omega S_x         (reduced-bath QFI),

with the same N, J, omega and initial state |+>_c |0>_b**N.  See lz_vs_full_core
for the conventions, the refinement strategy, and the ``--objective`` option.
t = 0 is excluded: every QFI vanishes there, so the optimal Omega is undefined.

Three panels are drawn: (a) the two optimal Omega curves with the sqrt(2JN/t)
reference, (b) the maximal QFI of each model, and (c) the transfer efficiency
F_full(Omega_LZ) / F_full(Omega_full), i.e. how much of the full-model optimum is
kept by using the cheap LZ-optimal drive.

Run from the repository root:
    python CRB/LandauZener/compare_optimal_omega_vs_time.py
    python CRB/LandauZener/compare_optimal_omega_vs_time.py --N 20 --t-max 3 --show

The figure and its numerical record are saved with smart_save; a values CSV and a
scan NPZ (including the full configuration) go under graphs/<script name>/.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.LandauZener.lz_vs_full_core import (
        MODEL_FIELDS, ModelConfig, add_model_arguments, at_search_boundary,
        full_qfi_at, full_qfi_trajectory, lz_qfi_at, lz_qfi_trajectory, omega_grid,
        refine_maximum, validate_model_config,
    )
except ModuleNotFoundError:  # Allow: python CRB/LandauZener/compare_optimal_omega_vs_time.py
    from lz_vs_full_core import (  # type: ignore[no-redef]
        MODEL_FIELDS, ModelConfig, add_model_arguments, at_search_boundary,
        full_qfi_at, full_qfi_trajectory, lz_qfi_at, lz_qfi_trajectory, omega_grid,
        refine_maximum, validate_model_config,
    )
from CRB.smart_save import PlotRecord, save_plot


@dataclass(frozen=True, slots=True)
class TimeSweepConfig:
    N: int = 15
    t_max: float = 3.0
    n_steps: int = 60  # Times are t_max * k / n_steps, k = 1, ..., n_steps.
    figure_width_in: float = 7.0
    figure_height_in: float = 9.0
    figure_dpi: int = 160
    show_figure: bool = False


@dataclass(frozen=True, slots=True)
class SweepResult:
    times: np.ndarray
    Omega_grid: np.ndarray
    lz_scan: np.ndarray  # (n_times, n_Omega)
    full_scan: np.ndarray
    Omega_lz: np.ndarray
    Omega_full: np.ndarray
    qfi_lz: np.ndarray  # LZ QFI at Omega_lz.
    qfi_full: np.ndarray  # Full QFI at Omega_full.
    qfi_full_at_lz: np.ndarray  # Full QFI at Omega_lz.
    efficiency: np.ndarray
    lz_at_boundary: np.ndarray
    full_at_boundary: np.ndarray


def validate_sweep_config(sweep: TimeSweepConfig) -> None:
    if not isinstance(sweep.N, int) or sweep.N < 1:
        raise ValueError("N must be a positive integer")
    if not np.isfinite(sweep.t_max) or sweep.t_max <= 0:
        raise ValueError("t_max must be positive")
    if sweep.n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if sweep.figure_width_in <= 0 or sweep.figure_height_in <= 0 or sweep.figure_dpi <= 0:
        raise ValueError("figure dimensions and dpi must be positive")


def run_sweep(cfg: ModelConfig, sweep: TimeSweepConfig) -> SweepResult:
    validate_model_config(cfg)
    validate_sweep_config(sweep)
    N = sweep.N
    all_times = np.linspace(0.0, sweep.t_max, sweep.n_steps + 1)
    times = all_times[1:]
    grid = omega_grid(cfg)

    # One trajectory per Omega serves every time of the coarse scan.
    lz_scan = np.empty((len(times), len(grid)))
    full_scan = np.empty_like(lz_scan)
    for j, Omega in enumerate(grid):
        lz_scan[:, j] = lz_qfi_trajectory(cfg, N, Omega, sweep.t_max, sweep.n_steps + 1)[1:]
        full_scan[:, j] = full_qfi_trajectory(cfg, N, Omega, times)
    print(f"Scanned {len(grid)} Omega values for N={N}; refining optima...", flush=True)

    Omega_lz, Omega_full, qfi_lz, qfi_full, qfi_full_at_lz = np.empty((5, len(times)))
    lz_boundary = np.zeros(len(times), dtype=bool)
    full_boundary = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        Omega_lz[i], qfi_lz[i] = refine_maximum(
            lambda Om, t=t: lz_qfi_at(cfg, N, Om, t), grid, lz_scan[i], cfg, f"LZ, t={t:g}")
        Omega_full[i], qfi_full[i] = refine_maximum(
            lambda Om, t=t: full_qfi_at(cfg, N, Om, t), grid, full_scan[i], cfg, f"full, t={t:g}")
        qfi_full_at_lz[i] = full_qfi_at(cfg, N, Omega_lz[i], t)
        lz_boundary[i] = at_search_boundary(Omega_lz[i], cfg)
        full_boundary[i] = at_search_boundary(Omega_full[i], cfg)
        flag = " (at search boundary)" if lz_boundary[i] or full_boundary[i] else ""
        print(f"t={t:6.3f}: Omega_LZ={Omega_lz[i]:.5g}, Omega_full={Omega_full[i]:.5g}, "
              f"F_full(Omega_LZ)/F_full(opt)={qfi_full_at_lz[i] / qfi_full[i]:.4f}{flag}",
              flush=True)
    efficiency = np.divide(qfi_full_at_lz, qfi_full, out=np.full_like(qfi_full, np.nan),
                           where=qfi_full > cfg.qfi_tol)
    return SweepResult(times, grid, lz_scan, full_scan, Omega_lz, Omega_full, qfi_lz,
                       qfi_full, qfi_full_at_lz, efficiency, lz_boundary, full_boundary)


def output_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "graphs" / Path(__file__).stem


def tag(cfg: ModelConfig, sweep: TimeSweepConfig) -> str:
    return (f"N={sweep.N}__t=0-{sweep.t_max:g}-n{sweep.n_steps}__obj={cfg.objective}"
            f"__om={cfg.omega:g}__Om={cfg.Omega_min:g}-{cfg.Omega_max:g}-n{cfg.Omega_points}")


def save_results(result: SweepResult, cfg: ModelConfig, sweep: TimeSweepConfig) -> tuple[Path, Path]:
    directory = output_directory()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"values__{tag(cfg, sweep)}.csv"
    scan_path = directory / f"scan__{tag(cfg, sweep)}.npz"
    np.savetxt(
        csv_path,
        np.column_stack((result.times, result.Omega_lz, result.Omega_full, result.qfi_lz,
                         result.qfi_full, result.qfi_full_at_lz, result.efficiency,
                         result.lz_at_boundary, result.full_at_boundary)),
        delimiter=",", fmt=("%.12g",) * 7 + ("%d",) * 2,
        header="t,Omega_LZ,Omega_full,FQ_LZ,FQ_full,FQ_full_at_Omega_LZ,efficiency,"
               "LZ_at_boundary,full_at_boundary",
        comments="",
    )
    np.savez_compressed(
        scan_path, **asdict(result),
        config_json=np.asarray(json.dumps({"model": asdict(cfg), "sweep": asdict(sweep)},
                                          sort_keys=True)),
    )
    return csv_path, scan_path


def plot_results(result: SweepResult, cfg: ModelConfig, sweep: TimeSweepConfig) -> PlotRecord:
    t = result.times
    reference = np.sqrt(0.5 * cfg.J_nominal * sweep.N / t) if cfg.J_nominal >= 0 else np.full_like(t, np.nan)
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(sweep.figure_width_in,
                                sweep.figure_height_in), layout="constrained")
    omega_axis, qfi_axis, eff_axis = axes
    omega_axis.plot(t, result.Omega_lz, "o-", ms=3, label=r"$\Omega_{\mathrm{opt}}$ (LZ)")
    omega_axis.plot(t, result.Omega_full, "s-", ms=3, label=r"$\Omega_{\mathrm{opt}}$ (full)")
    omega_axis.plot(t, reference, "k--", lw=1, label=r"$\sqrt{JN/2t}$")
    omega_axis.set_ylabel(r"Optimal drive $\Omega$")
    omega_axis.set_ylim(cfg.Omega_min, cfg.Omega_max)
    omega_axis.legend()
    omega_axis.set_title("(a) Optimal central drive", loc="left")

    qfi_axis.semilogy(t, result.qfi_lz, "o-", ms=3, label=r"LZ qubit $F_Q$")
    qfi_axis.semilogy(t, result.qfi_full, "s-", ms=3,
                      label=rf"full model $F_Q^{{\mathrm{{{cfg.objective}}}}}$")
    qfi_axis.set_ylabel(r"Maximal $F_Q$")
    qfi_axis.legend()
    qfi_axis.set_title(r"(b) QFI at the respective optimum", loc="left")

    eff_axis.plot(t, result.efficiency, "o-", ms=3, color="tab:green")
    eff_axis.axhline(1.0, color="0.5", ls="--", lw=1)
    eff_axis.set_ylabel(r"$F_Q(\Omega_{\mathrm{LZ}})/F_Q(\Omega_{\mathrm{full}})$")
    eff_axis.set_ylim(0, 1.05)
    eff_axis.set_xlabel(r"Interrogation time $t$")
    eff_axis.set_title("(c) Full-model QFI retained with the LZ-optimal drive", loc="left")
    for axis in axes:
        axis.grid(True, ls=":", alpha=0.6)
    figure.suptitle(rf"$N={sweep.N}$, $J={cfg.J_nominal:g}$, $\omega={cfg.omega:g}$, "
                    rf"$\Omega\in[{cfg.Omega_min:g},{cfg.Omega_max:g}]$")

    data = {
        "Omega_opt LZ": (t, result.Omega_lz),
        "Omega_opt full": (t, result.Omega_full),
        "sqrt(JN/2t)": (t, reference),
        "F_Q LZ (max)": (t, result.qfi_lz),
        f"F_Q full {cfg.objective} (max)": (t, result.qfi_full),
        "F_Q full at Omega_LZ": (t, result.qfi_full_at_lz),
        "efficiency": (t, result.efficiency),
    }
    try:
        record = save_plot(
            figure, system="central_spin", plot_type="optimal_omega_lz_vs_full_time",
            params={**asdict(cfg), **asdict(sweep)}, data=data,
            name=f"lz_vs_full_vs_time__{tag(cfg, sweep)}",
            xlabel=r"Interrogation time $t$", ylabel="Optimal Omega / QFI / efficiency (separate panels)",
            notes="Omega maximizes each model's QFI for J at each t (LZ qubit vs full model).",
            script_path=__file__, dpi=sweep.figure_dpi, bbox_inches="tight",
        )
        if sweep.show_figure:
            plt.show()
    finally:
        plt.close(figure)
    return record


def parse_config(argv: list[str] | None = None) -> tuple[ModelConfig, TimeSweepConfig]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = TimeSweepConfig()
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument("--t-max", type=float, default=defaults.t_max)
    parser.add_argument("--n-steps", type=int, default=defaults.n_steps)
    parser.add_argument("--figure-width-in", type=float, default=defaults.figure_width_in)
    parser.add_argument("--figure-height-in", type=float, default=defaults.figure_height_in)
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument("--show", dest="show_figure", action=argparse.BooleanOptionalAction,
                        default=defaults.show_figure)
    add_model_arguments(parser)
    values = vars(parser.parse_args(argv))
    sweep_names = {f.name for f in fields(TimeSweepConfig)}
    return (ModelConfig(**{k: v for k, v in values.items() if k in MODEL_FIELDS}),
            TimeSweepConfig(**{k: v for k, v in values.items() if k in sweep_names}))


def main(argv: list[str] | None = None) -> PlotRecord:
    cfg, sweep = parse_config(argv)
    result = run_sweep(cfg, sweep)
    for path in save_results(result, cfg, sweep):
        print(f"Saved {path}")
    record = plot_results(result, cfg, sweep)
    print(f"Saved {record.json_path}")
    if record.image_path is not None:
        print(f"Saved {record.image_path}")
    return record


if __name__ == "__main__":
    main()
