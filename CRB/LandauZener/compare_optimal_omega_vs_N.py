"""Optimal Omega of the Landau-Zener toy versus the full model, as a function of N.

At a single interrogation time (default t = 1) and for every N in [N_min, N_max],
the central drive Omega is chosen to maximize the QFI for J in

  * the LZ toy   H(t) = Omega X + N J cos(2 omega t) Z      (single qubit), and
  * the full model H = Omega X + J Z S_z + omega S_x         (reduced-bath QFI),

with the same N, J, omega and initial state |+>_c |0>_b**N.  See lz_vs_full_core
for the conventions, the refinement strategy, and the ``--objective`` option.

The plot is a single log-log panel: the full-model optimal Omega with the reference
0.66*sqrt(N) (``--ref-coeff``) and a least-squares sqrt(N) fit to it.  The LZ optimum
and the transfer efficiency are still computed and saved to the CSV/NPZ, but not drawn.

Run from the repository root:
    python CRB/LandauZener/compare_optimal_omega_vs_N.py
    python CRB/LandauZener/compare_optimal_omega_vs_N.py --N-max 60 --Omega-max 15 --show

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
        full_qfi_at, lz_qfi_at, omega_grid, refine_maximum, validate_model_config,
    )
except ModuleNotFoundError:  # Allow: python CRB/LandauZener/compare_optimal_omega_vs_N.py
    from lz_vs_full_core import (  # type: ignore[no-redef]
        MODEL_FIELDS, ModelConfig, add_model_arguments, at_search_boundary,
        full_qfi_at, lz_qfi_at, omega_grid, refine_maximum, validate_model_config,
    )
from CRB.smart_save import PlotRecord, save_plot


@dataclass(frozen=True, slots=True)
class NSweepConfig:
    N_min: int = 1
    N_max: int = 40
    t: float = 1.0
    ref_coeff: float = 0.66  # Reference curve Omega* = ref_coeff * sqrt(N).
    figure_width_in: float = 7.0
    figure_height_in: float = 5.5
    figure_dpi: int = 160
    show_figure: bool = False


@dataclass(frozen=True, slots=True)
class SweepResult:
    N_values: np.ndarray
    Omega_grid: np.ndarray
    lz_scan: np.ndarray  # (n_N, n_Omega)
    full_scan: np.ndarray
    Omega_lz: np.ndarray
    Omega_full: np.ndarray
    qfi_lz: np.ndarray
    qfi_full: np.ndarray
    qfi_full_at_lz: np.ndarray
    efficiency: np.ndarray
    lz_at_boundary: np.ndarray
    full_at_boundary: np.ndarray


def validate_sweep_config(sweep: NSweepConfig) -> None:
    if not isinstance(sweep.N_min, int) or not isinstance(sweep.N_max, int):
        raise ValueError("N_min and N_max must be integers")
    if not 1 <= sweep.N_min <= sweep.N_max:
        raise ValueError("require 1 <= N_min <= N_max")
    if not np.isfinite(sweep.t) or sweep.t <= 0:
        raise ValueError("t must be positive")
    if sweep.figure_width_in <= 0 or sweep.figure_height_in <= 0 or sweep.figure_dpi <= 0:
        raise ValueError("figure dimensions and dpi must be positive")


def run_sweep(cfg: ModelConfig, sweep: NSweepConfig) -> SweepResult:
    validate_model_config(cfg)
    validate_sweep_config(sweep)
    N_values = np.arange(sweep.N_min, sweep.N_max + 1)
    grid = omega_grid(cfg)
    n = len(N_values)
    lz_scan, full_scan = np.empty((2, n, len(grid)))
    Omega_lz, Omega_full, qfi_lz, qfi_full, qfi_full_at_lz = np.empty((5, n))
    lz_boundary = np.zeros(n, dtype=bool)
    full_boundary = np.zeros(n, dtype=bool)
    t = sweep.t
    for i, N in enumerate(map(int, N_values)):
        lz_scan[i] = [lz_qfi_at(cfg, N, Om, t) for Om in grid]
        full_scan[i] = [full_qfi_at(cfg, N, Om, t) for Om in grid]
        Omega_lz[i], qfi_lz[i] = refine_maximum(
            lambda Om, N=N: lz_qfi_at(cfg, N, Om, t), grid, lz_scan[i], cfg, f"LZ, N={N}")
        Omega_full[i], qfi_full[i] = refine_maximum(
            lambda Om, N=N: full_qfi_at(cfg, N, Om, t), grid, full_scan[i], cfg, f"full, N={N}")
        qfi_full_at_lz[i] = full_qfi_at(cfg, N, Omega_lz[i], t)
        lz_boundary[i] = at_search_boundary(Omega_lz[i], cfg)
        full_boundary[i] = at_search_boundary(Omega_full[i], cfg)
        flag = " (at search boundary; consider widening Omega bounds)" if (
            lz_boundary[i] or full_boundary[i]) else ""
        print(f"N={N:3d}: Omega_LZ={Omega_lz[i]:.5g}, Omega_full={Omega_full[i]:.5g}, "
              f"F_full(Omega_LZ)/F_full(opt)={qfi_full_at_lz[i] / qfi_full[i]:.4f}{flag}",
              flush=True)
    efficiency = np.divide(qfi_full_at_lz, qfi_full, out=np.full_like(qfi_full, np.nan),
                           where=qfi_full > cfg.qfi_tol)
    return SweepResult(N_values, grid, lz_scan, full_scan, Omega_lz, Omega_full, qfi_lz,
                       qfi_full, qfi_full_at_lz, efficiency, lz_boundary, full_boundary)


def output_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "graphs" / Path(__file__).stem


def tag(cfg: ModelConfig, sweep: NSweepConfig) -> str:
    return (f"N={sweep.N_min}-{sweep.N_max}__t={sweep.t:g}__obj={cfg.objective}"
            f"__om={cfg.omega:g}__Om={cfg.Omega_min:g}-{cfg.Omega_max:g}-n{cfg.Omega_points}")


def save_results(result: SweepResult, cfg: ModelConfig, sweep: NSweepConfig) -> tuple[Path, Path]:
    directory = output_directory()
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"values__{tag(cfg, sweep)}.csv"
    scan_path = directory / f"scan__{tag(cfg, sweep)}.npz"
    np.savetxt(
        csv_path,
        np.column_stack((result.N_values, result.Omega_lz, result.Omega_full, result.qfi_lz,
                         result.qfi_full, result.qfi_full_at_lz, result.efficiency,
                         result.lz_at_boundary, result.full_at_boundary)),
        delimiter=",", fmt=("%d",) + ("%.12g",) * 6 + ("%d",) * 2,
        header="N,Omega_LZ,Omega_full,FQ_LZ,FQ_full,FQ_full_at_Omega_LZ,efficiency,"
               "LZ_at_boundary,full_at_boundary",
        comments="",
    )
    np.savez_compressed(
        scan_path, **asdict(result),
        config_json=np.asarray(json.dumps({"model": asdict(cfg), "sweep": asdict(sweep)},
                                          sort_keys=True)),
    )
    return csv_path, scan_path


def plot_results(result: SweepResult, cfg: ModelConfig, sweep: NSweepConfig) -> PlotRecord:
    """Log-log plot of the full-model optimal Omega with sqrt(N) reference curves."""
    N = result.N_values
    reference = sweep.ref_coeff * np.sqrt(N)
    fit_coeff = float(np.sum(result.Omega_full * np.sqrt(N)) / np.sum(N))  # LSQ through origin.
    figure, axis = plt.subplots(figsize=(sweep.figure_width_in, sweep.figure_height_in),
                                layout="constrained")
    axis.loglog(N, result.Omega_full, "s-", ms=4, label=r"$\Omega_{\mathrm{opt}}$ (full model)")
    axis.loglog(N, reference, "k--", lw=1.2, label=rf"${sweep.ref_coeff:g}\sqrt{{N}}$")
    axis.loglog(N, fit_coeff * np.sqrt(N), ":", color="tab:orange", lw=1.2,
                label=rf"fit to full: ${fit_coeff:.2f}\sqrt{{N}}$")
    axis.set_xlabel(r"Number of bath spins $N$")
    axis.set_ylabel(r"Optimal drive $\Omega$")
    axis.grid(True, which="both", ls=":", alpha=0.6)
    axis.legend()
    axis.set_title(rf"$t={sweep.t:g}$, $J={cfg.J_nominal:g}$, $\omega={cfg.omega:g}$; "
                   rf"$\Omega$ optimized for $F_Q^{{\mathrm{{{cfg.objective}}}}}$ in "
                   rf"$[{cfg.Omega_min:g},{cfg.Omega_max:g}]$")

    data = {
        "Omega_opt full": (N, result.Omega_full),
        f"{sweep.ref_coeff:g}*sqrt(N)": (N, reference),
        f"{fit_coeff:.3f}*sqrt(N) (LSQ fit to full)": (N, fit_coeff * np.sqrt(N)),
    }
    try:
        record = save_plot(
            figure, system="central_spin", plot_type="optimal_omega_full_vs_N_loglog",
            params={**asdict(cfg), **asdict(sweep)}, data=data,
            name=f"optimal_omega_full_vs_N_loglog__{tag(cfg, sweep)}",
            xlabel=r"Number of bath spins $N$", ylabel=r"Optimal drive $\Omega$",
            notes="Omega maximizes the full-model QFI for J at fixed t for each N; log-log axes.",
            script_path=__file__, dpi=sweep.figure_dpi, bbox_inches="tight",
        )
        if sweep.show_figure:
            plt.show()
    finally:
        plt.close(figure)
    return record


def parse_config(argv: list[str] | None = None) -> tuple[ModelConfig, NSweepConfig]:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = NSweepConfig()
    parser.add_argument("--N-min", type=int, default=defaults.N_min)
    parser.add_argument("--N-max", type=int, default=defaults.N_max)
    parser.add_argument("--t", type=float, default=defaults.t, help="Interrogation time.")
    parser.add_argument("--ref-coeff", type=float, default=defaults.ref_coeff,
                        help="Coefficient c of the reference curve Omega* = c*sqrt(N).")
    parser.add_argument("--figure-width-in", type=float, default=defaults.figure_width_in)
    parser.add_argument("--figure-height-in", type=float, default=defaults.figure_height_in)
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument("--show", dest="show_figure", action=argparse.BooleanOptionalAction,
                        default=defaults.show_figure)
    add_model_arguments(parser)
    values = vars(parser.parse_args(argv))
    sweep_names = {f.name for f in fields(NSweepConfig)}
    return (ModelConfig(**{k: v for k, v in values.items() if k in MODEL_FIELDS}),
            NSweepConfig(**{k: v for k, v in values.items() if k in sweep_names}))


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
