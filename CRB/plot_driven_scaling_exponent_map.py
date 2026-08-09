"""Map the driven-protocol sensitivity scaling exponent over (omega, Omega).

For every pair of drive strengths, this script repeats the bath-QFI
optimization from ``compare_ramsey_driven_vs_N.py`` for N = 10, 20, and 40.
It minimizes the fixed-total-time sensitivity

    delta_J * sqrt(T_total) = sqrt(t / F_Q(t))

over 0 < t < 40 and fits

    min_t delta_J * sqrt(T_total) = A * N**p.

The plotted quantity is the fitted exponent ``p``.  Thus, more-negative
values indicate sensitivity that improves more rapidly with bath size.

The Hamiltonian and initial state are

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x,
    |psi(0)> = |+>_central |0>_bath**N,

with J = 1.  A checkpoint is written after every (omega, Omega) point so a
long sweep can be resumed by running the script again.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.compare_ramsey_driven_vs_N import (
        ComparisonConfig,
        Protocol,
        fit_power_law,
        optimize_protocol,
    )
except ModuleNotFoundError:  # Allow: python CRB/plot_driven_scaling_exponent_map.py
    from compare_ramsey_driven_vs_N import (
        ComparisonConfig,
        Protocol,
        fit_power_law,
        optimize_protocol,
    )


@dataclass(frozen=True)
class SweepConfig:
    """Physical parameters, numerical grid, and output filenames."""

    # N_values: tuple[int, ...] = (10, 20, 40)
    N_values: tuple[int, ...] = tuple(range(2, 60, 5))

    J_nominal: float = 0.1
    dJ: float = 1e-3

    drive_min: float = 0.25
    drive_max: float = 10.0
    drive_step: float = 0.25

    # The endpoints are excluded, as requested: 0 < t < 40.
    time_min: float = 0.9
    time_max: float = 1.0
    time_step: float = 0.05
    t_overhead: float = 0.0

    gamma: float = 0.0
    beta: float = 0.0
    qfi_tol: float = 1e-12

    output_figure: str = "driven_scaling_exponent_colormap_0.1_4.png"
    checkpoint_file: str = "driven_scaling_exponent_sweep_0.1_4.npz"


def output_path(filename: str) -> Path:
    """Resolve relative outputs beside this script."""
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return an endpoint-inclusive uniform grid without floating-point drift."""
    count = int(round((stop - start) / step))
    grid = start + step * np.arange(count + 1, dtype=float)
    if not np.isclose(grid[-1], stop):
        raise ValueError("drive range must contain an integer number of steps")
    return grid


def open_time_grid(cfg: SweepConfig) -> np.ndarray:
    """Return the uniform time grid strictly inside the configured interval."""
    times = np.arange(
        cfg.time_min + cfg.time_step,
        cfg.time_max,
        cfg.time_step,
        dtype=float,
    )
    if len(times) == 0:
        raise ValueError("the open time interval contains no samples")
    return times


def comparison_config(cfg: SweepConfig, tlist: np.ndarray) -> ComparisonConfig:
    """Build the configuration expected by the reference optimization code."""
    return ComparisonConfig(
        N_values=cfg.N_values,
        J_nominal=cfg.J_nominal,
        dJ=cfg.dJ,
        t_min=float(tlist[0]),
        t_max=float(tlist[-1]),
        n_steps=len(tlist),
        t_overhead=cfg.t_overhead,
        gamma=cfg.gamma,
        beta=cfg.beta,
        qfi_tol=cfg.qfi_tol,
    )


def initialize_results(
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
    N_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create empty result arrays for a new sweep."""
    map_shape = (len(omega_values), len(Omega_values))
    return {
        "qcrb_opt": np.full(map_shape + (len(N_values),), np.nan),
        "t_opt": np.full(map_shape + (len(N_values),), np.nan),
        "prefactor": np.full(map_shape, np.nan),
        "exponent": np.full(map_shape, np.nan),
        "r_squared": np.full(map_shape, np.nan),
        "completed": np.zeros(map_shape, dtype=bool),
    }


def load_checkpoint(
    path: Path,
    cfg: SweepConfig,
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
    N_values: np.ndarray,
    tlist: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load a compatible checkpoint or initialize a new result set."""
    if not path.exists():
        return initialize_results(omega_values, Omega_values, N_values)

    with np.load(path, allow_pickle=False) as saved:
        expected = {
            "omega_values": omega_values,
            "Omega_values": Omega_values,
            "N_values": N_values,
            "tlist": tlist,
            "physical_parameters": np.asarray(
                [
                    cfg.J_nominal,
                    cfg.dJ,
                    cfg.t_overhead,
                    cfg.gamma,
                    cfg.beta,
                    cfg.qfi_tol,
                ]
            ),
        }
        for name, values in expected.items():
            if name not in saved or not np.array_equal(saved[name], values):
                raise ValueError(
                    f"Checkpoint {path} is incompatible with the current {name}"
                )

        results = {
            name: saved[name].copy()
            for name in (
                "qcrb_opt",
                "t_opt",
                "prefactor",
                "exponent",
                "r_squared",
                "completed",
            )
        }

    print(f"Resuming checkpoint {path}")
    return results


def save_checkpoint(
    path: Path,
    cfg: SweepConfig,
    results: dict[str, np.ndarray],
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
    N_values: np.ndarray,
    tlist: np.ndarray,
) -> None:
    """Atomically save all sweep data."""
    temporary_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        omega_values=omega_values,
        Omega_values=Omega_values,
        N_values=N_values,
        tlist=tlist,
        physical_parameters=np.asarray(
            [
                cfg.J_nominal,
                cfg.dJ,
                cfg.t_overhead,
                cfg.gamma,
                cfg.beta,
                cfg.qfi_tol,
            ]
        ),
        **results,
    )
    temporary_path.replace(path)


def run_sweep(cfg: SweepConfig) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Compute or resume the full drive-parameter sweep."""
    omega_values = inclusive_grid(cfg.drive_min, cfg.drive_max, cfg.drive_step)
    Omega_values = inclusive_grid(cfg.drive_min, cfg.drive_max, cfg.drive_step)
    N_values = np.asarray(cfg.N_values, dtype=int)
    tlist = open_time_grid(cfg)
    optimization_cfg = comparison_config(cfg, tlist)
    checkpoint_path = output_path(cfg.checkpoint_file)
    results = load_checkpoint(
        checkpoint_path,
        cfg,
        omega_values,
        Omega_values,
        N_values,
        tlist,
    )

    total_points = len(omega_values) * len(Omega_values)
    for i_omega, omega in enumerate(omega_values):
        for i_Omega, Omega in enumerate(Omega_values):
            if results["completed"][i_omega, i_Omega]:
                continue

            point_number = i_omega * len(Omega_values) + i_Omega + 1
            print(
                f"[{point_number:3d}/{total_points}] "
                f"omega={omega:.2f}, Omega={Omega:.2f}"
            )
            protocol = Protocol(
                name="Driven",
                Omega=float(Omega),
                omega=float(omega),
                bath_theta=0.0,
                marker="s",
            )

            for i_N, N in enumerate(N_values):
                t_opt, qcrb_opt = optimize_protocol(
                    N=int(N),
                    protocol=protocol,
                    tlist=tlist,
                    cfg=optimization_cfg,
                )
                results["t_opt"][i_omega, i_Omega, i_N] = t_opt
                results["qcrb_opt"][i_omega, i_Omega, i_N] = qcrb_opt
                print(
                    f"    N={N:2d}: t_opt={t_opt:6.2f}, "
                    f"QCRB_opt={qcrb_opt:.6e}"
                )

            prefactor, exponent, r_squared = fit_power_law(
                N_values,
                results["qcrb_opt"][i_omega, i_Omega],
            )
            results["prefactor"][i_omega, i_Omega] = prefactor
            results["exponent"][i_omega, i_Omega] = exponent
            results["r_squared"][i_omega, i_Omega] = r_squared
            results["completed"][i_omega, i_Omega] = True
            print(f"    fit exponent p={exponent:+.4f}, R^2={r_squared:.4f}")

            save_checkpoint(
                checkpoint_path,
                cfg,
                results,
                omega_values,
                Omega_values,
                N_values,
                tlist,
            )

    return results, omega_values, Omega_values


def plot_exponent_map(
    results: dict[str, np.ndarray],
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
    cfg: SweepConfig,
) -> Path:
    """Plot the fitted QCRB scaling exponent over the two drive strengths."""
    exponent = np.ma.masked_invalid(results["exponent"])
    if exponent.count() == 0:
        raise RuntimeError("No finite scaling exponents were obtained")

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad(color="lightgray")

    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.pcolormesh(
        Omega_values,
        omega_values,
        exponent,
        shading="nearest",
        cmap=cmap,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"Scaling exponent $p$ in $\mathrm{QCRB}_{\min}\propto N^p$")

    axis.set_xlabel(r"Central-spin drive $\Omega/J$")
    axis.set_ylabel(r"Bath drive $\omega/J$")
    axis.set_xlim(cfg.drive_min, cfg.drive_max)
    axis.set_ylim(cfg.drive_min, cfg.drive_max)
    axis.set_aspect("equal")
    axis.set_xticks(np.arange(cfg.drive_min, cfg.drive_max + 0.5, 0.5))
    axis.set_yticks(np.arange(cfg.drive_min, cfg.drive_max + 0.5, 0.5))
    axis.set_title(
        "Driven bath-sensing scaling\n"
        rf"$J=1$, $N={list(cfg.N_values)}$, ${cfg.time_min:g}<t<{cfg.time_max:g}$"
    )
    figure.tight_layout()

    figure_path = output_path(cfg.output_figure)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return figure_path


def main() -> None:
    cfg = SweepConfig()
    results, omega_values, Omega_values = run_sweep(cfg)
    figure_path = plot_exponent_map(results, omega_values, Omega_values, cfg)
    print(f"Saved scaling-exponent map to {figure_path}")
    print(f"Saved numerical sweep to {output_path(cfg.checkpoint_file)}")


if __name__ == "__main__":
    main()
