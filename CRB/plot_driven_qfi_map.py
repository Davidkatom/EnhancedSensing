"""Sweep the driven-protocol bath QFI over central and bath drive strengths.

The estimated parameter is ``J`` in

    H = Omega * sigma_x + J * sigma_z * S_z + omega * S_x,

using the initial state

    |psi(0)> = |theta_c>_central |theta_b>_bath**N.

Each grid point evaluates the reduced-bath quantum Fisher information at one
fixed interrogation time.  Checkpoints make long sweeps safely resumable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    from CRB.crb_core import (
        coherent_bath_state,
        evolve_bath_density_matrix_noiseless,
        qfi_from_rho_and_drho,
        save_plot,
    )
except ModuleNotFoundError:  # Allow: python CRB/plot_driven_qfi_map.py
    from crb_core import (
        coherent_bath_state,
        evolve_bath_density_matrix_noiseless,
        qfi_from_rho_and_drho,
        save_plot,
    )


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Physics, numerical, checkpoint, and visualization parameters."""

    N: int = 15
    interrogation_time: float = 1.0
    J_nominal: float = 1.0
    dJ: float = 1e-3

    drive_min: float = 0.1
    drive_max: float = 5.0
    drive_step: float = 0.25

    central_theta_rad: float = np.pi / 2.0
    bath_theta_rad: float = 0.0
    qfi_tol: float = 1e-12

    colormap: str = "viridis"
    figure_width_in: float = 8.5
    figure_height_in: float = 7.0
    figure_dpi: int = 200
    figure_format: str = "png"

    checkpoint_every_points: int = 1
    checkpoint_replace_retries: int = 5
    checkpoint_retry_base_s: float = 0.05


def validate_config(cfg: SweepConfig) -> None:
    """Fail early with actionable messages for invalid configurations."""
    if cfg.N < 1:
        raise ValueError("N must be a positive integer")
    if cfg.interrogation_time < 0.0:
        raise ValueError("interrogation_time must be non-negative")
    if cfg.dJ <= 0.0:
        raise ValueError("dJ must be positive")
    if cfg.drive_min < 0.0:
        raise ValueError("drive_min must be non-negative")
    if cfg.drive_max < cfg.drive_min:
        raise ValueError("drive_max must be greater than or equal to drive_min")
    if cfg.drive_step <= 0.0:
        raise ValueError("drive_step must be positive")
    if cfg.qfi_tol <= 0.0:
        raise ValueError("qfi_tol must be positive")
    if cfg.figure_width_in <= 0.0 or cfg.figure_height_in <= 0.0:
        raise ValueError("figure dimensions must be positive")
    if cfg.figure_dpi <= 0:
        raise ValueError("figure_dpi must be positive")
    if not cfg.figure_format or cfg.figure_format.startswith("."):
        raise ValueError("figure_format must be an extension without a leading dot")
    if cfg.checkpoint_every_points < 1:
        raise ValueError("checkpoint_every_points must be positive")
    if cfg.checkpoint_replace_retries < 0:
        raise ValueError("checkpoint_replace_retries must be non-negative")
    if cfg.checkpoint_retry_base_s < 0.0:
        raise ValueError("checkpoint_retry_base_s must be non-negative")
    if cfg.colormap not in plt.colormaps():
        raise ValueError(f"unknown Matplotlib colormap: {cfg.colormap}")


def bounded_uniform_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Return ``start + k*step`` values that do not exceed ``stop``."""
    count = int(np.floor((stop - start) / step + 1e-12)) + 1
    return start + step * np.arange(count, dtype=float)


def format_number(value: float | int) -> str:
    """Format a number compactly and deterministically for a filename."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.12g}"


def format_angle(value: float) -> str:
    """Use readable names for common angles and compact numbers otherwise."""
    common_angles = {
        0.0: "0",
        np.pi / 2.0: "pi_2",
        np.pi: "pi",
        2.0 * np.pi: "2pi",
    }
    for angle, label in common_angles.items():
        if np.isclose(value, angle):
            return label
    return format_number(value)


def sanitize_tag(text: str) -> str:
    """Replace characters that are unsafe or awkward in filenames."""
    return re.sub(r"[^A-Za-z0-9._=+-]", "-", text).strip(".-")


def parameter_tags(cfg: SweepConfig) -> str:
    """Encode every configuration field in a stable, readable order."""
    tags = (
        f"N={cfg.N}",
        f"t={format_number(cfg.interrogation_time)}",
        f"J={format_number(cfg.J_nominal)}",
        f"dJ={format_number(cfg.dJ)}",
        (
            f"drive={format_number(cfg.drive_min)}-"
            f"{format_number(cfg.drive_max)}-step{format_number(cfg.drive_step)}"
        ),
        f"theta_c={format_angle(cfg.central_theta_rad)}",
        f"theta_b={format_angle(cfg.bath_theta_rad)}",
        f"tol={format_number(cfg.qfi_tol)}",
        f"cmap={cfg.colormap}",
        (
            f"figure={format_number(cfg.figure_width_in)}x"
            f"{format_number(cfg.figure_height_in)}in-"
            f"{cfg.figure_dpi}dpi-{cfg.figure_format}"
        ),
        (
            f"checkpoint=every{cfg.checkpoint_every_points}-"
            f"retries{cfg.checkpoint_replace_retries}-"
            f"base{format_number(cfg.checkpoint_retry_base_s)}s"
        ),
    )
    return "__".join(sanitize_tag(tag) for tag in tags)


def output_directory() -> Path:
    """Return ``graphs/<script-stem>/`` at the repository root."""
    repository_root = Path(__file__).resolve().parent.parent
    path = repository_root / "graphs" / Path(__file__).stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_paths(cfg: SweepConfig) -> tuple[Path, Path]:
    """Return traceable figure and checkpoint paths for ``cfg``."""
    tags = parameter_tags(cfg)
    directory = output_directory()
    figure_path = directory / f"qfi-map__{tags}.{cfg.figure_format}"
    checkpoint_path = directory / f"checkpoint__{tags}.npz"
    return figure_path, checkpoint_path


def config_signature(cfg: SweepConfig) -> str:
    """Return a deterministic checkpoint-compatibility signature."""
    return json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))


def load_legacy_checkpoint(
    cfg: SweepConfig,
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Import the old CRB checkpoint format when all physics data match."""
    legacy_path = Path(__file__).resolve().parent / "driven_qfi_sweep_N15_t1.npz"
    candidates = (
        legacy_path,
        legacy_path.with_suffix(".tmp.npz"),
    )
    expected_parameters = np.asarray(
        [
            cfg.N,
            cfg.interrogation_time,
            cfg.J_nominal,
            cfg.dJ,
            cfg.qfi_tol,
        ],
        dtype=float,
    )
    expected_shape = (len(omega_values), len(Omega_values))
    valid: list[tuple[int, int, Path, np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with np.load(candidate, allow_pickle=False) as saved:
                compatible = (
                    np.array_equal(saved["physical_parameters"], expected_parameters)
                    and np.array_equal(saved["omega_values"], omega_values)
                    and np.array_equal(saved["Omega_values"], Omega_values)
                )
                if not compatible:
                    continue
                qfi = saved["qfi"].copy()
                completed = saved["completed"].copy()
            if qfi.shape != expected_shape or completed.shape != expected_shape:
                continue
        except (KeyError, OSError, ValueError):
            continue

        valid.append(
            (
                int(np.count_nonzero(completed)),
                candidate.stat().st_mtime_ns,
                candidate,
                qfi,
                completed,
            )
        )

    if not valid:
        return None

    completed_count, _, selected, qfi, completed = max(valid)
    print(
        f"Importing compatible legacy checkpoint {selected} "
        f"({completed_count}/{completed.size} points complete)"
    )
    return qfi, completed


def load_checkpoint(
    path: Path,
    cfg: SweepConfig,
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the most complete valid main or temporary checkpoint."""
    shape = (len(omega_values), len(Omega_values))
    temporary_path = path.with_suffix(".tmp.npz")
    candidates = [candidate for candidate in (path, temporary_path) if candidate.exists()]
    if not candidates:
        legacy = load_legacy_checkpoint(cfg, omega_values, Omega_values)
        if legacy is not None:
            return legacy
        return np.full(shape, np.nan), np.zeros(shape, dtype=bool)

    expected_signature = config_signature(cfg)
    valid: list[tuple[int, int, Path, np.ndarray, np.ndarray]] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            with np.load(candidate, allow_pickle=False) as saved:
                if saved["config_signature"].item() != expected_signature:
                    raise ValueError("configuration does not match")
                if not np.array_equal(saved["omega_values"], omega_values):
                    raise ValueError("omega grid does not match")
                if not np.array_equal(saved["Omega_values"], Omega_values):
                    raise ValueError("Omega grid does not match")
                qfi = saved["qfi"].copy()
                completed = saved["completed"].copy()
            if qfi.shape != shape or completed.shape != shape:
                raise ValueError("result dimensions do not match")
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"{candidate}: {error}")
            continue

        valid.append(
            (
                int(np.count_nonzero(completed)),
                candidate.stat().st_mtime_ns,
                candidate,
                qfi,
                completed,
            )
        )

    if not valid:
        details = "\n".join(errors)
        raise ValueError(f"No compatible checkpoint could be loaded:\n{details}")

    completed_count, _, selected, qfi, completed = max(valid)
    print(
        f"Resuming checkpoint {selected} "
        f"({completed_count}/{completed.size} points complete)"
    )
    return qfi, completed


def save_checkpoint(
    path: Path,
    cfg: SweepConfig,
    qfi: np.ndarray,
    completed: np.ndarray,
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
    promote: bool,
) -> bool:
    """Save progress and atomically promote it when Windows permits."""
    temporary_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        config_signature=np.asarray(config_signature(cfg)),
        omega_values=omega_values,
        Omega_values=Omega_values,
        qfi=qfi,
        completed=completed,
    )
    if not promote:
        return False

    for attempt in range(cfg.checkpoint_replace_retries + 1):
        try:
            temporary_path.replace(path)
            return True
        except PermissionError:
            if attempt == cfg.checkpoint_replace_retries:
                break
            time.sleep(cfg.checkpoint_retry_base_s * 2**attempt)

    print(
        f"WARNING: Windows is locking {path}; current progress is safe in "
        f"{temporary_path}. The sweep will continue."
    )
    return False


def qfi_at_drive_pair(
    Omega: float,
    omega: float,
    cfg: SweepConfig,
    bath_state: np.ndarray,
) -> float:
    """Compute reduced-bath QFI for one drive pair."""
    common = {
        "Omega_0": Omega,
        "omega": omega,
        "time": cfg.interrogation_time,
        "N": cfg.N,
        "bath_state": bath_state,
        "central_theta": cfg.central_theta_rad,
    }
    rho_plus = evolve_bath_density_matrix_noiseless(
        J=cfg.J_nominal + cfg.dJ,
        **common,
    )
    rho_minus = evolve_bath_density_matrix_noiseless(
        J=cfg.J_nominal - cfg.dJ,
        **common,
    )
    rho = 0.5 * (rho_plus + rho_minus)
    drho = (rho_plus - rho_minus) / (2.0 * cfg.dJ)
    qfi, _ = qfi_from_rho_and_drho(rho, drho, tol=cfg.qfi_tol)
    return float(qfi)


def run_sweep(
    cfg: SweepConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Compute or resume the full drive-parameter sweep."""
    omega_values = bounded_uniform_grid(
        cfg.drive_min,
        cfg.drive_max,
        cfg.drive_step,
    )
    Omega_values = omega_values.copy()
    _, checkpoint_path = output_paths(cfg)
    qfi, completed = load_checkpoint(
        checkpoint_path,
        cfg,
        omega_values,
        Omega_values,
    )
    bath_state = coherent_bath_state(cfg.N, theta=cfg.bath_theta_rad)

    if not np.isclose(omega_values[-1], cfg.drive_max):
        print(
            f"Drive step does not land exactly on drive_max={cfg.drive_max:g}; "
            f"the final sampled value is {omega_values[-1]:g}."
        )

    total_points = completed.size
    completed_count = int(np.count_nonzero(completed))
    promotion_available = True
    for i_omega, omega in enumerate(omega_values):
        for i_Omega, Omega in enumerate(Omega_values):
            if completed[i_omega, i_Omega]:
                continue

            point_number = i_omega * len(Omega_values) + i_Omega + 1
            print(
                f"[{point_number:4d}/{total_points}] "
                f"omega={omega:.6g}, Omega={Omega:.6g}"
            )
            qfi[i_omega, i_Omega] = qfi_at_drive_pair(
                Omega=float(Omega),
                omega=float(omega),
                cfg=cfg,
                bath_state=bath_state,
            )
            completed[i_omega, i_Omega] = True
            completed_count += 1
            print(f"    F_Q={qfi[i_omega, i_Omega]:.6e}")

            if completed_count % cfg.checkpoint_every_points == 0:
                promotion_available = save_checkpoint(
                    checkpoint_path,
                    cfg,
                    qfi,
                    completed,
                    omega_values,
                    Omega_values,
                    promote=promotion_available,
                )

    save_checkpoint(
        checkpoint_path,
        cfg,
        qfi,
        completed,
        omega_values,
        Omega_values,
        promote=True,
    )
    return qfi, omega_values, Omega_values, checkpoint_path


def plot_qfi_map(
    qfi: np.ndarray,
    omega_values: np.ndarray,
    Omega_values: np.ndarray,
    cfg: SweepConfig,
) -> Path:
    """Save the QFI map under the traceable graph path for ``cfg``."""
    plotted_qfi = np.ma.masked_invalid(qfi)
    if plotted_qfi.count() == 0:
        raise RuntimeError("No finite QFI values were obtained")

    cmap = plt.colormaps[cfg.colormap].copy()
    cmap.set_bad(color="lightgray")
    figure, axis = plt.subplots(
        figsize=(cfg.figure_width_in, cfg.figure_height_in)
    )
    image = axis.pcolormesh(
        Omega_values,
        omega_values,
        plotted_qfi,
        shading="nearest",
        cmap=cmap,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"Bath quantum Fisher information $F_Q$")

    axis.set_xlabel(r"Central-spin drive $\Omega$")
    axis.set_ylabel(r"Bath drive $\omega$")
    axis.set_xlim(cfg.drive_min, cfg.drive_max)
    axis.set_ylim(cfg.drive_min, cfg.drive_max)
    axis.set_aspect("equal")
    axis.set_title(
        "Driven bath-sensing QFI\n"
        rf"$J={cfg.J_nominal:g}$, $N={cfg.N}$, "
        rf"$t={cfg.interrogation_time:g}$"
    )
    figure.tight_layout()

    figure_path, _ = output_paths(cfg)
    figure_path = save_plot(
        figure,
        figure_path,
        metadata={
            "config": cfg,
            "central_drive_values": Omega_values,
            "bath_drive_values": omega_values,
            "bath_qfi": qfi,
        },
        script_path=__file__,
        format=cfg.figure_format,
        dpi=cfg.figure_dpi,
        bbox_inches="tight",
    )
    plt.close(figure)
    return figure_path


def parse_config(argv: list[str] | None = None) -> SweepConfig:
    """Parse useful command-line overrides into the typed configuration."""
    defaults = SweepConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=defaults.N)
    parser.add_argument(
        "--time",
        dest="interrogation_time",
        type=float,
        default=defaults.interrogation_time,
    )
    parser.add_argument("--J", dest="J_nominal", type=float, default=defaults.J_nominal)
    parser.add_argument("--dJ", type=float, default=defaults.dJ)
    parser.add_argument("--drive-min", type=float, default=defaults.drive_min)
    parser.add_argument("--drive-max", type=float, default=defaults.drive_max)
    parser.add_argument("--drive-step", type=float, default=defaults.drive_step)
    parser.add_argument(
        "--central-theta-rad",
        type=float,
        default=defaults.central_theta_rad,
    )
    parser.add_argument(
        "--bath-theta-rad",
        type=float,
        default=defaults.bath_theta_rad,
    )
    parser.add_argument("--qfi-tol", type=float, default=defaults.qfi_tol)
    parser.add_argument("--colormap", default=defaults.colormap)
    parser.add_argument("--figure-width-in", type=float, default=defaults.figure_width_in)
    parser.add_argument("--figure-height-in", type=float, default=defaults.figure_height_in)
    parser.add_argument("--figure-dpi", type=int, default=defaults.figure_dpi)
    parser.add_argument("--figure-format", default=defaults.figure_format)
    parser.add_argument(
        "--checkpoint-every-points",
        type=int,
        default=defaults.checkpoint_every_points,
    )
    parser.add_argument(
        "--checkpoint-replace-retries",
        type=int,
        default=defaults.checkpoint_replace_retries,
    )
    parser.add_argument(
        "--checkpoint-retry-base-s",
        type=float,
        default=defaults.checkpoint_retry_base_s,
    )
    return SweepConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
    cfg = parse_config(argv)
    validate_config(cfg)
    qfi, omega_values, Omega_values, checkpoint_path = run_sweep(cfg)
    figure_path = plot_qfi_map(qfi, omega_values, Omega_values, cfg)
    print(f"Saved QFI map to {figure_path}")
    print(f"Saved numerical sweep to {checkpoint_path}")


if __name__ == "__main__":
    main()
