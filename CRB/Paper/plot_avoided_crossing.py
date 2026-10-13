"""Plot one avoided crossing of H(t) = Omega X + N J cos(omega t) Z.

This uses the Pauli-matrix convention of CRB.LandauZener.driven_qubit,
with hbar = 1 and J = 1 by default. Energies and angular frequencies use
the model's reference inverse-time unit. The first minimum gap occurs at
t_c = pi/(2 omega), where E_+ - E_- = 2 Omega. The diabatic branch slopes
are -/+ N J omega; their relative slope is v = 2 N J omega.

Run from the repository root::

    python CRB/Paper/plot_avoided_crossing.py
    python CRB/Paper/plot_avoided_crossing.py --no-show --format pdf

All exports use CRB.smart_save, under its configured
Graphs/plot_avoided_crossing/ directory. Filenames and embedded metadata
record every configuration field. The graph uses symbolic labels without
displaying numerical parameter values. No dynamics solver is needed for
these closed-form eigenenergies.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CRB.smart_save import save_plot


@dataclass(frozen=True, slots=True)
class CrossingConfig:
    """Model, sampling, and figure settings.

    N is dimensionless; J, Omega, and omega use reference inverse-time
    units. half_window_rad is the drive phase on either side of t_c.
    size contains the figure width and height in inches.
    """

    N: int = 15
    Omega: float = 2.5
    omega: float = 1.0
    J: float = 1.0
    half_window_rad: float = 0.7
    points: int = 1001
    size: tuple[float, float] = (7.4, 5.2)
    dpi: int = 200
    format: str = "png"
    show: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.N, int) or isinstance(self.N, bool) or self.N < 1:
            raise ValueError("N must be a positive integer.")
        for name in ("Omega", "omega", "J"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.half_window_rad) or not 0 < self.half_window_rad < np.pi / 2:
            raise ValueError("half_window_rad must lie between 0 and pi/2 to isolate one crossing.")
        if not isinstance(self.points, int) or self.points < 5:
            raise ValueError("points must be an integer of at least 5.")
        if len(self.size) != 2 or not all(np.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("size must contain two finite positive dimensions in inches.")
        if not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer.")
        if self.format not in ("png", "pdf", "svg"):
            raise ValueError("format must be png, pdf, or svg.")


def crossing_quantities(cfg: CrossingConfig) -> dict[str, float]:
    """Return the crossing time, minimum gap, and the two slope conventions."""
    slope = cfg.N * cfg.J * cfg.omega
    return {
        "t_c": np.pi / (2.0 * cfg.omega),
        "minimum_gap": 2.0 * cfg.Omega,
        "diabatic_plus_slope": -slope,
        "diabatic_minus_slope": slope,
        "relative_slope_v": 2.0 * slope,
    }


def compute_levels(cfg: CrossingConfig) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate the exact spectrum and the uncoupled (diabatic) energies."""
    t_c = crossing_quantities(cfg)["t_c"]
    half_window = cfg.half_window_rad / cfg.omega
    times = np.linspace(t_c - half_window, t_c + half_window, cfg.points)
    # Include the exact crossing even for an even sample count.
    times = np.unique(np.append(times, t_c))
    diabatic = cfg.N * cfg.J * np.cos(cfg.omega * times)
    diabatic[times == t_c] = 0.0  # Remove cos(pi/2) roundoff.
    energy = np.hypot(cfg.Omega, diabatic)
    return times, {
        "adiabatic_plus": energy,
        "adiabatic_minus": -energy,
        "diabatic_plus": diabatic,
        "diabatic_minus": -diabatic,
    }


def build_figure(cfg: CrossingConfig) -> Figure:
    """Draw one passage, its diabatic slopes, and the gap opened by Omega."""
    times, levels = compute_levels(cfg)
    quantities = crossing_quantities(cfg)
    t_c = quantities["t_c"]
    half_window = cfg.half_window_rad / cfg.omega
    adiabatic_color, diabatic_color = "#245c9e", "#bc6b35"
    figure, axis = plt.subplots(figsize=cfg.size, layout="constrained")

    axis.plot(
        times, levels["adiabatic_plus"], color=adiabatic_color, lw=2.4,
        label=r"Adiabatic: $E_\pm=\pm\sqrt{\Omega^2+[NJ\cos(\omega t)]^2}$",
    )
    axis.plot(times, levels["adiabatic_minus"], color=adiabatic_color, lw=2.4)
    axis.plot(
        times, levels["diabatic_plus"], color=diabatic_color, lw=1.7, ls="--",
        label=r"Diabatic: $\varepsilon_\pm=\pm NJ\cos(\omega t)$",
    )
    axis.plot(times, levels["diabatic_minus"], color=diabatic_color, lw=1.7, ls="--")

    # This is the only vertical guide: the minimum gap, not a drive extremum.
    axis.axvline(t_c, ymin=0.12, ymax=0.83, color="0.65", ls=":", lw=1.0, zorder=0)
    axis.scatter([t_c, t_c], [-cfg.Omega, cfg.Omega], s=22,
                 color=adiabatic_color, zorder=5)
    axis.annotate(
        "", xy=(t_c, cfg.Omega), xytext=(t_c, -cfg.Omega),
        arrowprops={"arrowstyle": "<->", "color": "0.15", "lw": 1.5,
                    "shrinkA": 0, "shrinkB": 0}, zorder=4,
    )
    axis.annotate(
        r"$\Delta_{\min}=2\Omega$",
        xy=(t_c, 0.0), xytext=(t_c + 0.12 * half_window, 0.0),
        va="center", fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
    )

    # A short tangent makes the slope at the diabatic intersection explicit.
    tangent_times = t_c + np.array([-0.30, 0.30]) * half_window
    tangent_energy = quantities["diabatic_plus_slope"] * (tangent_times - t_c)
    axis.plot(tangent_times, tangent_energy, color="0.35", ls=":", lw=1.3)
    slope_time = t_c - 0.25 * half_window
    axis.annotate(
        r"$\dot\varepsilon_\pm(t_c)=\mp NJ\omega$",
        xy=(slope_time, quantities["diabatic_plus_slope"] * (slope_time - t_c)),
        xytext=(0.055, 0.46), textcoords="axes fraction", fontsize=10,
        ha="left", va="center",
        arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.0,
                    "connectionstyle": "arc3,rad=-0.12"},
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
    )
    axis.text(
        0.5, 0.055,
        r"LZ relative slope: $v=|\dot\varepsilon_+-\dot\varepsilon_-|=2NJ\omega$",
        transform=axis.transAxes, ha="center", va="center", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 3.0},
    )

    energy_limit = 1.45 * float(levels["adiabatic_plus"].max())
    axis.set(
        xlim=(times[0], times[-1]), ylim=(-energy_limit, energy_limit),
        xlabel=r"Time $t$", ylabel=r"Energy $E$",
        title=r"$H(t)=\Omega X+NJ\cos(\omega t)Z$",
    )
    axis.set_xticks([t_c], labels=[r"$t_c=\frac{\pi}{2\omega}$"])
    axis.set_yticks([-cfg.Omega, cfg.Omega], labels=[r"$-\Omega$", r"$+\Omega$"])
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.15)
    axis.legend(loc="upper center", frameon=False, fontsize=10)
    return figure


def figure_filename(cfg: CrossingConfig) -> str:
    """Keep every configuration value in a deterministic, Windows-safe name."""
    def tag(value: object) -> str:
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return str(int(value)) if value.is_integer() else repr(value)
        if isinstance(value, tuple):
            return "x".join(tag(item) for item in value)
        return re.sub(r'[<>:"/\\|?*\s]+', "-", str(value))

    tags = "__".join(f"{field.name}={tag(getattr(cfg, field.name))}" for field in fields(cfg))
    return f"avoided-crossing__{tags}.{cfg.format}"


def plot_crossing(cfg: CrossingConfig) -> Path:
    """Save through the shared smart_save helper and close the figure."""
    figure = build_figure(cfg)
    try:
        path = save_plot(
            figure, figure_filename(cfg),
            metadata={
                "config": asdict(cfg),
                "hamiltonian": "Omega*X + N*J*cos(omega*t)*Z; Pauli matrices; hbar=1",
                "adiabatic_energies": "+/- sqrt(Omega**2 + (N*J*cos(omega*t))**2)",
                "diabatic_energies": "+/- N*J*cos(omega*t)",
                **crossing_quantities(cfg),
            },
            script_path=__file__, format=cfg.format, dpi=cfg.dpi, bbox_inches="tight",
        )
        if cfg.show:
            plt.show()
        return path
    finally:
        plt.close(figure)


def parse_config(argv: list[str] | None = None) -> CrossingConfig:
    """Expose every setting while keeping the dataclass as the default source."""
    defaults = CrossingConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in fields(defaults):
        default = getattr(defaults, field.name)
        flag = "--" + field.name.replace("_", "-")
        if field.name == "show":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif field.name == "size":
            parser.add_argument(flag, nargs=2, type=float, default=default, metavar=("WIDTH", "HEIGHT"))
        else:
            options = {"choices": ("png", "pdf", "svg")} if field.name == "format" else {}
            parser.add_argument(flag, type=type(default), default=default, **options)
    arguments = vars(parser.parse_args(argv))
    arguments["size"] = tuple(arguments["size"])
    return CrossingConfig(**arguments)


def main(argv: list[str] | None = None) -> Path:
    cfg = parse_config(argv)
    quantities = crossing_quantities(cfg)
    print(
        f"t_c={quantities['t_c']:.6g}; minimum gap={quantities['minimum_gap']:g}; "
        f"branch slopes=+/-{abs(quantities['diabatic_plus_slope']):g}; "
        f"LZ relative slope v={quantities['relative_slope_v']:g}"
    )
    path = plot_crossing(cfg)
    print(f"Saved avoided crossing to {path}")
    return path


if __name__ == "__main__":
    main()
