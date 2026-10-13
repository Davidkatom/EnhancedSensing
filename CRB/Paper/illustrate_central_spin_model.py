"""Create the paper's opening central-spin schematic as PNG, PDF, and SVG.

Run from the repository root::

    python CRB/Paper/illustrate_central_spin_model.py
    python CRB/Paper/illustrate_central_spin_model.py --spins 10 --dpi 600

The coupling convention matches ``CRB.crb_core``. Bath arrows indicate
polarization along +z; the central spin points transversely. No simulation is
performed. Only the model geometry and local drive symbols are shown.
All exports use smart_save's metadata-aware image writer and live under
``graphs/illustrate_central_spin_model/``. The lower-level writer is used
because the public numerical-plot API fixes its destination to Google Drive
and requires numerical curves; neither is appropriate for a paper schematic.
The shared smart_save configuration is never changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from matplotlib.path import Path as DrawingPath
from matplotlib.transforms import Affine2D
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from CRB import smart_save
from CRB.crb_core import build_spin_operators


@dataclass(frozen=True, slots=True)
class IllustrationConfig:
    """Model size and export settings; geometric coordinates are design units.

    N is the number of bath nodes drawn, with no indices displayed.
    The figure is designed for a single paper column at its default size.
    """

    N: int = 8
    w_in: float = 3.6
    h_in: float = 3.6
    dpi: int = 600
    font_pt: float = 9.0
    palette: str = "color"
    formats: tuple[str, ...] = ("png", "pdf", "svg")
    transparent: bool = False


def validate_config(cfg: IllustrationConfig) -> None:
    if not 3 <= cfg.N <= 16:
        raise ValueError("--spins must be between 3 and 16 for a legible schematic")
    sizes = (cfg.w_in, cfg.h_in, cfg.font_pt)
    if not all(np.isfinite(value) and value > 0 for value in sizes):
        raise ValueError("Figure dimensions and font size must be finite and positive")
    if cfg.dpi < 72:
        raise ValueError("--dpi must be at least 72 (600 is recommended for the paper)")
    if cfg.palette not in ("color", "mono"):
        raise ValueError("--palette must be 'color' or 'mono'")
    if not cfg.formats or any(fmt not in ("png", "pdf", "svg") for fmt in cfg.formats):
        raise ValueError("--formats must contain png, pdf, and/or svg")
    if len(set(cfg.formats)) != len(cfg.formats):
        raise ValueError("--formats must not contain duplicates")


def _colors(cfg: IllustrationConfig) -> dict[str, str]:
    if cfg.palette == "mono":
        return dict(ink="#242424", bath="#656565", central="#242424",
                    link="#9C9C9C")
    return dict(ink="#243644", bath="#167D9A", central="#D5763F",
                link="#9BAFB9")


def _spin(ax: Axes, xy: tuple[float, float], radius: float, color: str,
          label: str | None = None, font_pt: float = 9.0,
          direction: tuple[float, float] = (1.0, 0.0)) -> None:
    """A vector spin glyph with an explicit polarization direction."""
    ax.add_patch(Circle(xy, radius, facecolor=color, edgecolor="white",
                        linewidth=1.2, zorder=5))
    x, y = xy
    arrow_direction = np.asarray(direction, dtype=float)
    arrow_direction /= np.linalg.norm(arrow_direction)
    perpendicular = np.array([-arrow_direction[1], arrow_direction[0]])
    # Define the complete glyph in data coordinates. Equal tip/tail extents
    # keep its bounding box centered on the spin, independent of DPI or size.
    arrow_shape = np.array([
        (-0.64, -0.075), (0.12, -0.075), (0.12, -0.20), (0.64, 0.0),
        (0.12, 0.20), (0.12, 0.075), (-0.64, 0.075),
    ])
    vertices = np.array([x, y]) + radius * (
        arrow_shape[:, :1] * arrow_direction
        + arrow_shape[:, 1:] * perpendicular
    )
    ax.add_patch(Polygon(vertices, closed=True, facecolor="white",
                         edgecolor="none", zorder=6))
    if label:
        ax.text(x, y - 0.43 * radius, label, color="white", fontsize=font_pt,
                ha="center", va="center", zorder=7)


def _rotation(ax: Axes, xy: tuple[float, float], radius: float,
              start_deg: float, stop_deg: float, color: str) -> None:
    """Draw a single continuous circular arrow centered on the spin."""
    x, y = xy
    arc_path = DrawingPath.arc(start_deg, stop_deg).transformed(
        Affine2D().scale(radius).translate(x, y)
    )
    # One curved path aligns the arrowhead with the arc's actual endpoint.
    # Separate Arc/FancyArrowPatch objects introduced a gap and an overshoot.
    ax.add_patch(FancyArrowPatch(
        path=arc_path, arrowstyle="-|>", mutation_scale=9,
        shrinkA=0, shrinkB=0,
        linewidth=1.25, color=color, zorder=4,
    ))


def build_illustration(cfg: IllustrationConfig) -> Figure:
    """Return the single-panel model, with upward bath spins and local drives."""
    validate_config(cfg)
    colors = _colors(cfg)
    style = {
        "font.family": "DejaVu Sans", "font.size": cfg.font_pt,
        "mathtext.fontset": "stix", "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(cfg.w_in, cfg.h_in), facecolor="white")
        ax = fig.add_axes((0.025, 0.025, 0.95, 0.95))
        ax.set(xlim=(-3.9, 3.9), ylim=(-3.9, 3.9), aspect="equal")
        ax.axis("off")

        def text(x: float, y: float, value: str, **kwargs: object) -> None:
            ax.text(x, y, value, color=colors["ink"], va="center", **kwargs)

        center = np.array([0.0, 0.0])
        bath_radius = 2.50
        boundary_radius = 3.47
        ax.add_patch(Circle(center, boundary_radius, fill=False,
                            edgecolor=colors["bath"], linestyle=(0, (3.5, 3.5)),
                            linewidth=0.8, alpha=0.45, zorder=1))
        # Stagger nodes around the vertical axis for a balanced composition.
        angles = np.pi / 2 + 2 * np.pi * (
            np.arange(cfg.N) - cfg.N // 4 + 0.5) / cfg.N
        positions = center + bath_radius * np.column_stack(
            (np.cos(angles), np.sin(angles)))
        # Keep default nodes visually substantial while accommodating dense rings.
        # Leave enough room between the local drive glyphs on dense rings.
        node_radius = min(0.34, 0.55 * bath_radius * np.sin(np.pi / cfg.N))
        for xy in positions:
            ax.add_patch(FancyArrowPatch(
                center, xy, arrowstyle="-", shrinkA=0, shrinkB=0,
                linewidth=1.0, color=colors["link"], zorder=2,
            ))
            _spin(ax, tuple(xy), node_radius, colors["bath"], direction=(0.0, 1.0))
            drive_radius = 1.60 * node_radius
            _rotation(ax, tuple(xy), drive_radius, 210, 330, colors["bath"])
            ax.text(xy[0], xy[1] - drive_radius - 0.25, r"$\omega$",
                    color=colors["bath"], ha="center", va="center",
                    fontsize=cfg.font_pt + 1)

        _spin(ax, tuple(center), 0.60, colors["central"], r"$c$", cfg.font_pt)
        # Label one representative bond; all bonds have the same coefficient J.
        bond = center + 1.50 * np.array([np.cos(angles[0]), np.sin(angles[0])])
        text(*bond, r"$J$", ha="center", fontsize=cfg.font_pt + 3,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.4), zorder=8)

        _rotation(ax, tuple(center), 0.98, 210, 330, colors["central"])
        ax.text(center[0], center[1] - 1.31, r"$\Omega$",
                color=colors["central"], ha="center", va="center",
                fontsize=cfg.font_pt + 3)
        return fig


def parameter_tags(cfg: IllustrationConfig) -> str:
    """Encode the complete configuration in stable, Windows-safe file tags."""
    def value_tag(value: object) -> str:
        if isinstance(value, bool):
            return str(int(value))
        if isinstance(value, float):
            return str(value).removesuffix(".0")
        if isinstance(value, tuple):
            return "-".join(map(str, value))
        return str(value)

    return "__".join(
        f"{field.name}={re.sub(r'[^A-Za-z0-9.=_-]', '-', value_tag(getattr(cfg, field.name)))}"
        for field in fields(cfg)
    )


def save_illustration(fig: Figure, cfg: IllustrationConfig) -> tuple[Path, ...]:
    """Save all formats through smart_save, with embedded model provenance."""
    validate_config(cfg)
    script = Path(__file__).resolve()
    output_dir = script.parents[2] / "graphs" / script.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    operators = build_spin_operators(cfg.N)
    created = datetime.now(timezone.utc).isoformat()
    stem = f"fig1_central_spin__{parameter_tags(cfg)}"
    paths = []
    # Reuse the exact PNG/PDF/SVG metadata serializer used by save_plot.
    # No numerical data is invented to make an illustration look like a curve.
    for fmt in cfg.formats:
        path = output_dir / f"{stem}.{fmt}"
        payload = {
            "schema_version": 1, "saved_at_utc": created,
            "source_script": str(script), "script_name": script.stem,
            "figure": {"filename": path.name, "format": fmt, "dpi": cfg.dpi,
                       "size_inches": [cfg.w_in, cfg.h_in]},
            "parameters": asdict(cfg),
            "model": {
                "source": "CRB.crb_core.build_spin_operators",
                "hamiltonian": "Omega*sigma_x + J*sigma_z*S_z + omega*S_x",
                "collective_spin": "S_alpha = sum_i sigma_alpha^(i) = 2*j_alpha",
                "bath_symmetric_dimension": operators["dim_bath"],
                "estimated_parameter": "J", "bath_polarization": "+z (up)",
                "central_spin_direction": "+x (transverse)",
                "bath_drives": "one local omega control arrow per bath spin",
            },
        }
        with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
            smart_save._savefig_with_payload(
                fig, path, payload, figure_format=fmt, saved_at=created,
                format=fmt, dpi=cfg.dpi,
                facecolor="none" if cfg.transparent else "white",
                transparent=cfg.transparent,
                metadata={"Title": "Driven central spin with an upward-polarized bath"},
            )
        paths.append(path)
    return tuple(paths)


def parse_args(argv: list[str] | None = None) -> tuple[IllustrationConfig, bool]:
    defaults = IllustrationConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spins", dest="N", type=int, default=defaults.N,
                        help="Number of bath spins drawn, 3 to 16 (default: 8)")
    parser.add_argument("--width-in", dest="w_in", type=float, default=defaults.w_in)
    parser.add_argument("--height-in", dest="h_in", type=float, default=defaults.h_in)
    parser.add_argument("--dpi", type=int, default=defaults.dpi)
    parser.add_argument("--font-pt", type=float, default=defaults.font_pt)
    parser.add_argument("--palette", choices=("color", "mono"), default=defaults.palette)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=defaults.formats)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--show", action="store_true", help="Open a Matplotlib preview")
    args = vars(parser.parse_args(argv))
    show = args.pop("show")
    args["formats"] = tuple(args["formats"])
    cfg = IllustrationConfig(**args)
    validate_config(cfg)
    return cfg, show


def main(argv: list[str] | None = None) -> tuple[Path, ...]:
    cfg, show = parse_args(argv)
    fig = build_illustration(cfg)
    try:
        paths = save_illustration(fig, cfg)
        for path in paths:
            print(f"Saved {path}")
        if show:
            plt.show()
        return paths
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
